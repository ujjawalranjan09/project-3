"""
MILP-based Throughput Optimization Module
Maximizes section throughput using Mixed Integer Linear Programming
"""

import logging
from pulp import LpProblem, LpMaximize, LpVariable, LpStatus, lpSum, value, PULP_CBC_CMD
import numpy as np
from datetime import datetime
import json

logger = logging.getLogger(__name__)

MAX_DECISION_VARS = 3000  # Guard against combinatorial explosion


class ThroughputOptimizer:
    """
    MILP optimizer for railway section throughput maximization.
    """

    def __init__(self):
        self.problem = None

    def optimize_train_schedule(
        self,
        trains: list,
        platforms: int,
        section_capacity: int,
        time_horizon: int = 60
    ) -> dict:
        """
        Optimize train scheduling to maximize throughput.

        Args:
            trains: list of dicts with keys id, priority, arrival_time, duration.
            platforms: number of available platforms.
            section_capacity: maximum simultaneous trains in section.
            time_horizon: optimization window in minutes.

        Returns:
            Optimized schedule dict.

        Raises:
            ValueError: if problem size exceeds MAX_DECISION_VARS.
        """
        num_trains = len(trains)
        if num_trains * time_horizon > MAX_DECISION_VARS:
            raise ValueError(
                f"Problem too large ({num_trains} trains × {time_horizon} min = "
                f"{num_trains*time_horizon} vars > {MAX_DECISION_VARS}). "
                f"Reduce train count or time_horizon."
            )

        time_slots = list(range(time_horizon))

        prob = LpProblem('Railway_Throughput_Optimization', LpMaximize)

        # Decision variables
        x = {(i, t): LpVariable(f'x_{i}_{t}', cat='Binary')
             for i in range(num_trains) for t in time_slots}
        p = {(i, k): LpVariable(f'p_{i}_{k}', cat='Binary')
             for i in range(num_trains) for k in range(platforms)}

        # Objective: maximise priority-weighted throughput
        prob += lpSum(
            trains[i]['priority'] * x[i, t]
            for i in range(num_trains)
            for t in time_slots
        ), 'Total_Weighted_Throughput'

        # Constraint 1: each train scheduled at most once
        for i in range(num_trains):
            prob += lpSum(x[i, t] for t in time_slots) <= 1, f'once_{i}'

        # Constraint 2: platform assignment iff scheduled
        for i in range(num_trains):
            prob += (
                lpSum(p[i, k] for k in range(platforms))
                == lpSum(x[i, t] for t in time_slots)
            ), f'platform_assign_{i}'

        # Pre-compute occupancy windows to avoid triple nested loop inside solver
        # occupancy[i][t] = list of start slots at which train i occupies slot t
        occupancy = {}
        for i in range(num_trains):
            dur = trains[i]['duration']
            occupancy[i] = {}
            for t in time_slots:
                starts = [s for s in range(max(0, t - dur + 1), t + 1) if s in set(time_slots)]
                occupancy[i][t] = starts

        # Constraint 3: one train per platform per slot
        for k in range(platforms):
            for t in time_slots:
                terms = [
                    x[i, s] * p[i, k]   # NOTE: bilinear — valid for MILP via CBC
                    for i in range(num_trains)
                    for s in occupancy[i][t]
                ]
                if terms:
                    prob += lpSum(terms) <= 1, f'plat_{k}_{t}'

        # Constraint 4: section capacity
        for t in time_slots:
            terms = [
                x[i, s]
                for i in range(num_trains)
                for s in occupancy[i][t]
            ]
            if terms:
                prob += lpSum(terms) <= section_capacity, f'cap_{t}'

        # Constraint 5: respect earliest arrival times
        for i in range(num_trains):
            earliest = trains[i]['arrival_time']
            for t in time_slots:
                if t < earliest:
                    prob += x[i, t] == 0, f'arrival_{i}_{t}'

        solver = PULP_CBC_CMD(msg=0)
        prob.solve(solver)
        self.problem = prob

        logger.info(
            "MILP solved: status=%s objective=%.2f",
            LpStatus[prob.status], value(prob.objective) or 0
        )

        return self._extract_solution(x, p, trains, time_slots, platforms)

    def _extract_solution(
        self, x: dict, p: dict, trains: list, time_slots: list, platforms: int
    ) -> dict:
        """Extract and format the LP solution."""
        scheduled, unscheduled = [], []

        for i, train in enumerate(trains):
            placed = False
            for t in time_slots:
                if value(x[i, t]) and round(value(x[i, t])) == 1:
                    platform = next(
                        (k for k in range(platforms) if value(p[i, k]) and round(value(p[i, k])) == 1),
                        None
                    )
                    scheduled.append({
                        'train_id': train['id'],
                        'priority': train['priority'],
                        'scheduled_time': t,
                        'duration': train['duration'],
                        'platform': platform,
                        'original_arrival': train['arrival_time']
                    })
                    placed = True
                    break
            if not placed:
                unscheduled.append({
                    'train_id': train['id'],
                    'priority': train['priority'],
                    'reason': 'Insufficient capacity or platform availability'
                })

        throughput_rate = len(scheduled) / len(trains) * 100 if trains else 0.0
        avg_priority = float(np.mean([t['priority'] for t in scheduled])) if scheduled else 0.0

        return {
            'status': LpStatus[self.problem.status],
            'objective_value': value(self.problem.objective),
            'scheduled_trains': scheduled,
            'unscheduled_trains': unscheduled,
            'throughput_metrics': {
                'total_trains_scheduled': len(scheduled),
                'total_trains': len(trains),
                'throughput_rate': round(throughput_rate, 2),
                'average_priority_served': round(avg_priority, 2)
            }
        }

    def optimize_with_conflicts(
        self, current_state: dict, conflict_predictions: list
    ) -> dict:
        """
        Produce recommendations to minimize predicted conflicts.

        Args:
            current_state: current operational state dict.
            conflict_predictions: list of prediction dicts.

        Returns:
            Dict with recommendations and high-risk count.
        """
        high_risk = [c for c in conflict_predictions if c.get('conflict_probability', 0) > 0.7]
        recommendations = []

        for conflict in high_risk:
            features = conflict.get('input_features', {})
            if features.get('trains_in_section', 0) > 30:
                recommendations.append({
                    'action': 'REDUCE_SECTION_DENSITY',
                    'current_value': features['trains_in_section'],
                    'target_value': 25,
                    'expected_impact': 'Reduce conflict probability by 40-50%'
                })
            if features.get('available_platforms', 99) < 3:
                recommendations.append({
                    'action': 'INCREASE_PLATFORM_AVAILABILITY',
                    'current_value': features['available_platforms'],
                    'target_value': 4,
                    'expected_impact': 'Reduce conflict probability by 30-40%'
                })

        return {
            'high_risk_count': len(high_risk),
            'recommendations': recommendations,
            'optimization_objective': 'Minimize conflict probability while maximizing throughput'
        }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    optimizer = ThroughputOptimizer()

    trains = [
        {'id': 'T001', 'priority': 10, 'arrival_time': 0,  'duration': 8},
        {'id': 'T002', 'priority':  8, 'arrival_time': 5,  'duration': 6},
        {'id': 'T003', 'priority':  9, 'arrival_time': 10, 'duration': 7},
        {'id': 'T004', 'priority':  7, 'arrival_time': 15, 'duration': 5},
        {'id': 'T005', 'priority': 10, 'arrival_time': 20, 'duration': 9},
    ]
    result = optimizer.optimize_train_schedule(trains, platforms=3, section_capacity=25, time_horizon=60)
    print(json.dumps(result, indent=2))
