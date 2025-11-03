"""
MILP-based Throughput Optimization Module
Maximizes section throughput using Mixed Integer Linear Programming
"""

from pulp import *
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json


class ThroughputOptimizer:
    """
    MILP optimizer for railway section throughput maximization
    """

    def __init__(self):
        self.problem = None
        self.solution = None

    def optimize_train_schedule(self, trains, platforms, section_capacity, time_horizon=60):
        """
        Optimize train scheduling to maximize throughput

        Args:
            trains: list of train dicts with properties (id, priority, arrival_time, duration)
            platforms: number of available platforms
            section_capacity: maximum trains in section simultaneously
            time_horizon: optimization window in minutes

        Returns:
            optimized schedule
        """
        # Create optimization problem
        self.problem = LpProblem("Railway_Throughput_Optimization", LpMaximize)

        num_trains = len(trains)
        time_slots = list(range(time_horizon))

        # Decision variables: x[i][t] = 1 if train i is scheduled at time t
        x = {}
        for i in range(num_trains):
            for t in time_slots:
                x[i, t] = LpVariable(f"train_{i}_time_{t}", cat='Binary')

        # Platform assignment: p[i][k] = 1 if train i uses platform k
        p = {}
        for i in range(num_trains):
            for k in range(platforms):
                p[i, k] = LpVariable(f"train_{i}_platform_{k}", cat='Binary')

        # Objective: Maximize total trains scheduled weighted by priority
        self.problem += lpSum([
            trains[i]['priority'] * lpSum([x[i, t] for t in time_slots])
            for i in range(num_trains)
        ]), "Total_Weighted_Throughput"

        # Constraints

        # 1. Each train scheduled at most once
        for i in range(num_trains):
            self.problem += lpSum([x[i, t] for t in time_slots]) <= 1, f"Train_{i}_once"

        # 2. Each train uses exactly one platform if scheduled
        for i in range(num_trains):
            self.problem += lpSum([p[i, k] for k in range(platforms)]) == lpSum([x[i, t] for t in time_slots]), f"Train_{i}_platform"

        # 3. Platform capacity: at most one train per platform at any time
        for k in range(platforms):
            for t in time_slots:
                occupying_trains = []
                for i in range(num_trains):
                    duration = trains[i]['duration']
                    for start in range(max(0, t - duration + 1), t + 1):
                        if start in time_slots:
                            occupying_trains.append(x[i, start] * p[i, k])

                if occupying_trains:
                    self.problem += lpSum(occupying_trains) <= 1, f"Platform_{k}_time_{t}"

        # 4. Section capacity constraint
        for t in time_slots:
            trains_in_section = []
            for i in range(num_trains):
                duration = trains[i]['duration']
                for start in range(max(0, t - duration + 1), t + 1):
                    if start in time_slots:
                        trains_in_section.append(x[i, start])

            if trains_in_section:
                self.problem += lpSum(trains_in_section) <= section_capacity, f"Section_capacity_{t}"

        # 5. Respect earliest arrival times
        for i in range(num_trains):
            earliest = trains[i]['arrival_time']
            for t in range(earliest):
                if t in time_slots:
                    self.problem += x[i, t] == 0, f"Train_{i}_arrival_{t}"

        # Solve
        solver = PULP_CBC_CMD(msg=0)
        self.problem.solve(solver)

        # Extract solution
        schedule = self._extract_solution(x, p, trains, time_slots, platforms)

        return schedule

    def _extract_solution(self, x, p, trains, time_slots, platforms):
        """Extract and format optimization solution"""
        schedule = {
            'status': LpStatus[self.problem.status],
            'objective_value': value(self.problem.objective),
            'scheduled_trains': [],
            'unscheduled_trains': [],
            'platform_utilization': {},
            'throughput_metrics': {}
        }

        # Extract scheduled trains
        for i in range(len(trains)):
            scheduled = False
            for t in time_slots:
                if value(x[i, t]) == 1:
                    # Find assigned platform
                    assigned_platform = None
                    for k in range(platforms):
                        if value(p[i, k]) == 1:
                            assigned_platform = k
                            break

                    schedule['scheduled_trains'].append({
                        'train_id': trains[i]['id'],
                        'priority': trains[i]['priority'],
                        'scheduled_time': t,
                        'duration': trains[i]['duration'],
                        'platform': assigned_platform,
                        'original_arrival': trains[i]['arrival_time']
                    })
                    scheduled = True
                    break

            if not scheduled:
                schedule['unscheduled_trains'].append({
                    'train_id': trains[i]['id'],
                    'priority': trains[i]['priority'],
                    'reason': 'Insufficient capacity or platform availability'
                })

        # Calculate metrics
        schedule['throughput_metrics'] = {
            'total_trains_scheduled': len(schedule['scheduled_trains']),
            'total_trains': len(trains),
            'throughput_rate': len(schedule['scheduled_trains']) / len(trains) * 100,
            'average_priority_served': np.mean([t['priority'] for t in schedule['scheduled_trains']]) if schedule['scheduled_trains'] else 0
        }

        return schedule

    def optimize_with_conflicts(self, current_state, conflict_predictions):
        """
        Optimize considering predicted conflicts

        Args:
            current_state: current operational state
            conflict_predictions: list of predicted conflicts with probabilities

        Returns:
            optimized plan to minimize conflicts
        """
        recommendations = []

        # Analyze high-risk conflicts
        high_risk_conflicts = [c for c in conflict_predictions if c['conflict_probability'] > 0.7]

        for conflict in high_risk_conflicts:
            # Generate specific recommendations
            if conflict['input_features']['trains_in_section'] > 30:
                recommendations.append({
                    'action': 'REDUCE_SECTION_DENSITY',
                    'target': 'trains_in_section',
                    'current_value': conflict['input_features']['trains_in_section'],
                    'target_value': 25,
                    'expected_impact': 'Reduce conflict probability by 40-50%'
                })

            if conflict['input_features']['available_platforms'] < 3:
                recommendations.append({
                    'action': 'INCREASE_PLATFORM_AVAILABILITY',
                    'target': 'available_platforms',
                    'current_value': conflict['input_features']['available_platforms'],
                    'target_value': 4,
                    'expected_impact': 'Reduce conflict probability by 30-40%'
                })

        return {
            'high_risk_count': len(high_risk_conflicts),
            'recommendations': recommendations,
            'optimization_objective': 'Minimize conflict probability while maximizing throughput'
        }


if __name__ == "__main__":
    # Example usage
    optimizer = ThroughputOptimizer()

    # Sample trains
    trains = [
        {'id': 'T001', 'priority': 10, 'arrival_time': 0, 'duration': 8},
        {'id': 'T002', 'priority': 8, 'arrival_time': 5, 'duration': 6},
        {'id': 'T003', 'priority': 9, 'arrival_time': 10, 'duration': 7},
        {'id': 'T004', 'priority': 7, 'arrival_time': 15, 'duration': 5},
        {'id': 'T005', 'priority': 10, 'arrival_time': 20, 'duration': 9},
    ]

    result = optimizer.optimize_train_schedule(
        trains=trains,
        platforms=3,
        section_capacity=25,
        time_horizon=60
    )

    print("Optimization Result:")
    print(json.dumps(result, indent=2))
