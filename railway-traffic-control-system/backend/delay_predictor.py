"""
Delay Prediction Module
Predicts expected delay duration for railway operations
"""

import logging
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class DelayPredictor:
    """
    Real-time delay prediction using trained ML model.
    """

    FEATURE_COLUMNS = [
        'trains_in_section', 'available_platforms', 'platform_utilization',
        'weather_severity', 'rainfall_mm', 'fog_intensity', 'temperature_c',
        'hour', 'is_peak_hour', 'day_of_week', 'month',
        'hour_sin', 'hour_cos', 'train_density_risk',
        'weather_impact', 'congestion_score'
    ]

    def __init__(self, model_path: str = 'backend/models/delay_predictor.pkl'):
        self.model = joblib.load(model_path)
        logger.info("DelayPredictor loaded model from %s", model_path)

    def engineer_features(self, input_data: dict) -> dict:
        """Apply feature engineering — returns a new dict, does not mutate input."""
        data = input_data.copy()

        if 'timestamp' in data:
            ts = pd.to_datetime(data['timestamp'])
            data['day_of_week'] = ts.dayofweek
            data['month'] = ts.month
            data['hour'] = ts.hour
        else:
            now = datetime.utcnow()
            data.setdefault('day_of_week', now.weekday())
            data.setdefault('month', now.month)
            data.setdefault('hour', now.hour)

        h = data['hour']
        data['hour_sin'] = np.sin(2 * np.pi * h / 24)
        data['hour_cos'] = np.cos(2 * np.pi * h / 24)

        data['train_density_risk'] = data['trains_in_section'] / (data['available_platforms'] + 1)
        data['weather_impact'] = data['weather_severity'] * data['trains_in_section']
        data['congestion_score'] = data['platform_utilization'] * data['trains_in_section'] / 100

        return data

    def predict(self, input_data: dict) -> dict:
        """
        Predict expected delay for given operational state.

        Args:
            input_data: dict with operational parameters.

        Returns:
            dict with prediction results, severity, and mitigation strategies.
        """
        processed = self.engineer_features(input_data)

        feature_vector = pd.DataFrame([processed])[self.FEATURE_COLUMNS]

        predicted_delay = float(self.model.predict(feature_vector)[0])
        predicted_delay = max(0.0, predicted_delay)  # clamp to non-negative

        if predicted_delay < 5:
            severity, impact = 'MINIMAL', 'No significant impact expected'
        elif predicted_delay < 15:
            severity, impact = 'MODERATE', 'Minor schedule adjustments needed'
        elif predicted_delay < 30:
            severity, impact = 'SIGNIFICANT', 'Major schedule disruption expected'
        else:
            severity, impact = 'CRITICAL', 'Severe delays — immediate action required'

        mitigation = self._generate_mitigation(processed, predicted_delay)

        logger.debug("Delay prediction: %.1f min severity=%s", predicted_delay, severity)

        return {
            'predicted_delay_minutes': round(predicted_delay, 2),
            'severity': severity,
            'impact_description': impact,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'mitigation_strategies': mitigation,
            'operational_state': {
                k: (float(v) if isinstance(v, (int, float, np.number)) else v)
                for k, v in processed.items()
            }
        }

    def _generate_mitigation(self, data: dict, delay: float) -> list:
        """Generate mitigation strategies based on predicted delay."""
        strategies = []

        if delay > 15:
            strategies.append({
                'strategy': 'Activate priority routing',
                'expected_reduction': '5-10 minutes',
                'implementation': 'Prioritize express trains, hold local services'
            })

        if data['trains_in_section'] > 30:
            strategies.append({
                'strategy': 'Implement section throttling',
                'expected_reduction': '3-7 minutes',
                'implementation': f"Reduce entries to {int(data['trains_in_section'] * 0.7)} trains"
            })

        if data['weather_severity'] > 0.3:
            strategies.append({
                'strategy': 'Speed restrictions and safety protocols',
                'expected_reduction': '2-5 minutes (via accident prevention)',
                'implementation': 'Enforce 80% speed limit, increase signal spacing'
            })

        if data['platform_utilization'] > 90:
            strategies.append({
                'strategy': 'Dynamic platform reallocation',
                'expected_reduction': '4-8 minutes',
                'implementation': 'Reassign platforms based on train type and priority'
            })

        if delay > 30:
            strategies.append({
                'strategy': 'Emergency rerouting protocol',
                'expected_reduction': '10-15 minutes',
                'implementation': 'Divert trains to alternate routes/sections'
            })

        return strategies

    def simulate_scenario(self, base_data: dict, modifications: dict) -> dict:
        """
        Simulate what-if scenarios by modifying operational parameters.

        Args:
            base_data: baseline operational state dict.
            modifications: parameter overrides to test.

        Returns:
            Comparison dict with baseline, modified scenario, and improvement metrics.
        """
        baseline_result = self.predict(base_data)

        modified_data = {**base_data, **modifications}
        modified_result = self.predict(modified_data)

        base_delay = baseline_result['predicted_delay_minutes']
        mod_delay = modified_result['predicted_delay_minutes']
        delay_reduction = base_delay - mod_delay
        improvement_pct = (delay_reduction / base_delay * 100) if base_delay > 0 else 0.0

        return {
            'baseline': baseline_result,
            'modified_scenario': modified_result,
            'modifications_applied': modifications,
            'delay_reduction_minutes': round(delay_reduction, 2),
            'improvement_percentage': round(improvement_pct, 2)
        }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    predictor = DelayPredictor()

    scenario = {
        'timestamp': '2025-11-02 21:00:00',
        'trains_in_section': 35,
        'available_platforms': 3,
        'platform_utilization': 95.0,
        'weather_severity': 0.38,
        'rainfall_mm': 3.5,
        'fog_intensity': 0.6,
        'temperature_c': 23.0,
        'is_peak_hour': 1
    }

    result = predictor.predict(scenario)
    print(json.dumps(result, indent=2))

    sim = predictor.simulate_scenario(scenario, {'trains_in_section': 25, 'available_platforms': 4})
    print(f"Delay reduction: {sim['delay_reduction_minutes']:.2f} min ({sim['improvement_percentage']:.1f}%)")
