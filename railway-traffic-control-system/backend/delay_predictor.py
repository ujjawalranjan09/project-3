"""
Delay Prediction Module
Predicts expected delay duration for railway operations
"""

import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import json


class DelayPredictor:
    """
    Real-time delay prediction using trained ML model
    """

    def __init__(self, model_path='backend/models/delay_predictor.pkl'):
        self.model = joblib.load(model_path)
        self.feature_columns = [
            'trains_in_section', 'available_platforms', 'platform_utilization',
            'weather_severity', 'rainfall_mm', 'fog_intensity', 'temperature_c',
            'hour', 'is_peak_hour', 'day_of_week', 'month',
            'hour_sin', 'hour_cos', 'train_density_risk', 
            'weather_impact', 'congestion_score'
        ]

    def engineer_features(self, input_data):
        """Apply feature engineering to input data"""
        # Time-based features
        if 'timestamp' in input_data:
            timestamp = pd.to_datetime(input_data['timestamp'])
            input_data['day_of_week'] = timestamp.dayofweek
            input_data['month'] = timestamp.month
            input_data['hour'] = timestamp.hour

        # Trigonometric encoding for hour
        input_data['hour_sin'] = np.sin(2 * np.pi * input_data['hour'] / 24)
        input_data['hour_cos'] = np.cos(2 * np.pi * input_data['hour'] / 24)

        # Composite features
        input_data['train_density_risk'] = input_data['trains_in_section'] / (input_data['available_platforms'] + 1)
        input_data['weather_impact'] = input_data['weather_severity'] * input_data['trains_in_section']
        input_data['congestion_score'] = input_data['platform_utilization'] * input_data['trains_in_section'] / 100

        return input_data

    def predict(self, input_data):
        """
        Predict expected delay for given operational state

        Args:
            input_data: dict with operational parameters

        Returns:
            dict with prediction results
        """
        # Engineer features
        processed_data = self.engineer_features(input_data.copy())

        # Prepare feature vector
        feature_vector = pd.DataFrame([processed_data])[self.feature_columns]

        # Predict delay
        predicted_delay = self.model.predict(feature_vector)[0]
        predicted_delay = max(0, predicted_delay)  # Ensure non-negative

        # Categorize delay severity
        if predicted_delay < 5:
            severity = "MINIMAL"
            impact = "No significant impact expected"
        elif predicted_delay < 15:
            severity = "MODERATE"
            impact = "Minor schedule adjustments needed"
        elif predicted_delay < 30:
            severity = "SIGNIFICANT"
            impact = "Major schedule disruption expected"
        else:
            severity = "CRITICAL"
            impact = "Severe delays - immediate action required"

        # Generate mitigation strategies
        mitigation = self._generate_mitigation(processed_data, predicted_delay)

        return {
            'predicted_delay_minutes': float(predicted_delay),
            'severity': severity,
            'impact_description': impact,
            'timestamp': datetime.now().isoformat(),
            'mitigation_strategies': mitigation,
            'operational_state': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                                 for k, v in processed_data.items()}
        }

    def _generate_mitigation(self, data, delay):
        """Generate mitigation strategies based on predicted delay"""
        strategies = []

        # For significant delays
        if delay > 15:
            strategies.append({
                'strategy': 'Activate priority routing',
                'expected_reduction': '5-10 minutes',
                'implementation': 'Prioritize express trains, hold local services'
            })

        # High train density
        if data['trains_in_section'] > 30:
            strategies.append({
                'strategy': 'Implement section throttling',
                'expected_reduction': '3-7 minutes',
                'implementation': f"Reduce entries to {int(data['trains_in_section'] * 0.7)} trains"
            })

        # Weather-related delays
        if data['weather_severity'] > 0.3:
            strategies.append({
                'strategy': 'Speed restrictions and safety protocols',
                'expected_reduction': '2-5 minutes (via accident prevention)',
                'implementation': 'Enforce 80% speed limit, increase signal spacing'
            })

        # Platform optimization
        if data['platform_utilization'] > 90:
            strategies.append({
                'strategy': 'Dynamic platform reallocation',
                'expected_reduction': '4-8 minutes',
                'implementation': 'Reassign platforms based on train type and priority'
            })

        # Critical delays
        if delay > 30:
            strategies.append({
                'strategy': 'Emergency rerouting protocol',
                'expected_reduction': '10-15 minutes',
                'implementation': 'Divert trains to alternate routes/sections'
            })

        return strategies

    def simulate_scenario(self, base_data, modifications):
        """
        Simulate what-if scenarios by modifying operational parameters

        Args:
            base_data: baseline operational state
            modifications: dict of parameter changes to test

        Returns:
            comparison of scenarios
        """
        # Baseline prediction
        baseline_result = self.predict(base_data)

        # Modified scenario
        modified_data = base_data.copy()
        modified_data.update(modifications)
        modified_result = self.predict(modified_data)

        # Calculate improvement
        delay_reduction = baseline_result['predicted_delay_minutes'] - modified_result['predicted_delay_minutes']
        improvement_pct = (delay_reduction / baseline_result['predicted_delay_minutes'] * 100) if baseline_result['predicted_delay_minutes'] > 0 else 0

        return {
            'baseline': baseline_result,
            'modified_scenario': modified_result,
            'modifications_applied': modifications,
            'delay_reduction_minutes': float(delay_reduction),
            'improvement_percentage': float(improvement_pct)
        }


if __name__ == "__main__":
    # Example usage
    predictor = DelayPredictor()

    # Test scenario
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
    print("Delay Prediction:")
    print(json.dumps(result, indent=2))

    # What-if simulation
    print("\n\nWhat-If Simulation: Reduce train density")
    simulation = predictor.simulate_scenario(
        scenario,
        {'trains_in_section': 25, 'available_platforms': 4}
    )
    print(f"Delay reduction: {simulation['delay_reduction_minutes']:.2f} minutes")
    print(f"Improvement: {simulation['improvement_percentage']:.1f}%")
