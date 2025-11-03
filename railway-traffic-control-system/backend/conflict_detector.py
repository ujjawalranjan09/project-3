"""
Real-time Conflict Detection Module
Provides instant conflict risk assessment for railway operations
"""

import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import json


class ConflictDetector:
    """
    Real-time conflict detection using trained ML model
    """

    def __init__(self, model_path='backend/models/conflict_detector.pkl'):
        self.model = joblib.load(model_path)
        self.feature_columns = [
            'trains_in_section', 'available_platforms', 'platform_utilization_pct',
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
        input_data['congestion_score'] = input_data['platform_utilization_pct'] * input_data['trains_in_section'] / 100

        return input_data

    def predict(self, input_data):
        """
        Predict conflict probability for given operational state

        Args:
            input_data: dict with operational parameters

        Returns:
            dict with prediction results
        """
        # Engineer features
        processed_data = self.engineer_features(input_data.copy())

        # Prepare feature vector
        feature_vector = pd.DataFrame([processed_data])[self.feature_columns]

        # Predict
        conflict_prob = self.model.predict_proba(feature_vector)[0][1]
        conflict_predicted = self.model.predict(feature_vector)[0]

        # Determine risk level
        if conflict_prob < 0.3:
            risk_level = "LOW"
            alert_color = "green"
        elif conflict_prob < 0.7:
            risk_level = "MEDIUM"
            alert_color = "yellow"
        else:
            risk_level = "HIGH"
            alert_color = "red"

        # Generate recommendations
        recommendations = self._generate_recommendations(processed_data, conflict_prob)

        return {
            'conflict_predicted': bool(conflict_predicted),
            'conflict_probability': float(conflict_prob),
            'risk_level': risk_level,
            'alert_color': alert_color,
            'timestamp': datetime.now().isoformat(),
            'recommendations': recommendations,
            'input_features': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                             for k, v in processed_data.items()}
        }

    def _generate_recommendations(self, data, conflict_prob):
        """Generate actionable recommendations based on operational state"""
        recommendations = []

        # High train density
        if data['trains_in_section'] > 30:
            recommendations.append({
                'priority': 'HIGH',
                'action': 'Reduce train density in section',
                'details': f"Current: {data['trains_in_section']} trains (Threshold: 30)"
            })

        # Low platform availability
        if data['available_platforms'] < 3:
            recommendations.append({
                'priority': 'HIGH',
                'action': 'Increase platform availability',
                'details': f"Current: {data['available_platforms']} platforms (Target: ≥3)"
            })

        # High weather severity
        if data['weather_severity'] > 0.3:
            recommendations.append({
                'priority': 'MEDIUM',
                'action': 'Activate weather contingency protocol',
                'details': f"Weather severity: {data['weather_severity']:.2f} (Threshold: 0.3)"
            })

        # High platform utilization
        if data['platform_utilization_pct'] > 95:
            recommendations.append({
                'priority': 'HIGH',
                'action': 'Optimize platform allocation',
                'details': f"Utilization: {data['platform_utilization_pct']:.1f}% (Threshold: 95%)"
            })

        # General conflict risk
        if conflict_prob > 0.7:
            recommendations.append({
                'priority': 'CRITICAL',
                'action': 'Immediate intervention required',
                'details': f"Conflict probability: {conflict_prob:.1%}"
            })

        return recommendations

    def batch_predict(self, data_list):
        """Predict conflicts for multiple operational states"""
        results = []
        for data in data_list:
            results.append(self.predict(data))
        return results


if __name__ == "__main__":
    # Example usage
    detector = ConflictDetector()

    # Test case 1: High-risk scenario
    high_risk = {
        'timestamp': '2025-11-02 21:00:00',
        'trains_in_section': 38,
        'available_platforms': 2,
        'platform_utilization_pct': 98.5,
        'weather_severity': 0.45,
        'rainfall_mm': 4.2,
        'fog_intensity': 0.7,
        'temperature_c': 22.0,
        'is_peak_hour': 1
    }

    result = detector.predict(high_risk)
    print("High-Risk Scenario:")
    print(json.dumps(result, indent=2))
