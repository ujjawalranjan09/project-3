"""
Real-time Conflict Detection Module
Provides instant conflict risk assessment for railway operations
"""

import logging
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class ConflictDetector:
    """
    Real-time conflict detection using trained ML model.
    """

    FEATURE_COLUMNS = [
        'trains_in_section', 'available_platforms', 'platform_utilization',
        'weather_severity', 'rainfall_mm', 'fog_intensity', 'temperature_c',
        'hour', 'is_peak_hour', 'day_of_week', 'month',
        'hour_sin', 'hour_cos', 'train_density_risk',
        'weather_impact', 'congestion_score'
    ]

    # Rule thresholds
    DENSITY_THRESHOLD = 30
    PLATFORM_MIN = 3
    WEATHER_THRESHOLD = 0.3
    UTILIZATION_MAX = 95

    def __init__(self, model_path: str = 'backend/models/conflict_detector.pkl'):
        self.model = joblib.load(model_path)
        logger.info("ConflictDetector loaded model from %s", model_path)

    def engineer_features(self, input_data: dict) -> dict:
        """Apply feature engineering to a copy of input_data."""
        data = input_data.copy()

        # Time-based features from timestamp
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

        # Composite features
        data['train_density_risk'] = data['trains_in_section'] / (data['available_platforms'] + 1)
        data['weather_impact'] = data['weather_severity'] * data['trains_in_section']
        data['congestion_score'] = data['platform_utilization'] * data['trains_in_section'] / 100

        return data

    def predict(self, input_data: dict) -> dict:
        """
        Predict conflict probability for given operational state.

        Args:
            input_data: dict with operational parameters.

        Returns:
            dict with prediction results and recommendations.
        """
        processed = self.engineer_features(input_data)

        feature_vector = pd.DataFrame([processed])[self.FEATURE_COLUMNS]

        conflict_prob = float(self.model.predict_proba(feature_vector)[0][1])
        conflict_predicted = bool(self.model.predict(feature_vector)[0])

        if conflict_prob < 0.3:
            risk_level, alert_color = 'LOW', 'green'
        elif conflict_prob < 0.7:
            risk_level, alert_color = 'MEDIUM', 'yellow'
        else:
            risk_level, alert_color = 'HIGH', 'red'

        recommendations = self._generate_recommendations(processed, conflict_prob)

        logger.debug("Conflict prediction: prob=%.3f risk=%s", conflict_prob, risk_level)

        return {
            'conflict_predicted': conflict_predicted,
            'conflict_probability': round(conflict_prob, 4),
            'risk_level': risk_level,
            'alert_color': alert_color,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'recommendations': recommendations,
            'input_features': {
                k: (float(v) if isinstance(v, (int, float, np.number)) else v)
                for k, v in processed.items()
            }
        }

    def _generate_recommendations(self, data: dict, conflict_prob: float) -> list:
        """Generate actionable recommendations based on operational state."""
        recs = []

        if data['trains_in_section'] > self.DENSITY_THRESHOLD:
            recs.append({
                'priority': 'HIGH',
                'action': 'Reduce train density in section',
                'details': f"Current: {data['trains_in_section']} trains (Threshold: {self.DENSITY_THRESHOLD})"
            })

        if data['available_platforms'] < self.PLATFORM_MIN:
            recs.append({
                'priority': 'HIGH',
                'action': 'Increase platform availability',
                'details': f"Current: {data['available_platforms']} (Target: ≥{self.PLATFORM_MIN})"
            })

        if data['weather_severity'] > self.WEATHER_THRESHOLD:
            recs.append({
                'priority': 'MEDIUM',
                'action': 'Activate weather contingency protocol',
                'details': f"Weather severity: {data['weather_severity']:.2f} (Threshold: {self.WEATHER_THRESHOLD})"
            })

        if data['platform_utilization'] > self.UTILIZATION_MAX:
            recs.append({
                'priority': 'HIGH',
                'action': 'Optimize platform allocation',
                'details': f"Utilization: {data['platform_utilization']:.1f}% (Threshold: {self.UTILIZATION_MAX}%)"
            })

        if conflict_prob > 0.7:
            recs.append({
                'priority': 'CRITICAL',
                'action': 'Immediate intervention required',
                'details': f"Conflict probability: {conflict_prob:.1%}"
            })

        return recs

    def batch_predict(self, data_list: list) -> list:
        """Predict conflicts for multiple operational states."""
        results = []
        for i, data in enumerate(data_list):
            try:
                results.append(self.predict(data))
            except Exception as e:
                logger.warning("batch_predict item %d failed: %s", i, e)
                results.append({'error': str(e), 'index': i})
        return results


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    detector = ConflictDetector()

    high_risk = {
        'timestamp': '2025-11-02 21:00:00',
        'trains_in_section': 38,
        'available_platforms': 2,
        'platform_utilization': 98.5,
        'weather_severity': 0.45,
        'rainfall_mm': 4.2,
        'fog_intensity': 0.7,
        'temperature_c': 22.0,
        'is_peak_hour': 1
    }
    result = detector.predict(high_risk)
    print(json.dumps(result, indent=2))
