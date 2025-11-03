"""
Flask API Server for Railway Traffic Control System
Provides REST API endpoints for all system functionality
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import json
from datetime import datetime

# Import system modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from conflict_detector import ConflictDetector
from delay_predictor import DelayPredictor
from throughput_optimizer import ThroughputOptimizer

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Initialize components
try:
    conflict_detector = ConflictDetector('backend/models/conflict_detector.pkl')
    delay_predictor = DelayPredictor('backend/models/delay_predictor.pkl')
    throughput_optimizer = ThroughputOptimizer()
    print("✓ All models loaded successfully")
except Exception as e:
    print(f"Warning: Model loading failed - {e}")
    conflict_detector = None
    delay_predictor = None
    throughput_optimizer = None


# ============================================================================
# HEALTH CHECK & INFO
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """System health check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': {
            'conflict_detector': conflict_detector is not None,
            'delay_predictor': delay_predictor is not None,
            'optimizer': throughput_optimizer is not None
        }
    })


@app.route('/api/info', methods=['GET'])
def system_info():
    """Get system information"""
    return jsonify({
        'system_name': 'AI-Powered Railway Traffic Control System',
        'version': '1.0.0',
        'capabilities': [
            'Real-time conflict detection',
            'Delay prediction',
            'Throughput optimization',
            'What-if scenario simulation',
            'Decision support recommendations'
        ],
        'endpoints': {
            'conflict_detection': '/api/predict/conflict',
            'delay_prediction': '/api/predict/delay',
            'scenario_simulation': '/api/simulate',
            'optimization': '/api/optimize/schedule',
            'batch_analysis': '/api/batch/analyze'
        }
    })


# ============================================================================
# PREDICTION ENDPOINTS
# ============================================================================

@app.route('/api/predict/conflict', methods=['POST'])
def predict_conflict():
    """Predict conflict probability for given operational state"""
    try:
        data = request.json

        if not conflict_detector:
            return jsonify({'error': 'Conflict detector not loaded'}), 500

        # Validate required fields
        required_fields = ['trains_in_section', 'available_platforms', 
                            'platform_utilization', 'weather_severity',
                            'rainfall_mm', 'fog_intensity', 'temperature_c',
                            'is_peak_hour']

        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return jsonify({'error': f'Missing fields: {missing_fields}'}), 400

        # Add timestamp if not provided
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().isoformat()

        # Predict
        result = conflict_detector.predict(data)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/delay', methods=['POST'])
def predict_delay():
    """Predict expected delay for given operational state"""
    try:
        data = request.json

        if not delay_predictor:
            return jsonify({'error': 'Delay predictor not loaded'}), 500

        # Add timestamp if not provided
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().isoformat()

        # Predict
        result = delay_predictor.predict(data)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/combined', methods=['POST'])
def predict_combined():
    """Combined conflict and delay prediction"""
    try:
        data = request.json

        if not conflict_detector or not delay_predictor:
            return jsonify({'error': 'Models not loaded'}), 500

        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().isoformat()

        # Get both predictions
        conflict_result = conflict_detector.predict(data)
        delay_result = delay_predictor.predict(data)

        # Combine results
        combined = {
            'timestamp': datetime.now().isoformat(),
            'conflict_analysis': conflict_result,
            'delay_analysis': delay_result,
            'overall_risk_assessment': {
                'conflict_risk': conflict_result['risk_level'],
                'delay_severity': delay_result['severity'],
                'recommended_action': 'IMMEDIATE' if conflict_result['conflict_probability'] > 0.7 or delay_result['predicted_delay_minutes'] > 30 else 'MONITOR'
            }
        }

        return jsonify(combined)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# SIMULATION ENDPOINTS
# ============================================================================

@app.route('/api/simulate', methods=['POST'])
def simulate_scenario():
    """Simulate what-if scenarios"""
    try:
        data = request.json

        if not delay_predictor:
            return jsonify({'error': 'Delay predictor not loaded'}), 500

        # Extract baseline and modifications
        baseline = data.get('baseline', {})
        modifications = data.get('modifications', {})

        if not baseline or not modifications:
            return jsonify({'error': 'Both baseline and modifications required'}), 400

        # Run simulation
        result = delay_predictor.simulate_scenario(baseline, modifications)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# OPTIMIZATION ENDPOINTS
# ============================================================================

@app.route('/api/optimize/schedule', methods=['POST'])
def optimize_schedule():
    """Optimize train schedule for maximum throughput"""
    try:
        data = request.json

        if not throughput_optimizer:
            return jsonify({'error': 'Optimizer not loaded'}), 500

        # Extract parameters
        trains = data.get('trains', [])
        platforms = data.get('platforms', 3)
        section_capacity = data.get('section_capacity', 25)
        time_horizon = data.get('time_horizon', 60)

        if not trains:
            return jsonify({'error': 'Train list required'}), 400

        # Optimize
        result = throughput_optimizer.optimize_train_schedule(
            trains, platforms, section_capacity, time_horizon
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# BATCH PROCESSING
# ============================================================================

@app.route('/api/batch/analyze', methods=['POST'])
def batch_analyze():
    """Batch analysis of multiple operational states"""
    try:
        data = request.json
        states = data.get('states', [])

        if not states:
            return jsonify({'error': 'States list required'}), 400

        if not conflict_detector or not delay_predictor:
            return jsonify({'error': 'Models not loaded'}), 500

        results = []
        for state in states:
            conflict_res = conflict_detector.predict(state)
            delay_res = delay_predictor.predict(state)

            results.append({
                'state_id': state.get('id', 'unknown'),
                'conflict_probability': conflict_res['conflict_probability'],
                'risk_level': conflict_res['risk_level'],
                'predicted_delay': delay_res['predicted_delay_minutes'],
                'severity': delay_res['severity']
            })

        # Aggregate statistics
        high_risk_count = sum(1 for r in results if r['risk_level'] == 'HIGH')
        critical_delays = sum(1 for r in results if r['severity'] == 'CRITICAL')

        return jsonify({
            'total_analyzed': len(results),
            'high_risk_conflicts': high_risk_count,
            'critical_delays': critical_delays,
            'results': results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# KPI METRICS
# ============================================================================

@app.route('/api/metrics/kpi', methods=['GET'])
def get_kpi_metrics():
    """Get current KPI metrics"""
    # This would typically fetch from a database
    # For now, return sample metrics
    return jsonify({
        'timestamp': datetime.now().isoformat(),
        'throughput': {
            'trains_per_hour': 42,
            'target': 50,
            'percentage': 84.0
        },
        'punctuality': {
            'on_time_percentage': 87.5,
            'target': 90.0
        },
        'conflicts': {
            'detected_today': 12,
            'resolved': 10,
            'pending': 2
        },
        'average_delay': {
            'current': 8.3,
            'target': 5.0,
            'unit': 'minutes'
        },
        'platform_utilization': {
            'current': 92.5,
            'optimal_range': [85, 95],
            'unit': 'percentage'
        }
    })


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("="*60)
    print("AI-Powered Railway Traffic Control System")
    print("API Server Starting...")
    print("="*60)
    app.run(host='0.0.0.0', port=5000, debug=True)
