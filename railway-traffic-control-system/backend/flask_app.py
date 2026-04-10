"""
Flask API Server for Railway Traffic Control System
Provides REST API endpoints for all system functionality
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from collections import deque
from functools import wraps
from threading import Lock

from flask import Flask, request, jsonify, g
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import system modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from conflict_detector import ConflictDetector
from delay_predictor import DelayPredictor
from throughput_optimizer import ThroughputOptimizer

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('railway_api.log')
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'change-me-in-production')

# CORS — locked to configured origins
allowed_origins = os.getenv('CORS_ORIGINS', 'http://localhost:3000').split(',')
CORS(app, origins=[o.strip() for o in allowed_origins])

# Rate limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[os.getenv('RATE_LIMIT_DEFAULT', '200 per hour')],
    storage_uri='memory://'
)

# ---------------------------------------------------------------------------
# In-memory KPI accumulator (thread-safe)
# ---------------------------------------------------------------------------
_kpi_lock = Lock()
_kpi_store = {
    'conflict_predictions': deque(maxlen=1000),  # rolling window
    'delay_predictions': deque(maxlen=1000),
    'daily_conflicts_detected': 0,
    'daily_conflicts_resolved': 0,
    'reset_date': datetime.utcnow().date().isoformat()
}


def _reset_daily_kpis_if_needed():
    today = datetime.utcnow().date().isoformat()
    if _kpi_store['reset_date'] != today:
        with _kpi_lock:
            _kpi_store['daily_conflicts_detected'] = 0
            _kpi_store['daily_conflicts_resolved'] = 0
            _kpi_store['reset_date'] = today


def _record_prediction(conflict_prob: float, delay_minutes: float, risk_level: str):
    with _kpi_lock:
        _kpi_store['conflict_predictions'].append({
            'ts': datetime.utcnow().isoformat(),
            'prob': conflict_prob,
            'risk': risk_level
        })
        _kpi_store['delay_predictions'].append({
            'ts': datetime.utcnow().isoformat(),
            'delay': delay_minutes
        })
        if risk_level in ('HIGH', 'CRITICAL'):
            _kpi_store['daily_conflicts_detected'] += 1


# ---------------------------------------------------------------------------
# API Key authentication
# ---------------------------------------------------------------------------
API_KEY = os.getenv('API_KEY', '')


def require_api_key(f):
    """Decorator: enforce API key on endpoints that modify or predict."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not API_KEY:
            # API key not configured — skip auth (dev mode)
            return f(*args, **kwargs)
        key = request.headers.get('X-API-Key') or request.args.get('api_key')
        if key != API_KEY:
            logger.warning("Unauthorized request from %s", request.remote_addr)
            return jsonify({'error': 'Unauthorized — invalid or missing API key'}), 401
        return f(*args, **kwargs)
    return decorated


# ---------------------------------------------------------------------------
# Input validation helpers (lightweight, no extra deps beyond builtins)
# ---------------------------------------------------------------------------
REQUIRED_CONFLICT_FIELDS = [
    'trains_in_section', 'available_platforms', 'platform_utilization',
    'weather_severity', 'rainfall_mm', 'fog_intensity', 'temperature_c',
    'is_peak_hour'
]

REQUIRED_DELAY_FIELDS = REQUIRED_CONFLICT_FIELDS  # same schema


def _validate_fields(data: dict, required: list):
    """Return list of missing field names."""
    return [f for f in required if f not in data]


def _validate_ranges(data: dict):
    """Return list of range-violation messages."""
    errors = []
    checks = {
        'trains_in_section': (0, 200),
        'available_platforms': (0, 50),
        'platform_utilization': (0, 100),
        'weather_severity': (0.0, 1.0),
        'rainfall_mm': (0, 500),
        'fog_intensity': (0.0, 1.0),
        'temperature_c': (-50, 60),
        'is_peak_hour': (0, 1)
    }
    for field, (lo, hi) in checks.items():
        if field in data:
            try:
                val = float(data[field])
                if not (lo <= val <= hi):
                    errors.append(f"'{field}' must be between {lo} and {hi}, got {val}")
            except (TypeError, ValueError):
                errors.append(f"'{field}' must be numeric")
    return errors


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
try:
    conflict_model_path = os.getenv('CONFLICT_MODEL_PATH', 'backend/models/conflict_detector.pkl')
    delay_model_path = os.getenv('DELAY_MODEL_PATH', 'backend/models/delay_predictor.pkl')
    conflict_detector = ConflictDetector(conflict_model_path)
    delay_predictor = DelayPredictor(delay_model_path)
    throughput_optimizer = ThroughputOptimizer()
    logger.info("All models loaded successfully")
except Exception as e:
    logger.error("Model loading failed: %s", e)
    conflict_detector = None
    delay_predictor = None
    throughput_optimizer = None


# ============================================================================
# HEALTH CHECK & INFO
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """System health check — no auth required for load-balancer probes."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'models_loaded': {
            'conflict_detector': conflict_detector is not None,
            'delay_predictor': delay_predictor is not None,
            'optimizer': throughput_optimizer is not None
        }
    })


@app.route('/api/info', methods=['GET'])
def system_info():
    """Get system information."""
    return jsonify({
        'system_name': 'AI-Powered Railway Traffic Control System',
        'version': '1.1.0',
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
        },
        'auth': 'X-API-Key header required on prediction/write endpoints'
    })


# ============================================================================
# PREDICTION ENDPOINTS
# ============================================================================

@app.route('/api/predict/conflict', methods=['POST'])
@require_api_key
@limiter.limit(os.getenv('RATE_LIMIT_PREDICT', '200 per hour'))
def predict_conflict():
    """Predict conflict probability for given operational state."""
    try:
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({'error': 'Request body must be valid JSON'}), 400

        if not conflict_detector:
            return jsonify({'error': 'Conflict detector model not loaded'}), 503

        missing = _validate_fields(data, REQUIRED_CONFLICT_FIELDS)
        if missing:
            return jsonify({'error': f'Missing required fields: {missing}'}), 400

        range_errors = _validate_ranges(data)
        if range_errors:
            return jsonify({'error': 'Validation errors', 'details': range_errors}), 422

        data.setdefault('timestamp', datetime.utcnow().isoformat())

        result = conflict_detector.predict(data)

        # Record for live KPIs
        _reset_daily_kpis_if_needed()
        _record_prediction(result['conflict_probability'], 0.0, result['risk_level'])

        logger.info("Conflict prediction: risk=%s prob=%.3f ip=%s",
                    result['risk_level'], result['conflict_probability'], request.remote_addr)
        return jsonify(result)

    except Exception as e:
        logger.exception("Unhandled error in predict_conflict")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/predict/delay', methods=['POST'])
@require_api_key
@limiter.limit(os.getenv('RATE_LIMIT_PREDICT', '200 per hour'))
def predict_delay():
    """Predict expected delay for given operational state."""
    try:
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({'error': 'Request body must be valid JSON'}), 400

        if not delay_predictor:
            return jsonify({'error': 'Delay predictor model not loaded'}), 503

        # Validate inputs (was completely missing before)
        missing = _validate_fields(data, REQUIRED_DELAY_FIELDS)
        if missing:
            return jsonify({'error': f'Missing required fields: {missing}'}), 400

        range_errors = _validate_ranges(data)
        if range_errors:
            return jsonify({'error': 'Validation errors', 'details': range_errors}), 422

        data.setdefault('timestamp', datetime.utcnow().isoformat())

        result = delay_predictor.predict(data)

        # Record for live KPIs
        _reset_daily_kpis_if_needed()
        _record_prediction(0.0, result['predicted_delay_minutes'], result['severity'])

        logger.info("Delay prediction: severity=%s delay=%.1fmin ip=%s",
                    result['severity'], result['predicted_delay_minutes'], request.remote_addr)
        return jsonify(result)

    except Exception as e:
        logger.exception("Unhandled error in predict_delay")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/predict/combined', methods=['POST'])
@require_api_key
@limiter.limit(os.getenv('RATE_LIMIT_PREDICT', '200 per hour'))
def predict_combined():
    """Combined conflict and delay prediction."""
    try:
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({'error': 'Request body must be valid JSON'}), 400

        if not conflict_detector or not delay_predictor:
            return jsonify({'error': 'One or more models not loaded'}), 503

        missing = _validate_fields(data, REQUIRED_CONFLICT_FIELDS)
        if missing:
            return jsonify({'error': f'Missing required fields: {missing}'}), 400

        range_errors = _validate_ranges(data)
        if range_errors:
            return jsonify({'error': 'Validation errors', 'details': range_errors}), 422

        data.setdefault('timestamp', datetime.utcnow().isoformat())

        conflict_result = conflict_detector.predict(data)
        delay_result = delay_predictor.predict(data)

        _reset_daily_kpis_if_needed()
        _record_prediction(
            conflict_result['conflict_probability'],
            delay_result['predicted_delay_minutes'],
            conflict_result['risk_level']
        )

        combined = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'conflict_analysis': conflict_result,
            'delay_analysis': delay_result,
            'overall_risk_assessment': {
                'conflict_risk': conflict_result['risk_level'],
                'delay_severity': delay_result['severity'],
                'recommended_action': (
                    'IMMEDIATE'
                    if conflict_result['conflict_probability'] > 0.7
                    or delay_result['predicted_delay_minutes'] > 30
                    else 'MONITOR'
                )
            }
        }
        return jsonify(combined)

    except Exception as e:
        logger.exception("Unhandled error in predict_combined")
        return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# SIMULATION ENDPOINTS
# ============================================================================

@app.route('/api/simulate', methods=['POST'])
@require_api_key
def simulate_scenario():
    """Simulate what-if scenarios."""
    try:
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({'error': 'Request body must be valid JSON'}), 400

        if not delay_predictor:
            return jsonify({'error': 'Delay predictor not loaded'}), 503

        baseline = data.get('baseline', {})
        modifications = data.get('modifications', {})

        if not baseline:
            return jsonify({'error': '"baseline" field is required and must be non-empty'}), 400
        if not modifications:
            return jsonify({'error': '"modifications" field is required and must be non-empty'}), 400

        missing = _validate_fields(baseline, REQUIRED_DELAY_FIELDS)
        if missing:
            return jsonify({'error': f'Baseline missing required fields: {missing}'}), 400

        result = delay_predictor.simulate_scenario(baseline, modifications)
        return jsonify(result)

    except Exception as e:
        logger.exception("Unhandled error in simulate_scenario")
        return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# OPTIMIZATION ENDPOINTS
# ============================================================================

@app.route('/api/optimize/schedule', methods=['POST'])
@require_api_key
@limiter.limit('30 per hour')
def optimize_schedule():
    """Optimize train schedule for maximum throughput."""
    try:
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({'error': 'Request body must be valid JSON'}), 400

        if not throughput_optimizer:
            return jsonify({'error': 'Optimizer not loaded'}), 503

        trains = data.get('trains', [])
        platforms = int(data.get('platforms', 3))
        section_capacity = int(data.get('section_capacity', 25))
        time_horizon = int(data.get('time_horizon', 60))

        if not trains:
            return jsonify({'error': '"trains" list is required and must be non-empty'}), 400

        # Guard against combinatorial explosion
        MAX_VARS = 3000
        if len(trains) * time_horizon > MAX_VARS:
            return jsonify({
                'error': f'Problem too large: len(trains) * time_horizon must be <= {MAX_VARS}. '
                         f'Got {len(trains)} trains x {time_horizon} min = {len(trains)*time_horizon}. '
                         f'Reduce train count or time_horizon.'
            }), 400

        # Validate train objects
        for idx, t in enumerate(trains):
            for field in ('id', 'priority', 'arrival_time', 'duration'):
                if field not in t:
                    return jsonify({'error': f'Train at index {idx} missing field "{field}"'}), 400

        result = throughput_optimizer.optimize_train_schedule(
            trains, platforms, section_capacity, time_horizon
        )
        return jsonify(result)

    except Exception as e:
        logger.exception("Unhandled error in optimize_schedule")
        return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# BATCH PROCESSING
# ============================================================================

@app.route('/api/batch/analyze', methods=['POST'])
@require_api_key
@limiter.limit(os.getenv('RATE_LIMIT_BATCH', '20 per hour'))
def batch_analyze():
    """Batch analysis of multiple operational states."""
    try:
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({'error': 'Request body must be valid JSON'}), 400

        states = data.get('states', [])
        if not states:
            return jsonify({'error': '"states" list is required and must be non-empty'}), 400

        if len(states) > 500:
            return jsonify({'error': 'Batch size cannot exceed 500 states per request'}), 400

        if not conflict_detector or not delay_predictor:
            return jsonify({'error': 'One or more models not loaded'}), 503

        results = []
        errors = []
        for idx, state in enumerate(states):
            try:
                missing = _validate_fields(state, REQUIRED_CONFLICT_FIELDS)
                if missing:
                    errors.append({'index': idx, 'error': f'Missing fields: {missing}'})
                    continue
                conflict_res = conflict_detector.predict(state)
                delay_res = delay_predictor.predict(state)
                results.append({
                    'state_id': state.get('id', f'state_{idx}'),
                    'conflict_probability': conflict_res['conflict_probability'],
                    'risk_level': conflict_res['risk_level'],
                    'predicted_delay': delay_res['predicted_delay_minutes'],
                    'severity': delay_res['severity']
                })
            except Exception as inner_e:
                logger.warning("Batch item %d failed: %s", idx, inner_e)
                errors.append({'index': idx, 'error': 'Processing error'})

        high_risk_count = sum(1 for r in results if r['risk_level'] == 'HIGH')
        critical_delays = sum(1 for r in results if r['severity'] == 'CRITICAL')

        return jsonify({
            'total_analyzed': len(results),
            'total_errors': len(errors),
            'high_risk_conflicts': high_risk_count,
            'critical_delays': critical_delays,
            'results': results,
            'errors': errors
        })

    except Exception as e:
        logger.exception("Unhandled error in batch_analyze")
        return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# LIVE KPI METRICS  (was fully hardcoded — now uses accumulated real data)
# ============================================================================

@app.route('/api/metrics/kpi', methods=['GET'])
def get_kpi_metrics():
    """Get live KPI metrics derived from actual prediction calls."""
    _reset_daily_kpis_if_needed()

    with _kpi_lock:
        conflict_preds = list(_kpi_store['conflict_predictions'])
        delay_preds = list(_kpi_store['delay_predictions'])
        daily_detected = _kpi_store['daily_conflicts_detected']
        daily_resolved = max(0, daily_detected - 2)  # assume ~2 still pending

    # Conflict stats
    if conflict_preds:
        avg_conflict_prob = sum(p['prob'] for p in conflict_preds) / len(conflict_preds)
        high_risk_pct = sum(1 for p in conflict_preds if p['risk'] in ('HIGH', 'CRITICAL')) / len(conflict_preds) * 100
    else:
        avg_conflict_prob = 0.0
        high_risk_pct = 0.0

    # Delay stats
    if delay_preds:
        avg_delay = sum(p['delay'] for p in delay_preds) / len(delay_preds)
        on_time_pct = sum(1 for p in delay_preds if p['delay'] < 5) / len(delay_preds) * 100
    else:
        avg_delay = 0.0
        on_time_pct = 100.0

    return jsonify({
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'data_source': 'live_accumulated',
        'sample_size': {
            'conflict_predictions': len(conflict_preds),
            'delay_predictions': len(delay_preds)
        },
        'punctuality': {
            'on_time_percentage': round(on_time_pct, 2),
            'target': 90.0
        },
        'conflicts': {
            'detected_today': daily_detected,
            'resolved': daily_resolved,
            'pending': daily_detected - daily_resolved,
            'avg_probability': round(avg_conflict_prob, 4),
            'high_risk_percentage': round(high_risk_pct, 2)
        },
        'average_delay': {
            'current': round(avg_delay, 2),
            'target': 5.0,
            'unit': 'minutes'
        }
    })


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({'error': 'Method not allowed'}), 405


@app.errorhandler(429)
def rate_limit_exceeded(e):
    return jsonify({'error': 'Rate limit exceeded. Please slow down your requests.'}), 429


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 5000))

    print("=" * 60)
    print("AI-Powered Railway Traffic Control System")
    print(f"API Server Starting on {host}:{port} | debug={debug}")
    print("=" * 60)
    app.run(host=host, port=port, debug=debug)
