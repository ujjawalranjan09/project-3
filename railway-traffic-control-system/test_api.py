#!/usr/bin/env python3
"""
API Test Script
Tests all endpoints of the Railway Traffic Control System
"""

import requests
import json
import sys

API_BASE_URL = 'http://localhost:5000/api'

def print_header(text):
    print("\n" + "="*60)
    print(text)
    print("="*60)

def test_health():
    print_header("Testing Health Endpoint")
    try:
        response = requests.get(f'{API_BASE_URL}/health')
        print(f"Status: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_conflict_detection():
    print_header("Testing Conflict Detection")
    data = {
        "trains_in_section": 35,
        "available_platforms": 3,
        "platform_utilization": 95.0,
        "weather_severity": 0.4,
        "rainfall_mm": 3.5,
        "fog_intensity": 0.6,
        "temperature_c": 23.0,
        "is_peak_hour": 1
    }

    try:
        response = requests.post(f'{API_BASE_URL}/predict/conflict', json=data)
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Conflict Probability: {result.get('conflict_probability', 'N/A')}")
        print(f"Risk Level: {result.get('risk_level', 'N/A')}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_delay_prediction():
    print_header("Testing Delay Prediction")
    data = {
        "trains_in_section": 35,
        "available_platforms": 3,
        "platform_utilization": 95.0,
        "weather_severity": 0.4,
        "rainfall_mm": 3.5,
        "fog_intensity": 0.6,
        "temperature_c": 23.0,
        "is_peak_hour": 1
    }

    try:
        response = requests.post(f'{API_BASE_URL}/predict/delay', json=data)
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Predicted Delay: {result.get('predicted_delay_minutes', 'N/A')} minutes")
        print(f"Severity: {result.get('severity', 'N/A')}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_simulation():
    print_header("Testing What-If Simulation")
    data = {
        "baseline": {
            "trains_in_section": 35,
            "available_platforms": 3,
            "platform_utilization": 95.0,
            "weather_severity": 0.4,
            "rainfall_mm": 3.5,
            "fog_intensity": 0.6,
            "temperature_c": 23.0,
            "is_peak_hour": 1
        },
        "modifications": {
            "trains_in_section": 25,
            "available_platforms": 4
        }
    }

    try:
        response = requests.post(f'{API_BASE_URL}/simulate', json=data)
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Delay Reduction: {result.get('delay_reduction_minutes', 'N/A')} minutes")
        print(f"Improvement: {result.get('improvement_percentage', 'N/A')}%")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_kpi_metrics():
    print_header("Testing KPI Metrics")
    try:
        response = requests.get(f'{API_BASE_URL}/metrics/kpi')
        print(f"Status: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("Railway Traffic Control System - API Test Suite")
    print("="*60)

    tests = [
        ("Health Check", test_health),
        ("Conflict Detection", test_conflict_detection),
        ("Delay Prediction", test_delay_prediction),
        ("What-If Simulation", test_simulation),
        ("KPI Metrics", test_kpi_metrics)
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"Test failed with exception: {e}")
            results.append((name, False))

    # Summary
    print_header("Test Summary")
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:10s} {name}")

    print(f"\nTotal: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
