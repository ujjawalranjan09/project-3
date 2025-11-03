# AI-Powered Railway Traffic Control System

## Overview

An intelligent decision-support system for railway traffic controllers that uses machine learning and optimization algorithms to:

- **Detect conflicts** in real-time with probability assessment
- **Predict delays** with mitigation strategies
- **Optimize throughput** using Mixed Integer Linear Programming (MILP)
- **Simulate scenarios** for what-if analysis
- **Provide recommendations** for operational decisions

## System Architecture

```
├── backend/
│   ├── models/                    # Trained ML models
│   ├── optimization/              # MILP and RL optimizers
│   ├── api/                       # Flask REST API
│   └── utils/                     # Data processing utilities
├── frontend/
│   ├── dashboard.html             # Main dashboard
│   ├── components/                # JavaScript modules
│   └── styles/                    # CSS styling
├── config/
│   └── config.yaml                # System configuration
└── data/                          # Dataset storage
```

## Key Features

### 1. Real-time Conflict Detection
- **ML Model**: Ensemble of Random Forest, XGBoost, and LightGBM
- **Accuracy**: ~93% correlation with actual conflicts
- **Risk Levels**: LOW, MEDIUM, HIGH based on probability thresholds
- **Recommendations**: Automated action items for high-risk scenarios

### 2. Delay Prediction
- **Model**: Regression ensemble (RF, XGBoost, LightGBM)
- **Features**: 15+ engineered features including train density, weather, platform utilization
- **Output**: Predicted delay in minutes with severity categorization
- **Mitigation**: Context-aware strategies to reduce delays

### 3. Throughput Optimization
- **Algorithm**: Mixed Integer Linear Programming (MILP)
- **Objective**: Maximize section throughput while respecting constraints
- **Constraints**: Platform capacity, section capacity, train priorities, safety margins
- **Output**: Optimized train schedule with platform allocations

### 4. What-If Simulation
- **Capability**: Test operational changes before implementation
- **Metrics**: Delay reduction, conflict probability, throughput impact
- **Use Cases**: Resource allocation, weather contingency planning

## Installation

### Prerequisites
- Python 3.10+
- Docker (optional, for containerized deployment)
- 8GB+ RAM recommended

### Quick Start

1. **Clone the repository**
```bash
git clone <repository-url>
cd railway-traffic-control
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Train ML models**
```bash
# Place your dataset as: railway_OPTIMIZED_BALANCED_500K.csv
python train_ml_models.py
```

4. **Start the API server**
```bash
python flask_app.py
```

5. **Open the dashboard**
```bash
# Open frontend/dashboard.html in your browser
# Or serve via HTTP server:
cd frontend
python -m http.server 8080
```

## Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Access dashboard at: http://localhost
# API available at: http://localhost:5000
```

## API Endpoints

### Health Check
```
GET /api/health
```

### Conflict Detection
```
POST /api/predict/conflict
Body: {
  "trains_in_section": 35,
  "available_platforms": 3,
  "platform_utilization_pct": 95.0,
  "weather_severity": 0.4,
  "rainfall_mm": 3.5,
  "fog_intensity": 0.6,
  "temperature_c": 23.0,
  "is_peak_hour": 1
}
```

### Delay Prediction
```
POST /api/predict/delay
Body: (same as conflict detection)
```

### Combined Analysis
```
POST /api/predict/combined
Body: (same as conflict detection)
```

### Scenario Simulation
```
POST /api/simulate
Body: {
  "baseline": {...},
  "modifications": {
    "trains_in_section": 25,
    "available_platforms": 4
  }
}
```

### Schedule Optimization
```
POST /api/optimize/schedule
Body: {
  "trains": [
    {"id": "T001", "priority": 10, "arrival_time": 0, "duration": 8}
  ],
  "platforms": 3,
  "section_capacity": 25,
  "time_horizon": 60
}
```

## Dataset

The system is trained on a balanced dataset with:
- **500,000 records**
- **15 features** including operational metrics, weather conditions, and temporal data
- **Perfect balance**: 50% conflict, 50% no-conflict scenarios
- **Key predictors**: trains_in_section (0.88 correlation), weather_severity (0.82), available_platforms (-0.57)

## Model Performance

### Conflict Detection
- **F1-Score**: 0.92+
- **Precision**: 0.90+
- **Recall**: 0.93+

### Delay Prediction
- **MAE**: 3-5 minutes
- **R² Score**: 0.85+

## Configuration

Edit `config/config.yaml` to customize:
- Model thresholds
- API settings
- Optimization parameters
- Alert configurations

## Use Cases

1. **Section Controllers**: Real-time decision support for train precedence and crossings
2. **Operations Managers**: KPI monitoring and performance analysis
3. **Planning Teams**: Scenario simulation for capacity planning
4. **Maintenance Coordination**: Conflict-free scheduling of track work

## Performance Metrics (KPIs)

The system tracks:
- **Throughput**: Trains per hour
- **Punctuality**: On-time percentage
- **Average Delay**: Minutes
- **Active Conflicts**: Real-time count
- **Platform Utilization**: Percentage

## Contributing

This system was developed for Smart India Hackathon 2025, Problem Statement 25022.

## License

MIT License

## Support

For issues or questions, please open a GitHub issue or contact the development team.

## Acknowledgments

- Indian Railways for the problem statement
- Smart India Hackathon 2025
- Open-source ML and optimization libraries

---

**Version**: 1.0.0  
**Last Updated**: November 2025  
**Status**: Production Ready
