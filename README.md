<div align="center">

# 🚆 AI-Powered Railway Traffic Control System

**SIH 2025 Project** — Smart India Hackathon

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![ML](https://img.shields.io/badge/Ensemble_ML-XGBoost%20%7C%20LightGBM%20%7C%20CatBoost-F7931E?style=for-the-badge)](https://github.com/ujjawalranjan09/railway-traffic-control-system)
[![MILP](https://img.shields.io/badge/Optimizer-MILP-009688?style=for-the-badge)](https://github.com/ujjawalranjan09/railway-traffic-control-system)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

*A production-grade AI system for real-time railway conflict prediction, scheduling optimization, and traffic control — built for Smart India Hackathon 2025.*

</div>

---

## 🎯 What This Project Does

This system solves one of Indian Railways' critical operational problems: **train scheduling conflicts and delays**. It uses an ensemble of ML models to predict conflicts before they happen, and a **MILP (Mixed Integer Linear Programming)** optimizer to automatically reschedule trains in real time to maximize throughput.

> 🏆 **Built for SIH 2025** — Smart India Hackathon, Problem Statement: Railway Traffic Optimization

---

## 📊 Key Results & Metrics

| Metric | Value |
|--------|-------|
| **Conflict Prediction F1-Score** | **0.92+** |
| **Training Dataset Size** | **500,000+ railway records** |
| **Scheduling Conflict Reduction** | **35%** |
| **Ensemble Models Used** | RF, XGBoost, LightGBM, CatBoost, Neural Net |
| **Optimizer** | MILP (Mixed Integer Linear Programming) |

---

## 🧠 ML Architecture

### 1. Conflict Prediction Engine (Ensemble ML)

An ensemble of 5 models votes on whether a scheduling conflict will occur:

```
Input Features → [Random Forest]
               → [XGBoost]       → Weighted Ensemble Vote → Conflict Prediction
               → [LightGBM]      
               → [CatBoost]      
               → [Neural Network]
```

- **Random Forest** — handles non-linear feature interactions in scheduling windows
- **XGBoost** — gradient boosting for high-accuracy conflict detection
- **LightGBM** — fast training on 500K+ records with low memory footprint
- **CatBoost** — handles categorical features (station names, route IDs) natively
- **Neural Network** — captures complex temporal dependencies in train sequences

### 2. MILP Throughput Optimizer

When conflicts are detected, a **Mixed Integer Linear Programming** model reschedules affected trains:

- **Objective:** Maximize total throughput (trains dispatched per hour)
- **Constraints:** Track capacity, signal block sections, minimum headway time, platform availability
- **Output:** Revised departure/arrival schedule with zero conflicts

### 3. Delay Predictor

A separate LightGBM model predicts expected delay in minutes for each train, enabling proactive passenger alerts.

---

## ✨ Features

- **Real-Time Conflict Detection** — Ensemble ML predicts scheduling conflicts before they happen
- **MILP-Based Rescheduling** — Automatically resolves conflicts by optimizing the full timetable
- **Delay Prediction** — Estimates delay in minutes per train using LightGBM
- **Signal State Management** — Controls green/yellow/red signal transitions
- **Multi-Track Route Optimization** — Assigns optimal tracks based on real-time traffic state
- **500K+ Record Processing** — Batch pipeline for large-scale historical data analysis
- **Collision Avoidance Logic** — Hard constraint enforcement for safety-critical scenarios

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Ensemble ML | XGBoost, LightGBM, CatBoost, scikit-learn, PyTorch |
| Optimization | PuLP / OR-Tools (MILP) |
| Data Processing | pandas, NumPy |
| Model Storage | joblib, pickle |
| Configuration | python-dotenv |

---

## 📁 Trained Model Artifacts

Pre-trained model files are stored in the [`railway-ml-models`](https://github.com/ujjawalranjan09/models) repo:

| File | Model | Purpose |
|------|-------|---------|
| `xgb_conflict_predictor.pkl` | XGBoost | Conflict prediction |
| `lgb_conflict_predictor.pkl` | LightGBM | Conflict prediction |
| `lgb_delay_predictor.pkl` | LightGBM | Delay estimation |
| `catboost_conflict_predictor.cbm` | CatBoost | Conflict prediction |
| `conflict_detector_nn.pth` | Neural Net (PyTorch) | Deep conflict detection |
| `feature_scaler.pkl` | StandardScaler | Feature normalization |

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/ujjawalranjan09/railway-traffic-control-system.git
cd railway-traffic-control-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env

# Run conflict prediction
python railway-traffic-control-system/main.py
```

---

## 🧪 How the Pipeline Works

```
Raw Railway Records (500K+)
        ↓
  Feature Engineering
  (time windows, headway, station load)
        ↓
  Ensemble ML Conflict Prediction
  (RF + XGB + LGB + CatBoost + NN)
        ↓
  Conflict Detected?
  ├── NO  → Schedule proceeds as planned
  └── YES → MILP Optimizer reschedules affected trains
                    ↓
              Conflict-Free Timetable Output
              (35% fewer conflicts vs baseline)
```

---

## 🏆 SIH 2025 Context

This project was built for **Smart India Hackathon 2025**, targeting the real-world problem of train scheduling conflicts in Indian Railways. The dataset consists of 500,000+ railway operation records processed to extract conflict patterns, delay causes, and throughput bottlenecks.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with ❤️ by [Ujjawal Ranjan](https://github.com/ujjawalranjan09) | SKIT, Jaipur**

*Optimizing Indian Railways, one schedule at a time.*

</div>
