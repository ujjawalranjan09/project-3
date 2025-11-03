"""
AI-Powered Railway Traffic Control System
ML Model Training Pipeline
Author: Railway Optimization Team
Date: November 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, r2_score
import xgboost as xgb
from lightgbm import LGBMClassifier, LGBMRegressor
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class RailwayModelTrainer:
    """
    Comprehensive ML model training for conflict detection and delay prediction
    """

    def __init__(self, data_path='railway_OPTIMIZED_BALANCED_500K.csv'):
        self.data_path = data_path
        self.conflict_model = None
        self.delay_model = None
        self.feature_importance = {}
        self.metrics = {}

    def load_and_preprocess(self):
        """Load dataset and perform initial preprocessing"""
        print("Loading dataset...")
        df = pd.read_csv(self.data_path)

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Extract time-based features
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Create composite features based on correlation analysis
        df['train_density_risk'] = df['trains_in_section'] / (df['available_platforms'] + 1)
        df['weather_impact'] = df['weather_severity'] * df['trains_in_section']
        df['congestion_score'] = df['platform_utilization'] * df['trains_in_section'] / 100

        self.df = df
        print(f"Dataset loaded: {len(df)} records, {len(df.columns)} features")
        return df

    def prepare_features(self):
        """Prepare feature sets for training"""
        # Features for conflict detection (classification)
        
        conflict_features = [
            'trains_in_section', 'available_platforms', 'platform_utilization',
            'weather_severity', 'rainfall_mm', 'fog_intensity', 'temperature_c',
            'hour', 'is_peak_hour', 'day_of_week', 'month',
            'hour_sin', 'hour_cos', 'train_density_risk', 
            'weather_impact', 'congestion_score'
        ]


        # Features for delay prediction (regression)
        delay_features = conflict_features.copy()

        # Prepare datasets
        X_conflict = self.df[conflict_features]
        y_conflict = self.df['conflict_occurred']

        X_delay = self.df[delay_features]
        y_delay = self.df['delay_minutes']

        return X_conflict, y_conflict, X_delay, y_delay

    def train_conflict_detector(self, X, y):
        """Train ensemble model for conflict detection"""
        print("\n" + "="*60)
        print("TRAINING CONFLICT DETECTION MODEL")
        print("="*60)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train Random Forest
        print("\nTraining Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)

        # Train XGBoost
        print("Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)

        # Train LightGBM
        print("Training LightGBM...")
        lgbm_model = LGBMClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgbm_model.fit(X_train, y_train)

        # Evaluate models
        models = {
            'RandomForest': rf_model,
            'XGBoost': xgb_model,
            'LightGBM': lgbm_model
        }

        best_score = 0
        best_model = None
        best_name = None

        print("\nModel Evaluation:")
        print("-" * 60)

        for name, model in models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            print(f"\n{name}:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")

            if f1 > best_score:
                best_score = f1
                best_model = model
                best_name = name

        print(f"\n✓ Best Model: {best_name} (F1={best_score:.4f})")

        self.conflict_model = best_model
        self.metrics['conflict_detection'] = {
            'model': best_name,
            'f1_score': best_score,
            'accuracy': accuracy_score(y_test, best_model.predict(X_test))
        }

        # Feature importance
        if hasattr(best_model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': X.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)

            print("\nTop 10 Important Features:")
            print("-" * 60)
            for idx, row in importance.head(10).iterrows():
                print(f"  {row['feature']:30s} {row['importance']:.4f}")

            self.feature_importance['conflict'] = importance.to_dict('records')

        return best_model

    def train_delay_predictor(self, X, y):
        """Train ensemble model for delay prediction"""
        print("\n" + "="*60)
        print("TRAINING DELAY PREDICTION MODEL")
        print("="*60)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train Random Forest Regressor
        print("\nTraining Random Forest Regressor...")
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)

        # Train XGBoost Regressor
        print("Training XGBoost Regressor...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)

        # Train LightGBM Regressor
        print("Training LightGBM Regressor...")
        lgbm_model = LGBMRegressor(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgbm_model.fit(X_train, y_train)

        # Evaluate models
        models = {
            'RandomForest': rf_model,
            'XGBoost': xgb_model,
            'LightGBM': lgbm_model
        }

        best_score = float('inf')
        best_model = None
        best_name = None

        print("\nModel Evaluation:")
        print("-" * 60)

        for name, model in models.items():
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            print(f"\n{name}:")
            print(f"  MAE:       {mae:.4f} minutes")
            print(f"  R² Score:  {r2:.4f}")

            if mae < best_score:
                best_score = mae
                best_model = model
                best_name = name

        print(f"\n✓ Best Model: {best_name} (MAE={best_score:.4f} min)")

        self.delay_model = best_model
        self.metrics['delay_prediction'] = {
            'model': best_name,
            'mae': best_score,
            'r2_score': r2_score(y_test, best_model.predict(X_test))
        }

        # Feature importance
        if hasattr(best_model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': X.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)

            print("\nTop 10 Important Features:")
            print("-" * 60)
            for idx, row in importance.head(10).iterrows():
                print(f"  {row['feature']:30s} {row['importance']:.4f}")

            self.feature_importance['delay'] = importance.to_dict('records')

        return best_model

    def save_models(self, output_dir='backend/models'):
        """Save trained models and metadata"""
        print("\n" + "="*60)
        print("SAVING MODELS")
        print("="*60)

        # Save conflict detection model
        conflict_path = f'{output_dir}/conflict_detector.pkl'
        joblib.dump(self.conflict_model, conflict_path)
        print(f"✓ Conflict detector saved: {conflict_path}")

        # Save delay prediction model
        delay_path = f'{output_dir}/delay_predictor.pkl'
        joblib.dump(self.delay_model, delay_path)
        print(f"✓ Delay predictor saved: {delay_path}")

        # Save metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'dataset_size': len(self.df),
            'metrics': self.metrics,
            'feature_importance': self.feature_importance
        }

        metadata_path = f'{output_dir}/model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata saved: {metadata_path}")

        print("\n✓ All models saved successfully!")

    def train_all(self):
        """Complete training pipeline"""
        print("\n" + "="*60)
        print("AI-POWERED RAILWAY TRAFFIC CONTROL SYSTEM")
        print("ML Model Training Pipeline")
        print("="*60)

        # Load data
        self.load_and_preprocess()

        # Prepare features
        X_conflict, y_conflict, X_delay, y_delay = self.prepare_features()

        # Train models
        self.train_conflict_detector(X_conflict, y_conflict)
        self.train_delay_predictor(X_delay, y_delay)

        # Save models
        self.save_models()

        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print("\nModels ready for deployment.")


if __name__ == "__main__":
    trainer = RailwayModelTrainer()
    trainer.train_all()
