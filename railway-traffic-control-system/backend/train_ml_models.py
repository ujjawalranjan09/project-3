"""
AI-Powered Railway Traffic Control System
ML Model Training Pipeline
Author: Railway Optimization Team
Date: November 2025
"""

import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)
import xgboost as xgb
from lightgbm import LGBMClassifier, LGBMRegressor
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class RailwayModelTrainer:
    """
    Comprehensive ML model training for conflict detection and delay prediction.
    """

    # Conflict classifier features — does NOT include delay_minutes to prevent leakage
    CONFLICT_FEATURES = [
        'trains_in_section', 'available_platforms', 'platform_utilization',
        'weather_severity', 'rainfall_mm', 'fog_intensity', 'temperature_c',
        'hour', 'is_peak_hour', 'day_of_week', 'month',
        'hour_sin', 'hour_cos', 'train_density_risk',
        'weather_impact', 'congestion_score'
    ]

    # Delay regressor features — does NOT include conflict_occurred to prevent leakage
    DELAY_FEATURES = [
        'trains_in_section', 'available_platforms', 'platform_utilization',
        'weather_severity', 'rainfall_mm', 'fog_intensity', 'temperature_c',
        'hour', 'is_peak_hour', 'day_of_week', 'month',
        'hour_sin', 'hour_cos', 'train_density_risk',
        'weather_impact', 'congestion_score'
    ]

    def __init__(self, data_path: str = 'railway_OPTIMIZED_BALANCED_500K.csv'):
        self.data_path = data_path
        self.df = None
        self.conflict_model = None
        self.delay_model = None
        self.feature_importance = {}
        self.metrics = {}

    def load_and_preprocess(self) -> pd.DataFrame:
        """Load dataset and perform initial preprocessing."""
        logger.info("Loading dataset from %s ...", self.data_path)
        df = pd.read_csv(self.data_path)

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        df['train_density_risk'] = df['trains_in_section'] / (df['available_platforms'] + 1)
        df['weather_impact'] = df['weather_severity'] * df['trains_in_section']
        df['congestion_score'] = df['platform_utilization'] * df['trains_in_section'] / 100

        # Drop rows with NaN in key columns
        key_cols = self.CONFLICT_FEATURES + ['conflict_occurred', 'delay_minutes']
        initial_len = len(df)
        df = df.dropna(subset=key_cols)
        if len(df) < initial_len:
            logger.warning("Dropped %d rows with NaN values", initial_len - len(df))

        self.df = df
        logger.info("Dataset loaded: %d records, %d features", len(df), len(df.columns))
        return df

    def prepare_features(self):
        """Prepare separate feature sets for conflict and delay models."""
        # Conflict: classification — target is binary conflict_occurred
        X_conflict = self.df[self.CONFLICT_FEATURES]
        y_conflict = self.df['conflict_occurred']

        # Delay: regression — target is continuous delay_minutes
        # Intentionally excludes conflict_occurred to avoid data leakage
        X_delay = self.df[self.DELAY_FEATURES]
        y_delay = self.df['delay_minutes']

        return X_conflict, y_conflict, X_delay, y_delay

    def train_conflict_detector(self, X: pd.DataFrame, y: pd.Series):
        """Train and cross-validate ensemble models for conflict detection."""
        logger.info("=" * 60)
        logger.info("TRAINING CONFLICT DETECTION MODEL")
        logger.info("=" * 60)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200, max_depth=20, min_samples_split=10,
                min_samples_leaf=5, random_state=42, n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200, max_depth=10, learning_rate=0.1,
                random_state=42, n_jobs=-1, eval_metric='logloss'
            ),
            'LightGBM': LGBMClassifier(
                n_estimators=200, max_depth=10, learning_rate=0.1,
                random_state=42, n_jobs=-1, verbose=-1
            )
        }

        best_f1, best_model, best_name = 0.0, None, None
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        logger.info("Model Evaluation (test set + 5-fold CV):")
        logger.info("-" * 60)

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            # Cross-validation F1
            cv_f1 = cross_val_score(model, X, y, cv=cv, scoring='f1', n_jobs=-1).mean()

            logger.info("%s: Acc=%.4f Prec=%.4f Rec=%.4f F1=%.4f CV-F1=%.4f",
                        name, acc, prec, rec, f1, cv_f1)

            if f1 > best_f1:
                best_f1, best_model, best_name = f1, model, name

        logger.info("Best Model: %s (F1=%.4f)", best_name, best_f1)
        self.conflict_model = best_model
        self.metrics['conflict_detection'] = {
            'model': best_name,
            'f1_score': best_f1,
            'accuracy': accuracy_score(y_test, best_model.predict(X_test))
        }

        self._log_feature_importance(best_model, X, 'conflict')
        return best_model

    def train_delay_predictor(self, X: pd.DataFrame, y: pd.Series):
        """Train ensemble models for delay prediction with MAE + RMSE reporting."""
        logger.info("=" * 60)
        logger.info("TRAINING DELAY PREDICTION MODEL")
        logger.info("=" * 60)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=200, max_depth=20, min_samples_split=10,
                min_samples_leaf=5, random_state=42, n_jobs=-1
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=200, max_depth=10, learning_rate=0.1,
                random_state=42, n_jobs=-1
            ),
            'LightGBM': LGBMRegressor(
                n_estimators=200, max_depth=10, learning_rate=0.1,
                random_state=42, n_jobs=-1, verbose=-1
            )
        }

        best_mae, best_model, best_name = float('inf'), None, None

        logger.info("Model Evaluation:")
        logger.info("-" * 60)

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            # Cross-validation MAE
            cv_mae = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1).mean()

            logger.info("%s: MAE=%.4f RMSE=%.4f R2=%.4f CV-MAE=%.4f",
                        name, mae, rmse, r2, cv_mae)

            if mae < best_mae:
                best_mae, best_model, best_name = mae, model, name

        logger.info("Best Model: %s (MAE=%.4f min)", best_name, best_mae)
        self.delay_model = best_model

        y_best_pred = best_model.predict(X_test)
        self.metrics['delay_prediction'] = {
            'model': best_name,
            'mae': best_mae,
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_best_pred))),
            'r2_score': float(r2_score(y_test, y_best_pred))
        }

        self._log_feature_importance(best_model, X, 'delay')
        return best_model

    def _log_feature_importance(self, model, X: pd.DataFrame, label: str):
        """Log and store feature importances if the model supports them."""
        if not hasattr(model, 'feature_importances_'):
            return
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info("Top 10 features (%s):", label)
        for _, row in importance_df.head(10).iterrows():
            logger.info("  %-30s %.4f", row['feature'], row['importance'])

        self.feature_importance[label] = importance_df.to_dict('records')

    def save_models(self, output_dir: str = 'backend/models'):
        """Save trained models and metadata with versioning."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Version tag based on training date
        version = datetime.utcnow().strftime('%Y%m%d_%H%M%S')

        conflict_path = f'{output_dir}/conflict_detector.pkl'
        joblib.dump(self.conflict_model, conflict_path)
        logger.info("Conflict detector saved: %s", conflict_path)

        delay_path = f'{output_dir}/delay_predictor.pkl'
        joblib.dump(self.delay_model, delay_path)
        logger.info("Delay predictor saved: %s", delay_path)

        metadata = {
            'training_date': datetime.utcnow().isoformat(),
            'version': version,
            'dataset_size': len(self.df),
            'metrics': self.metrics,
            'feature_importance': self.feature_importance,
            'conflict_features': self.CONFLICT_FEATURES,
            'delay_features': self.DELAY_FEATURES
        }

        meta_path = f'{output_dir}/model_metadata.json'
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info("Metadata saved: %s (version=%s)", meta_path, version)

    def train_all(self):
        """Complete training pipeline."""
        logger.info("=" * 60)
        logger.info("AI-POWERED RAILWAY TRAFFIC CONTROL SYSTEM")
        logger.info("ML Model Training Pipeline")
        logger.info("=" * 60)

        self.load_and_preprocess()
        X_conflict, y_conflict, X_delay, y_delay = self.prepare_features()
        self.train_conflict_detector(X_conflict, y_conflict)
        self.train_delay_predictor(X_delay, y_delay)
        self.save_models()

        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE — models ready for deployment.")
        logger.info("=" * 60)


if __name__ == '__main__':
    trainer = RailwayModelTrainer()
    trainer.train_all()
