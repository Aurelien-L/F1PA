"""
F1PA - ML Configuration

Centralizes all ML pipeline parameters to facilitate experimentation.

MODEL OBJECTIVE:
Predict a driver's lap time on a circuit BEFORE they drive,
based on their historical performance and conditions.

This is NOT a final time predictor based on sector times
(which would be trivial since lap_duration ≈ sum(sector_times)).
"""
import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Dataset
DATASET_PATH = PROCESSED_DATA / "dataset_ml_lap_level_2023_2024_2025.csv"

# Train/Test split strategy: stratified 80/20 by circuit
SPLIT_STRATEGY = "stratified"
TEST_SIZE = 0.2
STRATIFY_BY = "circuit_key"

# Legacy temporal split parameters (kept for compatibility)
TRAIN_YEARS = [2023, 2024]
TEST_YEAR = 2025

# Features to exclude from model
EXCLUDE_FEATURES = [
    'meeting_key', 'session_key',
    'session_name', 'session_type', 'location', 'country_name',
    'date_start_session', 'date_end_session', 'wikipedia_circuit_url',
    'station_id', 'gmt_offset', '__source_file',
    'lap_hour_utc',
    'lap_duration',  # Target
    'duration_sector_1', 'duration_sector_2', 'duration_sector_3',  # Current lap data (not predictive)
    'total_sector_time', 'sector_1_ratio', 'sector_2_ratio',
    'prcp', 'wspd', 'cldc', 'wdir', 'weather_severity',  # Weak weather features
    'year_avg_laptime',  # Weak performance
    'driver_avg_laptime',  # Redundant with driver_perf_score
]

# Numerical features: speeds
SPORT_FEATURES = ['st_speed', 'i1_speed', 'i2_speed']

# Weather features
WEATHER_FEATURES = ['temp', 'rhum', 'pres']

# Categorical features
CATEGORICAL_FEATURES = ['circuit_key', 'driver_number', 'year']

# Derived features (created in preprocessing)
DERIVED_FEATURES = [
    'avg_speed',
    'lap_progress',
    'driver_perf_score',
    'circuit_avg_laptime',
]

# Target
TARGET = 'lap_duration'

# GridSearch hyperparameter grids (optimized for model size and training speed)
GRIDSEARCH_PARAMS = {
    'xgboost': {
        'n_estimators': [100, 200, 300],
        'max_depth': [7, 10],
        'learning_rate': [0.05, 0.1],
        'min_child_weight': [1, 3],
        'gamma': [0, 0.1],
    },  # 48 combinations
    'random_forest': {
        'n_estimators': [150, 200],         # Optimized for ~350 MB model size
        'max_depth': [15, 20],              # Limited depth prevents overfitting
        'min_samples_split': [2, 5],        # Standard regularization
        'min_samples_leaf': [1, 2],         # Standard regularization
        'max_features': ['sqrt', 0.7],
    }  # 32 combinations
}

# Fixed parameters (shared across all GridSearch combinations)
FIXED_PARAMS = {
    'xgboost': {
        'subsample': 0.75,             # Version 2.1: 0.75 (vs 0.7 V2.0, 0.8 V1)
        'colsample_bytree': 0.75,      # Version 2.1: 0.75 (vs 0.7 V2.0, 0.8 V1)
        'reg_alpha': 0.05,             # Version 2.1: reduced L1 (vs 0.1 V2.0)
        'reg_lambda': 0.5,             # Version 2.1: reduced L2 (vs 1.0 V2.0)
        'random_state': 42,
        'n_jobs': -1,
        'enable_categorical': True
    },
    'random_forest': {
        'random_state': 42,
        'n_jobs': -1
    }
}

# Baseline models (hyperparameters by default, for comparison)
BASELINE_MODELS = {
    'xgboost': {
        'n_estimators': 300,
        'max_depth': 7,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'enable_categorical': True
    },
    'random_forest': {
        'n_estimators': 300,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1
    }
}

# GridSearch configuration
GRIDSEARCH_CV_FOLDS = 3
GRIDSEARCH_SCORING = 'neg_mean_absolute_error'

# Cross-validation
CV_FOLDS = 5
CV_STRATIFY_BY = 'circuit_key'

# MLflow configuration
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "F1PA_LapTime_Prediction")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Evaluation metrics
METRICS = ['mae', 'rmse', 'r2', 'mape']

# Performance targets for validation
PERFORMANCE_TARGETS = {
    'mae_excellent': 2.0,
    'mae_good': 5.0,
    'mae_acceptable': 10.0,
    'r2_target': 0.85
}

# Random seed for reproducibility
RANDOM_STATE = 42
