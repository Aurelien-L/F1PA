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

# Train/Test split
# V2: Split 80/20 stratifié par circuit (au lieu de temporel strict)
# Permet d'inclure des données 2025 dans le train pour réduire le concept drift
SPLIT_STRATEGY = "stratified"  # "temporal" ou "stratified"
TEST_SIZE = 0.2  # 20% pour le test
STRATIFY_BY = "circuit_key"  # Assurer distribution circuits équilibrée

# Legacy (pour compatibilité si SPLIT_STRATEGY = "temporal")
TRAIN_YEARS = [2023, 2024]
TEST_YEAR = 2025

# Features
# Features à exclure (identifiants, metadata, target, et données du tour en cours)
EXCLUDE_FEATURES = [
    # Identifiants (gardés pour groupby mais pas comme features directes)
    'meeting_key', 'session_key',
    # Metadata non prédictives
    'session_name', 'session_type', 'location', 'country_name',
    'date_start_session', 'date_end_session', 'wikipedia_circuit_url',
    'station_id', 'gmt_offset', '__source_file',
    # Temporelles
    'lap_hour_utc',
    # Target
    'lap_duration',
    # ⚠️ EXCLUS: Temps secteurs (ce sont des données du tour en cours, pas prédictives!)
    # Le modèle doit prédire AVANT le tour, pas pendant
    'duration_sector_1', 'duration_sector_2', 'duration_sector_3',
    # Features dérivées des secteurs (également exclues)
    'total_sector_time', 'sector_1_ratio', 'sector_2_ratio',
    # Features météo faibles
    'prcp', 'wspd', 'cldc', 'wdir',
    'weather_severity',
    # year_avg_laptime (trop faible)
    'year_avg_laptime',
]

# Features numériques sport (vitesses uniquement - pas les temps secteurs!)
# Les vitesses sont des indicateurs de performance sans donner le temps directement
SPORT_FEATURES = ['st_speed', 'i1_speed', 'i2_speed']

# Features numériques météo (principales)
WEATHER_FEATURES = ['temp', 'rhum', 'pres']

# Features catégorielles (seront encodées)
CATEGORICAL_FEATURES = ['circuit_key', 'driver_number', 'year']

# Features dérivées (créées dans preprocessing)
DERIVED_FEATURES = [
    'avg_speed',              # Vitesse moyenne (performance globale)
    'lap_progress',           # Progression dans la session
    'driver_perf_score',      # Score de performance historique du pilote
    'circuit_avg_laptime',    # Temps moyen du circuit (difficulté)
    'driver_avg_laptime',     # Temps moyen du pilote (skill)
]

# Target
TARGET = 'lap_duration'

# GridSearch: Grilles de paramètres (Version 3.1 - optimisée pour vitesse)
# XGBoost: grille large car rapide à entraîner
# Random Forest: grille réduite car lent (focus sur les meilleurs paramètres)
GRIDSEARCH_PARAMS = {
    'xgboost': {
        'n_estimators': [100, 200, 300],    # 3 valeurs
        'max_depth': [7, 10],               # 2 valeurs (best=10)
        'learning_rate': [0.05, 0.1],       # 2 valeurs (best=0.05)
        'min_child_weight': [1, 3],         # 2 valeurs (best=1)
        'gamma': [0, 0.1],                  # 2 valeurs (best=0.1)
    },  # Total: 3×2×2×2×2 = 48 combinaisons
    'random_forest': {
        'n_estimators': [200, 300],         # 2 valeurs (baseline=300)
        'max_depth': [15, None],            # 2 valeurs (baseline=15)
        'min_samples_split': [2, 5],        # 2 valeurs (baseline=5)
        'min_samples_leaf': [1, 2],         # 2 valeurs (baseline=2)
        'max_features': ['sqrt', 0.7],      # 2 valeurs
    }  # Total: 2×2×2×2×2 = 32 combinaisons
}

# Paramètres fixes (communs à toutes les combinaisons GridSearch)
FIXED_PARAMS = {
    'xgboost': {
        'subsample': 0.75,             # Version 2.1: 0.75 (vs 0.7 V2.0, 0.8 V1)
        'colsample_bytree': 0.75,      # Version 2.1: 0.75 (vs 0.7 V2.0, 0.8 V1)
        'reg_alpha': 0.05,             # Version 2.1: réduit L1 (vs 0.1 V2.0)
        'reg_lambda': 0.5,             # Version 2.1: réduit L2 (vs 1.0 V2.0)
        'random_state': 42,
        'n_jobs': -1,
        'enable_categorical': True
    },
    'random_forest': {
        'random_state': 42,
        'n_jobs': -1
    }
}

# Modèles baseline (hyperparamètres par défaut, pour comparaison)
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
GRIDSEARCH_CV_FOLDS = 3  # 3-fold pour GridSearch (plus rapide que 5-fold)
GRIDSEARCH_SCORING = 'neg_mean_absolute_error'  # Métrique à optimiser

# Cross-validation
CV_FOLDS = 5
CV_STRATIFY_BY = 'circuit_key'  # Assurer distribution circuits équilibrée

# MLflow (from environment or defaults)
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "F1PA_LapTime_Prediction")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Métriques d'évaluation
METRICS = ['mae', 'rmse', 'r2', 'mape']

# Objectifs de performance (pour validation)
PERFORMANCE_TARGETS = {
    'mae_excellent': 2.0,   # < 2s = Excellent
    'mae_good': 5.0,        # < 5s = Bon
    'mae_acceptable': 10.0, # < 10s = Acceptable
    'r2_target': 0.85       # > 0.85 = Très bon
}

# Random seed pour reproductibilité
RANDOM_STATE = 42
