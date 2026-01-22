"""
F1PA - ML Configuration

Centralise tous les paramètres du pipeline ML pour faciliter les expérimentations.
"""
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
TRAIN_YEARS = [2023, 2024]  # 47,266 laps (66%)
TEST_YEAR = 2025             # 24,379 laps (34%)

# Features
# Features à exclure (identifiants, metadata, target)
EXCLUDE_FEATURES = [
    # Identifiants
    'meeting_key', 'session_key', 'driver_number', 'circuit_key',
    # Metadata non prédictives
    'session_name', 'session_type', 'location', 'country_name',
    'date_start_session', 'date_end_session', 'wikipedia_circuit_url',
    'station_id', 'gmt_offset', '__source_file',
    # Temporelles (utiles pour imputation mais pas en features)
    'lap_hour_utc',
    # Target
    'lap_duration',
    # Features faibles (importance < 0.001) - Solution anti-overfitting
    'year_avg_laptime',    # importance: 2.6e-07 (quasi nulle)
    'prcp',                # importance: 1.8e-05
    'wspd',                # importance: 5.2e-05
    'cldc',                # importance: 7.0e-05
    'weather_severity',    # importance: 0.00016
    'driver_avg_laptime',  # importance: 0.00029
    'wdir'                 # importance: 0.00048
]

# Features numériques sport
SPORT_FEATURES = ['st_speed', 'i1_speed', 'i2_speed',
                  'duration_sector_1', 'duration_sector_2', 'duration_sector_3']

# Features numériques météo
WEATHER_FEATURES = ['temp', 'rhum', 'pres', 'wspd', 'wdir', 'prcp', 'cldc']

# Features catégorielles
CATEGORICAL_FEATURES = ['circuit_key', 'driver_number', 'year']

# Features dérivées (créées dans preprocessing)
DERIVED_FEATURES = [
    'avg_speed',           # Vitesse moyenne
    'total_sector_time',   # Somme des secteurs
    'sector_1_ratio',      # % temps secteur 1
    'sector_2_ratio',      # % temps secteur 2
    'weather_severity',    # Composite vent + pluie
    'lap_progress'         # Progression dans la course
]

# Target
TARGET = 'lap_duration'

# GridSearch: Grilles de paramètres pour tuning léger avec régularisation équilibrée (Version 2.1)
GRIDSEARCH_PARAMS = {
    'xgboost': {
        'n_estimators': [50, 100, 150],     # Version 2.1: ajout 150 (vs [50, 100] V2.0)
        'max_depth': [4, 5],                # Version 2.1: 4-5 (vs [3, 5] V2.0)
        'learning_rate': [0.03, 0.05],      # Version 2.1: 0.03-0.05 (vs [0.01, 0.05] V2.0)
        'min_child_weight': [3, 5],         # OK: Régularisation
        'gamma': [0.05, 0.15],              # Version 2.1: réduit (vs [0.1, 0.3] V2.0)
    },
    'random_forest': {
        'n_estimators': [50, 100],          # Réduit (était [100, 300])
        'max_depth': [5, 8],                # Réduit (était [10, 15])
        'min_samples_split': [10, 20],      # Augmenté (était [2, 5])
        'min_samples_leaf': [5, 10],        # Augmenté (était implicite 2)
        'max_features': [0.5, 0.7],         # NOUVEAU: Réduire features par split
    }
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

# MLflow
MLFLOW_EXPERIMENT_NAME = "F1PA_LapTime_Prediction"
MLFLOW_TRACKING_URI = "http://localhost:5000"

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
