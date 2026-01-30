"""
Tests unitaires pour le preprocessing ML
"""
import pytest

def test_target_variable():
    """Test: checkr que la variable cible est définie"""
    from ml.config import TARGET
    
    assert TARGET == 'lap_duration'

def test_test_size_config():
    """Test: checkr la config du split train/test"""
    from ml.config import TEST_SIZE
    
    assert 0 < TEST_SIZE < 1
    assert TEST_SIZE == 0.2  # 80/20 split

def test_cv_folds():
    """Test: checkr le nombre de folds for CV"""
    from ml.config import CV_FOLDS
    
    assert CV_FOLDS >= 3
    assert isinstance(CV_FOLDS, int)

def test_mlflow_config():
    """Test: checkr la config MLflow"""
    from ml.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI
    
    assert MLFLOW_EXPERIMENT_NAME is not None
    assert MLFLOW_TRACKING_URI is not None
    assert isinstance(MLFLOW_EXPERIMENT_NAME, str)
    assert isinstance(MLFLOW_TRACKING_URI, str)

def test_gridsearch_params():
    """Test: checkr les parameters GridSearch"""
    from ml.config import GRIDSEARCH_PARAMS
    
    assert isinstance(GRIDSEARCH_PARAMS, dict)
    assert 'random_forest' in GRIDSEARCH_PARAMS
    assert 'xgboost' in GRIDSEARCH_PARAMS
    
    # Checkr que ce sont des dictionnaires with des listes
    for model_name, params in GRIDSEARCH_PARAMS.items():
        assert isinstance(params, dict)

def test_features_groups():
    """Test: checkr les groupes de features"""
    from ml.config import SPORT_FEATURES, WEATHER_FEATURES, CATEGORICAL_FEATURES
    
    assert isinstance(SPORT_FEATURES, list)
    assert isinstance(WEATHER_FEATURES, list)
    assert isinstance(CATEGORICAL_FEATURES, list)
    
    # Vérifier des features attendues
    assert 'st_speed' in SPORT_FEATURES
    assert 'temp' in WEATHER_FEATURES
    assert 'circuit_key' in CATEGORICAL_FEATURES

def test_random_state():
    """Test: checkr que le random state est fixé"""
    from ml.config import RANDOM_STATE
    
    assert RANDOM_STATE == 42
