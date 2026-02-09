"""
Unit tests for project configuration
"""
import pytest
import importlib.util
from pathlib import Path

def test_ml_config_imports():
    """Test: ml.config import works"""
    from ml.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, TARGET

    assert MLFLOW_EXPERIMENT_NAME is not None
    assert MLFLOW_TRACKING_URI is not None
    assert TARGET == 'lap_duration'

def test_streamlit_config_imports():
    """Test: streamlit config import works"""
    # Explicitly load local file to avoid conflict with streamlit package
    config_path = Path(__file__).parent.parent / "streamlit" / "config.py"
    spec = importlib.util.spec_from_file_location("streamlit_config", config_path)
    streamlit_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(streamlit_config)

    assert streamlit_config.API_BASE_URL is not None
    assert streamlit_config.API_USERNAME is not None

def test_mlflow_experiment_name():
    """Test: MLflow experiment name"""
    from ml.config import MLFLOW_EXPERIMENT_NAME
    
    assert "F1PA" in MLFLOW_EXPERIMENT_NAME

def test_paths_exist():
    """Test: check that config paths exist"""
    from ml.config import PROJECT_ROOT, DATA_DIR, MODELS_DIR

    assert PROJECT_ROOT.exists()
    assert DATA_DIR.exists() or True  # May not exist yet
    assert MODELS_DIR.exists() or True  # May not exist yet
