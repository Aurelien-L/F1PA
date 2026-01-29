"""
Extended tests for FastAPI API - Tests via real requests
"""
import pytest
import requests
from requests.auth import HTTPBasicAuth

@pytest.fixture
def base_url():
    """Base URL of the API"""
    return "http://localhost:8000"

def test_services_are_up(check_services):
    """Test: check that required services are UP"""
    assert check_services is True

# ============================================================================
# Tests endpoints /data/*
# ============================================================================

def test_drivers_list_complete(base_url, api_credentials):
    """Test: complete and well-structured drivers list"""
    response = requests.get(
        f"{base_url}/data/drivers",
        auth=HTTPBasicAuth(api_credentials["username"], api_credentials["password"])
    )
    assert response.status_code == 200
    drivers = response.json()

    assert isinstance(drivers, list)
    assert len(drivers) > 0, "Drivers list should not be empty"

    # Check first driver structure
    driver = drivers[0]
    required_fields = ["driver_number", "full_name", "name_acronym"]
    for field in required_fields:
        assert field in driver, f"Field '{field}' missing"

    # Check types
    assert isinstance(driver["driver_number"], int)
    assert isinstance(driver["full_name"], str)

    # Check that headshot_url is present
    assert "headshot_url" in driver
    assert "team_colour" in driver

def test_circuits_list_complete(base_url, api_credentials):
    """Test: complete and well-structured circuits list"""
    response = requests.get(
        f"{base_url}/data/circuits",
        auth=HTTPBasicAuth(api_credentials["username"], api_credentials["password"])
    )
    assert response.status_code == 200
    circuits = response.json()

    assert isinstance(circuits, list)
    assert len(circuits) > 0, "Circuits list should not be empty"

    # Check structure
    circuit = circuits[0]
    required_fields = ["circuit_key", "circuit_short_name", "country_name"]
    for field in required_fields:
        assert field in circuit

def test_prediction_lap_valid(base_url, api_credentials, sample_features):
    """Test: prediction with features valid return result consistent"""
    response = requests.post(
        f"{base_url}/predict/lap",
        json={"features": sample_features},
        auth=HTTPBasicAuth(api_credentials["username"], api_credentials["password"])
    )
    assert response.status_code == 200
    data = response.json()

    # Checkr structure complète
    assert "lap_duration_seconds" in data
    assert "lap_duration_formatted" in data
    assert "model_info" in data

    # Checkr cohérence des valeurs
    lap_time = data["lap_duration_seconds"]
    assert isinstance(lap_time, (int, float))
    assert 50 < lap_time < 200, f"Temps au tour inconsistent: {lap_time}s"

    # Checkr format mm:ss.xxx
    formatted = data["lap_duration_formatted"]
    assert ":" in formatted
    assert "." in formatted

    # Checkr model_info complet
    model_info = data["model_info"]
    assert "model_family" in model_info
    assert "source" in model_info
    assert model_info["source"] == "mlflow"  # Doit venir de MLflow

def test_prediction_batch(base_url, api_credentials, sample_features):
    """Test: prediction en batch fonctionne"""
    # Créer 3 prédictions différentes
    features_list = [sample_features.copy() for _ in range(3)]
    features_list[1]["driver_number"] = 16  # Leclerc
    features_list[2]["lap_number"] = 25      # Tour différent

    response = requests.post(
        f"{base_url}/predict/batch",
        json={"features": features_list},
        auth=HTTPBasicAuth(api_credentials["username"], api_credentials["password"])
    )
    assert response.status_code == 200
    data = response.json()

    assert "predictions" in data
    assert len(data["predictions"]) == 3
    assert "count" in data
    assert data["count"] == 3

    # Vérifier que chaque prédiction est un temps valide (float)
    for pred_time in data["predictions"]:
        assert isinstance(pred_time, (int, float))
        assert 50 < pred_time < 200

def test_model_info_complete(base_url, api_credentials):
    """Test: endpoint /predict/model return infos complètes"""
    response = requests.get(
        f"{base_url}/predict/model",
        auth=HTTPBasicAuth(api_credentials["username"], api_credentials["password"])
    )
    assert response.status_code == 200
    data = response.json()

    # Checkr tous les champs attendus
    required_fields = [
        "model_family", "strategy", "source",
        "test_mae", "test_r2", "cv_mae", "cv_r2"
    ]
    for field in required_fields:
        assert field in data, f"Champ '{field}' manquant"

    # Checkr que les métriques sont des nombres
    assert isinstance(data["test_mae"], (int, float))
    assert isinstance(data["test_r2"], (int, float))

    # Checkr cohérence des valeurs
    assert 0 < data["test_mae"] < 10, "MAE devrait être entre 0 et 10 secondes"
    assert 0 < data["test_r2"] < 1, "R² devrait être entre 0 et 1"

# ============================================================================
# Tests validation (utilise TestClient for rapidité)
# ============================================================================

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_prediction_missing_features(api_credentials):
    """Test: prediction with features manquantes échoue"""
    incomplete_features = {
        "driver_number": 1,
        "circuit_key": 9
        # Manque beaucoup de features
    }

    response = client.post(
        "/predict/lap",
        json={"features": incomplete_features},
        auth=(api_credentials["username"], api_credentials["password"])
    )
    assert response.status_code == 422  # Validation error

def test_prediction_invalid_values(api_credentials, sample_features):
    """Test: prediction with valeurs invalid échoue"""
    invalid_features = sample_features.copy()
    invalid_features["st_speed"] = -100  # Vitesse négative imposifble

    response = client.post(
        "/predict/lap",
        json={"features": invalid_features},
        auth=(api_credentials["username"], api_credentials["password"])
    )
    assert response.status_code == 422  # Validation error

def test_auth_wrong_username():
    """Test: username incorrect échoue"""
    response = client.get(
        "/predict/model",
        auth=("wrong_user", "f1pa")
    )
    assert response.status_code == 401

def test_auth_wrong_password(api_credentials):
    """Test: password incorrect échoue"""
    response = client.get(
        "/predict/model",
        auth=(api_credentials["username"], "wrong_password")
    )
    assert response.status_code == 401

def test_auth_missing():
    """Test: requête sans auth échoue"""
    response = client.get("/predict/model")
    assert response.status_code == 401

def test_driver_number_out_of_range(api_credentials, sample_features):
    """Test: numéro de pilote hors limites échoue"""
    invalid = sample_features.copy()
    invalid["driver_number"] = 999  # Hors limites

    response = client.post(
        "/predict/lap",
        json={"features": invalid},
        auth=(api_credentials["username"], api_credentials["password"])
    )
    assert response.status_code == 422

def test_year_out_of_range(api_credentials, sample_features):
    """Test: année hors limites échoue"""
    invalid = sample_features.copy()
    invalid["year"] = 2050  # Trop loin dans le futur

    response = client.post(
        "/predict/lap",
        json={"features": invalid},
        auth=(api_credentials["username"], api_credentials["password"])
    )
    assert response.status_code == 422
