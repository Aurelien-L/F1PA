"""
Tests étendus pour l'API FastAPI - Tests via requêtes réelles
"""
import pytest
import requests
from requests.auth import HTTPBasicAuth

@pytest.fixture
def base_url():
    """URL de base de l'API"""
    return "http://localhost:8000"

def test_services_are_up(check_services):
    """Test: vérifier que les services requis sont UP"""
    assert check_services is True

# ============================================================================
# Tests endpoints /data/*
# ============================================================================

def test_drivers_list_complete(base_url, api_credentials):
    """Test: liste des pilotes complète et bien structurée"""
    response = requests.get(
        f"{base_url}/data/drivers",
        auth=HTTPBasicAuth(api_credentials["username"], api_credentials["password"])
    )
    assert response.status_code == 200
    drivers = response.json()

    assert isinstance(drivers, list)
    assert len(drivers) > 0, "La liste des pilotes ne devrait pas être vide"

    # Vérifier la structure du premier pilote
    driver = drivers[0]
    required_fields = ["driver_number", "full_name", "name_acronym"]
    for field in required_fields:
        assert field in driver, f"Champ '{field}' manquant"

    # Vérifier les types
    assert isinstance(driver["driver_number"], int)
    assert isinstance(driver["full_name"], str)

    # Vérifier que headshot_url est présent (ajouté récemment)
    assert "headshot_url" in driver
    assert "team_colour" in driver

def test_circuits_list_complete(base_url, api_credentials):
    """Test: liste des circuits complète et bien structurée"""
    response = requests.get(
        f"{base_url}/data/circuits",
        auth=HTTPBasicAuth(api_credentials["username"], api_credentials["password"])
    )
    assert response.status_code == 200
    circuits = response.json()

    assert isinstance(circuits, list)
    assert len(circuits) > 0, "La liste des circuits ne devrait pas être vide"

    # Vérifier la structure
    circuit = circuits[0]
    required_fields = ["circuit_key", "circuit_short_name", "country_name"]
    for field in required_fields:
        assert field in circuit

def test_prediction_lap_valid(base_url, api_credentials, sample_features):
    """Test: prédiction avec features valides retourne résultat cohérent"""
    response = requests.post(
        f"{base_url}/predict/lap",
        json={"features": sample_features},
        auth=HTTPBasicAuth(api_credentials["username"], api_credentials["password"])
    )
    assert response.status_code == 200
    data = response.json()

    # Vérifier structure complète
    assert "lap_duration_seconds" in data
    assert "lap_duration_formatted" in data
    assert "model_info" in data

    # Vérifier cohérence des valeurs
    lap_time = data["lap_duration_seconds"]
    assert isinstance(lap_time, (int, float))
    assert 50 < lap_time < 200, f"Temps au tour incohérent: {lap_time}s"

    # Vérifier format mm:ss.xxx
    formatted = data["lap_duration_formatted"]
    assert ":" in formatted
    assert "." in formatted

    # Vérifier model_info complet
    model_info = data["model_info"]
    assert "model_family" in model_info
    assert "source" in model_info
    assert model_info["source"] == "mlflow"  # Doit venir de MLflow

def test_prediction_batch(base_url, api_credentials, sample_features):
    """Test: prédiction en batch fonctionne"""
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
    """Test: endpoint /predict/model retourne infos complètes"""
    response = requests.get(
        f"{base_url}/predict/model",
        auth=HTTPBasicAuth(api_credentials["username"], api_credentials["password"])
    )
    assert response.status_code == 200
    data = response.json()

    # Vérifier tous les champs attendus
    required_fields = [
        "model_family", "strategy", "source",
        "test_mae", "test_r2", "cv_mae", "cv_r2"
    ]
    for field in required_fields:
        assert field in data, f"Champ '{field}' manquant"

    # Vérifier que les métriques sont des nombres
    assert isinstance(data["test_mae"], (int, float))
    assert isinstance(data["test_r2"], (int, float))

    # Vérifier cohérence des valeurs
    assert 0 < data["test_mae"] < 10, "MAE devrait être entre 0 et 10 secondes"
    assert 0 < data["test_r2"] < 1, "R² devrait être entre 0 et 1"

# ============================================================================
# Tests validation (utilise TestClient pour rapidité)
# ============================================================================

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_prediction_missing_features(api_credentials):
    """Test: prédiction avec features manquantes échoue"""
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
    """Test: prédiction avec valeurs invalides échoue"""
    invalid_features = sample_features.copy()
    invalid_features["st_speed"] = -100  # Vitesse négative impossible

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
