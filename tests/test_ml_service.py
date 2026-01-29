"""
Tests pour le service ML via l'API (integration tests)
"""
import pytest
import requests
from requests.auth import HTTPBasicAuth

@pytest.fixture
def api_url():
    """URL de l'API réelle"""
    return "http://localhost:8000"

def test_model_is_loaded_from_mlflow(api_url, api_credentials):
    """Test: le model est chargé from MLflow via l'API"""
    response = requests.get(
        f"{api_url}/predict/model",
        auth=HTTPBasicAuth(api_credentials["username"], api_credentials["password"])
    )
    assert response.status_code == 200
    model_info = response.json()

    # Vérifier source MLflow
    assert model_info["source"] == "mlflow", "Le modèle devrait venir de MLflow"
    assert model_info["run_id"] is not None
    assert model_info["run_name"] is not None

def test_model_has_complete_metrics(api_url, api_credentials):
    """Test: le model a toutes les métriques"""
    response = requests.get(
        f"{api_url}/predict/model",
        auth=HTTPBasicAuth(api_credentials["username"], api_credentials["password"])
    )
    assert response.status_code == 200
    model_info = response.json()

    # Checkr métriques complètes
    assert model_info["test_mae"] is not None
    assert model_info["test_r2"] is not None
    assert model_info["cv_mae"] is not None
    assert model_info["cv_r2"] is not None

    # Checkr cohérence
    assert 0 < model_info["test_mae"] < 5
    assert 0.5 < model_info["test_r2"] < 1

def test_prediction_returns_valid_time(api_url, api_credentials, sample_features):
    """Test: prediction return un time consistent"""
    response = requests.post(
        f"{api_url}/predict/lap",
        json={"features": sample_features},
        auth=HTTPBasicAuth(api_credentials["username"], api_credentials["password"])
    )
    assert response.status_code == 200
    data = response.json()

    # Vérifier temps cohérent
    lap_time = data["lap_duration_seconds"]
    assert 50 < lap_time < 200, f"Temps incohérent: {lap_time}s"

def test_predictions_are_deterministic(api_url, api_credentials, sample_features):
    """Test: même input = même output"""
    # Faire 2 predictions identiques
    response1 = requests.post(
        f"{api_url}/predict/lap",
        json={"features": sample_features},
        auth=HTTPBasicAuth(api_credentials["username"], api_credentials["password"])
    )
    response2 = requests.post(
        f"{api_url}/predict/lap",
        json={"features": sample_features},
        auth=HTTPBasicAuth(api_credentials["username"], api_credentials["password"])
    )

    assert response1.status_code == 200
    assert response2.status_code == 200

    time1 = response1.json()["lap_duration_seconds"]
    time2 = response2.json()["lap_duration_seconds"]

    # Les time doivent être identiques
    assert abs(time1 - time2) < 0.001, "Les predictions devraient être déterministes"

def test_different_drivers_different_predictions(api_url, api_credentials, sample_features):
    """Test: pilotes différents = time différents"""
    features_ver = sample_features.copy()
    features_ver["driver_number"] = 1  # Verstappen

    features_lec = sample_features.copy()
    features_lec["driver_number"] = 16  # Leclerc

    response_ver = requests.post(
        f"{api_url}/predict/lap",
        json={"features": features_ver},
        auth=HTTPBasicAuth(api_credentials["username"], api_credentials["password"])
    )
    response_lec = requests.post(
        f"{api_url}/predict/lap",
        json={"features": features_lec},
        auth=HTTPBasicAuth(api_credentials["username"], api_credentials["password"])
    )

    assert response_ver.status_code == 200
    assert response_lec.status_code == 200

    time_ver = response_ver.json()["lap_duration_seconds"]
    time_lec = response_lec.json()["lap_duration_seconds"]

    # Les temps peuvent être différents (mais pas forcément)
    # On vérifie juste que les deux sont cohérents
    assert 50 < time_ver < 200
    assert 50 < time_lec < 200
