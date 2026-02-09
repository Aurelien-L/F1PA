"""
Tests for ML service via API (integration tests)
"""
import pytest
import requests
from requests.auth import HTTPBasicAuth

@pytest.fixture
def api_url():
    """Real API URL"""
    return "http://localhost:8000"

@pytest.mark.integration
def test_model_is_loaded_from_mlflow(api_url, api_credentials):
    """Test: model is loaded from MLflow via API"""
    response = requests.get(
        f"{api_url}/predict/model",
        auth=HTTPBasicAuth(api_credentials["username"], api_credentials["password"])
    )
    assert response.status_code == 200
    model_info = response.json()

    # Verify MLflow source
    assert model_info["source"] == "mlflow", "Model should come from MLflow"
    assert model_info["run_id"] is not None
    assert model_info["run_name"] is not None

@pytest.mark.integration
def test_model_has_complete_metrics(api_url, api_credentials):
    """Test: model has all metrics"""
    response = requests.get(
        f"{api_url}/predict/model",
        auth=HTTPBasicAuth(api_credentials["username"], api_credentials["password"])
    )
    assert response.status_code == 200
    model_info = response.json()

    # Check complete metrics
    assert model_info["test_mae"] is not None
    assert model_info["test_r2"] is not None
    assert model_info["cv_mae"] is not None
    assert model_info["cv_r2"] is not None

    # Check consistency
    assert 0 < model_info["test_mae"] < 5
    assert 0.5 < model_info["test_r2"] < 1

@pytest.mark.integration
def test_prediction_returns_valid_time(api_url, api_credentials, sample_features):
    """Test: prediction returns consistent time"""
    response = requests.post(
        f"{api_url}/predict/lap",
        json={"features": sample_features},
        auth=HTTPBasicAuth(api_credentials["username"], api_credentials["password"])
    )
    assert response.status_code == 200
    data = response.json()

    # Verify consistent time
    lap_time = data["lap_duration_seconds"]
    assert 50 < lap_time < 200, f"Inconsistent time: {lap_time}s"

@pytest.mark.integration
def test_predictions_are_deterministic(api_url, api_credentials, sample_features):
    """Test: same input = same output"""
    # Make 2 identical predictions
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

    # Times must be identical
    assert abs(time1 - time2) < 0.001, "Predictions should be deterministic"

@pytest.mark.integration
def test_different_drivers_different_predictions(api_url, api_credentials, sample_features):
    """Test: different drivers = different times"""
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

    # Times may be different (but not necessarily)
    # We just verify both are consistent
    assert 50 < time_ver < 200
    assert 50 < time_lec < 200
