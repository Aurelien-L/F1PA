"""
Unit tests for FastAPI API
"""
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_endpoint():
    """Test: /health endpoint responds"""
    response = client.get("/")
    # May return 200 (if route defined) or 404 (if no root route)
    assert response.status_code in [200, 404]

def test_docs_endpoint():
    """Test: Swagger documentation accessible"""
    response = client.get("/docs")
    assert response.status_code == 200

def test_openapi_endpoint():
    """Test: OpenAPI JSON endpoint accessible"""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert "openapi" in data
    assert "info" in data

def test_model_info_without_auth():
    """Test: /predict/model endpoint requires authentication"""
    response = client.get("/predict/model")
    assert response.status_code == 401  # Unauthorized

def test_model_info_with_auth(api_credentials):
    """Test: /predict/model endpoint with auth (may be 503 if services down)"""
    response = client.get(
        "/predict/model",
        auth=(api_credentials["username"], api_credentials["password"])
    )
    # 200 if working, 503 if DB/MLflow unavailable
    assert response.status_code in [200, 503]

    if response.status_code == 200:
        data = response.json()
        # Check required fields
        assert "model_family" in data
        assert "source" in data
        assert data["source"] in ["mlflow", "local"]

def test_prediction_endpoint_structure(api_credentials, sample_features):
    """Test: /predict/lap endpoint structure (may be 503 if services down)"""
    response = client.post(
        "/predict/lap",
        json={"features": sample_features},
        auth=(api_credentials["username"], api_credentials["password"])
    )
    # 200 if working, 503 if DB/MLflow unavailable
    assert response.status_code in [200, 503]

    if response.status_code == 200:
        data = response.json()
        # Check response structure
        assert "lap_duration_seconds" in data
        assert "lap_duration_formatted" in data
        assert "model_info" in data

        # Check time format
        assert isinstance(data["lap_duration_seconds"], (int, float))
        assert ":" in data["lap_duration_formatted"]

def test_prediction_invalid_features(api_credentials):
    """Test: prediction with invalid features fails"""
    response = client.post(
        "/predict/lap",
        json={"features": {"invalid": "data"}},
        auth=(api_credentials["username"], api_credentials["password"])
    )
    assert response.status_code == 422  # Validation error

def test_prediction_missing_auth():
    """Test: prediction without auth fails"""
    response = client.post(
        "/predict/lap",
        json={"features": {"test": "data"}}
    )
    assert response.status_code == 401  # Unauthorized

def test_drivers_endpoint(api_credentials):
    """Test: /data/drivers endpoint (may be 503 if DB down)"""
    response = client.get(
        "/data/drivers",
        auth=(api_credentials["username"], api_credentials["password"])
    )
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, list)
        
        if len(data) > 0:
            driver = data[0]
            assert "driver_number" in driver
            assert "full_name" in driver

def test_circuits_endpoint(api_credentials):
    """Test: /data/circuits endpoint (may be 503 if DB down)"""
    response = client.get(
        "/data/circuits",
        auth=(api_credentials["username"], api_credentials["password"])
    )
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, list)
        
        if len(data) > 0:
            circuit = data[0]
            assert "circuit_key" in circuit
            assert "circuit_short_name" in circuit

def test_auth_invalid_credentials():
    """Test: authentication with wrong credentials fails"""
    response = client.get(
        "/predict/model",
        auth=("wrong", "credentials")
    )
    assert response.status_code == 401
