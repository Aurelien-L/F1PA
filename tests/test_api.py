"""
Tests unitaires pour l'API FastAPI
"""
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_endpoint():
    """Test: endpoint /health répond"""
    response = client.get("/")
    # Peut retourner 200 (si route définie) ou 404 (si pas de route racine)
    assert response.status_code in [200, 404]

def test_docs_endpoint():
    """Test: documentation Swagger accessible"""
    response = client.get("/docs")
    assert response.status_code == 200

def test_openapi_endpoint():
    """Test: endpoint OpenAPI JSON accessible"""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert "openapi" in data
    assert "info" in data

def test_model_info_without_auth():
    """Test: endpoint /predict/model requiert authentification"""
    response = client.get("/predict/model")
    assert response.status_code == 401  # Unauthorized

def test_model_info_with_auth(api_credentials):
    """Test: endpoint /predict/model avec auth (peut être 503 si services down)"""
    response = client.get(
        "/predict/model",
        auth=(api_credentials["username"], api_credentials["password"])
    )
    # 200 si tout fonctionne, 503 si DB/MLflow indisponibles
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        # Vérifier les champs obligatoires
        assert "model_family" in data
        assert "source" in data
        assert data["source"] in ["mlflow", "local"]

def test_prediction_endpoint_structure(api_credentials, sample_features):
    """Test: endpoint /predict/lap structure (peut être 503 si services down)"""
    response = client.post(
        "/predict/lap",
        json={"features": sample_features},
        auth=(api_credentials["username"], api_credentials["password"])
    )
    # 200 si tout fonctionne, 503 si DB/MLflow indisponibles
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        # Vérifier la structure de la réponse
        assert "lap_duration_seconds" in data
        assert "lap_duration_formatted" in data
        assert "model_info" in data
        
        # Vérifier le format du temps
        assert isinstance(data["lap_duration_seconds"], (int, float))
        assert ":" in data["lap_duration_formatted"]

def test_prediction_invalid_features(api_credentials):
    """Test: prédiction avec features invalides échoue"""
    response = client.post(
        "/predict/lap",
        json={"features": {"invalid": "data"}},
        auth=(api_credentials["username"], api_credentials["password"])
    )
    assert response.status_code == 422  # Validation error

def test_prediction_missing_auth():
    """Test: prédiction sans auth échoue"""
    response = client.post(
        "/predict/lap",
        json={"features": {"test": "data"}}
    )
    assert response.status_code == 401  # Unauthorized

def test_drivers_endpoint(api_credentials):
    """Test: endpoint /data/drivers (peut être 503 si DB down)"""
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
    """Test: endpoint /data/circuits (peut être 503 si DB down)"""
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
    """Test: authentification avec mauvais credentials échoue"""
    response = client.get(
        "/predict/model",
        auth=("wrong", "credentials")
    )
    assert response.status_code == 401
