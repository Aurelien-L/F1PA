"""
Configuration pytest et fixtures pour les tests F1PA
"""
import pytest
import sys
import requests
from pathlib import Path
from requests.auth import HTTPBasicAuth

# Ajouter le répertoire racine au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

@pytest.fixture(scope="session")
def check_services():
    """Fixture: vérifier que les services Docker sont UP avant les tests"""
    services = {
        "API": "http://localhost:8000/docs",
        "PostgreSQL": "http://localhost:8000/data/drivers",  # Via API
        "MLflow": "http://localhost:5000"
    }
    
    failed_services = []
    
    for service_name, url in services.items():
        try:
            if service_name == "PostgreSQL":
                # Test via API avec auth
                response = requests.get(url, auth=HTTPBasicAuth("f1pa", "f1pa"), timeout=2)
            else:
                response = requests.get(url, timeout=2)
            
            if response.status_code >= 500:
                failed_services.append(service_name)
        except requests.exceptions.RequestException:
            failed_services.append(service_name)
    
    if failed_services:
        pytest.fail(
            f"❌ Services requis non disponibles: {', '.join(failed_services)}\n"
            f"→ Lancer: docker-compose up -d\n"
            f"→ Attendre 10-15 secondes que les services démarrent"
        )
    
    return True

@pytest.fixture
def sample_features():
    """Fixture: features pour test de prédiction"""
    return {
        "driver_number": 1,
        "circuit_key": 9,
        "st_speed": 320.5,
        "i1_speed": 290.2,
        "i2_speed": 285.1,
        "temp": 28.0,
        "rhum": 45.0,
        "pres": 1013.0,
        "lap_number": 15,
        "year": 2024,
        "circuit_avg_laptime": 106.5,
        "driver_avg_laptime": 105.2,
        "driver_perf_score": -1.2
    }

@pytest.fixture
def api_credentials():
    """Fixture: credentials API"""
    return {"username": "f1pa", "password": "f1pa"}

@pytest.fixture
def api_url():
    """Fixture: URL de base de l'API"""
    return "http://localhost:8000"
