# F1PA - Tests Unitaires

Suite de tests complète pour valider le bon fonctionnement du projet F1PA.

## 📊 Résultats

```
✅ 40 tests sur 40 passent (100%)
⚡ Exécution en 2.5s
```

## ✨ Points Clés

### Tests Requis Services UP
**Important** : Les tests vérifient que les services Docker sont disponibles AVANT de s'exécuter. Si PostgreSQL, API ou MLflow sont down, les tests échouent immédiatement avec un message clair.

```bash
# Lancer les services avant les tests
docker-compose up -d

# Attendre que tout démarre (10-15s)
sleep 15

# Exécuter les tests
pytest tests/ -v
```

### Coverage Ciblé

| Module | Coverage | Statut |
|--------|----------|--------|
| `api/auth.py` | 100% | ✅ Complet |
| `api/models.py` | 100% | ✅ Complet |
| `ml/config.py` | 100% | ✅ Complet |
| `streamlit/config.py` | 100% | ✅ Complet |
| `api/config.py` | 96% | ✅ Excellent |
| `api/endpoints/predictions.py` | 41% | ⚠️ Routes critiques testées |
| `api/services/ml_service.py` | 36% | ⚠️ Via tests d'intégration |
| `api/endpoints/data.py` | 28% | ⚠️ Endpoints principaux testés |

**Note** : Coverage bas pour preprocessing/train/streamlit est normal (scripts et UI).

## 🏗️ Structure

```
tests/
├── __init__.py                   # Package tests
├── conftest.py                   # Fixtures + vérification services
├── test_config.py                # 4 tests - Configurations
├── test_api.py                   # 11 tests - Endpoints FastAPI (TestClient)
├── test_api_extended.py          # 13 tests - Endpoints via HTTP réel
├── test_ml_service.py            # 5 tests - Service ML (intégration)
├── test_preprocessing.py         # 7 tests - Configuration ML
└── README.md                     # Ce fichier

pytest.ini                        # Configuration pytest
htmlcov/                          # Rapport coverage HTML
```

## 🧪 Tests par Catégorie

### test_config.py (4 tests)
- ✅ Import configurations ML (MLflow, target)
- ✅ Import configurations Streamlit
- ✅ Nom expérience MLflow
- ✅ Chemins projet

### test_api.py (11 tests - TestClient)
- ✅ Endpoint health/docs
- ✅ OpenAPI spec
- ✅ Authentification requise (401)
- ✅ Model info avec auth
- ✅ Prédiction structure
- ✅ Validation features invalides (422)
- ✅ Endpoints /data/drivers et /circuits
- ✅ Credentials invalides rejetés

### test_api_extended.py (13 tests - HTTP réel)
- ✅ **Vérification services UP** (fail si down)
- ✅ Liste pilotes complète (headshot_url, team_colour)
- ✅ Liste circuits complète
- ✅ Prédiction lap valide et cohérente
- ✅ Prédiction batch (3 prédictions)
- ✅ Model info complet (toutes métriques)
- ✅ Features manquantes/invalides (422)
- ✅ Auth incorrecte (401)
- ✅ Validation ranges (driver_number, year)

### test_ml_service.py (5 tests - Intégration)
- ✅ Modèle chargé depuis MLflow (source="mlflow")
- ✅ Métriques complètes (MAE, R², CV)
- ✅ Prédictions cohérentes (50-200s)
- ✅ Prédictions déterministes (même input = même output)
- ✅ Différents pilotes = temps dans range valide

### test_preprocessing.py (7 tests)
- ✅ Variable cible (lap_duration)
- ✅ Train/test split 80/20
- ✅ Cross-validation 5 folds
- ✅ Config MLflow
- ✅ Paramètres GridSearch
- ✅ Groupes features
- ✅ Random state = 42

## 🚀 Exécution

### Tous les tests
```bash
pytest tests/ -v
```

### Tests spécifiques
```bash
# Tests API uniquement
pytest tests/test_api.py tests/test_api_extended.py -v

# Tests configuration
pytest tests/test_config.py tests/test_preprocessing.py -v

# Tests ML service
pytest tests/test_ml_service.py -v
```

### Avec coverage
```bash
# Terminal
pytest tests/ --cov=api --cov=ml --cov=streamlit --cov-report=term-missing

# HTML (génère htmlcov/index.html)
pytest tests/ --cov=api --cov=ml --cov=streamlit --cov-report=html
```

### Tests rapides (sans services)
Si vous voulez tester les validations sans Docker :
```bash
# Seulement les tests de config et validation
pytest tests/test_config.py tests/test_preprocessing.py -v
```

**Note** : Les tests complets nécessitent `docker-compose up -d`.

## 🎯 Fixtures Disponibles

### `check_services` (session scope)
Vérifie que API, PostgreSQL et MLflow sont UP. **Fait échouer les tests si services down.**

### `sample_features`
Features valides pour test de prédiction (15 features complètes).

### `api_credentials`
Credentials API : `{"username": "f1pa", "password": "f1pa"}`

### `base_url` / `api_url`
URL de l'API : `http://localhost:8000`

## 📝 Ajouter un Test

### Test simple (config/validation)
```python
def test_my_config():
    """Test: vérifier une config"""
    from ml.config import MA_CONFIG
    assert MA_CONFIG == "valeur_attendue"
```

### Test API (validation)
```python
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_my_endpoint(api_credentials):
    """Test: endpoint retourne 422 si invalide"""
    response = client.post(
        "/endpoint",
        json={"invalid": "data"},
        auth=(api_credentials["username"], api_credentials["password"])
    )
    assert response.status_code == 422
```

### Test API (services réels)
```python
import requests
from requests.auth import HTTPBasicAuth

def test_my_endpoint_real(base_url, api_credentials):
    """Test: endpoint avec services UP"""
    response = requests.get(
        f"{base_url}/endpoint",
        auth=HTTPBasicAuth(api_credentials["username"], api_credentials["password"])
    )
    assert response.status_code == 200
    assert "expected_field" in response.json()
```

## 🐛 Debugging

### Test échoue
```bash
# Verbose + traceback complet
pytest tests/test_api.py::test_name -vv

# Arrêter au premier échec
pytest tests/ -x

# Debug interactif
pytest tests/ --pdb
```

### Services requis non disponibles
```
❌ Services requis non disponibles: PostgreSQL, MLflow
→ Lancer: docker-compose up -d
→ Attendre 10-15 secondes que les services démarrent
```

### Coverage manquant
```bash
# Identifier lignes non couvertes
pytest tests/ --cov=api --cov-report=term-missing
```

### Historique - Warning Pydantic V2 (résolu)
~~Un warning de dépréciation Pydantic apparaissait lors des tests.~~

**Résolu** : Le code a été migré vers la syntaxe Pydantic V2 (`ConfigDict`). Plus aucun warning de dépréciation.

## 📚 Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [Coverage.py](https://coverage.readthedocs.io/)
- [Requests Library](https://requests.readthedocs.io/)
