# F1PA API

API REST pour accéder aux données F1 et aux prédictions de temps au tour.

## Fonctionnalités

- **Prédictions ML** : Exposition du modèle XGBoost pour prédire les temps au tour
- **Accès aux données** : Endpoints pour consulter circuits, pilotes, sessions et tours
- **Authentification** : Sécurisation des endpoints via HTTP Basic Auth
- **Documentation** : Interface Swagger UI intégrée

## Démarrage

### Avec Docker Compose (recommandé)

```bash
docker-compose up api
```

L'API sera accessible sur `http://localhost:8000`

### En local

```bash
# Installer les dépendances
pip install fastapi uvicorn[standard] pydantic

# Lancer l'API
python -m api.main
```

## Authentification

Tous les endpoints (sauf `/health`) nécessitent une authentification HTTP Basic :

- **Username** : `f1pa`
- **Password** : `f1pa`

Exemple avec curl :
```bash
curl -u f1pa:f1pa http://localhost:8000/data/circuits
```

## Documentation interactive

Une fois l'API lancée, accédez à :
- **Swagger UI** : http://localhost:8000/docs
- **ReDoc** : http://localhost:8000/redoc

## Endpoints

### Health & Status

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | Informations de base de l'API |
| GET | `/health` | État de santé (modèle, BDD, MLflow) |

### Prédictions (`/predict`)

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/predict/model` | Informations sur le modèle chargé |
| POST | `/predict/lap` | Prédiction pour un tour (JSON body) |
| POST | `/predict/batch` | Prédictions par lot (jusqu'à 1000) |
| POST | `/predict/simple` | Prédiction via query params |

### Données (`/data`)

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/data/stats` | Statistiques globales du dataset |
| GET | `/data/circuits` | Liste des circuits |
| GET | `/data/circuits/{id}` | Détails d'un circuit |
| GET | `/data/circuits/{id}/laps` | Tours d'un circuit (triés par temps) |
| GET | `/data/circuits/{id}/avg-laptime` | Temps moyen du circuit |
| GET | `/data/drivers` | Liste des pilotes |
| GET | `/data/drivers/{id}` | Détails d'un pilote |
| GET | `/data/drivers/{id}/laps` | Tours d'un pilote |
| GET | `/data/sessions` | Sessions (filtrable par année/circuit) |
| GET | `/data/laps` | Tours avec filtres et pagination |

## Exemples d'utilisation

### Prédiction de temps au tour

```bash
curl -u f1pa:f1pa -X POST http://localhost:8000/predict/lap \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "st_speed": 310.5,
      "i1_speed": 295.2,
      "i2_speed": 288.1,
      "duration_sector_1": 28.5,
      "duration_sector_2": 35.2,
      "duration_sector_3": 26.8,
      "temp": 25.0,
      "rhum": 45.0,
      "pres": 1013.0,
      "lap_number": 15,
      "year": 2025,
      "circuit_avg_laptime": 92.5
    }
  }'
```

Réponse :
```json
{
  "lap_duration_seconds": 90.512,
  "lap_duration_formatted": "1:30.512",
  "model_info": {
    "model_family": "xgboost",
    "strategy": "robust",
    "test_mae": 0.285,
    "source": "mlflow"
  }
}
```

### Prédiction simple (query params)

```bash
curl -u f1pa:f1pa -X POST "http://localhost:8000/predict/simple?\
st_speed=310&i1_speed=295&i2_speed=288&\
duration_sector_1=28.5&duration_sector_2=35.2&duration_sector_3=26.8"
```

### Récupérer les circuits

```bash
curl -u f1pa:f1pa http://localhost:8000/data/circuits
```

Réponse :
```json
[
  {
    "circuit_key": 1,
    "circuit_short_name": "Bahrain",
    "location": "Sakhir",
    "country_name": "Bahrain",
    "country_code": "BH"
  },
  ...
]
```

### Récupérer les tours avec filtres

```bash
# Tours du pilote 1 (Verstappen) en 2024, page 2
curl -u f1pa:f1pa "http://localhost:8000/data/laps?driver_number=1&year=2024&page=2&page_size=50"
```

### Statistiques du dataset

```bash
curl -u f1pa:f1pa http://localhost:8000/data/stats
```

Réponse :
```json
{
  "total_laps": 71645,
  "total_circuits": 24,
  "total_drivers": 26,
  "total_sessions": 189,
  "years": [2023, 2024, 2025],
  "date_range": {
    "min": "2023-03-03",
    "max": "2025-03-16"
  }
}
```

## Features ML (pour les prédictions)

| Feature | Description | Plage |
|---------|-------------|-------|
| `st_speed` | Vitesse au speed trap (km/h) | 0-400 |
| `i1_speed` | Vitesse intermédiaire 1 (km/h) | 0-400 |
| `i2_speed` | Vitesse intermédiaire 2 (km/h) | 0-400 |
| `duration_sector_1` | Temps secteur 1 (s) | > 0 |
| `duration_sector_2` | Temps secteur 2 (s) | > 0 |
| `duration_sector_3` | Temps secteur 3 (s) | > 0 |
| `temp` | Température (°C) | -20 à 60 |
| `rhum` | Humidité relative (%) | 0-100 |
| `pres` | Pression atmosphérique (hPa) | 900-1100 |
| `lap_number` | Numéro du tour | >= 1 |
| `year` | Année | 2023-2030 |
| `circuit_avg_laptime` | Temps moyen du circuit (s) | 60-150 |

## Codes de réponse

| Code | Description |
|------|-------------|
| 200 | Succès |
| 401 | Non authentifié |
| 404 | Ressource non trouvée |
| 422 | Erreur de validation |
| 500 | Erreur serveur |
| 503 | Service indisponible (modèle/BDD) |

## Architecture

```
api/
├── __init__.py
├── main.py              # Application FastAPI
├── config.py            # Configuration
├── auth.py              # Authentification HTTP Basic
├── models.py            # Modèles Pydantic (request/response)
├── endpoints/
│   ├── predictions.py   # Routes /predict/*
│   └── data.py          # Routes /data/*
└── services/
    ├── ml_service.py    # Chargement modèle & prédictions
    └── db_service.py    # Accès base de données
```

## Configuration

Variables d'environnement (optionnelles) :

| Variable | Défaut | Description |
|----------|--------|-------------|
| `API_HOST` | `0.0.0.0` | Hôte de l'API |
| `API_PORT` | `8000` | Port de l'API |
| `DB_HOST` | `localhost` | Hôte PostgreSQL |
| `DB_PORT` | `5432` | Port PostgreSQL |
| `DB_NAME` | `f1pa_db` | Nom de la base |
| `DB_USER` | `f1pa` | Utilisateur BDD |
| `DB_PASSWORD` | `f1pa` | Mot de passe BDD |
| `MLFLOW_TRACKING_URI` | `http://localhost:5001` | URI MLflow |
