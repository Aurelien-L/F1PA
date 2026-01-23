# F1PA API

API REST pour accéder aux données F1 et aux prédictions de temps au tour.

## Objectif du modèle ML

Le modèle prédit le **temps au tour d'un pilote AVANT qu'il roule**, basé sur :
- Sa performance historique (`driver_perf_score`, `driver_avg_laptime`)
- Les caractéristiques du circuit (`circuit_avg_laptime`)
- Les conditions météo (`temp`, `rhum`, `pres`)
- Les vitesses attendues (`st_speed`, `i1_speed`, `i2_speed`)

**Important** : Les temps secteurs (`duration_sector_*`) ne sont PAS utilisés car ils représentent des données du tour en cours, ce qui rendrait la prédiction triviale (lap_time ≈ sum(sectors)).

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
| POST | `/predict/lap` | Prédiction pour un pilote/circuit (JSON body) |
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

### Prédiction de performance

```bash
curl -u f1pa:f1pa -X POST http://localhost:8000/predict/lap \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "driver_number": 1,
      "circuit_key": 7,
      "st_speed": 310.5,
      "i1_speed": 295.2,
      "i2_speed": 288.1,
      "temp": 25.0,
      "rhum": 45.0,
      "pres": 1013.0,
      "lap_number": 15,
      "year": 2025,
      "circuit_avg_laptime": 92.5,
      "driver_avg_laptime": 91.2,
      "driver_perf_score": -1.3
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
driver_number=1&circuit_key=7&\
st_speed=310&i1_speed=295&i2_speed=288"
```

### Récupérer les circuits

```bash
curl -u f1pa:f1pa http://localhost:8000/data/circuits
```

### Récupérer le temps moyen d'un circuit

```bash
curl -u f1pa:f1pa http://localhost:8000/data/circuits/7/avg-laptime
```

Réponse :
```json
{
  "circuit_key": 7,
  "avg_laptime_seconds": 92.543,
  "avg_laptime_formatted": "1:32.543"
}
```

### Statistiques du dataset

```bash
curl -u f1pa:f1pa http://localhost:8000/data/stats
```

## Features ML (pour les prédictions)

| Feature | Description | Source |
|---------|-------------|--------|
| `driver_number` | Numéro du pilote | Input |
| `circuit_key` | Clé du circuit | Input |
| `st_speed` | Vitesse speed trap attendue (km/h) | Input |
| `i1_speed` | Vitesse intermédiaire 1 (km/h) | Input |
| `i2_speed` | Vitesse intermédiaire 2 (km/h) | Input |
| `temp` | Température (°C) | Input |
| `rhum` | Humidité relative (%) | Input |
| `pres` | Pression atmosphérique (hPa) | Input |
| `lap_number` | Numéro du tour prévu | Input |
| `year` | Année | Input |
| `circuit_avg_laptime` | Temps moyen du circuit (s) | `/data/circuits/{id}/avg-laptime` |
| `driver_avg_laptime` | Temps moyen du pilote (s) | Calculé |
| `driver_perf_score` | Score de performance (négatif = plus rapide) | Calculé |

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
