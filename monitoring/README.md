# F1PA - Monitoring ML en Production

Stack complète de monitoring pour surveiller les performances du modèle ML et de l'API en production.

## 🎯 Objectif

Surveiller en temps réel :
- **Performances de l'API** : latence, throughput, erreurs
- **Performances du modèle ML** : temps de prédiction, qualité, drift
- **Santé des services** : database, MLflow, model

## 🏗️ Architecture

```
┌─────────────┐
│   F1PA API  │  ← Expose /metrics (Prometheus format)
└──────┬──────┘
       │
       │ scrape (10s)
       ▼
┌─────────────┐
│ Prometheus  │  ← Collecte et stocke les métriques
│  (port 9090)│
└──────┬──────┘
       │
       │ datasource
       ▼
┌─────────────┐
│   Grafana   │  ← Visualisation et alertes
│  (port 3000)│
└─────────────┘

┌─────────────┐
│  Evidently  │  ← Rapports de drift ML
└─────────────┘
```

## 📊 Stack Technique

| Composant | Version | Port | Description |
|-----------|---------|------|-------------|
| **Prometheus** | 3.2.1 | 9090 | Collecte et stockage des métriques time-series |
| **Grafana** | 11.5.0 | 3000 | Dashboards et alertes |
| **Evidently** | 0.4.33 | - | Monitoring du drift ML avec rapports HTML interactifs |
| **prometheus-client** | 0.24.1 | - | Bibliothèque Python pour Prometheus |

## 🚀 Démarrage Rapide

### 1. Lancer la stack complète

```bash
docker-compose up -d
```

Cela démarre :
- API (port 8000)
- Prometheus (port 9090)
- Grafana (port 3000)
- MLflow (port 5000)
- PostgreSQL (port 5432)

### 2. Accéder aux interfaces

| Service | URL | Credentials |
|---------|-----|-------------|
| **API Docs** | http://localhost:8000/docs | `f1pa` / `f1pa` |
| **Metrics** | http://localhost:8000/metrics | - |
| **Prometheus** | http://localhost:9090 | - |
| **Grafana** | http://localhost:3000 | `admin` / `admin` |

### 3. Visualiser le dashboard Grafana

1. Ouvrir http://localhost:3000
2. Login: `admin` / `admin`
3. Le dashboard **F1PA ML Model Monitoring** est automatiquement provisionné

## 📈 Métriques Disponibles

### Métriques HTTP

```promql
# Nombre total de requêtes HTTP
http_requests_total{method="GET", endpoint="/health", status="200"}

# Latence des requêtes (p50, p95, p99)
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
```

### Métriques ML - Prédictions

```promql
# Nombre de prédictions
f1pa_predictions_total{endpoint_type="single"}

# Durée des prédictions
f1pa_prediction_duration_seconds_sum / f1pa_prediction_duration_seconds_count

# Taux d'erreur des prédictions
rate(f1pa_prediction_errors_total[5m])
```

### Métriques de Statut

```promql
# Statut du modèle ML (1=loaded, 0=not loaded)
f1pa_model_loaded

# Statut database (1=connected, 0=disconnected)
f1pa_database_connected

# Statut MLflow (1=connected, 0=disconnected)
f1pa_mlflow_connected
```

### Métriques Database

```promql
# Durée des requêtes DB
f1pa_db_query_duration_seconds

# Requêtes par endpoint
f1pa_data_queries_total{endpoint="drivers"}
```

## 🚨 Alertes Configurées

Les alertes suivantes sont définies dans `prometheus/alerts.yml` :

### 1. **HighPredictionLatency**
- **Condition** : p95 latence > 1s pendant 5min
- **Impact** : Dégradation de l'expérience utilisateur
- **Action** : Vérifier les performances du modèle

### 2. **ModelNotLoaded**
- **Condition** : `f1pa_model_loaded == 0` pendant 1min
- **Impact** : API non fonctionnelle
- **Action** : Redémarrer l'API, vérifier MLflow

### 3. **DatabaseDisconnected**
- **Condition** : `f1pa_database_connected == 0` pendant 2min
- **Impact** : Endpoints /data/* non fonctionnels
- **Action** : Redémarrer PostgreSQL

### 4. **HighPredictionErrorRate**
- **Condition** : Taux d'erreur > 5% pendant 5min
- **Impact** : Prédictions échouent
- **Action** : Vérifier les logs API

## 📊 Dashboard Grafana

Le dashboard **F1PA ML Model Monitoring** contient :

### Ligne 1 - Activité
- **Prediction Requests/sec** : Débit de prédictions
- **Prediction Latency (p95)** : Latence 95e percentile

### Ligne 2 - Statuts
- **Model Status** : Modèle chargé ou non
- **Prediction Error Rate** : Taux d'erreur
- **Database Connection** : Statut DB
- **MLflow Connection** : Statut MLflow

### Ligne 3 - HTTP
- **HTTP Status Codes** : Distribution des codes HTTP

## 🔬 Monitoring ML Drift avec Evidently

✅ **Evidently 0.4.33** est installé et génère des **rapports HTML interactifs complets**.

### Générer un rapport de drift

**Option 1 : Via script Python depuis le container**

```bash
# Exécuter la génération du rapport (le script est déjà présent)
docker exec f1pa_api python scripts/generate_drift_report.py
```

**Option 2 : Via code Python**

```python
from monitoring.evidently.drift_monitor import DriftMonitor
import pandas as pd

# Charger les données (70% référence, 30% production)
df = pd.read_csv("data/processed/dataset_ml_lap_level_2023_2024_2025.csv")
split_idx = int(len(df) * 0.7)
reference_data = df[:split_idx]
current_data = df[split_idx:]

# Générer le rapport
monitor = DriftMonitor()
report_path = monitor.generate_data_drift_report(
    reference_data=reference_data,
    current_data=current_data,
    report_name="production_drift"
)

print(f"Rapport disponible: {report_path}")
```

### Rapports générés

Les rapports HTML interactifs sont stockés dans :
```
monitoring/evidently/reports/
  └── test_data_drift.html  (rapport complet avec graphiques interactifs)
```

**Visualisation** : Ouvrir le fichier `.html` dans un navigateur pour accéder à :
- **Graphiques interactifs de drift** par feature
- **Tests statistiques détaillés** (Kolmogorov-Smirnov, etc.)
- **Comparaison des distributions** référence vs production
- **Tableau récapitulatif** des features avec drift détecté
- **Métriques de qualité** des données

### Exemple de rapport

Le rapport de test inclut l'analyse de :
- **50 151 tours de référence** (70% du dataset)
- **21 494 tours actuels** (30% du dataset)
- **10 features analysées** : driver_number, circuit_key, vitesses, météo, etc.

## 🛠️ Configuration

### Prometheus

Configuration dans `monitoring/prometheus/prometheus.yml` :

```yaml
scrape_configs:
  - job_name: 'f1pa_api'
    static_configs:
      - targets: ['api:8000']
    scrape_interval: 10s  # Scrape toutes les 10 secondes
```

### Grafana

Provisioning automatique :
- **Datasource** : `monitoring/grafana/provisioning/datasources/prometheus.yml`
- **Dashboard** : `monitoring/grafana/provisioning/dashboards/json/f1pa_ml_monitoring.json`

## 📝 Logs et Debugging

### Vérifier que Prometheus scrape l'API

```bash
# Targets Prometheus
curl http://localhost:9090/api/v1/targets

# Doit montrer f1pa_api avec health="up"
```

### Vérifier les métriques de l'API

```bash
# Métriques brutes
curl http://localhost:8000/metrics | grep f1pa
```

### Logs des services

```bash
# API
docker logs f1pa_api

# Prometheus
docker logs f1pa_prometheus

# Grafana
docker logs f1pa_grafana
```

## 🧪 Tests

### Générer du trafic test

```bash
# Prédiction simple
for i in {1..100}; do
  curl -s -u f1pa:f1pa -X POST http://localhost:8000/predict/lap \
    -H "Content-Type: application/json" \
    -d '{"features":{"driver_number":1,"circuit_key":7,...}}'
done

# Vérifier les métriques
curl http://localhost:8000/metrics | grep f1pa_predictions_total
```

### Simuler une erreur

```bash
# Prédiction avec features invalides
curl -u f1pa:f1pa -X POST http://localhost:8000/predict/lap \
  -H "Content-Type: application/json" \
  -d '{"features":{"driver_number":-999}}'

# Vérifier le compteur d'erreurs
curl http://localhost:8000/metrics | grep f1pa_prediction_errors
```
