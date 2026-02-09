<p align="center">
  <img src="img/f1pa_banner.png" alt="F1PA Banner" width="800"/>
</p>

<h1 align="center">F1PA - Formula 1 Predictive Assistant</h1>

<p align="center">
  <strong>Pipeline ETL complet + Modèle ML pour prédire les temps au tour en Formule 1</strong><br>
  Projet d'IA appliquée au sport automobile
</p>

<p align="center">
  <a href="#-vue-densemble">Vue d'ensemble</a> •
  <a href="#-démarrage-rapide">Démarrage rapide</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-monitoring--mlops">Monitoring</a> •
  <a href="#-documentation">Documentation</a>
</p>

---

## 📊 Vue d'ensemble

**Objectif** : Prédire le temps au tour (`lap_duration`) d'un pilote, basé sur sa performance historique et les conditions.

**Données** :
- 71,645 tours de piste (2023-2025)
- 24 circuits × 32 pilotes
- Features : vitesses, météo, contexte circuit/pilote
- Sources : OpenF1 API, Wikipedia, Meteostat

**Stack** :
- **Backend** : Python 3.10+, PostgreSQL 15, FastAPI
- **ML** : scikit-learn, XGBoost, MLflow
- **Monitoring** : Prometheus, Grafana, Evidently
- **MLOps** : Docker, GitHub Actions (CI/CD)

---

## 🚀 Démarrage rapide

### Prérequis

```bash
# Cloner le projet
git clone https://github.com/<user>/F1PA.git
cd F1PA

# Créer environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Installer dépendances
pip install -r requirements.txt
```

### Lancer les services

```bash
# Démarrer tous les services (API, UI, Monitoring)
docker compose up -d

# Vérifier la santé des services
curl -u f1pa:f1pa http://localhost:8000/health
```

**Services disponibles** :
- 🔹 **API Documentation** : [http://localhost:8000/docs](http://localhost:8000/docs)
- 🔹 **Streamlit UI** : [http://localhost:8501](http://localhost:8501)
- 🔹 **MLflow** : [http://localhost:5000](http://localhost:5000)
- 🔹 **Grafana** : [http://localhost:3000](http://localhost:3000) (admin/admin)
- 🔹 **Prometheus** : [http://localhost:9090](http://localhost:9090)

### Exécuter le pipeline ETL

```bash
# Pipeline complet (Extract → Transform → Load)
python scripts/etl_pipeline.py --years 2023 2024 2025

# Options
python scripts/etl_pipeline.py --years 2024 2025 --skip-extract  # Si données déjà extraites
python scripts/etl_pipeline.py --verify-only                      # Vérification qualité uniquement
```

**Résultat** :
- Dataset ML : `data/processed/dataset_ml_lap_level_2023_2024_2025.csv`
- Base PostgreSQL peuplée (4 tables, 71k+ laps)

### Entraîner le modèle

```bash
# Entraînement avec GridSearchCV + MLflow tracking
python ml/run_ml_pipeline.py
```

**Résultats** :
- Modèle Random Forest (GridSearchCV) : MAE 1.08s, R² 0.79
- Tracking MLflow : [http://localhost:5000](http://localhost:5000)

---

## 🏗️ Architecture

```
F1PA/
├── data/
│   ├── extract/         # Données brutes (OpenF1, Wikipedia, Meteostat)
│   ├── transform/       # Données enrichies et nettoyées
│   └── processed/       # Dataset ML final (71k laps × 31 features)
│
├── etl/
│   ├── extract/         # Scripts extraction (API + scraping)
│   ├── transform/       # 6 étapes de transformation
│   └── load/            # Chargement PostgreSQL
│
├── ml/                  # Pipeline ML (preprocessing, training, inference)
├── api/                 # FastAPI REST (endpoints + auth)
├── streamlit/           # Interface utilisateur
├── monitoring/          # Evidently (drift detection)
├── tests/               # 53 tests (unitaires + intégration)
│
├── scripts/             # Scripts utilitaires (ETL, monitoring, déploiement)
├── .github/workflows/   # CI/CD GitHub Actions
└── docker-compose.yml   # Services (PostgreSQL, MLflow, Prometheus, Grafana)
```

### Pipeline ETL

**Extract** :
- OpenF1 API : sessions, circuits, laps, drivers
- Wikipedia : coordonnées géographiques circuits
- Meteostat : données météo horaires

**Transform** (6 étapes) :
1. Filtrage sessions Race
2. Extraction laps via OpenF1
3. Nettoyage outliers (quantiles)
4. Enrichissement contexte (circuit, pilote)
5. Jointure météo horaire
6. Construction dataset ML final

**Load** :
- PostgreSQL 15 (schéma en étoile)
- 4 tables : `fact_laps`, `dim_drivers`, `dim_circuits`, `dim_sessions`


### Modèle ML

**Objectif** : Prédire `lap_duration`

**Features principales** :
- Sport : `st_speed`, `i1_speed`, `i2_speed` (vitesses historiques)
- Météo : `temp`, `rhum`, `pres` (température, humidité, pression)
- Contexte : `circuit_avg_laptime`, `driver_perf_score`, `lap_progress`

**Modèle** : Random Forest (GridSearchCV)
- **MAE** : 1.08s (test)
- **R²** : 0.79 (test)
- **MAPE** : 0.90%
- **Features** : 14 features
- **Model size** : 335 MB (production-ready)
- **Tracking** : MLflow (hyperparams, metrics, feature importance)

### Optimisation du modèle

Le modèle a été optimisé à travers plusieurs itérations (v0 → v6) :
- **Réduction taille** : 1.5 GB → 335 MB (-78%)
- **Amélioration performance** : R² 0.77 → 0.79
- **Optimisation features** : Suppression redondance driver_avg_laptime (15 → 14 features)
- **Temps chargement API** : 19s → ~3s

📚 **Documentation détaillée** : [ml/MODEL_OPTIMIZATION.md](ml/MODEL_OPTIMIZATION.md)


### Scalabilité Big Data

**Architecture actuelle** : PostgreSQL 15 (adapté pour ~71k laps)

**Scale-up pour volumes > 10M rows** :
- **Apache Spark SQL** : Requêtes distribuées sur clusters Hadoop/HDFS
- **Apache Hive** : Data warehouse SQL sur Big Data avec partitionnement
- **Presto/Trino** : Requêtes SQL temps réel sur data lakes (S3, HDFS)

Le projet est conçu pour faciliter la migration : les requêtes SQL PostgreSQL sont compatibles Spark SQL avec adaptations mineures (types de données, fonctions de fenêtrage).

---

## 📊 Monitoring & MLOps

### CI/CD Pipeline

**GitHub Actions** - Workflow automatique sur chaque push :

```
Push → Lint → Tests → Build → Deploy
       ↓      ↓       ↓
    pylint  pytest  docker
           53 tests  images
```

**Workflows** :
- ✅ `.github/workflows/ci.yml` - Lint (pylint), test, build automatique
- ✅ `.github/workflows/release.yml` - Releases versionnées (tags `v*.*.*`)

**Tests locaux** :
```bash
pylint --rcfile=pyproject.toml api/ ml/ etl/ monitoring/ streamlit/ tests/ scripts/  # Code quality
pytest tests/ -v --cov=. --cov-report=term-missing  # 53 tests avec coverage
docker compose build            # Build images
docker compose up -d            # Lancer services
```

### Monitoring ML

**Prometheus + Grafana** :
- Métriques API : requêtes/sec, latence, erreurs
- Métriques modèle : prédictions, temps d'inférence
- Dashboard Grafana : [http://localhost:3000](http://localhost:3000)

**Evidently** - Détection de drift :
```bash
# Générer rapport de drift
docker exec f1pa_api python scripts/generate_drift_report.py

# Rapport HTML : monitoring/evidently/reports/test_data_drift.html
```

**Alertes** :
- Drift détecté sur features (threshold configurable)
- Performance dégradée (MAE > seuil)

📚 **Documentation détaillée** : [monitoring/README.md](monitoring/README.md)

---


## 📖 Documentation

**Guides essentiels** :

- 📘 [DEVELOPMENT.md](DEVELOPMENT.md) - **Guide complet** : développement, tests, CI/CD, déploiement
- 🤖 [ml/MODEL_OPTIMIZATION.md](ml/MODEL_OPTIMIZATION.md) - Optimisation du modèle ML (v0 → v6)
- 📊 [monitoring/MONITORING.md](monitoring/MONITORING.md) - Monitoring ML (Prometheus, Grafana, Evidently)
- 🔧 [scripts/README.md](scripts/README.md) - Scripts utilitaires (ETL, monitoring, déploiement)
- 🔒 [RGPD.md](RGPD.md) - Conformité RGPD

**API Documentation** :
- Swagger UI : [http://localhost:8000/docs](http://localhost:8000/docs)
- Endpoints : `/predict/lap`, `/data/drivers`, `/data/circuits`
- **Authentification** :
  - Dev/Démo : HTTP Basic Auth (username/password)
  - Production recommandée : JWT/OAuth2 pour sécurité renforcée

---