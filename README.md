# F1PA - Formula 1 Predictive Assistant

Projet de certification professionnelle **Développeur IA** consistant à construire un pipeline data ETL complet et un modèle ML pour prédire les performances en Formule 1 (temps au tour) en fonction des conditions contextuelles et météorologiques.

---

## Vue d'ensemble

**Objectif** : Estimer `lap_duration` (temps au tour en secondes) à partir de :
- Features sport : vitesses, temps secteurs
- Features météo : température, vent, pression, humidité
- Contexte : circuit, pilote, année

**Granularité** : lap-level (tour de piste)

**Période** : 2023-2025 (71,645 laps)

---

## Architecture projet

```
F1PA/
├── data/
│   ├── extract/         # Données brutes (OpenF1, Wikipedia, Meteostat)
│   ├── transform/       # Données transformées (enrichies, nettoyées)
│   └── processed/       # Dataset ML final
│
├── etl/
│   ├── extract/         # Scripts Extract
│   ├── transform/       # Scripts Transform
│   └── load/            # Scripts Load
│
├── notebooks/           # Exploration / Analyses
├── ml/                  # Modèles ML
├── api/                 # API REST
├── streamlit/           # UI Streamlit
├── monitoring/          # MLflow / Evidently
├── tests/               # Tests unitaires
│
├── docker-compose.yml   # PostgreSQL + MLflow + Airflow
└── requirements.txt
```

---

## Pipeline ETL

### 1. EXTRACT

**Sources de données** :
- **OpenF1 API** : sessions, circuits, laps, drivers
- **Wikipedia** (scraping) : localisation circuits (lat/lon)
- **Meteostat** : stations météo + météo horaire

**Sorties** :
- `data/extract/openf1/sessions_openf1_2023_2024_2025.csv`
- `data/extract/openf1/openf1_drivers_2023_2024_2025.csv` (32 pilotes)
- `data/extract/meteostat/hourly/` (météo par station/année)

**Orchestrateur** :
```bash
python -m etl/extract/run_extract_all.py --years 2023 2024 2025 --wiki-sleep 0.5 --top-n 15 --purge-raw
```

**Documentation** : Voir dossiers `etl/extract/*/`

---

### 2. TRANSFORM

**Pipeline en 6 étapes** :

1. `01_build_sessions_scope.py` : Filtre sessions Race uniquement
2. `02_extract_openf1_laps.py` : Extraction laps depuis OpenF1
3. `03_filter_clean_laps.py` : Nettoyage outliers
4. `04_enrich_laps_context.py` : Enrichissement contexte session/circuit
5. `05_join_weather_hourly.py` : Jointure météo horaire
6. `06_build_dataset_ml.py` : Construction dataset final

**Sortie** :
- `data/processed/dataset_ml_lap_level_2023_2024_2025.csv` (71,645 laps × 31 colonnes)
- Rapport qualité : `dataset_ml_lap_level_2023_2024_2025.report.json`

**Orchestrateur** :
```bash
python etl/transform/run_transform_all.py --years 2023 2024 2025
```

---

### 3. LOAD

**Base de données** : PostgreSQL 15 (Docker)

**Architecture** : Schéma en étoile simplifié (4 tables)

```
┌─────────────────┐
│  dim_circuits   │ (24 circuits)
└─────────────────┘
        │
        ↓ FK
┌─────────────────┐
│  dim_sessions   │ (71 sessions Race)
└─────────────────┘
        │
        ↓ FK
┌─────────────────┐      ┌─────────────────┐
│   fact_laps     │ ←────│  dim_drivers    │ (32 pilotes)
└─────────────────┘  FK  └─────────────────┘
    (71,645 laps)
```

**Chargement** :
```bash
# Démarrer PostgreSQL
docker-compose up -d postgres

# Charger les données
cd etl/load
python load_all_docker.py
```

**Documentation** : [etl/load/README.md](etl/load/README.md)

---

## Stack technique

**Langages** :
- Python 3.10+
- SQL (PostgreSQL)

**Librairies principales** :
```
pandas, numpy              # Data manipulation
requests, beautifulsoup4   # Web scraping
meteostat                  # Weather data
psycopg2-binary, sqlalchemy # Database
scikit-learn, xgboost      # ML
mlflow                     # ML tracking
evidently                  # Monitoring
```

**Infrastructure** :
- Docker + Docker Compose
- PostgreSQL 15
- MLflow server (port 5000)
- Apache Airflow (port 8080) - optionnel

---

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/<user>/F1PA.git
cd F1PA
```

### 2. Créer environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

### 3. Installer dépendances

```bash
pip install -r requirements.txt
```

### 4. Démarrer infrastructure Docker

```bash
docker-compose up -d postgres mlflow
```

---

## Utilisation

### Pipeline complet Extract → Transform → Load

```bash
# 1. Extract (sessions + circuits + météo)
python -m etl.extract.run_extract_all --years 2023 2024 2025 --wiki-sleep 0.5 --top-n 15 --purge-raw

# 2. Transform step 01 (crée sessions_scope, requis pour drivers)
python etl/transform/01_build_sessions_scope.py --years 2023 2024 2025

# 3. Extract drivers (dépend de Transform step 01)
python run_extract_drivers_standalone.py --years 2023 2024 2025

# 4. Transform complet (steps 02-06)
python etl/transform/run_transform_all.py --years 2023 2024 2025

# 5. Load PostgreSQL
docker-compose up -d postgres
python etl/load/load_all_docker.py
```

**Note** : Le script `extract_drivers.py` a une dépendance sur `sessions_scope` créé dans Transform step 01. Voir [etl/extract/README_DRIVERS.md](etl/extract/README_DRIVERS.md) pour l'explication architecturale.

### Requêtes SQL d'exemple

```sql
-- Top 10 pilotes par nombre de laps
SELECT
    d.name_acronym,
    d.full_name,
    COUNT(*) as total_laps
FROM fact_laps f
JOIN dim_drivers d ON f.driver_number = d.driver_number
GROUP BY d.driver_number, d.name_acronym, d.full_name
ORDER BY total_laps DESC
LIMIT 10;
```

---

## Prochaines étapes

### Bloc ML (à venir)

- [ ] Entraînement modèle XGBoost/Random Forest
- [ ] Hyperparameter tuning (Optuna)
- [ ] Feature engineering avancé
- [ ] Validation croisée temporelle
- [ ] Tracking MLflow

### Bloc API (à venir)

- [ ] FastAPI REST endpoint
- [ ] Endpoint `/predict` (lap_duration)
- [ ] Authentification
- [ ] Documentation OpenAPI
- [ ] Tests unitaires

### Bloc UI (à venir)

- [ ] Streamlit dashboard
- [ ] Sélection pilote/circuit/météo
- [ ] Prédiction en temps réel
- [ ] Visualisations (courbes, heatmaps)

### Bloc MLOps (à venir)

- [ ] CI/CD GitHub Actions
- [ ] Monitoring Evidently (data drift)
- [ ] Tests automatisés (pytest)
- [ ] Packaging Docker (API + UI)

---

## Compétences validées (Certification)

### Bloc E1 : Collecte et préparation des données

- ✅ C1 : Automatisation extraction (API, scraping, fichiers)
- ✅ C2 : Requêtes SQL (PostgreSQL)
- ✅ C3 : Agrégation et homogénéisation données
- ✅ C4 : Création BDD (modèle relationnel + RGPD)
- ✅ C5 : Partage dataset (SQL + API)

### Bloc E3 : Industrialisation IA (à venir)

- C9 : API REST + modèle IA
- C10 : Intégration API dans application
- C11 : Monitoring modèle (Evidently)
- C12 : Tests automatisés
- C13 : CI/CD MLOps

---

## Auteur

Projet réalisé dans le cadre de la certification **Développeur IA**

---

## Licence

Projet éducatif - Certification professionnelle
