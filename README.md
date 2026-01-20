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
├── docker-compose.yml   # PostgreSQL + MLflow
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

### Pipeline complet - Orchestrateur global (recommandé) ⭐

Le script [run_full_pipeline.py](run_full_pipeline.py) exécute automatiquement Extract → Transform → Load dans le bon ordre :

```bash
# Exécution complète du pipeline (15-20 minutes)
python run_full_pipeline.py --years 2023 2024 2025

# Options disponibles
python run_full_pipeline.py --years 2023 2024 2025 --skip-extract  # Skip Extract si données déjà extraites
python run_full_pipeline.py --years 2023 2024 2025 --skip-load     # Skip Load si DB déjà peuplée
python run_full_pipeline.py --years 2023 2024 2025 --force         # Force ré-exécution
python run_full_pipeline.py --verify-only                          # Vérification qualité uniquement
```

**Fonctionnalités** :
- ✅ Gère automatiquement l'ordre d'exécution et les dépendances
- ✅ Résout la dépendance `extract_drivers` ↔ `Transform step 01`
- ✅ Vérifie les prérequis (Docker, Python, venv)
- ✅ Détecte les données existantes (évite re-téléchargement)
- ✅ Affiche la progression en temps réel avec logs clairs
- ✅ Validation qualité des données à la fin (dataset + PostgreSQL)
- ✅ Gestion d'erreurs robuste avec messages explicites

**Validation automatique** :
- Dataset : 71,645 laps | 31 colonnes | 0 doublon | 0 target manquant
- PostgreSQL : Intégrité clés étrangères 100%
- Météo : 100% complète (températures 14.6°C - 33.0°C)

### Pipeline manuel (étape par étape)

Si vous préférez exécuter manuellement :

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

---

### Vérification qualité des données

```bash
# Vérifier la cohérence des données sans ré-exécuter le pipeline
python run_full_pipeline.py --verify-only
```

**Contrôles effectués** :
- ✅ Nombre de lignes/colonnes du dataset ML
- ✅ Doublons sur clé composite
- ✅ Valeurs target manquantes
- ✅ Comptage tables PostgreSQL
- ✅ Intégrité clés étrangères

---

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

## Troubleshooting

### Docker non démarré

```
[ERROR] Docker not found
```
**Solution** : Démarrer Docker Desktop avant d'exécuter le pipeline.

### PostgreSQL ne démarre pas

**Vérifier les logs** :
```bash
docker logs f1pa_postgres
```

**Redémarrer proprement** :
```bash
docker-compose down
docker-compose up -d postgres
```

### Réinitialiser complètement la base de données

```bash
# Supprimer le volume PostgreSQL
docker-compose down -v

# Redémarrer PostgreSQL
docker-compose up -d postgres

# Recharger les données
python run_full_pipeline.py --years 2023 2024 2025 --skip-extract --skip-transform
```

### Le pipeline est lent (Extract)

L'étape Extract peut prendre 10-15 minutes à cause des téléchargements API/Web.

**Solution 1** : Si données déjà extraites, utiliser `--skip-extract`
```bash
python run_full_pipeline.py --years 2023 2024 2025 --skip-extract
```

**Solution 2** : Réduire le nombre d'années
```bash
python run_full_pipeline.py --years 2024 2025
```

### Données manquantes

```
[ERROR] sessions_scope not found
```
**Solution** : Ne pas utiliser `--skip-transform` si les données Transform n'existent pas encore.

---

## Documentation détaillée

**README par phase ETL** :
- [etl/extract/README.md](etl/extract/README.md) - Explications extraction des données
- [etl/transform/README.md](etl/transform/README.md) - Traitement et enrichissement des données
- [etl/load/README.md](etl/load/README.md) - Architecture PostgreSQL, exemples SQL

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
