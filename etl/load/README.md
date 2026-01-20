# F1PA - LOAD Pipeline

Documentation de la partie **LOAD** du pipeline ETL du projet F1PA.

## Vue d'ensemble

La phase LOAD charge les données transformées dans une base de données **PostgreSQL** structurée selon un schéma en étoile simplifié, prête pour l'entraînement ML et l'exploitation via API.

### Architecture de la base de données

```
┌─────────────────┐
│  dim_circuits   │ ← Circuits F1
└─────────────────┘
        │
        ↓ FK
┌─────────────────┐
│  dim_sessions   │ ← Sessions Race
└─────────────────┘
        │
        ↓ FK
┌─────────────────┐      ┌─────────────────┐
│   fact_laps     │ ←────│  dim_drivers    │
└─────────────────┘  FK  └─────────────────┘
```

### Tables et volumétrie

| Table | Rôle | Lignes | Clé primaire |
|-------|------|--------|--------------|
| `dim_circuits` | Référence circuits F1 | 24 | `circuit_key` |
| `dim_drivers` | Référence pilotes | 32 | `driver_number` |
| `dim_sessions` | Contexte sessions Race | 71 | `session_key` |
| `fact_laps` | Table de faits lap-level | 71,645 | `(meeting_key, session_key, driver_number, lap_number)` |

---

## Schéma SQL

### dim_circuits
Référence des circuits utilisés (2023-2025).

```sql
CREATE TABLE dim_circuits (
    circuit_key INTEGER PRIMARY KEY,
    circuit_short_name VARCHAR(100),
    location VARCHAR(200),
    country_name VARCHAR(100),
    country_code VARCHAR(10),
    wikipedia_circuit_url TEXT,
    station_id VARCHAR(20)  -- Meteostat weather station
);
```

### dim_drivers
Données pilotes extraites depuis OpenF1 API.

```sql
CREATE TABLE dim_drivers (
    driver_number INTEGER PRIMARY KEY,
    full_name VARCHAR(200),
    broadcast_name VARCHAR(200),
    name_acronym VARCHAR(5),  -- VER, HAM, etc.
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    country_code VARCHAR(10),
    team_name VARCHAR(200),
    team_colour VARCHAR(10),
    headshot_url TEXT
);
```

### dim_sessions
Contexte des sessions Race uniquement.

```sql
CREATE TABLE dim_sessions (
    session_key INTEGER PRIMARY KEY,
    meeting_key INTEGER NOT NULL,
    year INTEGER NOT NULL,
    session_name VARCHAR(100),
    session_type VARCHAR(50),
    date_start TIMESTAMP WITH TIME ZONE,
    date_end TIMESTAMP WITH TIME ZONE,
    gmt_offset INTERVAL,
    circuit_key INTEGER REFERENCES dim_circuits(circuit_key)
);
```

### fact_laps
Table centrale lap-level avec features ML.

```sql
CREATE TABLE fact_laps (
    -- Clé composite
    meeting_key INTEGER NOT NULL,
    session_key INTEGER NOT NULL REFERENCES dim_sessions,
    driver_number INTEGER NOT NULL REFERENCES dim_drivers,
    lap_number INTEGER NOT NULL,

    -- Contexte
    year INTEGER NOT NULL,
    circuit_key INTEGER NOT NULL REFERENCES dim_circuits,
    lap_hour_utc TIMESTAMP WITH TIME ZONE,

    -- Features sport (OpenF1)
    st_speed NUMERIC(10, 2),
    i1_speed NUMERIC(10, 2),
    i2_speed NUMERIC(10, 2),
    duration_sector_1 NUMERIC(10, 3),
    duration_sector_2 NUMERIC(10, 3),
    duration_sector_3 NUMERIC(10, 3),

    -- Features météo (Meteostat)
    temp NUMERIC(10, 2),      -- Température (°C)
    rhum NUMERIC(10, 2),      -- Humidité (%)
    pres NUMERIC(10, 2),      -- Pression (hPa)
    wspd NUMERIC(10, 2),      -- Vent vitesse (km/h)
    wdir NUMERIC(10, 2),      -- Vent direction (°)
    prcp NUMERIC(10, 2),      -- Précipitations (mm)
    cldc NUMERIC(10, 2),      -- Couverture nuageuse (%)

    -- Target ML
    lap_duration NUMERIC(10, 3) NOT NULL,  -- Temps au tour (s)

    -- Metadata
    source_file VARCHAR(200),

    CONSTRAINT pk_fact_laps PRIMARY KEY (meeting_key, session_key, driver_number, lap_number)
);
```

**Index créés** :
- `idx_laps_session` sur `session_key`
- `idx_laps_driver` sur `driver_number`
- `idx_laps_circuit` sur `circuit_key`
- `idx_laps_year` sur `year`
- `idx_laps_hour` sur `lap_hour_utc`

---

## Scripts LOAD

### Fichiers principaux

```
etl/load/
├── schema.sql                  # DDL PostgreSQL (tables, indexes, constraints)
├── load_all_docker.py          # Script principal de chargement
├── db_config.py                # Utilitaires connexion DB (optionnel)
├── 01_init_schema.py           # Initialisation schéma (optionnel)
├── 02_load_dim_circuits.py     # Chargement circuits (optionnel)
├── 03_load_dim_drivers.py      # Chargement drivers (optionnel)
├── 04_load_dim_sessions.py     # Chargement sessions (optionnel)
├── 05_load_fact_laps.py        # Chargement laps (optionnel)
└── run_load_all.py             # Orchestrateur (optionnel, problème Windows)
```

### Script recommandé : `load_all_docker.py`

Ce script utilise `docker exec` + `COPY SQL` pour contourner les problèmes de connexion psycopg2 sur Windows.

**Prérequis** :
- PostgreSQL en cours d'exécution via Docker Compose :
  ```bash
  docker-compose up -d postgres
  ```

**Exécution** :
```bash
cd etl/load
python load_all_docker.py
```

**Étapes exécutées** :
1. Initialisation schéma via `schema.sql`
2. Chargement `dim_circuits` (24 circuits)
3. Chargement `dim_drivers` (32 pilotes)
4. Chargement `dim_sessions` (71 sessions Race)
5. Chargement `fact_laps` (71,645 laps)

---

## Initialisation manuelle du schéma

Si besoin de réinitialiser la base :

```bash
docker exec -i f1pa_postgres psql -U f1pa -d f1pa_db < etl/load/schema.sql
```

---

## Exemples de requêtes SQL

### Top 10 pilotes par nombre de laps

```sql
SELECT
    d.name_acronym,
    d.full_name,
    d.team_name,
    COUNT(f.lap_number) as total_laps
FROM fact_laps f
JOIN dim_drivers d ON f.driver_number = d.driver_number
GROUP BY d.driver_number, d.name_acronym, d.full_name, d.team_name
ORDER BY total_laps DESC
LIMIT 10;
```

### Temps moyen au tour par circuit

```sql
SELECT
    c.circuit_short_name,
    c.location,
    ROUND(AVG(f.lap_duration)::numeric, 3) as avg_lap_time_seconds,
    COUNT(*) as total_laps
FROM fact_laps f
JOIN dim_circuits c ON f.circuit_key = c.circuit_key
GROUP BY c.circuit_key, c.circuit_short_name, c.location
ORDER BY avg_lap_time_seconds;
```

### Conditions météo moyennes par année

```sql
SELECT
    f.year,
    ROUND(AVG(f.temp)::numeric, 1) as avg_temp_c,
    ROUND(AVG(f.wspd)::numeric, 1) as avg_wind_kmh,
    ROUND(AVG(f.rhum)::numeric, 1) as avg_humidity_pct
FROM fact_laps f
WHERE f.temp IS NOT NULL
GROUP BY f.year
ORDER BY f.year;
```

### Laps les plus rapides de Verstappen en 2024

```sql
SELECT
    s.session_name,
    c.circuit_short_name,
    f.lap_number,
    f.lap_duration,
    f.temp,
    f.wspd
FROM fact_laps f
JOIN dim_drivers d ON f.driver_number = d.driver_number
JOIN dim_sessions s ON f.session_key = s.session_key
JOIN dim_circuits c ON f.circuit_key = c.circuit_key
WHERE d.name_acronym = 'VER'
AND f.year = 2024
ORDER BY f.lap_duration
LIMIT 10;
```

---

## Justification de l'architecture

### Choix du schéma en étoile simplifié

1. **MVP** : 4 tables suffisent pour démontrer :
   - Modélisation SQL (dimensions + fait)
   - Clés primaires / étrangères
   - Contraintes de qualité
   - Index de performance

2. **Pas de sur-modélisation** :
   - On aurait pu créer `dim_teams`, `dim_circuits_detailed`, etc.
   - Non nécessaire pour le scope MVP
   - Évite la complexité inutile

3. **Normalisation équilibrée** :
   - Pilotes et sessions normalisés (évite redondance)
   - Circuit info dans dimension dédiée
   - Fact table dénormalisée pour performance ML

### Contraintes de qualité

```sql
-- Lap duration positive et raisonnable
CHECK (lap_duration > 0 AND lap_duration < 3600)

-- Années valides
CHECK (year >= 2020 AND year <= 2030)

-- Driver number positif
CHECK (driver_number > 0)
```

### Index pour performance

- **Requêtes ML** : index sur `year`, `circuit_key`
- **Jointures** : index sur FK (`session_key`, `driver_number`)
- **Analyses temporelles** : index sur `lap_hour_utc`

---

## Prochaines étapes

La base de données PostgreSQL est maintenant **prête pour** :

1. **Entraînement ML** : `fact_laps` contient toutes les features + target
2. **API REST** : Données structurées accessibles via SQLAlchemy
3. **UI Streamlit** : Affichage noms pilotes, équipes, circuits
4. **Monitoring** : Historique complet 2023-2025

### Exemple d'utilisation en Python

```python
from sqlalchemy import create_engine, text

engine = create_engine("postgresql://f1pa:f1pa@localhost:5432/f1pa_db")

with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT d.name_acronym, AVG(f.lap_duration) as avg_lap
        FROM fact_laps f
        JOIN dim_drivers d ON f.driver_number = d.driver_number
        WHERE f.year = 2024
        GROUP BY d.name_acronym
        ORDER BY avg_lap
        LIMIT 5
    """))
    for row in result:
        print(f"{row[0]}: {row[1]:.3f}s")
```

---

## Troubleshooting

### PostgreSQL non démarré

```bash
docker-compose up -d postgres
docker ps  # Vérifier que f1pa_postgres est UP
```

### Problème de connexion psycopg2 sur Windows

Utiliser `load_all_docker.py` qui utilise `docker exec` au lieu de connexion directe.

### Réinitialiser complètement la base

```bash
# Supprimer le volume
docker-compose down -v

# Redémarrer
docker-compose up -d postgres

# Réinitialiser le schéma
docker exec -i f1pa_postgres psql -U f1pa -d f1pa_db < etl/load/schema.sql

# Recharger les données
python etl/load/load_all_docker.py
```

---

**Date de création** : 2026-01-20
**Volumétrie totale** : 71,645 laps | 2023-2025 | 32 pilotes | 24 circuits | 71 sessions
