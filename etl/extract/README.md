# F1PA — Extract (ETL)

Ce dossier contient l’ensemble des scripts **Extract** du projet **Formula 1 Predictive Assistant (F1PA)**.

Objectif : produire des artefacts **reproductibles** et **traçables** à partir de trois sources :
- **OpenF1 API** (données sportives, sessions)
- **Wikipedia** (référentiel circuits + coordonnées)
- **Meteostat** (météo horaire par station)

Aucune transformation métier n’est effectuée ici (pas de feature engineering, pas de jointures finales ML).  
La sortie Extract est conçue pour alimenter **Transform** puis **Load (PostgreSQL)**.

---

## 1) Prérequis

- Python 3.10+
- Dépendances installées (`requirements.txt`)
- Exécution depuis la racine du projet `F1PA/`

Les scripts s’exécutent en local et produisent des fichiers sous `data/extract/**` (généralement ignorés par git).

---

## 2) Vue d’ensemble des scripts (rôle et dépendances)

### Ordre logique d’exécution

1. **OpenF1**
   - Extraire les sessions
   - Identifier les circuits réellement utilisés (2023–2025)

2. **Wikipedia**
   - Scraper la liste des circuits F1
   - Récupérer les coordonnées (lat/lon)
   - Faire le matching OpenF1 ↔ Wikipedia
   - Filtrer les circuits utiles uniquement

3. **Meteostat**
   - Télécharger la base des stations
   - Associer chaque circuit à une station météo pertinente
   - Télécharger la météo horaire par station et par année

### Orchestrateurs (entrée recommandée)

#### `etl/extract/run_extract_openf1.py`
Orchestre l’extraction OpenF1 :
1. `openf1/extract_sessions.py`
2. `openf1/build_circuits_used.py`

Sorties principales :
- `data/extract/openf1/sessions_openf1_<year>.csv`
- `data/extract/openf1/sessions_openf1_2022_2025.csv` (concat des années dispo)
- `data/extract/openf1/openf1_year_availability.csv`
- `data/extract/openf1/openf1_circuits_used_2022_2025.csv`

#### `etl/extract/run_extract_wikipedia.py`
Orchestre Wikipedia + matching OpenF1 ↔ Wikipedia :
1. `wikipedia/extract_circuits.py` (scraping circuits + lat/lon)
2. `matching/build_openf1_wikipedia_candidates.py` (candidats)
3. `matching/finalize_openf1_wikipedia_mapping.py` (mapping final + overrides)
4. `wikipedia/filter_circuits_for_openf1.py` (filtre circuits utiles)

Sorties principales :
- `data/extract/wikipedia/circuits_wikipedia_extract.csv`
- `data/extract/matching/openf1_wikipedia_match_candidates.csv`
- `data/extract/matching/openf1_to_wikipedia_circuit_map.csv`
- `data/extract/wikipedia/circuits_wikipedia_filtered_2023_2025.csv`

#### `etl/extract/run_extract_meteostat.py`
Orchestre Meteostat (stations + mapping + téléchargement hourly) :
1. `meteostat/download_stations_db.py`
2. `meteostat/build_circuit_station_mapping.py` (sélection station “coverage-aware”)
3. `meteostat/download_hourly_by_station.py` (hourly par station / année)

Sorties principales :
- `data/extract/meteostat/stations/stations.db` (local, volumineux)
- `data/extract/meteostat/mapping/circuit_station_mapping_2023_2025.csv`
- `data/extract/meteostat/mapping/circuit_station_mapping_decisions_2023_2025.csv`
- `data/extract/meteostat/hourly/<station>__<country>__<locality>/{2023,2024,2025}.csv`
- `data/extract/meteostat/hourly_download_report.csv`

#### `etl/extract/run_extract_all.py`
Orchestrateur global : exécute dans l’ordre
1) OpenF1  
2) Wikipedia + matching  
3) Meteostat

>C’est la commande recommandée pour reproduire l’Extract complet.

---

### Scripts unitaires (exécutés par les orchestrateurs)

#### OpenF1
- `etl/extract/openf1/extract_sessions.py`  
  Télécharge les sessions OpenF1 par année via API et écrit des CSV.
- `etl/extract/openf1/build_circuits_used.py`  
  Construit la liste des circuits effectivement utilisés (périmètre 2023–2025) à partir des sessions.

#### Wikipedia
- `etl/extract/wikipedia/extract_circuits.py`  
  Scrape la page “List of Formula One circuits” et récupère :
  `circuit_name, circuit_url, locality, country, latitude, longitude, scraped_at_utc`
- `etl/extract/wikipedia/filter_circuits_for_openf1.py`  
  Filtre les circuits Wikipedia pour ne conserver que ceux utilisés par OpenF1 (via mapping validé).

#### Matching OpenF1 ↔ Wikipedia
- `etl/extract/matching/build_openf1_wikipedia_candidates.py`  
  Génère une liste de candidats Wikipedia par circuit OpenF1 (matching approximatif).
- `etl/extract/matching/finalize_openf1_wikipedia_mapping.py`  
  Construit le mapping final “OpenF1 circuit_key -> Wikipedia circuit_url” en appliquant :
  - une sélection par défaut (best score)
  - des **overrides** si nécessaires (fichier versionné, voir plus bas)

#### Meteostat
- `etl/extract/meteostat/download_stations_db.py`  
  Télécharge `stations.db` (référentiel des stations Meteostat).
- `etl/extract/meteostat/inspect_stations_db.py`  
  Script utilitaire pour inspecter la structure SQL (debug / exploration).
- `etl/extract/meteostat/build_circuit_station_mapping.py`  
  Sélectionne la station Meteostat par circuit (logique “nearest + couverture 2023–2025”).
  Produit des fichiers “candidates” et “decisions”.
- `etl/extract/meteostat/download_hourly_by_station.py`  
  Télécharge `hourly/<year>/<station_id>.csv.gz`, décompresse vers CSV, et peut purger les raw.

---

## 3) Fichier d’overrides (important)

Certains circuits peuvent être ambigus lors du matching automatique OpenF1 ↔ Wikipedia
(ex : Catalunya, Montreal). Pour verrouiller le résultat, on utilise un fichier versionné :

`etl/extract/matching/openf1_wikipedia_overrides.csv`

Format attendu :

```csv
circuit_key,chosen_candidate_rank
15,2
23,2
```

- `circuit_key` : identifiant OpenF1 du circuit
- `chosen_candidate_rank` : rang du candidat choisi dans `openf1_wikipedia_match_candidates.csv`

## 4) Commandes d’exécution

### A) Exécuter tout Extract (recommandé)

```bash
python -m etl.extract.run_extract_all --years 2023 2024 2025 --wiki-sleep 0.5 --top-n 15 --purge-raw
```


#### Paramètres utiles :

`--years` : années ciblées (OpenF1 est disponible à partir de 2023)

`--wiki-sleep` : pause entre requêtes Wikipedia (respect des serveurs)

`--top-n` : nombre de stations candidates Meteostat évaluées avant décision

`--purge-raw` : supprime les dossiers raw restants (hourly_raw/**) en fin de run

### B) Exécuter par source

#### OpenF1 :

`python -m etl.extract.run_extract_openf1 --years 2023 2024 2025`


#### Wikipedia + matching :

`python -m etl.extract.run_extract_wikipedia --years 2023 2024 2025 --sleep 0.5`


#### Meteostat :

`python -m etl.extract.run_extract_meteostat --years 2023 2024 2025 --top-n 15 --purge-raw`


## 5) Sorties attendues (contrôles rapides)

### OpenF1

- Sessions CSV par année (2023–2025)

- `openf1_year_availability.csv` confirme l’absence de 2022

- `openf1_circuits_used_2022_2025.csv` contient ~24 circuits


### Wikipedia + Matching

- `circuits_wikipedia_extract.csv` (liste complète + coords)

- `openf1_to_wikipedia_circuit_map.csv` (24 lignes)

- `circuits_wikipedia_filtered_2023_2025.csv` (24 lignes, lat/lon non nulles)


### Meteostat

- `stations.db` présent en local

- `circuit_station_mapping_decisions_2023_2025.csv` : years_missing vide

- `hourly_download_report.csv` : OK = 72 (24 circuits x 3 années)

- Dossiers `data/extract/meteostat/hourly/<station>__<country>__<locality>/2023.csv|2024.csv|2025.csv`


## 6) Notes de versioning (Git)

Le dossier data/** est ignoré par défaut (volume élevé).
Les éléments suivants doivent être versionnés car ils assurent la reproductibilité logique :

- scripts `etl/**`
- orchestrateurs
- `etl/extract/matching/openf1_wikipedia_overrides.csv` (si utilisé)

Les gros artefacts (stations.db, hourly) restent locaux.


## 7) Prochaine étape (Transform)

À la fin de l’Extract, on dispose :
- des circuits et coordonnées (Wikipedia filtré)
- des sessions (OpenF1)
- d’une météo horaire par circuit/station (Meteostat)
  

La suite du projet consiste à :
- normaliser et jointer (Transform)
- charger en PostgreSQL (Load)
- construire le dataset ML
- entraîner un modèle de régression + exposer via API/Streamlit