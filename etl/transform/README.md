# F1PA — Transform (ETL)

Ce dossier contient l’ensemble des scripts **Transform** du projet **Formula 1 Predictive Assistant (F1PA)**.

Objectif : construire un **dataset lap-level prêt pour le Machine Learning** à partir des artefacts produits en **Extract**, en appliquant :
- des règles de nettoyage explicites et auditées,
- des enrichissements de contexte (session/circuit),
- une jointure météo **horaire** via Meteostat,
- une sortie unique et traçable dans `data/processed/`.

Le périmètre est volontairement **MVP** : pipeline simple, reproductible, sans sur-ingénierie.

---

## 1) Prérequis

- Python 3.10+
- Dépendances installées (`requirements.txt`)
- Exécution depuis la racine du projet : `F1PA/`
- La partie **Extract** doit avoir été exécutée au préalable (scripts et artefacts disponibles sous `data/extract/**`)

---

## 2) Entrées / sorties de Transform

### Entrées (provenant d’Extract)

- **OpenF1 sessions**  
  `data/extract/openf1/sessions_openf1_<year>.csv`  
  ou `data/extract/openf1/sessions_openf1_2022_2025.csv`

- **Mapping OpenF1 ↔ Wikipedia** (validé en Extract)  
  `data/extract/matching/openf1_to_wikipedia_circuit_map.csv`  
  (utilisé pour relier un `circuit_key` OpenF1 à une `wikipedia_circuit_url`)

- **Mapping Wikipedia → station Meteostat** (décisions “coverage-aware”)  
  `data/extract/meteostat/mapping/circuit_station_mapping_2023_2025.csv`  
  (clé utilisée en Transform : `circuit_url` = URL Wikipedia)

- **Météo Meteostat hourly**  
  `data/extract/meteostat/hourly/<station>__<country>__<locality>/<year>.csv`

### Sorties (Transform)

- Intermédiaires : `data/transform/**`
- Dataset final ML : `data/processed/dataset_ml_lap_level_2023_2024_2025.csv`
- Rapport dataset : `data/processed/dataset_ml_lap_level_2023_2024_2025.report.json`

---

## 3) Granularité et clés (rappel)

### Granularité retenue : **lap-level**
Cible ML : `lap_duration` (temps au tour) — variable continue.

Clé composite unique (lap-level) :
- `meeting_key`
- `session_key`
- `driver_number`
- `lap_number`

Météo : granularité **horaire**  
Alignement temporel par :
- `lap_hour_utc = floor(date_start, 'H')`

---

## 4) Vue d’ensemble des scripts Transform

L’ordre logique est le suivant :

1) Construire le périmètre des sessions (Race-only)  
2) Extraire les laps OpenF1 (ciblé sur le périmètre)  
3) Nettoyer les laps (règles lap-level + outliers)  
4) Enrichir avec le contexte session/circuit  
5) Joindre la météo hourly Meteostat  
6) Concaténer et produire le dataset ML final

---

## 5) Description détaillée des scripts

### `etl/transform/01_build_sessions_scope.py`
**Rôle :** construire la liste des sessions **Race** à traiter.

- Filtre : `session_type == "Race"`
- Parsing dates (UTC)
- Déduplication
- Export scope

**Sortie :**
- `data/transform/sessions_scope_<years>.csv`

**Commande :**
```bash
python -m etl.transform.01_build_sessions_scope --years 2023 2024 2025
```


### `etl/transform/02_extract_openf1_laps.py`

**Rôle :** extraire les laps OpenF1 uniquement pour les sessions du scope.

- Requête ciblée par `session_key`
- Une sortie par session
- Écrit un manifest de run (sessions OK/KO)

**Sorties :**

- `data/transform/laps_raw_by_session/laps_session_<session_key>.csv`
- `data/transform/laps_raw_by_session/manifest_laps_extract.json`

**Commande (exemple) :**
```bash
python -m etl.transform.02_extract_openf1_laps --scope data/transform/sessions_scope_2023_2024_2025.csv --overwrite
```

>Note : certaines sessions peuvent être “KO” (course annulée, données absentes).
Le pipeline est conçu pour rester robuste : ces cas sont tracés dans le manifest.


### `etl/transform/03_filter_clean_laps.py`

**Rôle :** nettoyage lap-level, règles explicites + filtrage outliers.

Règles strictes MVP :
- `lap_duration` non nul
- `lap_duration > 0`
- exclusion des `is_pit_out_lap == True`


Outliers (recommandé) :
- filtrage par **quantiles au niveau session** (ex : 1%–99%), robuste aux circuits rapides/lents

**Sorties :**

- `data/transform/laps_clean_by_session/laps_session_<session_key>.csv`
- `data/transform/laps_clean_by_session/report_laps_cleaning.csv`
- `data/transform/laps_clean_by_session/manifest_cleaning.json`

**Commande (recommandée) :**
```bash
python -m etl.transform.03_filter_clean_laps --use-quantiles --q-low 0.01 --q-high 0.99 --overwrite
```


### `etl/transform/04_enrich_laps_context.py`

**Rôle :** enrichir chaque lap nettoyé avec les métadonnées de session/circuit du scope.

Ajouts principaux :

- colonnes de contexte (`year`, `meeting_key`, `location`, `country_name`, etc.)
- `date_start_session` / `date_end_session`
- `lap_hour_utc` (clé de jointure météo horaire)

**Sorties :**

- `data/transform/laps_with_context_by_session/laps_session_<session_key>.csv`
- `data/transform/laps_with_context_by_session/report_laps_context.csv`
- `data/transform/laps_with_context_by_session/manifest_context.json`

**Commande :**
```bash
python -m etl.transform.04_enrich_laps_context --scope data/transform/sessions_scope_2023_2024_2025.csv --overwrite
```


### `etl/transform/05_join_weather_hourly.py`

**Rôle :** joindre la météo Meteostat hourly à chaque lap.

Principe :

- chaque session a un `circuit_key` OpenF1
- `circuit_key` → `wikipedia_circuit_url` via `openf1_to_wikipedia_circuit_map.csv`
- `wikipedia_circuit_url` → `station_id` via `circuit_station_mapping_2023_2025.csv` (clé `circuit_url`)
- jointure sur `lap_hour_utc` = heure UTC de la météo

Décision MVP :

- pas d’interpolation
- les laps sans météo sont exclus (mais tracés via les stats)

**Sorties :**

- `data/transform/laps_with_weather_by_session/laps_session_<session_key>.csv`
- `data/transform/laps_with_weather_by_session/report_laps_weather_join.csv`
- `data/transform/laps_with_weather_by_session/manifest_weather_join.json`

**Commande :**
```bash
python -m etl.transform.05_join_weather_hourly --overwrite
```


### `etl/transform/06_build_dataset_ml.py`

**Rôle :** concaténer toutes les sessions enrichies météo et produire le dataset ML final.

- Concat des fichiers `laps_session_*.csv`
- Sélection explicite des features (ID / contexte / sport / météo)
- Cast des types
- Contrôles qualité (doublons, NA)
- Export dataset + report JSON

**Sorties :**

- `data/processed/dataset_ml_lap_level_2023_2024_2025.csv`
- `data/processed/dataset_ml_lap_level_2023_2024_2025.report.json`

**Commande :**
```bash
python -m etl.transform.06_build_dataset_ml
```


## 6) Orchestrateur Transform (recommandé)

### `etl/transform/run_transform_all.py`

**Rôle :** exécuter toutes les étapes Transform dans l’ordre, avec options :

- `--overwrite` : régénère les sorties
- `--purge-intermediate` : purge progressive des intermédiaires (optionnel)
tolérance aux étapes “partially successful” (ex : sessions KO tracées)

**Commande recommandée (rebuild complet) :**.
```bash
python -m etl.transform.run_transform_all --years 2023 2024 2025 --use-quantiles --overwrite
```

**Commande recommandée (rebuild + nettoyage intermédiaire) :**
```bash
python -m etl.transform.run_transform_all --years 2023 2024 2025 --use-quantiles --overwrite --purge-intermediate
```

>Recommandation : en phase de debug, ne pas purger.
Une fois validé, `--purge-intermediate` permet de ne conserver que l’essentiel pour la suite (Load / ML / API).


## 7) Contrôles rapides (à faire après Transform)
### A) Le dataset final existe

- `data/processed/dataset_ml_lap_level_2023_2024_2025.csv`

### B) Le report JSON est cohérent

- `n_duplicates_key` idéalement à 0
- `missing_target` doit être 0 (après filtres)
- `missing_weather_all` doit être 0 (sinon jointure météo invalide)

### C) Quelques lignes du dataset “ont du sens”

- `lap_duration` plausible
- `lap_hour_utc` présent
- variables météo présentes


## 8) Notes de design (justifications MVP)

- **ETL simple**
- **Traçabilité :** chaque étape produit un report/manifest
- **Robustesse :** une session “KO” n’arrête pas tout le pipeline (cas réel : course annulée)
- **Clé de référence circuits :** `wikipedia_circuit_url` (stable, versionnée, et déjà validée en Extract)
- **Météo :** alignement horaire (simplicité, cohérence avec Meteostat hourly)


## 9) Prochaine étape : Load (PostgreSQL)

À l’issue de Transform, le projet dispose :

- d’un dataset ML unique prêt à charger en base,
- d’une chaîne de préparation entièrement reproductible,
- d’artefacts explicables et auditables pour un jury.

La suite consiste à :

- définir un schéma minimal PostgreSQL (MVP),
- charger le dataset,
- préparer l’API ML et le prototype Streamlit.