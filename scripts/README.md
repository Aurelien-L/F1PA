# F1PA - Scripts Utilitaires

Scripts d'automatisation pour le projet F1PA.

---

## 📜 Scripts disponibles

### `etl_pipeline.py` - Pipeline ETL Complet

**Description** : Orchestre l'exécution complète du pipeline ETL (Extract → Transform → Load).

**Usage** :
```bash
# Exécution complète
python scripts/etl_pipeline.py --years 2023 2024 2025

# Skip Extract (données déjà téléchargées)
python scripts/etl_pipeline.py --years 2023 2024 2025 --skip-extract

# Skip Load (base de données déjà peuplée)
python scripts/etl_pipeline.py --years 2023 2024 2025 --skip-load

# Forcer la ré-exécution
python scripts/etl_pipeline.py --years 2023 2024 2025 --force

# Vérification qualité uniquement
python scripts/etl_pipeline.py --verify-only
```

**Dépendances** : Gère automatiquement l'ordre d'exécution et les dépendances entre phases.

---

### `extract_drivers.py` - Extraction Drivers Standalone

**Description** : Extrait les données des pilotes après Transform step 01 (dépendance architecturale).

**Usage** :
```bash
python scripts/extract_drivers.py --years 2023 2024 2025
```

**Note** : Ce script nécessite que Transform step 01 soit déjà exécuté car il dépend de `sessions_scope`.

---

### `generate_drift_report.py` - Génération Rapport Evidently

**Description** : Génère un rapport HTML interactif de drift ML avec Evidently.

**Usage** :
```bash
# Depuis le container API
docker exec f1pa_api python scripts/generate_drift_report.py

# Ou localement (si environnement configuré)
python scripts/generate_drift_report.py
```

**Output** : `monitoring/evidently/reports/test_data_drift.html`

**Configuration** :
- Split 70/30 (référence/production)
- Dataset : `data/processed/dataset_ml_lap_level_2023_2024_2025.csv`
- Features analysées : 10 features (vitesses, météo, contexte)

---

### `deploy.sh` - Déploiement Automatisé

**Description** : Script de déploiement interactif (local ou production).

**Usage** :
```bash
bash scripts/deploy.sh
```

**Options** :
1. **Local development** : Build + démarrage des services
2. **Production** : Déploiement via SSH sur serveur distant


---

## ⚠️ Notes

- Les scripts ETL nécessitent une connexion réseau (API OpenF1, Meteostat)
- Le script de drift nécessite le dataset complet (~72k tours)
- Le script de déploiement nécessite Docker et Docker Compose
