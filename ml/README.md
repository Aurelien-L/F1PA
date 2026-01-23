# F1PA - Machine Learning Pipeline

**Prédiction de Performance des Temps au Tour en Formule 1**

---

## Table des Matières

1. [Vue d'ensemble](#vue-densemble)
2. [Architecture](#architecture)
3. [Installation & Prérequis](#installation--prérequis)
4. [Exécution Rapide](#exécution-rapide)
5. [Dataset](#dataset)
6. [Feature Engineering](#feature-engineering)
7. [Modèles & Stratégie](#modèles--stratégie)
8. [Résultats](#résultats)
9. [Utilisation des Modèles](#utilisation-des-modèles)
10. [MLflow](#mlflow)
11. [Améliorations Futures](#améliorations-futures)

---

## Vue d'ensemble

Le projet F1PA implémente un système de **prédiction de performance** des temps au tour en Formule 1. L'objectif est de prédire le `lap_duration` (durée du tour en secondes) **AVANT que le pilote ne roule**, basé sur ses performances historiques et les conditions.

### Objectif

Prédire le temps au tour d'un pilote en fonction de:
- **Performance historique**: `driver_perf_score`, `driver_avg_laptime`
- **Caractéristiques circuit**: `circuit_avg_laptime`, `circuit_key`
- **Données météo**: Température, humidité, pression
- **Vitesses attendues**: `st_speed`, `i1_speed`, `i2_speed`
- **Contexte**: Numéro du tour, année

### Important: Prédiction vs Calcul

**Les temps secteurs (`duration_sector_*`) ne sont PAS utilisés** car ils représentent des données du tour en cours. Utiliser ces données rendrait la prédiction triviale :

```
lap_duration ≈ sector_1 + sector_2 + sector_3
```

Notre modèle prédit la **performance** attendue, pas un simple calcul de somme.

### Contraintes

- Données publiques uniquement (pas de télémétrie voiture)
- Split stratifié 80/20 par circuit
- Régularisation anti-overfitting
- Tracking MLflow professionnel

---

## Architecture

```
ml/
├── config.py              # Configuration (chemins, hyperparamètres)
├── preprocessing.py       # Feature engineering et préparation des données
├── train.py              # Pipeline d'entraînement complet
├── load_model_simple.py  # Chargement dynamique des modèles
├── run_ml_pipeline.py    # Script d'exécution tout-en-un (RECOMMANDÉ)
└── README.md            # Ce fichier
```

### Flux de Travail

```
Données brutes (CSV)
    ↓
[preprocessing.py] → Feature engineering (15 features)
    ↓
[train.py] → Entraînement (XGBoost + Random Forest)
    ↓
[MLflow] → Tracking (métriques, artifacts)
    ↓
[load_model_simple.py] → Chargement pour inférence
```

---

## Installation & Prérequis

### 1. Environnement Python

```bash
# Créer environnement virtuel
python -m venv .venv

# Activer (Windows)
.venv\Scripts\activate

# Activer (Linux/Mac)
source .venv/bin/activate

# Installer dépendances
pip install -r requirements.txt
```

### 2. MLflow avec Docker

```bash
# Démarrer PostgreSQL + MLflow
docker-compose up -d

# Vérifier
curl http://localhost:5000/health
# Expected: OK
```

### 3. Données

Le dataset doit être présent dans:
```
data/processed/dataset_ml_lap_level_2023_2024_2025.csv
```

---

## Exécution Rapide

### Option 1: Script Tout-en-Un (RECOMMANDÉ)

```bash
# Exécute le pipeline complet
python ml/run_ml_pipeline.py
```

Ce script:
1. Vérifie les prérequis (dataset, MLflow)
2. Lance l'entraînement complet
3. Affiche les résultats
4. Guide pour charger les modèles

### Option 2: Étape par Étape

```bash
# 1. Entraîner tous les modèles
python -m ml.train

# 2. Voir les modèles disponibles
python -m ml.load_model_simple

# 3. Accéder à l'interface MLflow
# http://localhost:5000
```

---

## Dataset

### Caractéristiques

| Attribut | Valeur |
|----------|--------|
| **Période** | 2023-2025 (3 saisons) |
| **Samples totaux** | 71,645 tours |
| **Train** | 57,316 tours (80%) |
| **Test** | 14,329 tours (20%) |
| **Split** | Stratifié par circuit |
| **Features** | 15 |

### Variables Clés

**Identifiants**:
- `circuit_key`: Identifiant circuit
- `driver_number`: Numéro pilote
- `year`: Saison

**Vitesses (indicateurs de performance)**:
- `st_speed`: Vitesse speed trap (km/h)
- `i1_speed`: Vitesse intermédiaire 1 (km/h)
- `i2_speed`: Vitesse intermédiaire 2 (km/h)

**Météo**:
- `temp`: Température (°C)
- `pres`: Pression atmosphérique (hPa)
- `rhum`: Humidité relative (%)

**Performance encodings**:
- `circuit_avg_laptime`: Temps moyen du circuit
- `driver_avg_laptime`: Temps moyen du pilote
- `driver_perf_score`: Score de performance (négatif = plus rapide)

---

## Feature Engineering

### 1. Target Encoding (Performance)

Le `driver_perf_score` encode la performance relative du pilote:

```python
# Score = différence entre temps pilote et moyenne circuit
driver_perf_score = driver_avg_laptime - circuit_avg_laptime
```

- **Score négatif** = Pilote plus rapide que la moyenne
- **Score positif** = Pilote plus lent que la moyenne

### 2. Features Dérivées

| Feature | Formule | Utilité |
|---------|---------|---------|
| `avg_speed` | Moyenne(st, i1, i2) | Vitesse moyenne globale |
| `lap_progress` | lap_number / 70 | Progression dans la course |

>`lap_progress`est calculé avec une estimation de *70* comme nombre total de tours sur le GP. 70 est un **compromis arbitraire**  venant d'une valeur médiane approximative du nombre de tours. Cette solution permet tout de même de capturer la **tendance générale de dégradation des pneus** en compensant via `circuit_avg_laptime`, sans ajout de feature d'entrée ou de table.

### 3. Features Finales (15)

| # | Feature | Type | Importance |
|---|---------|------|------------|
| 1 | `circuit_avg_laptime` | Encoding | 23.9% |
| 2 | `avg_speed` | Derived | 16.8% |
| 3 | `lap_number` | Context | 12.3% |
| 4 | `lap_progress` | Derived | 11.4% |
| 5 | `st_speed` | Speed | 6.7% |
| 6 | `pres` | Weather | 6.1% |
| 7 | `i2_speed` | Speed | 5.4% |
| 8 | `i1_speed` | Speed | 4.5% |
| 9 | `driver_perf_score` | Encoding | 3.6% |
| 10 | `rhum` | Weather | 2.1% |
| 11 | `driver_avg_laptime` | Encoding | 1.9% |
| 12 | `temp` | Weather | 1.8% |
| 13 | `circuit_key` | Context | 1.8% |
| 14 | `driver_number` | Context | 1.1% |
| 15 | `year` | Context | 0.7% |

---

## Modèles & Stratégie

### Modèles Testés

4 modèles entraînés à chaque run:

1. **XGBoost Baseline**: Configuration par défaut
2. **XGBoost GridSearch**: Hyperparamètres optimisés
3. **Random Forest Baseline**: Configuration par défaut
4. **Random Forest GridSearch**: Hyperparamètres optimisés (MEILLEUR)

### Random Forest GridSearch (RECOMMANDÉ)

**Hyperparamètres optimaux**:

```python
{
    'n_estimators': 300,        # Nombre d'arbres
    'max_depth': None,          # Pas de limite
    'max_features': 0.7,        # 70% des features par split
    'min_samples_split': 2,     # Minimum pour split
    'min_samples_leaf': 1       # Minimum par feuille
}
```

### Stratégie Anti-Overfitting

1. **Split stratifié** par circuit (équilibre train/test)
2. **Cross-validation 3-fold** pour validation robuste
3. **GridSearch** pour optimisation automatique
4. **Métriques multiples** (MAE, RMSE, R², overfitting ratio)

---

## Résultats

### Comparaison des Modèles

| Modèle | Test MAE | Test R² | CV MAE | CV R² |
|--------|----------|---------|--------|-------|
| **RF GridSearch** | **1.070s** | 0.755 | **1.016s** | 0.800 |
| XGBoost GridSearch | 1.127s | 0.698 | 1.069s | 0.797 |
| RF Baseline | 1.130s | 0.780 | 1.090s | 0.805 |
| XGBoost Baseline | 1.230s | 0.696 | 1.158s | 0.765 |

### Métriques Finales (Random Forest GridSearch)

**Test Set (20%)**:
- **MAE**: 1.070s (erreur moyenne)
- **RMSE**: 12.56s (pénalise erreurs extrêmes)
- **R²**: 0.755 (75.5% variance expliquée)
- **MAPE**: 0.86%

**Cross-Validation**:
- **CV MAE**: 1.016s ± 0.035s
- **CV R²**: 0.800 ± 0.023

### Analyse des Résultats

**Points Forts**:
- Excellente généralisation (CV ≈ Test)
- R² 0.755 avec données publiques uniquement
- Prédiction à ±1.07s du temps réel
- Pas de data leakage (split stratifié)

**Limitations**:
- Erreur plus élevée que les écuries F1 (MAE ~0.1-0.2s avec télémétrie)
- RMSE élevé dû aux outliers (tours lents: SC, VSC, problèmes)

---

## Utilisation des Modèles

### 1. Charger le Meilleur Modèle (Automatique)

```python
from ml.load_model_simple import load_model_from_mlflow

# Stratégie "mae" : sélectionne le meilleur MAE absolu
model, info = load_model_from_mlflow(strategy='mae', model_family=None)

print(f"Model: {info.get('model_family')}")
print(f"Test MAE: {info['test_mae']:.3f}s")

# Prédiction
import pandas as pd
X_new = pd.DataFrame([...])  # Vos features
predictions = model.predict(X_new)
```

### 2. Stratégies de Chargement

**Stratégie "mae"** (recommandé pour API):
- Sélectionne le modèle avec le meilleur Test MAE absolu
- Actuellement: Random Forest GridSearch (1.070s)

**Stratégie "robust"**:
- Sélectionne le modèle avec le meilleur compromis robustesse/performance
- Critères: Overfitting < 5.0
- Tri par overfitting croissant

```python
# Performance absolue (API default)
model, info = load_model_from_mlflow(strategy='mae', model_family=None)

# Spécifier une famille
model, info = load_model_from_mlflow(strategy='mae', model_family='random_forest')
```

### 3. Chargement d'un Run Spécifique

```python
# Pour reproductibilité exacte
run_id = "1b311597c5e94874a616a71cf9d10e5d"
model, info = load_model_from_mlflow(run_id=run_id)
```

### 4. Fallback Local (Sans MLflow)

```python
from ml.load_model_simple import load_model_local

# Si MLflow indisponible
model, info = load_model_local(model_family='random_forest')
```

---

## MLflow

### Configuration

MLflow est configuré avec Docker pour:
- **Backend store**: SQLite (`./mlflow_db/mlflow.db`) - Persistant
- **Artifact store**: File system (`./mlartifacts/`) - Persistant
- **Tracking URI**: http://localhost:5000

### Artifacts Loggés

Pour chaque run:
- `model/model_artifact.pkl`: Modèle sérialisé
- `feature_importance.csv`: Importance des features
- `feature_importance.png`: Graphique importance
- `predictions_vs_actual.png`: Scatter plot prédictions
- `residuals_distribution.png`: Distribution des résidus
- `training_report.json`: Rapport complet (métriques, params)

### Métriques Trackées

- `test_mae`, `test_rmse`, `test_r2`, `test_mape`: Métriques test
- `train_mae`, `train_rmse`, `train_r2`: Métriques train
- `cv_mae`, `cv_r2`: Cross-validation (moyenne ± std)
- `overfitting_ratio`: test_mae / train_mae

### Interface Web

```bash
# Accéder à l'interface
open http://localhost:5000

# Voir les runs
# → Experiments → F1PA_LapTime_Prediction
```

---

## Améliorations Futures

### Court Terme

1. **Données Supplémentaires**
   - SafetyCar/VSC (interruptions)
   - Compound pneus (Soft/Medium/Hard)
   - Position qualification

2. **Feature Engineering**
   - Moving average sur derniers tours
   - Interaction features (temp × rhum)

3. **Ensembling**
   - Stacking XGBoost + Random Forest
   - Voting regressor

### Long Terme

1. **Télémétrie** (données privées)
   - Setup aérodynamique
   - Pression pneus, température freins

2. **Deep Learning**
   - LSTM pour séries temporelles
   - Transformer avec attention

---

## Résumé

**F1PA ML Pipeline** - Prédiction de performance des temps au tour F1:

- **Dataset**: 71,645 tours (2023-2025), split stratifié 80/20
- **Features**: 15 features (performance encodings, vitesses, météo)
- **Meilleur Modèle**: Random Forest GridSearch
- **Performance**: MAE 1.070s, R² 0.755
- **Tracking**: MLflow (http://localhost:5000)
- **API**: Auto-sélection du meilleur modèle
