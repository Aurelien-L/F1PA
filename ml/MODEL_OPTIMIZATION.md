# Optimisation du Modèle ML - F1PA

Documentation détaillée du processus d'optimisation du modèle Random Forest pour la prédiction des temps au tour.

---

## 🎯 Objectif

Développer un modèle Random Forest performant tout en respectant les contraintes de production :
- Taille modèle < 500 MB (pour upload MLflow)
- Temps de chargement API < 5s
- Performance prédictive optimale (MAE, R²)

---

## 📊 Parcours d'Optimisation

### Défi Initial

**Problème** : Modèle Random Forest trop volumineux (1.5 GB) causant des échecs d'upload vers MLflow.

### Itérations

| Version | Problème | Solution appliquée | Résultat |
|---------|----------|-------------------|----------|
| **v0** (baseline) | `max_depth=None` → 1.5 GB, crash MLflow | - | ❌ Bloquant production |
| **v1** | Profondeur limitée mais toujours lourd | `max_depth=[15,20]` au lieu de `[15,None]` | 674 MB, ⚠️ encore lourd |
| **v2** | Besoin d'un modèle plus léger | `n_estimators=[150,200]` au lieu de `[200,300]` | 449 MB, ⚠️ acceptable |
| **v3** | Cohérence train/inference + optimisation | lap_progress circuit-based + `n_estimators=150` | 351 MB, MAE 1.08s, R² 0.77 |
| **v6** (final) | Redondance feature driver_avg_laptime | Suppression driver_avg_laptime (14 features) | ✅ **335 MB** (-78%), R² **0.79** |

---

## 📈 Résultats Finaux

### Gains Mesurés

- **Réduction taille** : 1.5 GB → 335 MB (-78%)
- **Amélioration performance** : R² 0.77 → 0.79 (+2.6%, meilleure généralisation)
- **Réduction features** : 15 → 14 (suppression redondance driver_avg_laptime)
- **Temps de chargement** : 19s → ~3s dans l'API (-84%)

### Métriques du Modèle v6

- **MAE** : 1.08s (test)
- **R²** : 0.79 (test)
- **RMSE** : 11.56s (test)
- **MAPE** : 0.90%
- **Taille** : 335 MB
- **Features** : 14

---

## 🔍 Apprentissages Clés

### 1. Profondeur des Arbres

**Constat** : Profondeur illimitée (`max_depth=None`) crée un overfitting massif avec 300 arbres.

**Solution** : GridSearch a sélectionné `max_depth=20` comme optimal (équilibre précision/généralisation).

### 2. Nombre d'Estimateurs

**Constat** : 300 arbres = modèle trop lourd sans gain significatif de performance.

**Solution** : Réduire `n_estimators` à 150 offre le meilleur compromis taille/performance.

### 3. Cohérence Train/Inference

**Problème initial** : Utilisation d'un `max_lap=70` fixe pour tous les circuits (Monaco=78 laps, Spa=44 laps).

**Solution implémentée** : Calcul dynamique basé sur le max_lap typique du circuit
- Calcul du **max_lap typique** par circuit (moyenne des max_laps historiques)
- **Training** : `lap_progress = lap_number / avg(max_lap) par circuit`
- **Inference** : Même logique via requête DB avec cache
- Requête : `SELECT AVG(MAX(lap_number)) FROM fact_laps WHERE circuit_key = ? GROUP BY session_key`

**Résultat** : Cohérence training/inference pour prédictions hypothétiques.

### 4. Redondance des Features

**Constat** : `driver_avg_laptime` et `driver_perf_score` sont redondants.
- `driver_perf_score` = `driver_avg_laptime` - `circuit_avg_laptime`
- Les deux features encodent la même information relative

**Solution** : Suppression de `driver_avg_laptime`, conservation de `driver_perf_score` uniquement.

**Impact** :
- Réduction overfitting → +2.6% R² (0.77 → 0.79)
- Modèle plus léger (351 MB → 335 MB)
- Prédictions plus logiques (Verstappen > Stroll)

---

## ⚙️ Configuration de Production

### Hyperparamètres GridSearch

```python
# ml/config.py - Paramètres GridSearch Random Forest
'n_estimators': [150, 200],      # Modèle plus léger (cible ~450 MB)
'max_depth': [15, 20],            # Évite l'overfitting (était [15, None])
'min_samples_leaf': [1, 2],       # Paramètres standard
'min_samples_split': [2, 5],      # Paramètres standard
'max_features': [0.7, 0.9],       # Feature sampling
```

### Paramètres Sélectionnés (v6)

```python
{
    'n_estimators': 150,
    'max_depth': 20,
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'max_features': 0.7
}
```

---

## 📦 Features Finales (14 total)

### Contexte (4)
- `year` : Année de la session
- `circuit_key` : Identifiant du circuit
- `driver_number` : Numéro du pilote
- `lap_number` : Numéro du tour dans la session

### Vitesses (3)
- `st_speed` : Vitesse au speed trap (km/h)
- `i1_speed` : Vitesse intermédiaire 1 (km/h)
- `i2_speed` : Vitesse intermédiaire 2 (km/h)

### Météo (3)
- `temp` : Température (°C)
- `rhum` : Humidité relative (%)
- `pres` : Pression atmosphérique (hPa)

### Performance (4)
- `circuit_avg_laptime` : Temps moyen du circuit (s)
- `avg_speed` : Vitesse moyenne calculée (km/h)
- `lap_progress` : Progression dans la session (0-1)
- `driver_perf_score` : Score de performance pilote (négatif = plus rapide)

**Feature supprimée** : `driver_avg_laptime` (redondance avec driver_perf_score)

---

## 🔄 Processus de Réentraînement

### Quand réentraîner ?

1. **Nouveaux données disponibles** : Nouvelle saison F1
2. **Drift détecté** : Rapport Evidently signale un drift significatif
3. **Performance dégradée** : MAE > 1.5s sur données récentes
4. **Changements réglementaires** : Nouveaux règlements F1 impactant les performances

### Commande

```bash
python ml/run_ml_pipeline.py
```

### Validation

Vérifier que le nouveau modèle :
- MAE < 1.2s
- R² > 0.75
- Taille < 500 MB
- Prédictions logiques (top pilotes > pilotes moyens)

---

## 📚 Références

- MLflow Run ID v6 : `3261c2b8d2f440848ca459cd35e67e14`
- Rapport d'entraînement : `reports/random_forest_gridsearch/training_report.json`
- Tracking MLflow : http://localhost:5000
