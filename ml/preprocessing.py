"""
F1PA - Preprocessing Pipeline

Gère:
1. Imputation des valeurs manquantes (stratégie intelligente par type de feature)
2. Feature engineering (création de 6 features dérivées)
3. Encodage des variables catégorielles (Target Encoding)
4. Préparation train/test split temporel

Justifications techniques détaillées dans ml/README.md
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import warnings

warnings.filterwarnings('ignore')


def log(msg: str) -> None:
    """Simple logging."""
    print(f"[preprocessing] {msg}")


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """
    Charge le dataset ML.

    Returns:
        DataFrame avec 71,645 laps × 31 colonnes
    """
    log(f"Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    log(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputation intelligente des valeurs manquantes.

    STRATÉGIE:
    1. Sport features (vitesses, secteurs):
       - Imputation par MÉDIANE du groupe (circuit_key, driver_number)
       - Rationale: Un pilote sur un circuit donné a des caractéristiques stables
       - Fallback: Médiane globale si groupe trop petit

    2. Weather features (prcp, cldc):
       - Forward fill temporel (par session_key + lap_number)
       - Rationale: La météo change graduellement dans une session
       - Fallback: Médiane globale

    3. Autres: Médiane globale

    JUSTIFICATION vs autres méthodes:
    - ❌ Suppression lignes → Perte de 16% des données (12,000 laps)
    - ❌ Suppression features → Perte de prédicteurs forts (vitesses)
    - ❌ Imputation globale simple → Ignore le contexte circuit/pilote
    - ✅ Imputation par groupe → Conserve les patterns réels
    """
    df = df.copy()
    initial_nulls = df.isnull().sum().sum()
    log(f"Initial missing values: {initial_nulls:,}")

    # 1. Sport features: Imputation par groupe (circuit, driver)
    sport_features = ['st_speed', 'i1_speed', 'i2_speed',
                      'duration_sector_1', 'duration_sector_2', 'duration_sector_3']

    for feat in sport_features:
        if df[feat].isnull().any():
            # Médiane par groupe
            df[feat] = df.groupby(['circuit_key', 'driver_number'])[feat].transform(
                lambda x: x.fillna(x.median())
            )
            # Fallback: médiane globale
            df[feat].fillna(df[feat].median(), inplace=True)
            log(f"  {feat}: Imputed by (circuit, driver) group")

    # 2. Weather features: Forward fill temporel + fallback
    weather_features = ['temp', 'rhum', 'pres', 'wspd', 'wdir', 'prcp', 'cldc']

    for feat in weather_features:
        if df[feat].isnull().any():
            # Forward fill par session (les conditions météo persistent)
            df = df.sort_values(['session_key', 'lap_number'])
            df[feat] = df.groupby('session_key')[feat].ffill()
            # Fallback: médiane globale
            df[feat].fillna(df[feat].median(), inplace=True)
            log(f"  {feat}: Forward-filled by session + global median")

    # 3. Autres features numériques: médiane globale
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    final_nulls = df.isnull().sum().sum()
    log(f"Remaining missing values: {final_nulls}")

    return df


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Création de 6 features dérivées.

    FEATURES CRÉÉES:

    1. avg_speed: Vitesse moyenne (st_speed + i1_speed + i2_speed) / 3
       → Mesure globale de performance vitesse

    2. total_sector_time: duration_sector_1 + duration_sector_2 + duration_sector_3
       → Approximation de lap_duration (très corrélé)
       → Utile pour les arbres de décision (split direct)

    3. sector_1_ratio: duration_sector_1 / total_sector_time
    4. sector_2_ratio: duration_sector_2 / total_sector_time
       → Capture le STYLE de pilotage:
         - Ratio S1 élevé → Pilote agressif en début de tour
         - Ratio S3 élevé → Pilote conservateur (économie pneus)
       → Permet de différencier les pilotes au-delà de leur vitesse brute

    5. weather_severity: (wspd / 41) + (prcp / 1.8)
       → Score composite météo difficile (0-2)
       → Combine vent fort + pluie
       → Normalisé par max observé (wspd_max=41, prcp_max=1.8)

    6. lap_progress: lap_number / max_lap_number (par session)
       → Progression dans la course (0-1)
       → Capture dégradation pneus + fatigue pilote
       → Temps augmente typiquement en fin de course

    JUSTIFICATION:
    - Ces features sont difficiles à capturer par le modèle brut
    - Elles encodent de la CONNAISSANCE MÉTIER (F1)
    - Impact attendu: +5-10% R² par rapport à features brutes seules
    """
    df = df.copy()
    log("Creating 6 derived features...")

    # 1. Vitesse moyenne
    df['avg_speed'] = (df['st_speed'] + df['i1_speed'] + df['i2_speed']) / 3

    # 2. Temps secteurs total
    df['total_sector_time'] = (
        df['duration_sector_1'] +
        df['duration_sector_2'] +
        df['duration_sector_3']
    )

    # 3-4. Ratios secteurs (style pilotage)
    df['sector_1_ratio'] = df['duration_sector_1'] / df['total_sector_time']
    df['sector_2_ratio'] = df['duration_sector_2'] / df['total_sector_time']
    # Note: sector_3_ratio = 1 - sector_1_ratio - sector_2_ratio (redondant)

    # 5. Météo composite (0-2 scale)
    # Normalisation par max observé
    df['weather_severity'] = (df['wspd'] / 41.0) + (df['prcp'] / 1.8)

    # 6. Progression course (0-1 scale)
    df['lap_progress'] = df.groupby('session_key')['lap_number'].transform(
        lambda x: x / x.max()
    )

    log("  avg_speed: Average of 3 speed measurements")
    log("  total_sector_time: Sum of 3 sector durations")
    log("  sector_1_ratio, sector_2_ratio: Driving style indicators")
    log("  weather_severity: Composite wind + rain difficulty")
    log("  lap_progress: Position in race (tire degradation)")

    return df


def target_encode_categorical(
    df: pd.DataFrame,
    train_mask: pd.Series,
    categorical_cols: list[str],
    target_col: str = 'lap_duration'
) -> pd.DataFrame:
    """
    Target Encoding des variables catégorielles.

    PRINCIPE:
    - Remplacer circuit_key par lap_duration MOYEN sur ce circuit
    - Remplacer driver_number par lap_duration MOYEN pour ce pilote
    - Encoder l'information de performance directement dans la feature

    JUSTIFICATION vs One-Hot Encoding:

    1. Dimensionnalité:
       - One-Hot: 24 circuits + 32 drivers = 56 colonnes binaires
       - Target Encoding: 2 colonnes numériques
       → Gain: 96% de colonnes en moins

    2. Performance arbres:
       - Arbres de décision (XGBoost, RF) gèrent mal le one-hot
       - Ils doivent apprendre: "circuit_1 OU circuit_5 OU circuit_9 → lap lent"
       - Avec target encoding: "circuit_avg > 95s → lap lent" (split direct)

    3. Généralisation:
       - One-Hot: Chaque circuit/pilote est indépendant
       - Target Encoding: Encode la DIFFICULTÉ du circuit et SKILL du pilote
       → Meilleure généralisation sur nouveaux circuits similaires

    4. Éviter le data leakage:
       - ⚠️ CRUCIAL: Encoder uniquement sur train set
       - Test set utilise les moyennes du train
       → Pas de fuite d'information futur → passé

    LIMITATION:
    - Sensible au overfitting si peu de samples par catégorie
    - Solution: Smoothing (non implémenté ici, pas nécessaire avec 71k samples)

    Args:
        df: DataFrame complet
        train_mask: Masque booléen indiquant les lignes de train
        categorical_cols: ['circuit_key', 'driver_number', 'year']
        target_col: 'lap_duration'

    Returns:
        DataFrame avec colonnes encodées: circuit_avg_laptime, driver_avg_laptime, year_avg_laptime
    """
    df = df.copy()
    log(f"Target encoding {len(categorical_cols)} categorical features...")

    for col in categorical_cols:
        # Calculer moyenne target PAR CATÉGORIE sur train set uniquement
        train_means = df.loc[train_mask].groupby(col)[target_col].mean()

        # Nom de la nouvelle colonne
        new_col = f"{col.replace('_key', '').replace('_number', '')}_avg_laptime"

        # Mapper sur tout le dataset (train + test)
        df[new_col] = df[col].map(train_means)

        # Fallback: Si catégorie inconnue (ne devrait pas arriver), utiliser moyenne globale train
        global_mean = df.loc[train_mask, target_col].mean()
        df[new_col].fillna(global_mean, inplace=True)

        log(f"  {col} -> {new_col} (mean lap_duration per category)")

    # Garder aussi les colonnes catégorielles originales pour XGBoost
    # (XGBoost peut utiliser enable_categorical=True)
    # On aura donc: circuit_key (catégoriel) ET circuit_avg_laptime (numérique)
    for col in categorical_cols:
        df[col] = df[col].astype('category')

    return df


def prepare_train_test_split(
    df: pd.DataFrame,
    train_years: list[int],
    test_year: int,
    target_col: str = 'lap_duration'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split temporel Train/Test.

    JUSTIFICATION vs split aléatoire:

    ❌ Split aléatoire (shuffle=True):
    - Mélange 2023, 2024, 2025
    - Le modèle "voit le futur" pendant l'entraînement
    - DATA LEAKAGE: Lap de 2025 dans train → Test sur 2025 biaisé
    - Métriques optimistes mais modèle inutilisable en production

    ✅ Split temporel (année):
    - Train: 2023-2024 (passé)
    - Test: 2025 (futur)
    - Simule la PRODUCTION: Prédire les courses de 2026 avec data 2023-2025
    - Métriques réalistes
    - Détecte le concept drift (changements réglementaires F1)

    DISTRIBUTION:
    - Train: 47,266 laps (66%) → 2023 + 2024
    - Test: 24,379 laps (34%) → 2025
    - Ratio 66/34 acceptable (généralement 70/30 ou 80/20)

    Args:
        df: DataFrame preprocessé
        train_years: [2023, 2024]
        test_year: 2025
        target_col: 'lap_duration'

    Returns:
        X_train, X_test, y_train, y_test
    """
    log(f"Splitting train ({train_years}) / test ({test_year})...")

    # Masque train/test
    train_mask = df['year'].isin(train_years)
    test_mask = df['year'] == test_year

    # Colonnes features (exclure target + metadata)
    from ml.config import EXCLUDE_FEATURES
    feature_cols = [c for c in df.columns if c not in EXCLUDE_FEATURES]

    # Split
    X_train = df.loc[train_mask, feature_cols]
    X_test = df.loc[test_mask, feature_cols]
    y_train = df.loc[train_mask, target_col]
    y_test = df.loc[test_mask, target_col]

    log(f"Train: {len(X_train):,} samples ({len(X_train)/len(df)*100:.1f}%)")
    log(f"Test:  {len(X_test):,} samples ({len(X_test)/len(df)*100:.1f}%)")
    log(f"Features: {len(feature_cols)}")

    return X_train, X_test, y_train, y_test


def preprocess_pipeline(dataset_path: Path, train_years: list[int], test_year: int):
    """
    Pipeline preprocessing complet.

    Étapes:
    1. Chargement dataset
    2. Imputation valeurs manquantes
    3. Création features dérivées
    4. Target encoding catégorielles
    5. Split train/test temporel

    Returns:
        X_train, X_test, y_train, y_test, df_preprocessed
    """
    log("=" * 80)
    log("PREPROCESSING PIPELINE")
    log("=" * 80)

    # 1. Load
    df = load_dataset(dataset_path)

    # 2. Handle missing values
    df = handle_missing_values(df)

    # 3. Create derived features
    df = create_derived_features(df)

    # 4. Target encoding (nécessite de connaître train/test split)
    train_mask = df['year'].isin(train_years)
    df = target_encode_categorical(
        df, train_mask,
        categorical_cols=['circuit_key', 'driver_number', 'year'],
        target_col='lap_duration'
    )

    # 5. Split train/test
    X_train, X_test, y_train, y_test = prepare_train_test_split(
        df, train_years, test_year
    )

    log("=" * 80)
    log("PREPROCESSING COMPLETE")
    log("=" * 80)

    return X_train, X_test, y_train, y_test, df


if __name__ == "__main__":
    # Test preprocessing
    from ml.config import DATASET_PATH, TRAIN_YEARS, TEST_YEAR

    X_train, X_test, y_train, y_test, df = preprocess_pipeline(
        DATASET_PATH, TRAIN_YEARS, TEST_YEAR
    )

    print("\n=== PREPROCESSING SUMMARY ===")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"\nFeatures: {list(X_train.columns)}")
