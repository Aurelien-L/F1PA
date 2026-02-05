"""
F1PA - Preprocessing Pipeline

Handles:
1. Missing value imputation (intelligent strategy by feature type)
2. Feature engineering (creation of PREDICTIVE derived features)
3. Categorical variable encoding (Target Encoding)
4. Temporal train/test split preparation

IMPORTANT: This pipeline prepares data for a PERFORMANCE prediction model,
NOT a final time calculation model.

Sector times (duration_sector_*) are EXCLUDED as they represent
current lap data, not predictors before the lap.

Detailed technical justifications in ml/README.md
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
    Load ML dataset.

    Returns:
        DataFrame with 71,645 laps × 31 columns
    """
    log(f"Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    log(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Intelligent missing value imputation.

    STRATEGY:
    1. Sport features (speeds, sectors):
       - Imputation by group MEDIAN (circuit_key, driver_number)
       - Rationale: A driver on a given circuit has stable characteristics
       - Fallback: Global median if group too small

    2. Weather features (prcp, cldc):
       - Temporal forward fill (by session_key + lap_number)
       - Rationale: Weather changes gradually within a session
       - Fallback: Global median

    3. Others: Global median

    JUSTIFICATION vs other methods:
    - Row deletion: Loss of 16% of data (12,000 laps)
    - Feature deletion: Loss of strong predictors (speeds)
    - Simple global imputation: Ignores circuit/driver context
    - Group imputation: Preserves real patterns
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
            # Forward fill by session (weather conditions persist)
            df = df.sort_values(['session_key', 'lap_number'])
            df[feat] = df.groupby('session_key')[feat].ffill()
            # Fallback: global median
            df[feat].fillna(df[feat].median(), inplace=True)
            log(f"  {feat}: Forward-filled by session + global median")

    # 3. Other numeric features: global median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    final_nulls = df.isnull().sum().sum()
    log(f"Remaining missing values: {final_nulls}")

    return df


def create_derived_features(df: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    """
    Creation of PREDICTIVE derived features.

    IMPORTANT: These features must be calculable BEFORE the lap,
    so NOT based on sector times of the current lap.

    CREATED FEATURES:

    1. avg_speed: Historical average speed (st_speed + i1_speed + i2_speed) / 3
       - Global performance indicator

    2. lap_progress: lap_number / max_lap_number (per session)
       - Progress in session (0-1)
       - Captures tire degradation + evolving conditions

    3. driver_perf_score: Driver performance score
       - Based on difference between driver time and circuit average time
       - Calculated on TRAIN SET only (no data leakage)
       - Negative score = driver faster than average

    Args:
        df: DataFrame with raw data
        train_mask: Boolean mask for train set (avoid data leakage)

    EXCLUDED (current lap data):
    - total_sector_time (almost directly lap_duration)
    - sector_1_ratio, sector_2_ratio (based on lap sectors)
    - weather_severity (low impact per feature importance)
    """
    df = df.copy()
    log("Creating derived features (predictive only)...")

    # 1. Average speed (performance indicator, not direct time)
    df['avg_speed'] = (df['st_speed'] + df['i1_speed'] + df['i2_speed']) / 3
    log("  avg_speed: Average of 3 speed measurements")

    # 2. Circuit-based lap progress (0-1 scale)
    # Uses typical max_lap per circuit (average across sessions)
    # This matches inference behavior (circuit-based, not session-specific)
    circuit_max_laps = df.groupby(['circuit_key', 'session_key'])['lap_number'].max().groupby('circuit_key').mean()
    df['lap_progress'] = df.apply(
        lambda row: row['lap_number'] / circuit_max_laps.get(row['circuit_key'], 70),
        axis=1
    ).clip(upper=1.0)
    log("  lap_progress: Circuit-based position (typical max_lap per circuit)")

    # 3. Driver Performance Score
    # Calculated as: driver average time - circuit average time
    # Negative score = driver faster than average
    # Calculated on TRAIN SET only to avoid data leakage

    # First calculate circuit means (on train)
    circuit_means = df.loc[train_mask].groupby('circuit_key')['lap_duration'].mean()

    # Then calculate driver-circuit means (on train)
    driver_circuit_means = df.loc[train_mask].groupby(
        ['driver_number', 'circuit_key']
    )['lap_duration'].mean()

    # Create performance score
    def calc_driver_perf(row):
        driver = row['driver_number']
        circuit = row['circuit_key']
        circuit_avg = circuit_means.get(circuit, df.loc[train_mask, 'lap_duration'].mean())

        if (driver, circuit) in driver_circuit_means.index:
            driver_avg = driver_circuit_means[(driver, circuit)]
        else:
            # Unknown driver on this circuit: use global average
            driver_global = df.loc[
                train_mask & (df['driver_number'] == driver), 'lap_duration'
            ].mean()
            if pd.isna(driver_global):
                driver_avg = circuit_avg  # Fallback: neutral
            else:
                driver_avg = driver_global

        return driver_avg - circuit_avg

    df['driver_perf_score'] = df.apply(calc_driver_perf, axis=1)
    log("  driver_perf_score: Driver performance vs circuit average (negative = faster)")

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

    # Garder ausif les colonnes catégorielles originales for XGBoost
    # (XGBoost peut utiliser enable_categorical=True)
    # On aura donc: circuit_key (catégoriel) ET circuit_avg_laptime (numérique)
    for col in categorical_cols:
        df[col] = df[col].astype('category')

    return df


def prepare_train_test_split_temporal(
    df: pd.DataFrame,
    train_years: list[int],
    test_year: int,
    target_col: str = 'lap_duration'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split temporel Train/Test (legacy).

    Train: 2023-2024 | Test: 2025
    """
    log(f"Splitting TEMPORAL: train ({train_years}) / test ({test_year})...")

    train_mask = df['year'].isin(train_years)
    test_mask = df['year'] == test_year

    from ml.config import EXCLUDE_FEATURES
    feature_cols = [c for c in df.columns if c not in EXCLUDE_FEATURES]

    X_train = df.loc[train_mask, feature_cols]
    X_test = df.loc[test_mask, feature_cols]
    y_train = df.loc[train_mask, target_col]
    y_test = df.loc[test_mask, target_col]

    log(f"Train: {len(X_train):,} samples ({len(X_train)/len(df)*100:.1f}%)")
    log(f"Test:  {len(X_test):,} samples ({len(X_test)/len(df)*100:.1f}%)")
    log(f"Features: {len(feature_cols)}")

    return X_train, X_test, y_train, y_test


def prepare_train_test_split_stratified(
    df: pd.DataFrame,
    test_size: float = 0.2,
    stratify_by: str = 'circuit_key',
    target_col: str = 'lap_duration',
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split stratifié 80/20 incluant toutes les années.

    AVANTAGES:
    - Inclut des données 2025 dans le train → réduit le concept drift
    - Distribution équilibrée des circuits dans train et test
    - Meilleure généralisation

    INCONVÉNIENT:
    - Légère fuite temporelle (acceptable pour ce cas d'usage)

    Args:
        df: DataFrame preprocessé
        test_size: Proportion du test set (0.2 = 20%)
        stratify_by: Colonne pour stratification ('circuit_key')
        target_col: 'lap_duration'
        random_state: Seed pour reproductibilité

    Returns:
        X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split
    from ml.config import EXCLUDE_FEATURES

    log(f"Splitting STRATIFIED: {(1-test_size)*100:.0f}% train / {test_size*100:.0f}% test")
    log(f"Stratified by: {stratify_by}")

    feature_cols = [c for c in df.columns if c not in EXCLUDE_FEATURES]

    X = df[feature_cols]
    y = df[target_col]

    # Stratification par circuit pour assurer une bonne distribution
    stratify_col = df[stratify_by]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=stratify_col,
        random_state=random_state
    )

    log(f"Train: {len(X_train):,} samples ({len(X_train)/len(df)*100:.1f}%)")
    log(f"Test:  {len(X_test):,} samples ({len(X_test)/len(df)*100:.1f}%)")
    log(f"Features: {len(feature_cols)}")

    # Checkr la distribution des années
    train_years = df.loc[X_train.index, 'year'].value_counts().sort_index()
    test_years = df.loc[X_test.index, 'year'].value_counts().sort_index()
    log(f"Train years distribution: {train_years.to_dict()}")
    log(f"Test years distribution: {test_years.to_dict()}")

    return X_train, X_test, y_train, y_test


def preprocess_pipeline(dataset_path: Path, train_years: list[int] = None, test_year: int = None):
    """
    Pipeline preprocessing complet pour modèle de PRÉDICTION de performance.

    Étapes:
    1. Chargement dataset
    2. Imputation valeurs manquantes
    3. Target encoding catégorielles (circuit_avg_laptime, driver_avg_laptime)
    4. Création features dérivées (driver_perf_score, etc.)
    5. Split train/test (stratifié ou temporel selon config)

    IMPORTANT: Les temps secteurs sont EXCLUS car ce sont des données
    du tour en cours, pas des prédicteurs.

    Returns:
        X_train, X_test, y_train, y_test, df_preprocessed
    """
    from ml.config import SPLIT_STRATEGY, TEST_SIZE, STRATIFY_BY, TRAIN_YEARS, TEST_YEAR, RANDOM_STATE

    # Valeurs par défaut from config
    if train_years is None:
        train_years = TRAIN_YEARS
    if test_year is None:
        test_year = TEST_YEAR

    log("=" * 80)
    log("PREPROCESSING PIPELINE (Performance Prediction Model)")
    log(f"Split strategy: {SPLIT_STRATEGY}")
    log("=" * 80)

    # 1. Load
    df = load_dataset(dataset_path)

    # 2. Handle missing values
    df = handle_missing_values(df)

    # 3. Définir le masque train for les encodages
    # Pour le split stratifié, on utilise 80% des data for l'encodage
    if SPLIT_STRATEGY == "stratified":
        # Pour l'encodage, on utilise un sample aléatoire de 80%
        from sklearn.model_selection import train_test_split
        train_idx, _ = train_test_split(
            df.index, test_size=TEST_SIZE,
            stratify=df[STRATIFY_BY],
            random_state=RANDOM_STATE
        )
        train_mask = df.index.isin(train_idx)
    else:
        train_mask = df['year'].isin(train_years)

    # 4. Target encoding (calcule circuit_avg_laptime, driver_avg_laptime)
    df = target_encode_categorical(
        df, train_mask,
        categorical_cols=['circuit_key', 'driver_number', 'year'],
        target_col='lap_duration'
    )

    # 5. Create derived features (utilise train_mask for éviter leakage)
    df = create_derived_features(df, train_mask)

    # 6. Split train/test selon la stratégie
    if SPLIT_STRATEGY == "stratified":
        X_train, X_test, y_train, y_test = prepare_train_test_split_stratified(
            df, test_size=TEST_SIZE, stratify_by=STRATIFY_BY, random_state=RANDOM_STATE
        )
    else:
        X_train, X_test, y_train, y_test = prepare_train_test_split_temporal(
            df, train_years, test_year
        )

    # Log features finales
    log(f"Final features ({len(X_train.columns)}): {list(X_train.columns)}")

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
