"""
F1PA - Model Training with GridSearchCV + MLflow Tracking

Pipeline d'entraînement avec:
1. Baseline models (hyperparamètres par défaut)
2. GridSearchCV pour tuning léger
3. Comparaison XGBoost vs Random Forest
4. Tracking MLflow complet

MLflow logs pour chaque run:
- Hyperparamètres
- Métriques (MAE, RMSE, R², MAPE) sur train/val/test
- Feature importance
- Modèle sérialisé
- Metadata
"""
from __future__ import annotations

# IMPORTANT: Set matplotlib backend BEFORE importing pyplot
# This prevents tkinter errors when running with parallel jobs (n_jobs=-1) on Windows
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend, no GUI required

import time
import json
import pickle
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

import mlflow
import mlflow.sklearn
import mlflow.xgboost

from ml.config import (
    BASELINE_MODELS, GRIDSEARCH_PARAMS, FIXED_PARAMS, GRIDSEARCH_CV_FOLDS, GRIDSEARCH_SCORING,
    CV_FOLDS, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI,
    MODELS_DIR, REPORTS_DIR, RANDOM_STATE
)
from ml.preprocessing import preprocess_pipeline


def log(msg: str) -> None:
    """Simple logging with timestamp."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcule les 4 métriques principales."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }


def cross_validate_model(model, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> dict:
    """Cross-validation K-fold."""
    log(f"Running {cv_folds}-fold cross-validation...")

    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    mae_scores = -cross_val_score(model, X, y, cv=kfold,
                                   scoring='neg_mean_absolute_error', n_jobs=-1)
    rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=kfold,
                                            scoring='neg_mean_squared_error', n_jobs=-1))
    r2_scores = cross_val_score(model, X, y, cv=kfold,
                                 scoring='r2', n_jobs=-1)

    cv_metrics = {
        'cv_mae_mean': mae_scores.mean(),
        'cv_mae_std': mae_scores.std(),
        'cv_rmse_mean': rmse_scores.mean(),
        'cv_rmse_std': rmse_scores.std(),
        'cv_r2_mean': r2_scores.mean(),
        'cv_r2_std': r2_scores.std()
    }

    log(f"  CV MAE: {cv_metrics['cv_mae_mean']:.3f} +/- {cv_metrics['cv_mae_std']:.3f}s")
    log(f"  CV R2:  {cv_metrics['cv_r2_mean']:.3f} +/- {cv_metrics['cv_r2_std']:.3f}")

    return cv_metrics


def plot_feature_importance(model, feature_names: list[str], model_name: str,
                             save_path: Path) -> pd.DataFrame:
    """Génère graphique feature importance."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        log(f"  Warning: {model_name} has no feature_importances_ attribute")
        return None

    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=feat_imp.head(20), x='importance', y='feature', palette='viridis')
    plt.title(f'{model_name} - Feature Importance (Top 20)', fontsize=14, fontweight='bold')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    log(f"  Feature importance plot saved: {save_path}")

    csv_path = save_path.with_suffix('.csv')
    feat_imp.to_csv(csv_path, index=False)
    log(f"  Feature importance CSV saved: {csv_path}")

    return feat_imp


def run_gridsearch(model_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    """
    Execute GridSearchCV pour tuning léger.

    Returns:
        (best_model, best_params, grid_results)
    """
    log("=" * 80)
    log(f"GRIDSEARCH - {model_name.upper()}")
    log("=" * 80)

    # Instancier modèle base
    if model_name == 'xgboost':
        base_model = XGBRegressor(**FIXED_PARAMS[model_name])
    elif model_name == 'random_forest':
        base_model = RandomForestRegressor(**FIXED_PARAMS[model_name])
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # GridSearch
    param_grid = GRIDSEARCH_PARAMS[model_name]

    log(f"Parameter grid: {param_grid}")
    log(f"Total combinations: {np.prod([len(v) for v in param_grid.values()])}")
    log(f"CV folds: {GRIDSEARCH_CV_FOLDS}")

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=GRIDSEARCH_CV_FOLDS,
        scoring=GRIDSEARCH_SCORING,
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )

    start_time = time.time()
    grid_search.fit(X_train, y_train)
    elapsed = time.time() - start_time

    log(f"GridSearch completed in {elapsed:.1f}s")
    log(f"Best params: {grid_search.best_params_}")
    log(f"Best CV MAE: {-grid_search.best_score_:.3f}s")

    # Extraire résultats
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df['mean_test_mae'] = -results_df['mean_test_score']
    results_df['mean_train_mae'] = -results_df['mean_train_score']

    return grid_search.best_estimator_, grid_search.best_params_, results_df


def train_model_with_gridsearch(
    model_name: str,
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
    use_gridsearch: bool = True
) -> dict:
    """
    Entraîne un modèle (avec ou sans GridSearch) + tracking MLflow.

    Pipeline:
    1. GridSearchCV (si use_gridsearch=True) ou baseline params
    2. Entraînement sur train complet avec meilleurs params
    3. Évaluation sur test
    4. Feature importance
    5. Logging MLflow complet
    """
    run_name = f"{model_name}_gridsearch" if use_gridsearch else f"{model_name}_baseline"

    log("=" * 80)
    log(f"TRAINING {run_name.upper()}")
    log("=" * 80)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        log(f"MLflow run ID: {run_id}")

        # Set description
        description = f"""
F1 Lap Time PERFORMANCE Prediction - {model_name.upper()} {'with GridSearchCV' if use_gridsearch else 'Baseline'}

OBJECTIF: Prédire le temps au tour d'un pilote AVANT qu'il roule
(basé sur performance historique + conditions, PAS sur les temps secteurs)

Dataset: 71,645 laps (2023-2025)
Train: 47,266 laps (2023-2024) | Test: 24,379 laps (2025)
Features: vitesses + météo + driver_perf_score + circuit_avg_laptime

Note: Les temps secteurs (duration_sector_*) sont EXCLUS car ils représentent
des données du tour en cours, pas des prédicteurs.
        """.strip()
        mlflow.set_tag("mlflow.note.content", description)

        # Set tags for filtering
        mlflow.set_tag("model_family", model_name)
        mlflow.set_tag("tuning_method", "gridsearch" if use_gridsearch else "baseline")
        mlflow.set_tag("split_strategy", "temporal_2023-2024_train_2025_test")
        mlflow.set_tag("target_metric", "MAE")
        mlflow.set_tag("experiment_phase", "MVP")

        mlflow.log_param('model_type', model_name)
        mlflow.log_param('use_gridsearch', use_gridsearch)

        start_time = time.time()

        # 1. Obtenir modèle (GridSearch ou baseline)
        if use_gridsearch:
            model, best_params, grid_results = run_gridsearch(model_name, X_train, y_train)

            # Log meilleurs params
            for key, value in best_params.items():
                mlflow.log_param(f"best_{key}", value)

            # Log tous les params finaux (best + fixed)
            all_params = {**FIXED_PARAMS[model_name], **best_params}
            for key, value in all_params.items():
                mlflow.log_param(key, value)

            # Sauvegarder résultats GridSearch
            grid_csv_path = REPORTS_DIR / model_name / 'gridsearch_results.csv'
            grid_csv_path.parent.mkdir(parents=True, exist_ok=True)
            grid_results.to_csv(grid_csv_path, index=False)
            mlflow.log_artifact(str(grid_csv_path))
            log(f"  GridSearch results saved: {grid_csv_path}")

        else:
            # Baseline: hyperparamètres par défaut
            params = BASELINE_MODELS[model_name]

            if model_name == 'xgboost':
                model = XGBRegressor(**params)
            else:
                model = RandomForestRegressor(**params)

            # Entraîner
            log("Training baseline model on full train set...")
            model.fit(X_train, y_train)

            # Log params
            for key, value in params.items():
                mlflow.log_param(key, value)

        train_time = time.time() - start_time
        log(f"Training completed in {train_time:.1f}s")
        mlflow.log_metric('train_time_seconds', train_time)

        # 2. Cross-validation 5-fold (sur train)
        cv_metrics = cross_validate_model(model, X_train, y_train, cv_folds=CV_FOLDS)
        for key, value in cv_metrics.items():
            mlflow.log_metric(key, value)

        # 3. Prédictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # 4. Métriques train
        train_metrics = calculate_metrics(y_train, y_train_pred)
        log("Train metrics:")
        for key, value in train_metrics.items():
            log(f"  {key}: {value:.4f}")
            mlflow.log_metric(f'train_{key}', value)

        # 5. Métriques test
        test_metrics = calculate_metrics(y_test, y_test_pred)
        log("Test metrics:")
        for key, value in test_metrics.items():
            log(f"  {key}: {value:.4f}")
            mlflow.log_metric(f'test_{key}', value)

        # 5b. Métriques dérivées (overfitting, concept drift)
        overfitting_ratio = test_metrics['mae'] / train_metrics['mae']
        mlflow.log_metric('overfitting_ratio', overfitting_ratio)

        concept_drift_score = abs(cv_metrics['cv_r2_mean'] - test_metrics['r2'])
        mlflow.log_metric('concept_drift_score', concept_drift_score)

        log(f"  overfitting_ratio: {overfitting_ratio:.4f} (ideal: 1.0-1.5)")
        log(f"  concept_drift_score: {concept_drift_score:.4f} (lower is better)")

        # 6. Metadata
        mlflow.log_param('n_train_samples', len(X_train))
        mlflow.log_param('n_test_samples', len(X_test))
        mlflow.log_param('n_features', X_train.shape[1])
        mlflow.log_param('train_test_ratio', f"{len(X_train)}/{len(X_test)}")
        mlflow.log_param('feature_engineering', 'yes_6_derived_features')
        mlflow.log_param('categorical_encoding', 'target_encoding')

        # 7. Feature importance
        reports_model_dir = REPORTS_DIR / run_name
        reports_model_dir.mkdir(parents=True, exist_ok=True)

        feat_imp_path = reports_model_dir / 'feature_importance.png'
        feat_imp_df = plot_feature_importance(
            model, X_train.columns.tolist(), run_name, feat_imp_path
        )

        if feat_imp_df is not None:
            mlflow.log_artifact(str(feat_imp_path))
            mlflow.log_artifact(str(feat_imp_path.with_suffix('.csv')))

        # 7b. Prediction plots
        # Plot 1: Predictions vs Actual (Test set)
        pred_plot_path = reports_model_dir / 'predictions_vs_actual.png'
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_test_pred, alpha=0.3, s=10)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Lap Duration (s)')
        plt.ylabel('Predicted Lap Duration (s)')
        plt.title(f'{run_name} - Predictions vs Actual (Test Set)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(pred_plot_path, dpi=150)
        plt.close()
        mlflow.log_artifact(str(pred_plot_path))
        log(f"  Prediction plot saved: {pred_plot_path}")

        # Plot 2: Residuals distribution
        residuals_path = reports_model_dir / 'residuals_distribution.png'
        residuals = y_test - y_test_pred
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
        plt.xlabel('Residuals (Actual - Predicted)')
        plt.ylabel('Frequency')
        plt.title(f'{run_name} - Residuals Distribution (Test Set)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(residuals_path, dpi=150)
        plt.close()
        mlflow.log_artifact(str(residuals_path))
        log(f"  Residuals plot saved: {residuals_path}")

        # 8. Log modèle MLflow avec signature et input example
        try:
            # Utiliser pyfunc pour éviter l'erreur 404 de logged-models
            import cloudpickle

            # Créer un artifact temporaire du modèle
            model_artifact_path = reports_model_dir / "model_artifact.pkl"
            with open(model_artifact_path, 'wb') as f:
                cloudpickle.dump(model, f)

            # Logger le modèle comme artifact
            mlflow.log_artifact(str(model_artifact_path), artifact_path="model")

            log("  Model logged to MLflow successfully as artifact")
        except Exception as e:
            log(f"  Warning: Could not log model to MLflow: {e}")
            log("  Model will still be saved locally")

        # 9. Sauvegarder modèle localement
        model_filename = f"{run_name}_model.pkl"
        model_path = MODELS_DIR / model_filename
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        log(f"Model saved: {model_path}")

        # 10. Rapport JSON
        report = {
            'model_name': model_name,
            'run_name': run_name,
            'run_id': run_id,
            'use_gridsearch': use_gridsearch,
            'train_time_seconds': train_time,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_metrics': cv_metrics,
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'n_features': X_train.shape[1],
            'feature_names': X_train.columns.tolist(),
            'top_10_features': feat_imp_df.head(10).to_dict('records') if feat_imp_df is not None else [],
            'timestamp': datetime.now().isoformat()
        }

        if use_gridsearch:
            report['best_params'] = best_params

        report_path = reports_model_dir / 'training_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        log(f"Report saved: {report_path}")
        mlflow.log_artifact(str(report_path))

        log("=" * 80)
        log(f"{run_name.upper()} TRAINING COMPLETE")
        log("=" * 80)

        return {
            'model': model,
            'model_name': model_name,
            'run_name': run_name,
            'run_id': run_id,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_metrics': cv_metrics,
            'report': report
        }


def compare_models(results: list[dict]) -> None:
    """Compare les performances de tous les modèles."""
    log("=" * 80)
    log("MODEL COMPARISON")
    log("=" * 80)

    comparison = []
    for res in results:
        comparison.append({
            'Model': res['run_name'],
            'Test MAE': res['test_metrics']['mae'],
            'Test RMSE': res['test_metrics']['rmse'],
            'Test R2': res['test_metrics']['r2'],
            'Test MAPE': res['test_metrics']['mape'],
            'CV MAE': res['cv_metrics']['cv_mae_mean'],
            'CV R2': res['cv_metrics']['cv_r2_mean']
        })

    df_comp = pd.DataFrame(comparison)
    print(df_comp.to_string(index=False))

    # Meilleur modèle = MAE test le plus faible
    best_idx = df_comp['Test MAE'].idxmin()
    best_model = df_comp.loc[best_idx, 'Model']

    log("=" * 80)
    log(f"BEST MODEL: {best_model} (lowest Test MAE)")
    log("=" * 80)

    # Sauvegarder comparaison
    comp_path = REPORTS_DIR / 'model_comparison.csv'
    df_comp.to_csv(comp_path, index=False)
    log(f"Comparison saved: {comp_path}")


def main():
    """Pipeline d'entraînement complet avec GridSearch."""
    # Fix Windows encoding for MLflow emoji output
    import sys
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    log("=" * 80)
    log("F1PA - MODEL TRAINING PIPELINE WITH GRIDSEARCH")
    log("=" * 80)

    # 1. Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    log(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    log(f"MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")

    # 2. Preprocessing
    from ml.config import DATASET_PATH, TRAIN_YEARS, TEST_YEAR
    X_train, X_test, y_train, y_test, df = preprocess_pipeline(
        DATASET_PATH, TRAIN_YEARS, TEST_YEAR
    )

    # 3. Entraîner les modèles
    results = []

    for model_name in ['xgboost', 'random_forest']:
        # Baseline (sans GridSearch)
        result_baseline = train_model_with_gridsearch(
            model_name, X_train, y_train, X_test, y_test,
            use_gridsearch=False
        )
        results.append(result_baseline)

        # Avec GridSearch
        result_gridsearch = train_model_with_gridsearch(
            model_name, X_train, y_train, X_test, y_test,
            use_gridsearch=True
        )
        results.append(result_gridsearch)

    # 4. Comparer tous les modèles
    compare_models(results)

    log("=" * 80)
    log("TRAINING PIPELINE COMPLETE")
    log("=" * 80)
    log(f"View results: {MLFLOW_TRACKING_URI}")
    log(f"Total models trained: {len(results)}")
    log("  - 2 baselines (xgboost, random_forest)")
    log("  - 2 with GridSearchCV tuning")


if __name__ == "__main__":
    main()
