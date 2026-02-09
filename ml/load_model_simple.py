"""
F1PA - Model Loading

Models can be loaded in 3 ways:
1. From MLflow (latest run) - RECOMMENDED to always use the latest model
2. From MLflow (specific Run ID) - For reproducibility
3. From local .pkl file - BACKUP if MLflow unavailable

Models are automatically searched in MLflow without depending on predefined IDs.
"""
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import pickle
from pathlib import Path
import mlflow
import cloudpickle

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "F1PA_LapTime_Prediction")

# Local model paths (backup)
MODELS_DIR = Path(__file__).parent.parent / "models"
MODEL_XGBOOST_PATH = MODELS_DIR / "xgboost_gridsearch_model.pkl"
MODEL_RF_PATH = MODELS_DIR / "random_forest_gridsearch_model.pkl"


def get_best_model_from_mlflow(strategy="robust", model_family=None):
    """
    Retrieve the best model from MLflow according to a strategy.

    Args:
        strategy: "robust" (best overfitting) or "mae" (best absolute MAE)
        model_family: "xgboost", "random_forest" or None (all models)

    Returns:
        run_id: Best run ID
        metrics: Dictionary of metrics
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Get experiment
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if not experiment:
        raise ValueError(f"Experiment '{MLFLOW_EXPERIMENT_NAME}' not found. Have you run training?")

    # Search ALL runs (baseline + gridsearch) for the specified model family
    if model_family:
        filter_string = f"tags.model_family = '{model_family}'"
    else:
        filter_string = ""  # All models

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string,
        order_by=["metrics.test_mae ASC"]
    )

    if runs.empty:
        raise ValueError(f"No runs found in MLflow for family '{model_family}'")

    if strategy == "robust":
        # Robust strategy: best overfitting ratio (generalization)
        robust_runs = runs[
            runs['metrics.overfitting_ratio'] < 5.0
        ].sort_values('metrics.overfitting_ratio')

        if robust_runs.empty:
            print("Warning: No robust model found (overfitting < 5.0), using best MAE instead")
            best_run = runs.iloc[0]
        else:
            best_run = robust_runs.iloc[0]
    else:  # strategy == "mae"
        # MAE strategy: best absolute MAE, but favors R² if MAE very close
        # If top 2 runs have very close MAE (<0.01s), choose the one with better R²
        best_mae_run = runs.iloc[0]
        if len(runs) > 1:
            second_run = runs.iloc[1]
            mae_diff = abs(best_mae_run['metrics.test_mae'] - second_run['metrics.test_mae'])

            # If MAE very close (<0.01s), compare by R²
            if mae_diff < 0.01:
                r2_best = best_mae_run.get('metrics.test_r2', 0.0)
                r2_second = second_run.get('metrics.test_r2', 0.0)

                if r2_second > r2_best:
                    print(f"MAE very close ({mae_diff:.4f}s), selecting model with better R² ({r2_second:.3f} vs {r2_best:.3f})")
                    best_run = second_run
                else:
                    best_run = best_mae_run
            else:
                best_run = best_mae_run
        else:
            best_run = best_mae_run

    # Get actual model family from the selected run's tags
    actual_model_family = best_run.get('tags.model_family', 'unknown')
    run_name = best_run.get('tags.mlflow.runName', 'unknown')

    print(f"Selected model: {run_name} (MAE: {best_run['metrics.test_mae']:.3f}s)")

    return best_run['run_id'], {
        'test_mae': best_run.get('metrics.test_mae', 0.0),
        'test_r2': best_run.get('metrics.test_r2', 0.0),
        'test_rmse': best_run.get('metrics.test_rmse', 0.0),
        'overfitting_ratio': best_run.get('metrics.overfitting_ratio', 0.0),
        'cv_mae': best_run.get('metrics.cv_mae_mean', 0.0),
        'cv_r2': best_run.get('metrics.cv_r2_mean', 0.0),
        'model_family': actual_model_family,
        'run_name': run_name
    }


def load_model_from_mlflow(strategy="robust", model_family="xgboost", run_id=None):
    """
    Load a model from MLflow.

    Args:
        strategy: "robust" (best overfitting) or "mae" (best MAE)
        model_family: "xgboost" or "random_forest"
        run_id: Specific run ID (optional, for reproducibility)

    Returns:
        model: Model loaded from MLflow
        info: Dictionary with model metadata
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # If no run_id specified, find the best
    if run_id is None:
        run_id, metrics = get_best_model_from_mlflow(strategy, model_family)
    else:
        # Get metrics for specified run
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        metrics = {
            'test_mae': run.data.metrics.get('test_mae'),
            'test_r2': run.data.metrics.get('test_r2'),
            'test_rmse': run.data.metrics.get('test_rmse'),
            'overfitting_ratio': run.data.metrics.get('overfitting_ratio'),
            'cv_mae': run.data.metrics.get('cv_mae_mean'),
            'cv_r2': run.data.metrics.get('cv_r2_mean'),
            'model_family': run.data.tags.get('model_family'),
            'run_name': run.data.tags.get('mlflow.runName')
        }

    # Download model artifact
    client = mlflow.tracking.MlflowClient()
    artifact_path = client.download_artifacts(run_id, "model/model_artifact.pkl")

    # Load model
    with open(artifact_path, 'rb') as f:
        model = cloudpickle.load(f)

    # Use actual model_family from metrics if available (auto-selection case)
    actual_family = metrics.get('model_family', model_family) or 'unknown'
    run_name = metrics.get('run_name', 'unknown')

    print(f"Model loaded from MLflow ({strategy} strategy)")
    print(f"  Model Family: {actual_family}")
    print(f"  Run Name: {run_name}")
    print(f"  Run ID: {run_id}")

    if metrics.get('test_mae') is not None:
        print(f"  Test MAE: {metrics['test_mae']:.3f}s")
    if metrics.get('test_r2') is not None:
        print(f"  Test R²: {metrics['test_r2']:.3f}")
    if metrics.get('test_rmse') is not None:
        print(f"  Test RMSE: {metrics['test_rmse']:.3f}s")
    if metrics.get('overfitting_ratio') is not None:
        print(f"  Overfitting: {metrics['overfitting_ratio']:.2f}")
    if metrics.get('cv_mae') is not None and metrics['cv_mae'] > 0:
        print(f"  CV MAE: {metrics['cv_mae']:.3f}s")
    if metrics.get('cv_r2') is not None and metrics['cv_r2'] > 0:
        print(f"  CV R²: {metrics['cv_r2']:.3f}")

    info = {
        'run_id': run_id,
        'strategy': strategy,
        'model_family': actual_family,
        'run_name': run_name,
        **{k: v for k, v in metrics.items() if k not in ('model_family', 'run_name')}
    }

    return model, info


def load_model_local(model_family="xgboost"):
    """
    Load a model from a local file (backup if MLflow unavailable).

    Args:
        model_family: "xgboost" or "random_forest"

    Returns:
        model: Loaded model
        info: Dictionary with basic metadata
    """
    if model_family == "xgboost":
        model_path = MODEL_XGBOOST_PATH
    elif model_family == "random_forest":
        model_path = MODEL_RF_PATH
    else:
        raise ValueError(f"Model family unknown: {model_family}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    print(f"Model loaded from local file")
    print(f"  Model Family: {model_family}")
    print(f"  Path: {model_path}")
    print(f"  Warning: No metrics available (use MLflow for full tracking)")

    info = {
        'source': 'local',
        'path': str(model_path),
        'model_family': model_family
    }

    return model, info


def show_models_info():
    """Display information about all available models in MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    print("=" * 80)
    print("F1PA MODELS AVAILABLE IN MLFLOW")
    print("=" * 80)

    try:
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if not experiment:
            print(f"\n⚠️ Experiment '{MLFLOW_EXPERIMENT_NAME}' not found")
            print("Have you run the training pipeline? (python -m ml.train)")
            return

        # Retrieve all runs with GridSearch
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.tuning_method = 'gridsearch'",
            order_by=["metrics.test_mae ASC"]
        )

        if runs.empty:
            print("\n⚠️ No GridSearch runs found")
            return

        print(f"\nFound {len(runs)} GridSearch runs\n")

        for idx, run in runs.iterrows():
            model_family = run.get('tags.model_family', 'unknown')
            run_id = run['run_id']

            print(f"{model_family.upper()} GridSearch")
            print(f"  Run ID: {run_id}")
            print(f"  Test MAE: {run.get('metrics.test_mae', 0.0):.3f}s")
            print(f"  Test R²: {run.get('metrics.test_r2', 0.0):.3f}")
            print(f"  Test RMSE: {run.get('metrics.test_rmse', 0.0):.3f}s")
            print(f"  Overfitting: {run.get('metrics.overfitting_ratio', 0.0):.2f}")

            cv_mae = run.get('metrics.cv_mae')
            cv_r2 = run.get('metrics.cv_r2')
            if cv_mae is not None and cv_r2 is not None:
                print(f"  CV MAE: {cv_mae:.3f}s")
                print(f"  CV R²: {cv_r2:.3f}")
            print()

        # Recommendations
        print("=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)

        xgb_runs = runs[runs['tags.model_family'] == 'xgboost']
        if not xgb_runs.empty:
            robust_xgb = xgb_runs[
                (xgb_runs['metrics.overfitting_ratio'] < 5.0) &
                (xgb_runs['metrics.test_mae'] < 1.5)
            ].sort_values('metrics.overfitting_ratio')

            if not robust_xgb.empty:
                best = robust_xgb.iloc[0]
                print(f"\n✅ RECOMMENDED (XGBoost Robust):")
                print(f"   Run ID: {best['run_id']}")
                print(f"   Test MAE: {best['metrics.test_mae']:.3f}s")
                print(f"   Overfitting: {best['metrics.overfitting_ratio']:.2f}")
                print(f"\n   Usage:")
                print(f"   model, info = load_model_from_mlflow(strategy='robust', model_family='xgboost')")

    except Exception as e:
        print(f"\n❌ Error connecting to MLflow: {e}")
        print(f"Make sure MLflow is running at {MLFLOW_TRACKING_URI}")
        print("\nLocal models available:")
        print(f"  - XGBoost: {MODEL_XGBOOST_PATH} (exists: {MODEL_XGBOOST_PATH.exists()})")
        print(f"  - Random Forest: {MODEL_RF_PATH} (exists: {MODEL_RF_PATH.exists()})")


# Example usage
if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("F1PA - Model Loading")
    print("=" * 80)

    # Display available models
    show_models_info()

    # Load recommended model (Robust XGBoost)
    print("\n" + "=" * 80)
    print("LOADING RECOMMENDED MODEL")
    print("=" * 80 + "\n")

    print("Method 1: From MLflow (strategy 'robust' - RECOMMENDED)")
    try:
        model, info = load_model_from_mlflow(strategy="robust", model_family="xgboost")
        print("\n✅ Model loaded from MLflow successfully")
        print(f"\nUsage:")
        print(f"  predictions = model.predict(X_new)")
    except Exception as e:
        print(f"\n⚠️ MLflow unavailable: {e}")
        print("\nTrying local method as backup...")
        print("\n" + "-" * 80 + "\n")
        print("Method 2: From local .pkl file (backup)")
        try:
            model, info = load_model_local(model_family="xgboost")
            print("\n✅ Model loaded from local file")
        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n" + "=" * 80)
    print("OTHER USAGE EXAMPLES")
    print("=" * 80)
    print("\n# Load best MAE (absolute performance)")
    print("model, info = load_model_from_mlflow(strategy='mae', model_family='xgboost')")
    print("\n# Load Random Forest")
    print("model, info = load_model_from_mlflow(strategy='robust', model_family='random_forest')")
    print("\n# Load specific run (reproducibility)")
    print("model, info = load_model_from_mlflow(run_id='<RUN_ID>')")
    print("\n# Load from local file")
    print("model, info = load_model_local(model_family='xgboost')")
