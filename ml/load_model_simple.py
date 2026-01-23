"""
F1PA - Charger les Modèles

Les modèles peuvent être chargés de 3 façons:
1. Depuis MLflow (dernier run) - RECOMMANDÉ pour toujours avoir le dernier modèle
2. Depuis MLflow (via Run ID spécifique) - Pour reproductibilité
3. Depuis fichier local .pkl - BACKUP si MLflow indisponible

Les modèles sont automatiquement recherchés dans MLflow sans dépendre d'IDs prédéfinis.
"""
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pickle
from pathlib import Path
import mlflow
import cloudpickle

# Configuration MLflow
MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT_NAME = "F1PA_LapTime_Prediction"

# Chemins des modèles locaux (backup)
MODELS_DIR = Path(__file__).parent.parent / "models"
MODEL_XGBOOST_PATH = MODELS_DIR / "xgboost_gridsearch_model.pkl"
MODEL_RF_PATH = MODELS_DIR / "random_forest_gridsearch_model.pkl"


def get_best_model_from_mlflow(strategy="robust", model_family=None):
    """
    Récupère le meilleur modèle depuis MLflow selon une stratégie.

    Args:
        strategy: "robust" (meilleur overfitting) ou "mae" (meilleur MAE absolu)
        model_family: "xgboost", "random_forest" ou None (tous les modèles)

    Returns:
        run_id: ID du meilleur run
        metrics: Dictionnaire des métriques
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Récupérer l'expériment
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if not experiment:
        raise ValueError(f"Experiment '{MLFLOW_EXPERIMENT_NAME}' not found. Have you run training?")

    # Chercher TOUS les runs (baseline + gridsearch) du bon modèle
    if model_family:
        filter_string = f"tags.model_family = '{model_family}'"
    else:
        filter_string = ""  # Tous les modèles

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string,
        order_by=["metrics.test_mae ASC"]
    )

    if runs.empty:
        raise ValueError(f"No runs found in MLflow for family '{model_family}'")

    if strategy == "robust":
        # Stratégie robust: meilleur overfitting ratio (généralisation)
        robust_runs = runs[
            runs['metrics.overfitting_ratio'] < 5.0
        ].sort_values('metrics.overfitting_ratio')

        if robust_runs.empty:
            print("Warning: No robust model found (overfitting < 5.0), using best MAE instead")
            best_run = runs.iloc[0]
        else:
            best_run = robust_runs.iloc[0]
    else:  # strategy == "mae"
        # Stratégie MAE: meilleur MAE absolu sur test
        best_run = runs.iloc[0]

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
    Charge un modèle depuis MLflow.

    Args:
        strategy: "robust" (meilleur overfitting) ou "mae" (meilleur MAE)
        model_family: "xgboost" ou "random_forest"
        run_id: ID spécifique d'un run (optionnel, pour reproductibilité)

    Returns:
        model: Modèle chargé depuis MLflow
        info: Dictionnaire avec les métadonnées du modèle
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Si pas de run_id spécifié, chercher le meilleur
    if run_id is None:
        run_id, metrics = get_best_model_from_mlflow(strategy, model_family)
    else:
        # Récupérer les métriques du run spécifié
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

    # Télécharger l'artifact du modèle
    client = mlflow.tracking.MlflowClient()
    artifact_path = client.download_artifacts(run_id, "model/model_artifact.pkl")

    # Charger le modèle
    with open(artifact_path, 'rb') as f:
        model = cloudpickle.load(f)

    # Use actual model_family from metrics if available (auto-selection case)
    actual_family = metrics.get('model_family', model_family) or 'unknown'
    run_name = metrics.get('run_name', 'unknown')

    print(f"Modèle chargé depuis MLflow ({strategy} strategy)")
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
    Charge un modèle depuis un fichier local (backup si MLflow indisponible).

    Args:
        model_family: "xgboost" ou "random_forest"

    Returns:
        model: Modèle chargé
        info: Dictionnaire avec les métadonnées basiques
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
    """Affiche les informations sur tous les modèles disponibles dans MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    print("=" * 80)
    print("MODÈLES F1PA DISPONIBLES DANS MLFLOW")
    print("=" * 80)

    try:
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if not experiment:
            print(f"\n⚠️ Experiment '{MLFLOW_EXPERIMENT_NAME}' not found")
            print("Have you run the training pipeline? (python -m ml.train)")
            return

        # Récupérer tous les runs avec GridSearch
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

        # Recommandations
        print("=" * 80)
        print("RECOMMANDATIONS")
        print("=" * 80)

        xgb_runs = runs[runs['tags.model_family'] == 'xgboost']
        if not xgb_runs.empty:
            robust_xgb = xgb_runs[
                (xgb_runs['metrics.overfitting_ratio'] < 5.0) &
                (xgb_runs['metrics.test_mae'] < 1.5)
            ].sort_values('metrics.overfitting_ratio')

            if not robust_xgb.empty:
                best = robust_xgb.iloc[0]
                print(f"\n✅ RECOMMANDÉ (XGBoost Robust):")
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


# Exemple d'utilisation
if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("F1PA - Chargement des Modèles")
    print("=" * 80)

    # Afficher les modèles disponibles
    show_models_info()

    # Charger le modèle recommandé (Robust XGBoost)
    print("\n" + "=" * 80)
    print("CHARGEMENT MODÈLE RECOMMANDÉ")
    print("=" * 80 + "\n")

    print("Méthode 1: Depuis MLflow (stratégie 'robust' - RECOMMANDÉ)")
    try:
        model, info = load_model_from_mlflow(strategy="robust", model_family="xgboost")
        print("\n✅ Modèle chargé depuis MLflow avec succès")
        print(f"\nUtilisation:")
        print(f"  predictions = model.predict(X_new)")
    except Exception as e:
        print(f"\n⚠️ MLflow indisponible: {e}")
        print("\nEssayer la méthode locale comme backup...")
        print("\n" + "-" * 80 + "\n")
        print("Méthode 2: Depuis fichier local .pkl (backup)")
        try:
            model, info = load_model_local(model_family="xgboost")
            print("\n✅ Modèle chargé depuis fichier local")
        except Exception as e:
            print(f"❌ Erreur: {e}")

    print("\n" + "=" * 80)
    print("AUTRES EXEMPLES D'UTILISATION")
    print("=" * 80)
    print("\n# Charger meilleur MAE (performance absolue)")
    print("model, info = load_model_from_mlflow(strategy='mae', model_family='xgboost')")
    print("\n# Charger Random Forest")
    print("model, info = load_model_from_mlflow(strategy='robust', model_family='random_forest')")
    print("\n# Charger un run spécifique (reproductibilité)")
    print("model, info = load_model_from_mlflow(run_id='<RUN_ID>')")
    print("\n# Charger depuis fichier local")
    print("model, info = load_model_local(model_family='xgboost')")
