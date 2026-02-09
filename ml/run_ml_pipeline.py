"""
F1PA - Run Complete ML Pipeline

All-in-one script to execute the complete ML pipeline:
1. Prerequisites check
2. Model training
3. Display results
4. Usage guide

Usage:
    python ml/run_ml_pipeline.py
"""
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import subprocess
from pathlib import Path
import urllib.request
import urllib.error


def log(msg: str, level: str = "INFO") -> None:
    """Display formatted message."""
    symbols = {
        "INFO": "ℹ️",
        "SUCCESS": "✅",
        "WARNING": "⚠️",
        "ERROR": "❌",
        "STEP": "▶️"
    }
    symbol = symbols.get(level, "•")
    print(f"{symbol} {msg}")


def check_dataset() -> bool:
    """Check if dataset exists."""
    dataset_path = Path("data/processed/dataset_ml_lap_level_2023_2024_2025.csv")
    if dataset_path.exists():
        log(f"Dataset found: {dataset_path}", "SUCCESS")
        return True
    else:
        log(f"Dataset NOT found: {dataset_path}", "ERROR")
        log("Please run the ETL pipeline first (extract + transform + load)", "INFO")
        return False


def check_mlflow() -> bool:
    """Check if MLflow is running."""
    try:
        req = urllib.request.Request("http://localhost:5000/health", method='GET')
        with urllib.request.urlopen(req, timeout=2) as response:
            if response.status == 200:
                log("MLflow is running: http://localhost:5000", "SUCCESS")
                return True
            else:
                log("MLflow responded but with error", "WARNING")
                return False
    except (urllib.error.URLError, urllib.error.HTTPError, Exception) as e:
        log(f"MLflow is NOT running: {e}", "WARNING")
        log("Start it with: docker-compose up -d mlflow", "INFO")
        log("Training will still work but without tracking", "WARNING")
        return False


def run_training() -> bool:
    """Launch model training."""
    log("Starting ML training pipeline...", "STEP")
    log("This will take 5-10 minutes (4 models + GridSearch)", "INFO")
    print()

    try:
        result = subprocess.run(
            [sys.executable, "-m", "ml.train"],
            capture_output=False,
            text=True
        )

        if result.returncode == 0:
            log("Training completed successfully!", "SUCCESS")
            return True
        else:
            log(f"Training failed with exit code {result.returncode}", "ERROR")
            return False

    except Exception as e:
        log(f"Error running training: {e}", "ERROR")
        return False


def show_results() -> None:
    """Display results and instructions."""
    log("=" * 80, "INFO")
    log("TRAINING COMPLETE - NEXT STEPS", "SUCCESS")
    log("=" * 80, "INFO")
    print()

    log("1. View Models in MLflow UI:", "STEP")
    print("   → http://localhost:5000")
    print("   → Go to: Experiments → F1PA_LapTime_Prediction")
    print()

    log("2. Load a Model in Python:", "STEP")
    print("""
from ml.load_model_simple import load_model_from_mlflow

# Load best robust model (RECOMMENDED)
model, info = load_model_from_mlflow(strategy='robust', model_family='xgboost')

print(f"Run ID: {info['run_id']}")
print(f"Test MAE: {info['test_mae']:.3f}s")
print(f"Overfitting: {info['overfitting_ratio']:.2f}")

# Make predictions
predictions = model.predict(X_new)
""")
    print()

    log("3. View All Available Models:", "STEP")
    print("   python -m ml.load_model_simple")
    print()

    log("4. Model Files Saved:", "STEP")
    print("   → models/xgboost_baseline_model.pkl")
    print("   → models/xgboost_gridsearch_model.pkl")
    print("   → models/random_forest_baseline_model.pkl")
    print("   → models/random_forest_gridsearch_model.pkl")
    print()

    log("5. Reports Generated:", "STEP")
    print("   → reports/model_comparison.csv")
    print("   → reports/xgboost_gridsearch/  (plots + metrics)")
    print("   → reports/random_forest_gridsearch/  (plots + metrics)")
    print()

    log("📚 Full documentation: ml/README.md", "INFO")
    print()


def main():
    """Main entry point."""
    print()
    log("=" * 80, "INFO")
    log("F1PA - COMPLETE ML PIPELINE", "INFO")
    log("=" * 80, "INFO")
    print()

    # Step 1: Check prerequisites
    log("STEP 1: Checking Prerequisites", "STEP")
    print()

    dataset_ok = check_dataset()
    mlflow_ok = check_mlflow()

    if not dataset_ok:
        log("Cannot proceed without dataset", "ERROR")
        sys.exit(1)

    if not mlflow_ok:
        log("Proceeding without MLflow (models will still be saved locally)", "WARNING")

    print()

    # Step 2: Run training
    log("STEP 2: Training Models", "STEP")
    print()

    training_ok = run_training()

    if not training_ok:
        log("Training failed - check errors above", "ERROR")
        sys.exit(1)

    print()

    # Step 3: Show results
    show_results()

    log("=" * 80, "INFO")
    log("PIPELINE COMPLETE ✨", "SUCCESS")
    log("=" * 80, "INFO")
    print()


if __name__ == "__main__":
    main()
