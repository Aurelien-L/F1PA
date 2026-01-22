"""
F1PA API - ML Service

Service for loading and using ML models for predictions.
"""
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


class MLService:
    """Service for ML model management and predictions."""

    def __init__(self):
        self.model = None
        self.model_info: Dict[str, Any] = {}
        self._initialized = False

    def load_model(
        self,
        strategy: str = "robust",
        model_family: str = "xgboost",
        run_id: Optional[str] = None
    ) -> bool:
        """
        Load ML model from MLflow or local file.

        Args:
            strategy: "robust" (low overfitting) or "mae" (best performance)
            model_family: "xgboost" or "random_forest"
            run_id: Specific MLflow run ID (optional)

        Returns:
            True if model loaded successfully
        """
        try:
            # Try loading from MLflow first
            from ml.load_model_simple import load_model_from_mlflow
            self.model, self.model_info = load_model_from_mlflow(
                strategy=strategy,
                model_family=model_family,
                run_id=run_id
            )
            self.model_info["source"] = "mlflow"
            self._initialized = True
            return True

        except Exception as mlflow_error:
            print(f"MLflow loading failed: {mlflow_error}")

            # Fallback to local model
            try:
                from ml.load_model_simple import load_model_local
                self.model, self.model_info = load_model_local(model_family=model_family)
                self.model_info["source"] = "local"
                self._initialized = True
                return True

            except Exception as local_error:
                print(f"Local model loading failed: {local_error}")
                self._initialized = False
                return False

    def is_ready(self) -> bool:
        """Check if model is loaded and ready."""
        return self._initialized and self.model is not None

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.is_ready():
            return {"error": "Model not loaded"}
        return self.model_info

    def predict(self, features: Dict[str, float]) -> float:
        """
        Make a single prediction.

        Args:
            features: Dictionary of input features

        Returns:
            Predicted lap duration in seconds
        """
        if not self.is_ready():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Prepare features in the correct order
        X = self._prepare_features(features)

        # Make prediction
        prediction = self.model.predict(X)[0]
        return float(prediction)

    def predict_batch(self, features_list: List[Dict[str, float]]) -> List[float]:
        """
        Make batch predictions.

        Args:
            features_list: List of feature dictionaries

        Returns:
            List of predicted lap durations
        """
        if not self.is_ready():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Prepare all features
        X = np.vstack([self._prepare_features(f) for f in features_list])

        # Make predictions
        predictions = self.model.predict(X)
        return [float(p) for p in predictions]

    def _prepare_features(self, features: Dict[str, float]) -> np.ndarray:
        """
        Prepare features for prediction.

        Transforms raw input features into the format expected by the model,
        including derived features.
        """
        # Calculate derived features
        avg_speed = (features["st_speed"] + features["i1_speed"] + features["i2_speed"]) / 3
        total_sector_time = (
            features["duration_sector_1"] +
            features["duration_sector_2"] +
            features["duration_sector_3"]
        )

        # Avoid division by zero
        if total_sector_time > 0:
            sector_1_ratio = features["duration_sector_1"] / total_sector_time
            sector_2_ratio = features["duration_sector_2"] / total_sector_time
        else:
            sector_1_ratio = 0.33
            sector_2_ratio = 0.33

        # Lap progress (normalized lap number, assuming ~70 laps max)
        lap_progress = min(features["lap_number"] / 70.0, 1.0)

        # Feature vector in expected order (17 features after selection)
        # Based on ml/config.py and preprocessing.py
        feature_vector = [
            features["st_speed"],
            features["i1_speed"],
            features["i2_speed"],
            features["duration_sector_1"],
            features["duration_sector_2"],
            features["duration_sector_3"],
            features["temp"],
            features["rhum"],
            features["pres"],
            features["lap_number"],
            features["year"],
            features["circuit_avg_laptime"],
            avg_speed,
            total_sector_time,
            sector_1_ratio,
            sector_2_ratio,
            lap_progress,
        ]

        return np.array([feature_vector])

    @staticmethod
    def format_lap_time(seconds: float) -> str:
        """Format lap time as MM:SS.mmm"""
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}:{remaining_seconds:06.3f}"


# Global service instance
ml_service = MLService()
