"""
F1PA API - Configuration
"""
import os
from dataclasses import dataclass


@dataclass
class APIConfig:
    """API configuration."""

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # Database
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "f1pa_db"
    db_user: str = "f1pa"
    db_password: str = "f1pa"

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "F1PA_LapTime_Prediction"

    # Model selection: automatic selection of best model across all families
    # Set to None to auto-select best model (recommended)
    # Or specify "xgboost" / "random_forest" to restrict to one family
    default_model_family: str = None  # None = auto-select best across all
    default_model_strategy: str = "mae"  # "mae" = best absolute performance
    model_run_id: str = None  # Specific run ID to load (overrides strategy)

    @classmethod
    def from_env(cls) -> "APIConfig":
        """Load configuration from environment variables."""
        # Handle model_family: "auto" or empty string means None (auto-select)
        model_family_env = os.getenv("DEFAULT_MODEL_FAMILY", "auto")
        model_family = None if model_family_env in ("auto", "", "none") else model_family_env

        # Handle model_run_id: empty string or "none" means None
        model_run_id_env = os.getenv("MODEL_RUN_ID", "")
        model_run_id = None if model_run_id_env in ("", "none") else model_run_id_env

        return cls(
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            debug=os.getenv("API_DEBUG", "false").lower() == "true",
            db_host=os.getenv("POSTGRES_HOST", "localhost"),
            db_port=int(os.getenv("POSTGRES_PORT", "5432")),
            db_name=os.getenv("POSTGRES_DB", "f1pa_db"),
            db_user=os.getenv("POSTGRES_USER", "f1pa"),
            db_password=os.getenv("POSTGRES_PASSWORD", "f1pa"),
            mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
            mlflow_experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "F1PA_LapTime_Prediction"),
            default_model_family=model_family,
            default_model_strategy=os.getenv("DEFAULT_MODEL_STRATEGY", "mae"),
            model_run_id=model_run_id,
        )

    @property
    def database_url(self) -> str:
        """Build PostgreSQL connection string."""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"


# Global config instance
config = APIConfig.from_env()
