"""
F1PA API - Pydantic Models

Request/Response models for the API endpoints.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


# =============================================================================
# PREDICTION MODELS (C9)
# =============================================================================

class LapFeatures(BaseModel):
    """
    Input features for lap time PERFORMANCE prediction.

    OBJECTIF: Prédire le temps au tour d'un pilote AVANT qu'il roule,
    basé sur sa performance historique et les conditions.

    NOTE: Les temps secteurs ne sont PAS utilisés car ils représentent
    des données du tour en cours (ce serait trivial: lap_time ≈ sum(sectors)).
    """

    # Driver identification
    driver_number: int = Field(..., description="Driver number (e.g., 1 for Verstappen)", ge=1, le=99)

    # Circuit identification
    circuit_key: int = Field(..., description="Circuit key from the database", ge=1)

    # Speed measurements (km/h) - historical/expected performance
    st_speed: float = Field(..., description="Expected speed trap (km/h)", ge=0, le=400)
    i1_speed: float = Field(..., description="Expected intermediate 1 speed (km/h)", ge=0, le=400)
    i2_speed: float = Field(..., description="Expected intermediate 2 speed (km/h)", ge=0, le=400)

    # Weather conditions
    temp: float = Field(..., description="Temperature (Celsius)", ge=-20, le=60)
    rhum: float = Field(..., description="Relative humidity (%)", ge=0, le=100)
    pres: float = Field(..., description="Atmospheric pressure (hPa)", ge=900, le=1100)

    # Context
    lap_number: int = Field(..., description="Expected lap number in the session", ge=1)
    year: int = Field(..., description="Year of the race", ge=2023, le=2030)

    # Performance encodings (can be fetched from /data endpoints)
    circuit_avg_laptime: float = Field(..., description="Average lap time for this circuit (seconds)", ge=60, le=150)
    driver_avg_laptime: float = Field(..., description="Driver's average lap time (seconds)", ge=60, le=150)
    driver_perf_score: float = Field(0.0, description="Driver performance score (negative = faster than average)")

    # Pydantic V2 configuration
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "driver_number": 1,
                "circuit_key": 7,
                "st_speed": 310.5,
                "i1_speed": 295.2,
                "i2_speed": 288.1,
                "temp": 25.0,
                "rhum": 45.0,
                "pres": 1013.0,
                "lap_number": 15,
                "year": 2025,
                "circuit_avg_laptime": 92.5,
                "driver_avg_laptime": 91.2,
                "driver_perf_score": -1.3
            }
        }
    )


class PredictionRequest(BaseModel):
    """Request body for single prediction."""
    features: LapFeatures


class BatchPredictionRequest(BaseModel):
    """Request body for batch predictions."""
    features: List[LapFeatures] = Field(..., min_length=1, max_length=1000)


class PredictionResponse(BaseModel):
    """Response for lap time prediction."""
    lap_duration_seconds: float = Field(..., description="Predicted lap time in seconds")
    lap_duration_formatted: str = Field(..., description="Predicted lap time formatted (MM:SS.mmm)")
    model_info: Dict[str, Any] = Field(..., description="Information about the model used")


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    predictions: List[float] = Field(..., description="Predicted lap times in seconds")
    count: int = Field(..., description="Number of predictions")
    model_info: Dict[str, Any] = Field(..., description="Information about the model used")


class ModelInfoResponse(BaseModel):
    """Response with model information."""
    model_family: str = Field(..., description="Model type (xgboost, random_forest)")
    strategy: str = Field(..., description="Selection strategy (mae, robust)")
    run_name: Optional[str] = Field(None, description="MLflow run name")
    run_id: Optional[str] = Field(None, description="MLflow run ID")
    test_mae: Optional[float] = Field(None, description="Test Mean Absolute Error (seconds)")
    test_r2: Optional[float] = Field(None, description="Test R² score")
    test_rmse: Optional[float] = Field(None, description="Test Root Mean Square Error")
    cv_mae: Optional[float] = Field(None, description="Cross-validation MAE")
    cv_r2: Optional[float] = Field(None, description="Cross-validation R²")
    overfitting_ratio: Optional[float] = Field(None, description="Overfitting ratio (test/train)")
    source: str = Field("mlflow", description="Model source (mlflow, local)")


# =============================================================================
# DATA ACCESS MODELS (C5)
# =============================================================================

class CircuitResponse(BaseModel):
    """Circuit information."""
    circuit_key: int
    circuit_short_name: str
    location: Optional[str] = None
    country_name: Optional[str] = None
    country_code: Optional[str] = None


class DriverResponse(BaseModel):
    """Driver information."""
    driver_number: int
    full_name: str
    name_acronym: str
    team_name: Optional[str] = None
    team_colour: Optional[str] = None
    country_code: Optional[str] = None
    headshot_url: Optional[str] = None


class SessionResponse(BaseModel):
    """Session information."""
    session_key: int
    meeting_key: int
    year: int
    session_name: str
    session_type: Optional[str] = None
    circuit_key: int
    date_start: Optional[str] = None


class LapResponse(BaseModel):
    """Lap data from the database."""
    meeting_key: int
    session_key: int
    driver_number: int
    lap_number: int
    year: int
    circuit_key: int
    lap_duration: float
    st_speed: Optional[float] = None
    i1_speed: Optional[float] = None
    i2_speed: Optional[float] = None
    duration_sector_1: Optional[float] = None
    duration_sector_2: Optional[float] = None
    duration_sector_3: Optional[float] = None
    temp: Optional[float] = None
    rhum: Optional[float] = None
    pres: Optional[float] = None


class PaginatedResponse(BaseModel):
    """Generic paginated response."""
    data: List[Any]
    total: int
    page: int
    page_size: int
    total_pages: int


class DatasetStatsResponse(BaseModel):
    """Dataset statistics."""
    total_laps: int
    total_circuits: int
    total_drivers: int
    total_sessions: int
    years: List[int]
    date_range: Dict[str, str]


# =============================================================================
# HEALTH & STATUS MODELS
# =============================================================================

class HealthResponse(BaseModel):
    """API health status."""
    status: str
    version: str
    model_loaded: bool
    database_connected: bool
    mlflow_connected: bool


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
