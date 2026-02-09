"""
F1PA API - Main Application

FastAPI service exposing:
- ML model predictions
- Data access endpoints
"""
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from api.config import config
from api.models import HealthResponse, ErrorResponse
from api.services.ml_service import ml_service
from api.services.db_service import db_service
from api.endpoints.predictions import router as predictions_router
from api.endpoints.data import router as data_router
from api.endpoints.monitoring import router as monitoring_router
from api.middleware import PrometheusMiddleware, metrics_endpoint
from api.middleware.metrics import (
    update_model_status,
    update_database_status,
    update_mlflow_status
)


# =============================================================================
# LIFESPAN (Startup/Shutdown)
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown events."""
    # Startup
    print("=" * 60)
    print("F1PA API - Starting up...")
    print("=" * 60)

    # Load ML model
    print("\n[Startup] Loading ML model...")
    if config.model_run_id:
        print(f"[Startup] Using specific run ID: {config.model_run_id}")
        model_loaded = ml_service.load_model(run_id=config.model_run_id)
    else:
        model_loaded = ml_service.load_model(
            strategy=config.default_model_strategy,
            model_family=config.default_model_family
        )
    if model_loaded:
        info = ml_service.get_model_info()
        print(f"[Startup] Model loaded: {info.get('model_family')} ({info.get('source')})")
        if info.get('test_mae'):
            print(f"[Startup] Model MAE: {info['test_mae']:.3f}s")
    else:
        print("[Startup] WARNING: Model could not be loaded!")

    # Update Prometheus gauge for model status
    update_model_status(model_loaded)

    # Connect to database
    print("\n[Startup] Connecting to database...")
    db_connected = db_service.connect(config.database_url)
    if db_connected:
        print(f"[Startup] Database connected: {config.db_host}:{config.db_port}/{config.db_name}")
    else:
        print("[Startup] WARNING: Database connection failed!")

    # Update Prometheus gauge for database status
    update_database_status(db_connected)

    # Check MLflow connection
    print("\n[Startup] Checking MLflow connection...")
    mlflow_connected = False
    try:
        import mlflow
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        experiment = mlflow.get_experiment_by_name(config.mlflow_experiment_name)
        mlflow_connected = experiment is not None
        if mlflow_connected:
            print(f"[Startup] MLflow connected: {config.mlflow_tracking_uri}")
    except Exception as e:
        print(f"[Startup] WARNING: MLflow connection failed: {e}")

    # Update Prometheus gauge for MLflow status
    update_mlflow_status(mlflow_connected)

    print("\n" + "=" * 60)
    print(f"F1PA API ready at http://{config.host}:{config.port}")
    print(f"Documentation: http://{config.host}:{config.port}/docs")
    print(f"Metrics: http://{config.host}:{config.port}/metrics")
    print("=" * 60 + "\n")

    yield

    # Shutdown
    print("\n[Shutdown] F1PA API shutting down...")


# =============================================================================
# APPLICATION
# =============================================================================

app = FastAPI(
    title="F1PA API",
    description="""
## F1PA - Formula 1 Predictive Assistant

API REST for F1 lap time predictions and data access.

### Features

**ML Predictions**
- Lap time prediction before the lap starts (Random Forest model)
- Single and batch predictions
- Hypothetical predictions with year fixed to 2025 (last training year)
- Auto-calculated performance metrics (circuit avg, driver performance)

**Data Access**
- Circuits: list, details, average lap times
- Drivers: list, details, performance history
- Sessions: search by year/circuit
- Laps: complete data with filters and pagination

### Model Information
- **Dataset**: 71,645 laps (2023-2025)
- **Features**: 14 features (speeds, weather, circuit/driver performance)
- **Performance**: MAE 1.08s, R² 0.79
- **Tracking**: MLflow for full traceability
    """,
    version="1.0.0",
    lifespan=lifespan,
    responses={
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
        503: {"model": ErrorResponse, "description": "Service Unavailable"},
    }
)

# CORS middleware (allows cross-origin requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics middleware
app.add_middleware(PrometheusMiddleware)

# Include routers
app.include_router(predictions_router)
app.include_router(data_router)
app.include_router(monitoring_router)

# Metrics endpoint for Prometheus scraping
app.add_route("/metrics", metrics_endpoint)


# =============================================================================
# HEALTH & ROOT ENDPOINTS
# =============================================================================

@app.get("/", tags=["Health"])
async def root():
    """API root - redirect to documentation."""
    return {
        "name": "F1PA API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Check API health status.

    Returns status of the API, ML model, and database connections.
    """
    # Check MLflow connection
    mlflow_connected = False
    try:
        import mlflow
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        experiment = mlflow.get_experiment_by_name(config.mlflow_experiment_name)
        mlflow_connected = experiment is not None
    except Exception:
        pass

    return HealthResponse(
        status="healthy" if ml_service.is_ready() and db_service.is_ready() else "degraded",
        version="1.0.0",
        model_loaded=ml_service.is_ready(),
        database_connected=db_service.is_ready(),
        mlflow_connected=mlflow_connected
    )


# =============================================================================
# MAIN (for direct execution)
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug
    )
