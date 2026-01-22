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

    # Connect to database
    print("\n[Startup] Connecting to database...")
    db_connected = db_service.connect(config.database_url)
    if db_connected:
        print(f"[Startup] Database connected: {config.db_host}:{config.db_port}/{config.db_name}")
    else:
        print("[Startup] WARNING: Database connection failed!")

    print("\n" + "=" * 60)
    print(f"F1PA API ready at http://{config.host}:{config.port}")
    print(f"Documentation: http://{config.host}:{config.port}/docs")
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
## F1PA - Formula 1 Predictive Assistant API

API REST pour accéder aux données F1 et aux prédictions de temps au tour.

### Fonctionnalités

**Prédictions ML**
- Prédiction de temps au tour via modèle XGBoost/RandomForest
- Prédictions unitaires et par batch
- Informations sur le modèle chargé

**Accès aux Données**
- Circuits: liste, détails, temps moyens
- Pilotes: liste, détails, historique
- Sessions: recherche par année/circuit
- Tours: données complètes avec filtres et pagination

### Dataset
- 71,645 tours (2023-2025)
- 17 features ML
- Tracking MLflow pour la traçabilité
    """,
    version="1.0.0",
    lifespan=lifespan,
    responses={
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
        503: {"model": ErrorResponse, "description": "Service Unavailable"},
    }
)

# CORS middleware (permet les appels cross-origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifier les origines autorisées
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predictions_router)
app.include_router(data_router)


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
