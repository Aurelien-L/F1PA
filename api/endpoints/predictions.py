"""
F1PA API - Prediction Endpoints (C9)

Endpoints for exposing the ML model for lap time predictions.
"""
from fastapi import APIRouter, HTTPException, Depends

from api.models import (
    PredictionRequest,
    BatchPredictionRequest,
    PredictionResponse,
    BatchPredictionResponse,
    ModelInfoResponse,
    LapFeatures,
)
from api.services.ml_service import ml_service
from api.auth import get_current_user

router = APIRouter(prefix="/predict", tags=["Predictions"])


@router.get("/model", response_model=ModelInfoResponse)
async def get_model_info(username: str = Depends(get_current_user)):
    """
    Get information about the currently loaded ML model.

    Returns model family, strategy, performance metrics, and source.
    """
    if not ml_service.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service is starting up."
        )

    info = ml_service.get_model_info()
    return ModelInfoResponse(
        model_family=info.get("model_family", "unknown"),
        strategy=info.get("strategy", "unknown"),
        run_id=info.get("run_id"),
        test_mae=info.get("test_mae"),
        test_r2=info.get("test_r2"),
        test_rmse=info.get("test_rmse"),
        overfitting_ratio=info.get("overfitting_ratio"),
        source=info.get("source", "unknown")
    )


@router.post("/lap", response_model=PredictionResponse)
async def predict_lap_time(request: PredictionRequest, username: str = Depends(get_current_user)):
    """
    Predict lap time for a single lap.

    Provide lap features (speeds, sector times, weather, context) and
    receive the predicted lap duration.

    **Example Request:**
    ```json
    {
        "features": {
            "st_speed": 310.5,
            "i1_speed": 295.2,
            "i2_speed": 288.1,
            "duration_sector_1": 28.5,
            "duration_sector_2": 35.2,
            "duration_sector_3": 26.8,
            "temp": 25.0,
            "rhum": 45.0,
            "pres": 1013.0,
            "lap_number": 15,
            "year": 2025,
            "circuit_avg_laptime": 92.5
        }
    }
    ```
    """
    if not ml_service.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service is starting up."
        )

    try:
        # Convert Pydantic model to dict
        features_dict = request.features.model_dump()

        # Make prediction
        prediction = ml_service.predict(features_dict)

        return PredictionResponse(
            lap_duration_seconds=round(prediction, 3),
            lap_duration_formatted=ml_service.format_lap_time(prediction),
            model_info=ml_service.get_model_info()
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest, username: str = Depends(get_current_user)):
    """
    Predict lap times for multiple laps in a single request.

    Accepts up to 1000 laps per request for efficient batch processing.

    **Example Request:**
    ```json
    {
        "features": [
            {"st_speed": 310.5, "i1_speed": 295.2, ...},
            {"st_speed": 308.2, "i1_speed": 292.1, ...}
        ]
    }
    ```
    """
    if not ml_service.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service is starting up."
        )

    try:
        # Convert Pydantic models to dicts
        features_list = [f.model_dump() for f in request.features]

        # Make batch predictions
        predictions = ml_service.predict_batch(features_list)

        return BatchPredictionResponse(
            predictions=[round(p, 3) for p in predictions],
            count=len(predictions),
            model_info=ml_service.get_model_info()
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


@router.post("/simple")
async def predict_simple(
    st_speed: float,
    i1_speed: float,
    i2_speed: float,
    duration_sector_1: float,
    duration_sector_2: float,
    duration_sector_3: float,
    temp: float = 25.0,
    rhum: float = 50.0,
    pres: float = 1013.0,
    lap_number: int = 1,
    year: int = 2025,
    circuit_avg_laptime: float = 90.0,
    username: str = Depends(get_current_user)
):
    """
    Simple prediction endpoint with query parameters.

    Useful for quick testing without building a JSON body.

    **Example:**
    ```
    POST /predict/simple?st_speed=310&i1_speed=295&i2_speed=288&duration_sector_1=28.5&duration_sector_2=35.2&duration_sector_3=26.8
    ```
    """
    if not ml_service.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service is starting up."
        )

    try:
        features = LapFeatures(
            st_speed=st_speed,
            i1_speed=i1_speed,
            i2_speed=i2_speed,
            duration_sector_1=duration_sector_1,
            duration_sector_2=duration_sector_2,
            duration_sector_3=duration_sector_3,
            temp=temp,
            rhum=rhum,
            pres=pres,
            lap_number=lap_number,
            year=year,
            circuit_avg_laptime=circuit_avg_laptime
        )

        prediction = ml_service.predict(features.model_dump())

        return {
            "lap_duration_seconds": round(prediction, 3),
            "lap_duration_formatted": ml_service.format_lap_time(prediction)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
