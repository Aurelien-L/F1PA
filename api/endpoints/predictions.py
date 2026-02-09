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
from api.services.db_service import db_service
from api.auth import get_current_user
from api.middleware.metrics import track_prediction, track_prediction_error

router = APIRouter(prefix="/predict", tags=["Predictions"])


def enrich_features(features_dict: dict) -> dict:
    """
    Enrich features with auto-calculated performance metrics if not provided.

    Auto-calculates from database:
    - circuit_avg_laptime: Average lap time for the circuit
    - driver_perf_score: Driver performance score (driver_circuit_avg - circuit_avg)
    - year: Fixed to 2025 (last training year) for hypothetical predictions

    This ensures predictions use accurate driver performance data instead of
    requiring users to manually provide these complex metrics.
    """
    driver_number = features_dict["driver_number"]
    circuit_key = features_dict["circuit_key"]

    # Auto-fill year to last training year (2025) for hypothetical predictions
    # This makes the API a prediction tool independent of the actual calendar year
    if features_dict.get("year") is None:
        features_dict["year"] = 2025

    # Auto-calculate circuit_avg_laptime if not provided
    if features_dict.get("circuit_avg_laptime") is None:
        circuit_avg = db_service.get_circuit_avg_laptime(circuit_key)
        if circuit_avg is not None:
            features_dict["circuit_avg_laptime"] = circuit_avg
        else:
            # Fallback to reasonable default
            features_dict["circuit_avg_laptime"] = 90.0

    # Auto-calculate driver_perf_score if not provided
    if features_dict.get("driver_perf_score") is None:
        perf_score = db_service.get_driver_perf_score(driver_number, circuit_key)
        features_dict["driver_perf_score"] = perf_score

    return features_dict


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
        model_family=info.get("model_family") or "unknown",
        strategy=info.get("strategy") or "unknown",
        run_name=info.get("run_name"),
        run_id=info.get("run_id"),
        test_mae=info.get("test_mae"),
        test_r2=info.get("test_r2"),
        test_rmse=info.get("test_rmse"),
        cv_mae=info.get("cv_mae"),
        cv_r2=info.get("cv_r2"),
        overfitting_ratio=info.get("overfitting_ratio"),
        source=info.get("source") or "unknown"
    )


@router.post("/lap", response_model=PredictionResponse)
async def predict_lap_time(request: PredictionRequest, username: str = Depends(get_current_user)):
    """
    Predict lap time for a driver on a circuit.

    The model predicts lap time BEFORE the lap starts, based on:
    - Expected speeds (st_speed, i1_speed, i2_speed)
    - Weather conditions (temp, rhum, pres)
    - Driver historical performance (auto-calculated)
    - Circuit characteristics (auto-calculated)

    **Example Request:**
    ```json
    {
        "features": {
            "driver_number": 1,
            "circuit_key": 7,
            "st_speed": 310.5,
            "i1_speed": 295.2,
            "i2_speed": 288.1,
            "temp": 25.0,
            "rhum": 45.0,
            "pres": 1013.0,
            "lap_number": 15
        }
    }
    ```

    **Notes:**
    - Year is fixed to 2025 (last training year) for hypothetical predictions
    - circuit_avg_laptime and driver_perf_score are optional (auto-calculated if omitted)
    - Sector times are NOT used (would make prediction trivial)
    """
    if not ml_service.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service is starting up."
        )

    try:
        # Convert Pydantic model to dict
        features_dict = request.features.model_dump()

        # Enrich with auto-calculated performance metrics
        features_dict = enrich_features(features_dict)

        # Make prediction with metrics tracking
        with track_prediction("single"):
            prediction = ml_service.predict(features_dict)

        return PredictionResponse(
            lap_duration_seconds=round(prediction, 3),
            lap_duration_formatted=ml_service.format_lap_time(prediction),
            model_info=ml_service.get_model_info()
        )

    except Exception as e:
        track_prediction_error("single")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest, username: str = Depends(get_current_user)):
    """
    Predict lap times for multiple laps in a single request.

    Efficient batch processing (up to 1000 laps per request).
    Same parameters as /lap endpoint for each feature set.
    """
    if not ml_service.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service is starting up."
        )

    try:
        # Convert Pydantic models to dicts
        features_list = [f.model_dump() for f in request.features]

        # Enrich each feature set with auto-calculated performance metrics
        features_list = [enrich_features(f) for f in features_list]

        # Make batch predictions with metrics tracking
        with track_prediction("batch"):
            predictions = ml_service.predict_batch(features_list)

        return BatchPredictionResponse(
            predictions=[round(p, 3) for p in predictions],
            count=len(predictions),
            model_info=ml_service.get_model_info()
        )

    except Exception as e:
        track_prediction_error("batch")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


@router.post("/simple")
async def predict_simple(
    driver_number: int,
    circuit_key: int,
    st_speed: float,
    i1_speed: float,
    i2_speed: float,
    temp: float = 25.0,
    rhum: float = 50.0,
    pres: float = 1013.0,
    lap_number: int = 1,
    circuit_avg_laptime: float = None,
    driver_perf_score: float = None,
    username: str = Depends(get_current_user)
):
    """
    Simple prediction endpoint with query parameters.

    Quick testing endpoint without JSON body. Year is fixed to 2025 (last training year).
    Performance metrics are auto-calculated if not provided.

    **Example:**
    ```
    POST /predict/simple?driver_number=1&circuit_key=7&st_speed=310&i1_speed=295&i2_speed=288
    ```
    """
    if not ml_service.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service is starting up."
        )

    try:
        features = LapFeatures(
            driver_number=driver_number,
            circuit_key=circuit_key,
            st_speed=st_speed,
            i1_speed=i1_speed,
            i2_speed=i2_speed,
            temp=temp,
            rhum=rhum,
            pres=pres,
            lap_number=lap_number,
            circuit_avg_laptime=circuit_avg_laptime,
            driver_perf_score=driver_perf_score
        )

        # Enrich with auto-calculated performance metrics
        features_dict = enrich_features(features.model_dump())

        prediction = ml_service.predict(features_dict)

        return {
            "lap_duration_seconds": round(prediction, 3),
            "lap_duration_formatted": ml_service.format_lap_time(prediction)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
