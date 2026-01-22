"""
F1PA API - Data Access Endpoints (C5)

Endpoints for accessing F1 lap data from the database.
"""
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query, Depends

from api.models import (
    CircuitResponse,
    DriverResponse,
    SessionResponse,
    LapResponse,
    PaginatedResponse,
    DatasetStatsResponse,
)
from api.services.db_service import db_service
from api.auth import get_current_user

router = APIRouter(prefix="/data", tags=["Data Access"])


# =============================================================================
# STATISTICS
# =============================================================================

@router.get("/stats", response_model=DatasetStatsResponse)
async def get_dataset_stats(username: str = Depends(get_current_user)):
    """
    Get overall dataset statistics.

    Returns total counts for laps, circuits, drivers, sessions,
    available years, and date range.
    """
    if not db_service.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Database not connected."
        )

    try:
        stats = db_service.get_dataset_stats()
        return DatasetStatsResponse(**stats)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get statistics: {str(e)}"
        )


# =============================================================================
# CIRCUITS
# =============================================================================

@router.get("/circuits", response_model=List[CircuitResponse])
async def get_circuits(username: str = Depends(get_current_user)):
    """
    Get all circuits in the database.

    Returns circuit key, name, location, and country information.
    """
    if not db_service.is_ready():
        raise HTTPException(status_code=503, detail="Database not connected.")

    try:
        circuits = db_service.get_circuits()
        return [CircuitResponse(**c) for c in circuits]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get circuits: {str(e)}")


@router.get("/circuits/{circuit_key}", response_model=CircuitResponse)
async def get_circuit(circuit_key: int, username: str = Depends(get_current_user)):
    """
    Get a specific circuit by its key.
    """
    if not db_service.is_ready():
        raise HTTPException(status_code=503, detail="Database not connected.")

    try:
        circuit = db_service.get_circuit(circuit_key)
        if circuit is None:
            raise HTTPException(status_code=404, detail=f"Circuit {circuit_key} not found")
        return CircuitResponse(**circuit)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get circuit: {str(e)}")


@router.get("/circuits/{circuit_key}/laps", response_model=List[LapResponse])
async def get_circuit_laps(
    circuit_key: int,
    year: Optional[int] = Query(None, description="Filter by year"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of laps to return"),
    username: str = Depends(get_current_user)
):
    """
    Get laps for a specific circuit, sorted by fastest lap time.

    Useful for analyzing circuit characteristics and finding fastest laps.
    """
    if not db_service.is_ready():
        raise HTTPException(status_code=503, detail="Database not connected.")

    try:
        laps = db_service.get_circuit_laps(circuit_key, year=year, limit=limit)
        return [LapResponse(**lap) for lap in laps]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get circuit laps: {str(e)}")


@router.get("/circuits/{circuit_key}/avg-laptime")
async def get_circuit_avg_laptime(circuit_key: int, username: str = Depends(get_current_user)):
    """
    Get average lap time for a circuit.

    This value can be used as the `circuit_avg_laptime` feature for predictions.
    """
    if not db_service.is_ready():
        raise HTTPException(status_code=503, detail="Database not connected.")

    try:
        avg_laptime = db_service.get_circuit_avg_laptime(circuit_key)
        if avg_laptime is None:
            raise HTTPException(status_code=404, detail=f"No laps found for circuit {circuit_key}")
        return {
            "circuit_key": circuit_key,
            "avg_laptime_seconds": round(avg_laptime, 3),
            "avg_laptime_formatted": f"{int(avg_laptime // 60)}:{avg_laptime % 60:06.3f}"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get average lap time: {str(e)}")


# =============================================================================
# DRIVERS
# =============================================================================

@router.get("/drivers", response_model=List[DriverResponse])
async def get_drivers(username: str = Depends(get_current_user)):
    """
    Get all drivers in the database.

    Returns driver number, name, acronym, team, and country.
    """
    if not db_service.is_ready():
        raise HTTPException(status_code=503, detail="Database not connected.")

    try:
        drivers = db_service.get_drivers()
        return [DriverResponse(**d) for d in drivers]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get drivers: {str(e)}")


@router.get("/drivers/{driver_number}", response_model=DriverResponse)
async def get_driver(driver_number: int, username: str = Depends(get_current_user)):
    """
    Get a specific driver by their race number.
    """
    if not db_service.is_ready():
        raise HTTPException(status_code=503, detail="Database not connected.")

    try:
        driver = db_service.get_driver(driver_number)
        if driver is None:
            raise HTTPException(status_code=404, detail=f"Driver {driver_number} not found")
        return DriverResponse(**driver)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get driver: {str(e)}")


@router.get("/drivers/{driver_number}/laps", response_model=List[LapResponse])
async def get_driver_laps(
    driver_number: int,
    year: Optional[int] = Query(None, description="Filter by year"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of laps to return"),
    username: str = Depends(get_current_user)
):
    """
    Get laps for a specific driver.

    Returns recent laps for the driver, useful for driver performance analysis.
    """
    if not db_service.is_ready():
        raise HTTPException(status_code=503, detail="Database not connected.")

    try:
        laps = db_service.get_driver_laps(driver_number, year=year, limit=limit)
        return [LapResponse(**lap) for lap in laps]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get driver laps: {str(e)}")


# =============================================================================
# SESSIONS
# =============================================================================

@router.get("/sessions", response_model=List[SessionResponse])
async def get_sessions(
    year: Optional[int] = Query(None, description="Filter by year"),
    circuit_key: Optional[int] = Query(None, description="Filter by circuit"),
    limit: int = Query(100, ge=1, le=500, description="Maximum number of sessions"),
    username: str = Depends(get_current_user)
):
    """
    Get race sessions with optional filters.

    Returns session information including date, circuit, and session type.
    """
    if not db_service.is_ready():
        raise HTTPException(status_code=503, detail="Database not connected.")

    try:
        sessions = db_service.get_sessions(year=year, circuit_key=circuit_key, limit=limit)
        return [SessionResponse(**s) for s in sessions]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get sessions: {str(e)}")


# =============================================================================
# LAPS (Main Data Endpoint)
# =============================================================================

@router.get("/laps", response_model=PaginatedResponse)
async def get_laps(
    year: Optional[int] = Query(None, description="Filter by year (2023, 2024, 2025)"),
    circuit_key: Optional[int] = Query(None, description="Filter by circuit key"),
    driver_number: Optional[int] = Query(None, description="Filter by driver number"),
    session_key: Optional[int] = Query(None, description="Filter by session key"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(100, ge=1, le=1000, description="Items per page"),
    username: str = Depends(get_current_user)
):
    """
    Get lap data with flexible filtering and pagination.

    This is the main endpoint for accessing the F1 lap dataset.
    Supports filtering by year, circuit, driver, and session.

    **Example Requests:**
    - All laps: `GET /data/laps`
    - 2025 laps: `GET /data/laps?year=2025`
    - Driver 1 in 2024: `GET /data/laps?driver_number=1&year=2024`
    - Paginated: `GET /data/laps?page=2&page_size=50`
    """
    if not db_service.is_ready():
        raise HTTPException(status_code=503, detail="Database not connected.")

    try:
        result = db_service.get_laps(
            year=year,
            circuit_key=circuit_key,
            driver_number=driver_number,
            session_key=session_key,
            page=page,
            page_size=page_size
        )

        return PaginatedResponse(
            data=[LapResponse(**lap).model_dump() for lap in result["data"]],
            total=result["total"],
            page=result["page"],
            page_size=result["page_size"],
            total_pages=result["total_pages"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get laps: {str(e)}")
