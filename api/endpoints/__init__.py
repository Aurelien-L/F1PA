"""F1PA API Endpoints."""
from .predictions import router as predictions_router
from .data import router as data_router

__all__ = ["predictions_router", "data_router"]
