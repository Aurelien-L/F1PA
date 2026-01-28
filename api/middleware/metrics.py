"""
F1PA API - Prometheus Metrics Middleware

Instruments the FastAPI application with Prometheus metrics for monitoring.
"""
import time
from contextlib import contextmanager
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

# =============================================================================
# METRICS DEFINITIONS
# =============================================================================

# HTTP Requests
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

# ML Predictions
f1pa_predictions_total = Counter(
    "f1pa_predictions_total",
    "Total number of ML predictions",
    ["endpoint_type"]  # lap, batch
)

f1pa_prediction_duration_seconds = Histogram(
    "f1pa_prediction_duration_seconds",
    "ML prediction duration in seconds",
    ["endpoint_type"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)
)

f1pa_prediction_errors_total = Counter(
    "f1pa_prediction_errors_total",
    "Total number of prediction errors",
    ["error_type"]
)

# Model Status
f1pa_model_loaded = Gauge(
    "f1pa_model_loaded",
    "Whether the ML model is loaded (1) or not (0)"
)

# Database Status
f1pa_database_connected = Gauge(
    "f1pa_database_connected",
    "Whether database is connected (1) or not (0)"
)

f1pa_db_query_duration_seconds = Histogram(
    "f1pa_db_query_duration_seconds",
    "Database query duration in seconds",
    ["query_type"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
)

# MLflow Status
f1pa_mlflow_connected = Gauge(
    "f1pa_mlflow_connected",
    "Whether MLflow is connected (1) or not (0)"
)

# Data Endpoints
f1pa_data_queries_total = Counter(
    "f1pa_data_queries_total",
    "Total data endpoint queries",
    ["endpoint"]
)

# =============================================================================
# MIDDLEWARE
# =============================================================================

class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware to collect Prometheus metrics for all HTTP requests."""

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        # Skip metrics endpoint itself
        if request.url.path == "/metrics":
            return await call_next(request)

        # Start timer
        start_time = time.time()

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration = time.time() - start_time

        # Record metrics
        endpoint = request.url.path
        method = request.method
        status = response.status_code

        # Increment request counter
        http_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status=status
        ).inc()

        # Record request duration
        http_request_duration_seconds.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)

        return response


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@contextmanager
def track_prediction(endpoint_type: str):
    """Track a prediction with automatic timing."""
    start_time = time.time()
    try:
        yield
        duration = time.time() - start_time
        f1pa_predictions_total.labels(endpoint_type=endpoint_type).inc()
        f1pa_prediction_duration_seconds.labels(endpoint_type=endpoint_type).observe(duration)
    except Exception:
        duration = time.time() - start_time
        f1pa_prediction_duration_seconds.labels(endpoint_type=endpoint_type).observe(duration)
        raise


def track_prediction_error(error_type: str):
    """Track a prediction error."""
    f1pa_prediction_errors_total.labels(error_type=error_type).inc()


def update_model_status(loaded: bool):
    """Update model loaded status."""
    f1pa_model_loaded.set(1 if loaded else 0)


def update_database_status(connected: bool):
    """Update database connection status."""
    f1pa_database_connected.set(1 if connected else 0)


def update_mlflow_status(connected: bool):
    """Update MLflow connection status."""
    f1pa_mlflow_connected.set(1 if connected else 0)


@contextmanager
def track_db_query(query_type: str):
    """Track a database query with automatic timing."""
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        f1pa_db_query_duration_seconds.labels(query_type=query_type).observe(duration)


def track_data_query(endpoint: str):
    """Track data endpoint queries."""
    f1pa_data_queries_total.labels(endpoint=endpoint).inc()


async def metrics_endpoint(request: Request):
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
