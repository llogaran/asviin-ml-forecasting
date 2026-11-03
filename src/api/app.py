"""FastAPI application factory for the Asviin ML Forecasting API (SPEC 4.7).

The app is a thin HTTP adapter over :class:`src.api.service.ForecastService`: it
owns authentication, CORS, request validation, error mapping and metrics, and it
never loads model artifacts itself. ``create_app`` accepts an optional service so
tests (and the agent, if needed) can inject a small fixture model.

Endpoint map
------------
``GET  /health``         liveness, always open and independent from the model
``GET  /ready``          readiness, forces the lazy model load (503 if missing)
``GET  /model``          champion registry metadata
``GET  /products?q=``    product catalog with an optional free-text filter
``POST /forecast``       recursive p10/p50/p90 forecast
``POST /recommendation`` reorder-point recommendation
``GET  /metrics``        JSON counters (requests, errors, uptime)

Design notes
------------
* The module-level ``app`` only exists for ``uvicorn src.api.app:app``. It is
  built from :func:`src.config.get_settings` at import time but does not touch
  the model until the first request (the service is lazy).
* When ``settings.api_key`` is set, every path except ``/health`` (including
  ``/docs`` and ``/openapi.json``) requires ``X-API-Key``. CORS is registered
  after the auth/metrics middleware, so preflight requests are answered by the
  CORS layer and 401 responses still carry CORS headers.
* Metrics are process-local counters created per app instance (no external
  monitoring dependency) and reset on restart; a ``/metrics`` request is counted
  after its own snapshot is taken.
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
from collections.abc import Callable
from typing import Annotated, Any, TypeVar

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src import __version__
from src.config import get_settings

from .models import (
    ErrorResponse,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    MetricsResponse,
    ModelInfoResponse,
    ProductListResponse,
    ProductSummary,
    ReadyResponse,
    RecommendationRequest,
    RecommendationResponse,
)
from .service import ForecastService, ModelUnavailableError

LOGGER = logging.getLogger(__name__)

HEALTH_PATH = "/health"
AUTH_ERROR_DETAIL = "Missing or invalid API key: send the X-API-Key header."
T = TypeVar("T")

#: OpenAPI documentation for the cross-cutting error responses.
STANDARD_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    401: {"model": ErrorResponse, "description": "Missing or invalid API key."},
    404: {"model": ErrorResponse, "description": "Unknown product or branch."},
    503: {"model": ErrorResponse, "description": "Model artifacts are not available."},
}


def _normalize_path(path: str) -> str:
    """Normalize a request path so ``/health/`` is treated like ``/health``."""
    return path.rstrip("/") or "/"


def _api_key_valid(provided: str | None, expected: str) -> bool:
    """Constant-time comparison of the header value against the configured key."""
    if not provided:
        return False
    return secrets.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


def _error_detail(exc: KeyError) -> str:
    """Avoid the extra quotes ``str(KeyError(...))`` adds around the message."""
    return str(exc.args[0]) if exc.args else str(exc)


def _translate(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Map service/forecaster exceptions to the HTTP status codes of SPEC 4.7."""
    try:
        return func(*args, **kwargs)
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=_error_detail(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


class _Metrics:
    """Thread-safe in-process request counters (one instance per app)."""

    def __init__(self) -> None:
        self._started_at = time.monotonic()
        self._lock = threading.Lock()
        self.requests_total = 0
        self.forecast_requests = 0
        self.recommendation_requests = 0
        self.errors_total = 0

    def record(self, path: str, status_code: int) -> None:
        """Count one completed request and, when applicable, the endpoint bucket."""
        with self._lock:
            self.requests_total += 1
            if path == "/forecast":
                self.forecast_requests += 1
            elif path == "/recommendation":
                self.recommendation_requests += 1
            if status_code >= 400:
                self.errors_total += 1

    def snapshot(self) -> dict[str, int | float]:
        """Return a consistent copy of the counters plus the app uptime."""
        with self._lock:
            return {
                "requests_total": self.requests_total,
                "forecast_requests": self.forecast_requests,
                "recommendation_requests": self.recommendation_requests,
                "errors_total": self.errors_total,
                "uptime_seconds": round(time.monotonic() - self._started_at, 3),
            }


def create_app(service: ForecastService | None = None) -> FastAPI:
    """Build a configured FastAPI application (SPEC 4.7).

    Args:
        service: Optional service to inject; by default a lazy service is created
            from the ``MLFC_DATA_DIR``/``MLFC_ARTIFACTS_DIR`` settings.
    """
    settings = get_settings()
    forecast_service = service if service is not None else ForecastService()
    metrics = _Metrics()

    app = FastAPI(
        title="Asviin ML Forecasting API",
        version=__version__,
        summary="Demand forecasting and inventory recommendations for the Asviin ERP.",
        description=__doc__,
    )

    async def auth_and_metrics(request: Request, call_next: Callable[[Request], Any]) -> Any:
        """Enforce the optional API key, then count the completed request."""
        path = _normalize_path(request.url.path)
        if (
            settings.api_key is not None
            and path != HEALTH_PATH
            and request.method != "OPTIONS"
            and not _api_key_valid(request.headers.get("X-API-Key"), settings.api_key)
        ):
            metrics.record(path, 401)
            return JSONResponse(
                status_code=401,
                content={"detail": AUTH_ERROR_DETAIL},
                headers={"WWW-Authenticate": "X-API-Key"},
            )
        try:
            response = await call_next(request)
        except Exception:
            metrics.record(path, 500)
            raise
        metrics.record(path, response.status_code)
        return response

    # Middleware added later wraps the earlier ones: CORS stays outermost so that
    # preflight requests never require an API key and every response (including
    # 401s from the auth guard) gets the CORS headers.
    app.middleware("http")(auth_and_metrics)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins) or ["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health() -> HealthResponse:
        """Liveness probe: always open and independent from the model artifacts."""
        return HealthResponse(status="ok")

    @app.get("/ready", response_model=ReadyResponse, responses=STANDARD_ERROR_RESPONSES, tags=["ops"])
    def ready() -> ReadyResponse:
        """Readiness probe: loads the model lazily and returns 503 when unavailable."""
        info = _translate(forecast_service.model_info)
        return ReadyResponse(
            status="ready",
            model=str(info["model"]),
            model_version=str(info["model_version"]),
        )

    @app.get("/model", response_model=ModelInfoResponse, responses=STANDARD_ERROR_RESPONSES, tags=["model"])
    def model_info() -> ModelInfoResponse:
        """Return the champion model registry metadata and training metrics."""
        info = _translate(forecast_service.model_info)
        return ModelInfoResponse.model_validate(info)

    @app.get(
        "/products",
        response_model=ProductListResponse,
        responses=STANDARD_ERROR_RESPONSES,
        tags=["catalog"],
    )
    def products(
        q: Annotated[
            str | None,
            Query(max_length=100, description="Case/accent-insensitive filter on id, name or category."),
        ] = None,
    ) -> ProductListResponse:
        """List products known to the training dataset, optionally filtered by ``q``."""
        rows = _translate(forecast_service.list_products, q)
        return ProductListResponse(
            query=q,
            count=len(rows),
            products=[ProductSummary.model_validate(row) for row in rows],
        )

    @app.post(
        "/forecast",
        response_model=ForecastResponse,
        responses=STANDARD_ERROR_RESPONSES,
        tags=["inference"],
    )
    def forecast(payload: ForecastRequest) -> ForecastResponse:
        """Forecast p10/p50/p90 units per day for a product at a branch."""
        result = _translate(
            forecast_service.forecast,
            payload.product_id,
            payload.branch_id,
            horizon_days=payload.horizon_days,
        )
        return ForecastResponse.model_validate(result)

    @app.post(
        "/recommendation",
        response_model=RecommendationResponse,
        responses=STANDARD_ERROR_RESPONSES,
        tags=["inference"],
    )
    def recommendation(payload: RecommendationRequest) -> RecommendationResponse:
        """Recommend a reorder quantity from the horizon forecast and current stock."""
        result = _translate(
            forecast_service.recommend,
            payload.product_id,
            payload.branch_id,
            horizon_days=payload.horizon_days,
            service_level=payload.service_level,
        )
        return RecommendationResponse.model_validate(result)

    @app.get("/metrics", response_model=MetricsResponse, tags=["ops"])
    def metrics_endpoint() -> MetricsResponse:
        """Return process-local request counters (reset when the process restarts)."""
        return MetricsResponse.model_validate(metrics.snapshot())

    return app


#: ASGI entry point for ``uvicorn src.api.app:app`` (lazy: no artifacts at import).
app = create_app()
