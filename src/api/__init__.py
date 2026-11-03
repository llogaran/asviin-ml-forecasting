"""HTTP serving layer for the Asviin ML Forecasting API (SPEC 4.7).

The public surface is :func:`create_app` (application factory), the module-level
``app`` used by ``uvicorn src.api.app:app``, and :class:`ForecastService` (the
injectable serving facade). Request/response schemas live in ``models.py``.
"""

from .app import app, create_app
from .models import (
    ForecastRequest,
    ForecastResponse,
    RecommendationRequest,
    RecommendationResponse,
)
from .service import ForecastService, ModelUnavailableError

__all__ = [
    "ForecastRequest",
    "ForecastResponse",
    "ForecastService",
    "ModelUnavailableError",
    "RecommendationRequest",
    "RecommendationResponse",
    "app",
    "create_app",
]
