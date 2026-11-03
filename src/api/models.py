"""Pydantic request/response models for the public HTTP API (SPEC 4.7).

These schemas are the single source of truth for the HTTP contract. Request
models are strict (``extra="forbid"``): a misspelled field fails with 422 instead
of being silently ignored, which is much easier to debug for an ERP integrator.
Response models mirror exactly the dictionaries produced by
:class:`src.pipelines.forecast.Forecaster`; the API layer never invents fields,
it only validates and filters them.

Response envelopes are deliberately small:

* ``GET /products`` returns ``{"query", "count", "products"}`` instead of a bare
  list so the result can grow (pagination, branches) without breaking clients.
* ``GET /health`` returns only ``{"status": "ok"}`` because it must stay usable
  even when the model is missing; readiness is reported by ``GET /ready``.
"""

from __future__ import annotations

import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

#: JSON schema example reused by the two inference endpoints' request models.
_EXAMPLE_PRODUCT_ID = "P-001"
_EXAMPLE_BRANCH_ID = "B-01"


class ForecastRequest(BaseModel):
    """Body of ``POST /forecast``."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "product_id": _EXAMPLE_PRODUCT_ID,
                "branch_id": _EXAMPLE_BRANCH_ID,
                "horizon_days": 28,
            }
        },
    )

    product_id: str = Field(min_length=1, max_length=64, description="Product id, e.g. `P-001`.")
    branch_id: str = Field(min_length=1, max_length=64, description="Branch id, e.g. `B-01`.")
    horizon_days: int = Field(
        default=28,
        ge=1,
        le=90,
        description="Forecast horizon in calendar days (1-90).",
    )


class RecommendationRequest(BaseModel):
    """Body of ``POST /recommendation``."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "product_id": _EXAMPLE_PRODUCT_ID,
                "branch_id": _EXAMPLE_BRANCH_ID,
                "horizon_days": 30,
                "service_level": 0.95,
            }
        },
    )

    product_id: str = Field(min_length=1, max_length=64, description="Product id, e.g. `P-001`.")
    branch_id: str = Field(min_length=1, max_length=64, description="Branch id, e.g. `B-01`.")
    horizon_days: int = Field(
        default=30,
        ge=1,
        le=90,
        description="Horizon used to estimate demand and uncertainty (1-90 days).",
    )
    service_level: float = Field(
        default=0.95,
        gt=0.0,
        lt=1.0,
        description="Target cycle service level, strictly between 0 and 1.",
    )


class ForecastPoint(BaseModel):
    """One forecasted day with its 10/50/90 quantiles in product units."""

    date: datetime.date = Field(description="Forecasted calendar day (ISO-8601 date).")
    p50: float = Field(ge=0.0, description="Median (p50) forecast in units.")
    p10: float = Field(ge=0.0, description="Conservative (p10) forecast in units.")
    p90: float = Field(ge=0.0, description="Optimistic (p90) forecast in units.")


class ForecastResponse(BaseModel):
    """Response of ``POST /forecast`` (SPEC 4.5)."""

    product_id: str
    branch_id: str
    model: str
    model_version: str
    generated_at: str = Field(description="UTC timestamp of the inference run.")
    horizon_days: int = Field(ge=1, le=90)
    points: list[ForecastPoint]


class RecommendationResponse(BaseModel):
    """Response of ``POST /recommendation`` (SPEC 4.5)."""

    product_id: str
    branch_id: str
    model: str
    model_version: str
    generated_at: str = Field(description="UTC timestamp of the inference run.")
    horizon_days: int = Field(ge=1, le=90)
    service_level: float = Field(gt=0.0, lt=1.0)
    lead_time_days: int = Field(ge=0)
    daily_mean: float = Field(ge=0.0, description="Mean daily demand over the horizon.")
    sigma: float = Field(ge=0.0, description="Mean daily demand standard deviation.")
    daily_mean_units: float = Field(ge=0.0, description="Alias of `daily_mean` (explicit units).")
    sigma_units: float = Field(ge=0.0, description="Alias of `sigma` (explicit units).")
    z_score: float = Field(description="Standard-normal quantile of the service level.")
    safety_stock: float = Field(description="z * sigma * sqrt(lead_time).")
    reorder_point: float = Field(description="daily_mean * lead_time + safety_stock.")
    current_stock: float = Field(ge=0.0, description="Latest closing stock on hand.")
    reorder_quantity: float = Field(ge=0.0, description="max(0, reorder_point - current_stock).")
    days_of_cover: float | None = Field(
        default=None, ge=0.0, description="current_stock / daily_mean, or null when demand is zero."
    )
    rationale: str = Field(description="Human-readable explanation of the calculation.")


class ProductSummary(BaseModel):
    """Product catalog entry exposed by ``GET /products``."""

    product_id: str
    name: str
    category: str
    unit: str
    price_mxn: float = Field(ge=0.0)
    lead_time_days: int = Field(ge=0)
    demand_pattern: str
    abc_class: str
    is_ingredient: bool


class ProductListResponse(BaseModel):
    """Envelope returned by ``GET /products``."""

    query: str | None = Field(default=None, description="Echo of the `q` filter, if any.")
    count: int = Field(ge=0)
    products: list[ProductSummary]


class ModelInfoResponse(BaseModel):
    """Response of ``GET /model`` (registry metadata, SPEC 5)."""

    model: str
    model_version: str
    champion: str
    schema_version: str
    artifact: str
    trained_at: str | None = None
    holdout_days: int | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    baselines: dict[str, Any] = Field(default_factory=dict)
    dataset: dict[str, Any] = Field(default_factory=dict)
    features: list[str] = Field(default_factory=list)
    target_transform: str
    quick: bool
    n_series: int = Field(ge=0)


class HealthResponse(BaseModel):
    """Response of ``GET /health``; never depends on the model artifacts."""

    status: Literal["ok"] = "ok"


class ReadyResponse(BaseModel):
    """Response of ``GET /ready`` once the model has been loaded successfully."""

    status: Literal["ready"] = "ready"
    model: str
    model_version: str


class MetricsResponse(BaseModel):
    """Response of ``GET /metrics``: process-local counters since app creation."""

    requests_total: int = Field(ge=0)
    forecast_requests: int = Field(ge=0)
    recommendation_requests: int = Field(ge=0)
    errors_total: int = Field(ge=0)
    uptime_seconds: float = Field(ge=0.0)


class ErrorResponse(BaseModel):
    """Standard FastAPI error body used by 401/404/422/503 responses."""

    detail: str
