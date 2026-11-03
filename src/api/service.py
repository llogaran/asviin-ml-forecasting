"""Serving facade between the HTTP layer and the trained model (SPEC 4.7).

``ForecastService`` is the only object the API needs. It is intentionally lazy:
constructing it never touches the filesystem, so the API can boot and answer
``/health``/``/metrics`` before (or without) model artifacts being mounted. The
first call that needs data triggers :meth:`Forecaster.load` once, under a lock.

Loading failures are **not** cached: an operator can run ``make train`` while the
process is up and the next request will pick the new artifacts up. Domain errors
are delegated untouched: unknown products/branches raise ``KeyError`` (the API
maps them to 404) and invalid horizons raise ``ValueError`` (422). Only failures
at the artifact-loading boundary are wrapped in :class:`ModelUnavailableError`,
which the API maps to 503 with an actionable message.
"""

from __future__ import annotations

import logging
import threading
import unicodedata
from pathlib import Path
from typing import Any

from src.config import get_settings
from src.pipelines.forecast import Forecaster

LOGGER = logging.getLogger(__name__)

#: Fields searched by :meth:`ForecastService.list_products` (case/accent-insensitive).
SEARCHABLE_PRODUCT_FIELDS: tuple[str, ...] = (
    "product_id",
    "name",
    "category",
    "unit",
    "demand_pattern",
    "abc_class",
)


class ModelUnavailableError(RuntimeError):
    """Raised when the model artifacts cannot be loaded (mapped to HTTP 503)."""


def _fold(value: str) -> str:
    """Lowercase and strip accents so `materia prima` matches `Matería Prima`."""
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(char for char in decomposed if not unicodedata.combining(char)).casefold().strip()


class ForecastService:
    """Lazy, thread-safe serving facade over :class:`Forecaster`."""

    def __init__(
        self,
        data_dir: str | Path | None = None,
        artifacts_dir: str | Path | None = None,
    ) -> None:
        """Store the directories to load from; nothing is read until first use."""
        settings = get_settings()
        self.data_dir = Path(data_dir) if data_dir is not None else settings.data_dir
        self.artifacts_dir = (
            Path(artifacts_dir) if artifacts_dir is not None else settings.artifacts_dir
        )
        self._forecaster: Forecaster | None = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        """Whether the model has already been loaded in this process."""
        return self._forecaster is not None

    def _forecaster_or_raise(self) -> Forecaster:
        """Return the cached forecaster, loading it on first use (double-checked lock)."""
        forecaster = self._forecaster
        if forecaster is not None:
            return forecaster
        with self._lock:
            if self._forecaster is None:
                try:
                    self._forecaster = Forecaster.load(self.data_dir, self.artifacts_dir)
                except Exception as exc:  # any artifact problem becomes a 503, not a 500
                    raise ModelUnavailableError(self._unavailable_message(exc)) from exc
                LOGGER.info(
                    "Loaded forecaster from data_dir=%s artifacts_dir=%s (n_series=%s)",
                    self.data_dir,
                    self.artifacts_dir,
                    self._forecaster.model_info().get("n_series"),
                )
            return self._forecaster

    def _unavailable_message(self, exc: Exception) -> str:
        return (
            f"Forecast model is not available: {exc} "
            "Build the dataset and train the model first "
            "(`make data && make train`), or point MLFC_DATA_DIR "
            f"({self.data_dir}) and MLFC_ARTIFACTS_DIR ({self.artifacts_dir}) "
            "to existing artifacts."
        )

    def list_products(self, query: str | None = None) -> list[dict[str, Any]]:
        """Return the product catalog, optionally filtered by a free-text query."""
        products = self._forecaster_or_raise().products()
        needle = _fold(query or "")
        if not needle:
            return products
        return [
            product
            for product in products
            if any(
                needle in _fold(str(product.get(field, "")))
                for field in SEARCHABLE_PRODUCT_FIELDS
            )
        ]

    def forecast(self, product_id: str, branch_id: str, horizon_days: int = 28) -> dict[str, Any]:
        """Delegate to :meth:`Forecaster.forecast` after ensuring the model is loaded."""
        return self._forecaster_or_raise().forecast(product_id, branch_id, horizon_days)

    def recommend(
        self,
        product_id: str,
        branch_id: str,
        horizon_days: int = 30,
        service_level: float = 0.95,
    ) -> dict[str, Any]:
        """Delegate to :meth:`Forecaster.recommend` after ensuring the model is loaded."""
        return self._forecaster_or_raise().recommend(
            product_id, branch_id, horizon_days, service_level
        )

    def model_info(self) -> dict[str, Any]:
        """Return registry metadata for the loaded champion model."""
        return self._forecaster_or_raise().model_info()
