"""Serving-side recursive forecaster and inventory recommendation (SPEC 4.5).

The API and the agent never load XGBoost directly: they call this class, which
owns the train/serve contract. The forecast walks the horizon one day at a time:
:func:`src.pipelines.features.features_for_next_day` builds the next feature row
from the latest available history, the three quantile models predict
``p10``/``p50``/``p90``, and the ``p50`` is appended to the history so the
following day can reuse the freshly predicted demand. Inventory is depleted by
the predicted sales (no replenishment is assumed inside a short horizon), which
keeps the ``stock_to_demand`` feature realistic.

The recommendation follows the SPEC formula exactly::

    sigma = mean((p90 - p10) / 2.562) over the horizon
    z = NormalDist().inv_cdf(service_level)
    safety_stock = z * sigma * sqrt(lead_time)
    reorder_point = daily_mean * lead_time + safety_stock
    reorder_quantity = max(0, reorder_point - current_stock)

All public lookups are deterministic. Unknown products/branches raise ``KeyError``
so HTTP adapters can map them to 404, and horizons outside ``1..90`` raise
``ValueError``.
"""

from __future__ import annotations

import json
import logging
import math
import warnings
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.data_contracts.validation import load_csv
from src.pipelines.features import FEATURES, SERIES_KEYS, features_for_next_day

LOGGER = logging.getLogger(__name__)

MODEL_NAME = "xgboost_global"
ARTIFACT_FILENAME = "xgboost_global.joblib"
REGISTRY_FILENAME = "registry.json"
DATA_FILENAME = Path("processed") / "demand_daily.csv"
DEFAULT_SCHEMA_VERSION = "1.0.0"
TARGET_TRANSFORM = "log1p"
MIN_HORIZON_DAYS = 1
MAX_HORIZON_DAYS = 90
#: 2.562 = 2 * 1.281, the 10-90 quantile span of a standard normal.
NORMAL_P90_P10_SPAN = 2.562
#: Weight of the p90 in the planning demand signal used by ``recommend``.
#: The p50 of a zero-inflated series is often 0, which understates expected
#: demand; blending 30% of the p90 corrects the low bias without paying the
#: inventory cost of ordering at the p90. See docs/DECISIONS.md (ADR-009).
PLANNING_P90_BLEND = 0.3


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_optional_csv(path: Path) -> pd.DataFrame | None:
    """Load optional master data without making serving depend on the raw files."""
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        LOGGER.warning("Ignoring unreadable master file %s: %s", path, exc)
        return None


def _normalize_daily(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any():
        raise ValueError("demand_daily.csv contains unparseable dates")
    numeric = (
        "units_sold",
        "revenue_mxn",
        "cost_mxn",
        "cancelled_units",
        "pending_units",
        "stock_on_hand",
        "stockout_flag",
        "promo",
        "unit_price_mxn",
        "unit_cost_mxn",
        "lead_time_days",
    )
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame["product_id"] = frame["product_id"].astype(str)
    frame["branch_id"] = frame["branch_id"].astype(str)
    frame["business_id"] = frame["business_id"].astype(str)
    return frame


class Forecaster:
    """Load a trained global model and serve forecasts/recommendations."""

    def __init__(
        self,
        *,
        daily: pd.DataFrame,
        models: dict[str, Any],
        meta: dict[str, Any],
        registry: dict[str, Any],
        products_master: pd.DataFrame | None = None,
        branches_master: pd.DataFrame | None = None,
    ) -> None:
        self._daily = _normalize_daily(daily)
        self._models = models
        self._meta = meta
        self._registry = registry
        self.model_version = str(
            registry.get("schema_version") or meta.get("model_version") or DEFAULT_SCHEMA_VERSION
        )

        self._series: dict[tuple[str, str, str], pd.DataFrame] = {}
        self._pair_index: dict[tuple[str, str], tuple[str, str, str]] = {}
        for key, group in self._daily.groupby(list(SERIES_KEYS), sort=True):
            series_key = tuple(key) if isinstance(key, tuple) else (key,)
            ordered = group.sort_values("date", kind="stable").reset_index(drop=True)
            self._series[series_key] = ordered  # type: ignore[index]
            self._pair_index[(str(series_key[2]), str(series_key[1]))] = series_key  # type: ignore[index]
        self._product_ids = {key[2] for key in self._series}
        self._branch_ids = {key[1] for key in self._series}
        self._products = self._build_products(products_master)
        self._branches = self._build_branches(branches_master)

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, data_dir: Path | str, artifacts_dir: Path | str) -> Forecaster:
        """Load the model, registry and daily panel; fail with actionable errors."""
        data_path = Path(data_dir)
        artifact_path = Path(artifacts_dir)
        daily_path = data_path / DATA_FILENAME
        model_path = artifact_path / "models" / ARTIFACT_FILENAME
        registry_path = artifact_path / "models" / REGISTRY_FILENAME

        for path, hint in (
            (daily_path, "build the dataset first (run `make data` or the ETL)"),
            (model_path, "train a model first (run `make train`)"),
            (registry_path, "train a model first (run `make train`)"),
        ):
            if not path.exists():
                raise FileNotFoundError(f"Missing {path}; {hint}.")

        daily = load_csv(daily_path, "demand_daily")
        with warnings.catch_warnings():
            # Some sandboxed environments cannot access /dev/shm and joblib
            # prints an absolute-path warning on load. It is environmental noise,
            # not a model problem, and would leak local paths into CLI output.
            warnings.filterwarnings("ignore", message=".*joblib will operate in serial mode.*")
            payload = joblib.load(model_path)
        if not isinstance(payload, dict) or "models" not in payload or "meta" not in payload:
            raise ValueError(f"Malformed model artifact {model_path}; retrain with `make train`.")
        models = payload["models"]
        missing = {"p10", "p50", "p90"} - set(models)
        if missing:
            raise ValueError(
                f"Model artifact {model_path} is missing quantiles {sorted(missing)}; "
                "retrain with `make train`."
            )
        meta = payload["meta"]
        if not isinstance(meta, dict) or "features" not in meta:
            raise ValueError(
                f"Model artifact {model_path} has no feature metadata; retrain with `make train`."
            )
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid registry {registry_path}: {exc}; retrain with `make train`."
            ) from exc

        return cls(
            daily=daily,
            models=models,
            meta=meta,
            registry=registry,
            products_master=_read_optional_csv(data_path / "raw" / "products.csv"),
            branches_master=_read_optional_csv(data_path / "raw" / "branches.csv"),
        )

    # ------------------------------------------------------------------ metadata
    def _build_products(self, master: pd.DataFrame | None) -> list[dict[str, Any]]:
        master_by_id: dict[str, pd.Series] = {}
        if master is not None and "product_id" in master.columns:
            master_by_id = {str(row["product_id"]): row for _, row in master.iterrows()}
        products: list[dict[str, Any]] = []
        for product_id, group in self._daily.groupby("product_id", sort=True):
            last = group.sort_values("date", kind="stable").iloc[-1]
            row = master_by_id.get(str(product_id))
            price = float(last["unit_price_mxn"])
            name = str(product_id)
            is_ingredient = bool(last.get("is_ingredient", 0))
            if row is not None:
                if pd.notna(row.get("name")):
                    name = str(row["name"])
                if pd.notna(row.get("price_mxn")):
                    price = float(row["price_mxn"])
                if "is_ingredient" in row.index and pd.notna(row["is_ingredient"]):
                    is_ingredient = bool(int(row["is_ingredient"]))
            products.append(
                {
                    "product_id": str(product_id),
                    "name": name,
                    "category": str(last["category"]),
                    "unit": str(last["unit"]),
                    "price_mxn": round(price, 2),
                    "lead_time_days": int(round(float(last["lead_time_days"]))),
                    "demand_pattern": str(last["demand_pattern"]),
                    "abc_class": str(last["abc_class"]),
                    "is_ingredient": is_ingredient,
                }
            )
        return products

    def _build_branches(self, master: pd.DataFrame | None) -> list[dict[str, Any]]:
        master_by_id: dict[str, pd.Series] = {}
        if master is not None and "branch_id" in master.columns:
            master_by_id = {str(row["branch_id"]): row for _, row in master.iterrows()}
        branches: list[dict[str, Any]] = []
        for branch_id, group in self._daily.groupby("branch_id", sort=True):
            row = master_by_id.get(str(branch_id))
            name = str(branch_id)
            region = ""
            if row is not None:
                if pd.notna(row.get("name")):
                    name = str(row["name"])
                if pd.notna(row.get("region")):
                    region = str(row["region"])
            branches.append(
                {
                    "branch_id": str(branch_id),
                    "business_id": str(group["business_id"].iloc[0]),
                    "name": name,
                    "region": region,
                }
            )
        return branches

    def products(self) -> list[dict[str, Any]]:
        """Return the product catalog known to the loaded dataset."""
        return [dict(product) for product in self._products]

    def branches(self) -> list[dict[str, Any]]:
        """Return the branches known to the loaded dataset."""
        return [dict(branch) for branch in self._branches]

    def model_info(self) -> dict[str, Any]:
        """Return the registry metadata for the champion model."""
        entry = self._registry.get("models", {}).get(MODEL_NAME, {})
        return {
            "model": MODEL_NAME,
            "model_version": self.model_version,
            "champion": self._registry.get("champion", MODEL_NAME),
            "schema_version": self._registry.get("schema_version", DEFAULT_SCHEMA_VERSION),
            "artifact": ARTIFACT_FILENAME,
            "trained_at": entry.get("trained_at") or self._meta.get("trained_at"),
            "holdout_days": entry.get("holdout_days", self._meta.get("holdout_days")),
            "metrics": entry.get("metrics", {}),
            "baselines": entry.get("baselines", {}),
            "dataset": self._registry.get("dataset", {}),
            "features": list(self._meta.get("features") or FEATURES),
            "target_transform": self._meta.get("target_transform", TARGET_TRANSFORM),
            "quick": bool(self._meta.get("quick", False)),
            "n_series": len(self._series),
        }

    # ------------------------------------------------------------------ internals
    def _require_series(self, product_id: str, branch_id: str) -> tuple[str, str, str]:
        product = str(product_id)
        branch = str(branch_id)
        if product not in self._product_ids:
            known = ", ".join(sorted(self._product_ids)[:10])
            raise KeyError(f"Unknown product_id '{product}'. Known products: {known}")
        if branch not in self._branch_ids:
            known = ", ".join(sorted(self._branch_ids))
            raise KeyError(f"Unknown branch_id '{branch}'. Known branches: {known}")
        pair = (product, branch)
        if pair not in self._pair_index:
            raise KeyError(f"No history for product '{product}' at branch '{branch}'")
        return self._pair_index[pair]

    @staticmethod
    def _validate_horizon(horizon_days: int) -> int:
        if isinstance(horizon_days, bool) or not isinstance(horizon_days, (int, np.integer)):
            raise ValueError("horizon_days must be an integer between 1 and 90")
        horizon = int(horizon_days)
        if not MIN_HORIZON_DAYS <= horizon <= MAX_HORIZON_DAYS:
            raise ValueError(
                f"horizon_days must be between {MIN_HORIZON_DAYS} and {MAX_HORIZON_DAYS}; "
                f"got {horizon}"
            )
        return horizon

    @staticmethod
    def _validate_service_level(service_level: float) -> float:
        level = float(service_level)
        if not 0.0 < level < 1.0:
            raise ValueError("service_level must be strictly between 0 and 1")
        return level

    def _predict_step(self, feature_row: pd.Series) -> dict[str, float]:
        features = list(self._meta.get("features") or FEATURES)
        matrix = np.asarray(
            [[float(feature_row.get(name, 0.0)) for name in features]], dtype=np.float32
        )
        raw = {
            name: float(np.asarray(model.predict(matrix), dtype=float)[0])
            for name, model in self._models.items()
        }
        lower = min(raw["p10"], raw["p90"])
        upper = max(raw["p10"], raw["p90"])
        middle = min(max(raw["p50"], lower), upper)
        return {
            "p10": max(0.0, float(np.expm1(lower))),
            "p50": max(0.0, float(np.expm1(middle))),
            "p90": max(0.0, float(np.expm1(upper))),
        }

    def _append_synthetic_day(
        self, history: pd.DataFrame, next_date: pd.Timestamp, planning_units: float
    ) -> pd.DataFrame:
        """Extend the history with a predicted day so the recursion can continue.

        The recursion advances with the **planning signal** (the same
        70/30 p50/p90 blend used by :meth:`recommend`), not with the raw
        median. Appending the median makes a zero-inflated series feed zeros
        back into its own lags and the forecast collapses to zero after a few
        days; the blend keeps the simulated history demographically consistent
        with the series' observed demand level.
        """
        last = history.iloc[-1]
        row = last.to_dict()
        price = max(0.0, float(last.get("unit_price_mxn", 0.0)))
        cost = max(0.0, float(last.get("unit_cost_mxn", 0.0)))
        units = max(0.0, float(planning_units))
        # Stock is kept flat on purpose. Subtracting every forecasted day from
        # the synthetic history fabricates a stock-out after a few steps; the
        # model then reads ``stockout_flag`` and ``stock_to_demand`` as
        # "demand is censored" and collapses the forecast. This endpoint
        # forecasts unconstrained demand; the replenishment policy lives in
        # ``recommend`` and in the impact simulation.
        stock = max(0.0, float(last.get("stock_on_hand", 0.0)))
        row.update(
            {
                "date": pd.Timestamp(next_date),
                "units_sold": units,
                "revenue_mxn": units * price,
                "cost_mxn": units * cost,
                "cancelled_units": 0.0,
                "pending_units": 0.0,
                "stock_on_hand": stock,
                "stockout_flag": 0,
                "promo": 0,
            }
        )
        return pd.concat([history, pd.DataFrame([row])], ignore_index=True)

    def _raw_forecast(self, series_key: tuple[str, str, str], horizon: int) -> list[dict[str, Any]]:
        history = self._series[series_key].copy()
        points: list[dict[str, Any]] = []
        for _ in range(horizon):
            feature_row = features_for_next_day(history, self._meta)
            quantiles = self._predict_step(feature_row)
            next_date = pd.Timestamp(history["date"].iloc[-1]) + pd.Timedelta(days=1)
            points.append(
                {
                    "date": next_date.date().isoformat(),
                    "p50": quantiles["p50"],
                    "p10": quantiles["p10"],
                    "p90": quantiles["p90"],
                }
            )
            planning_units = (
                (1.0 - PLANNING_P90_BLEND) * quantiles["p50"]
                + PLANNING_P90_BLEND * quantiles["p90"]
            )
            history = self._append_synthetic_day(history, next_date, planning_units)
        return points

    # ------------------------------------------------------------------- public
    def forecast(self, product_id: str, branch_id: str, horizon_days: int = 28) -> dict:
        """Recursive multi-step forecast with p10/p50/p90 per day."""
        series_key = self._require_series(product_id, branch_id)
        horizon = self._validate_horizon(horizon_days)
        raw_points = self._raw_forecast(series_key, horizon)
        points = [
            {
                "date": point["date"],
                "p50": round(float(point["p50"]), 2),
                "p10": round(float(point["p10"]), 2),
                "p90": round(float(point["p90"]), 2),
            }
            for point in raw_points
        ]
        return {
            "product_id": series_key[2],
            "branch_id": series_key[1],
            "model": MODEL_NAME,
            "model_version": self.model_version,
            "generated_at": _utc_now(),
            "horizon_days": horizon,
            "points": points,
        }

    def recommend(
        self,
        product_id: str,
        branch_id: str,
        horizon_days: int = 30,
        service_level: float = 0.95,
    ) -> dict:
        """Translate the horizon forecast into a reorder recommendation."""
        series_key = self._require_series(product_id, branch_id)
        horizon = self._validate_horizon(horizon_days)
        level = self._validate_service_level(service_level)
        points = self._raw_forecast(series_key, horizon)

        p10 = np.asarray([point["p10"] for point in points], dtype=float)
        p50 = np.asarray([point["p50"] for point in points], dtype=float)
        p90 = np.asarray([point["p90"] for point in points], dtype=float)
        # Median of a zero-inflated series is often 0, so planning on the p50
        # alone under-stocks intermittent SKUs. The blend keeps the median's
        # stability while pricing the chance of a high-demand day.
        planning = (1.0 - PLANNING_P90_BLEND) * p50 + PLANNING_P90_BLEND * p90
        daily_mean = float(planning.mean())
        sigma = max(0.0, float(np.mean((p90 - p10) / NORMAL_P90_P10_SPAN)))
        z_score = float(NormalDist().inv_cdf(level))

        last = self._series[series_key].iloc[-1]
        lead_time = max(0, int(round(float(last["lead_time_days"]))))
        current_stock = max(0.0, float(last["stock_on_hand"]))
        safety_stock = z_score * sigma * math.sqrt(lead_time)
        reorder_point = daily_mean * lead_time + safety_stock
        reorder_quantity = max(0.0, reorder_point - current_stock)
        days_of_cover = current_stock / daily_mean if daily_mean > 0 else None

        safety_stock = round(safety_stock, 2)
        reorder_point = round(reorder_point, 2)
        reorder_quantity = round(reorder_quantity, 2)
        daily_mean = round(daily_mean, 2)
        sigma = round(sigma, 2)
        current_stock = round(current_stock, 2)
        z_score = round(z_score, 4)
        days_of_cover = round(days_of_cover, 2) if days_of_cover is not None else None
        rationale = (
            f"Plan demand blends {1 - PLANNING_P90_BLEND:.0%} of the median with "
            f"{PLANNING_P90_BLEND:.0%} of the p90 to correct median bias on intermittent "
            f"SKUs. At a {level:.0%} service level the forecast averages {daily_mean:.2f} "
            f"units/day with uncertainty {sigma:.2f} (z={z_score:.3f}). Lead time is "
            f"{lead_time} days, so safety stock = {z_score:.3f} * {sigma:.2f} * "
            f"sqrt({lead_time}) = {safety_stock:.2f} and the reorder point = "
            f"{daily_mean:.2f} * {lead_time} + {safety_stock:.2f} = {reorder_point:.2f}. "
            f"Current stock is {current_stock:.2f}, therefore the recommended reorder "
            f"quantity is {reorder_quantity:.2f} units."
        )
        return {
            "product_id": series_key[2],
            "branch_id": series_key[1],
            "model": MODEL_NAME,
            "model_version": self.model_version,
            "generated_at": _utc_now(),
            "horizon_days": horizon,
            "service_level": level,
            "lead_time_days": lead_time,
            # ``daily_mean``/``sigma`` mirror the SPEC's mu/sigma notation; the
            # unit-suffixed keys are kept as explicit aliases for API clients.
            "daily_mean": daily_mean,
            "sigma": sigma,
            "daily_mean_units": daily_mean,
            "sigma_units": sigma,
            "z_score": z_score,
            "safety_stock": safety_stock,
            "reorder_point": reorder_point,
            "current_stock": current_stock,
            "reorder_quantity": reorder_quantity,
            "days_of_cover": days_of_cover,
            "rationale": rationale,
        }
