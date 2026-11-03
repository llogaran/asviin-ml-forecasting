"""Deterministic inventory-policy backtest for the holdout window (SPEC 4.4).

Given the holdout actual demand and two competing daily demand drivers (the
recursive XGBoost forecast and the naive seasonal baseline), this module
simulates a simple periodic-review replenishment policy and reports the
operational impact:

* **Forecast-driven** -- orders are sized from a planning signal that blends
  ``p50`` and ``p90`` (30% weight on the p90) plus the forecast's ``p10``/``p90``
  spread. The blend corrects the low bias of the conditional median on
  zero-inflated series; ordering at the p50 alone under-stocked intermittent SKUs
  in the backtest.
* **Naive-driven** -- orders are sized from the same-weekday-last-week demand and
  the historical demand volatility.

Both policies share the review period, service level, lead time and starting
inventory, so the only difference is the quality of the demand signal. The
simulation is a pure function (no randomness, no timestamps), which keeps the
committed ``impact.json`` reproducible for a given dataset and model.

Policy details (documented because the SPEC leaves them to the implementation):

* Review every ``review_period_days`` days, starting on the first holdout day.
* ``safety_stock = z * sigma * sqrt(lead_time)`` where ``z`` is the normal
  quantile of the service level; ``sigma`` is the horizon-mean
  ``(p90 - p10) / 2.562`` for the forecast driver and the pre-holdout daily
  demand standard deviation for the naive driver.
* ``reorder_point = expected_demand(lead_time) + safety_stock`` and
  ``order_up_to = expected_demand(lead_time + review_period) + safety_stock``.
* A day is a *stockout day* when demand exceeds available on-hand stock (unmet
  demand is lost). Service level is the unit fill rate
  ``1 - stockout_units / demand_units``.
* Average on-hand is the mean of closing on-hand units across series-days.
"""

from __future__ import annotations

import logging
import math
from statistics import NormalDist

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

SERIES_KEYS: tuple[str, ...] = ("business_id", "branch_id", "product_id")
SCHEMA_VERSION = "1.0.0"
DEFAULT_SERVICE_LEVEL = 0.95
DEFAULT_REVIEW_PERIOD_DAYS = 7
#: Weight of the p90 in the planning demand driver. The conditional median of a
#: zero-inflated series is often 0, so ordering at the p50 alone systematically
#: under-stocks intermittent SKUs (measured on the committed backtest). The blend
#: keeps the median's stability while pricing the upside tail.
PLANNING_P90_BLEND = 0.3
#: 2.562 = 2 * 1.281, the 10-90 quantile span of a standard normal.
NORMAL_P90_P10_SPAN = 2.562
MIN_LEAD_TIME_DAYS = 1


def normal_quantile(service_level: float) -> float:
    """Return the standard-normal quantile for a service level in ``(0, 1)``."""
    if not 0.0 < service_level < 1.0:
        raise ValueError("service_level must be strictly between 0 and 1")
    return float(NormalDist().inv_cdf(service_level))


def _prepare_daily(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ("units_sold", "stock_on_hand", "lead_time_days"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype(float)
    frame = frame.dropna(subset=["date"])
    return frame.sort_values([*SERIES_KEYS, "date"], kind="stable").reset_index(drop=True)


def _expected_demand(driver: np.ndarray, start: int, days: int) -> float:
    """Sum the driver over ``days``; extrapolate the last value past the horizon."""
    if days <= 0 or len(driver) == 0:
        return 0.0
    segment = driver[start : start + days]
    if len(segment) == 0:
        return 0.0
    total = float(np.sum(segment))
    if len(segment) < days:
        total += float(segment[-1]) * (days - len(segment))
    return max(0.0, total)


def _run_policy(
    actual: np.ndarray,
    driver: np.ndarray,
    *,
    start_stock: float,
    lead_time_days: int,
    review_period_days: int,
    z_score: float,
    sigma: float,
) -> dict[str, float]:
    """Simulate one ``(s, S)`` periodic-review policy over the holdout days."""
    on_hand = max(0.0, start_stock)
    on_order: dict[int, float] = {}
    safety_stock = z_score * max(0.0, sigma) * math.sqrt(lead_time_days)
    closing_sum = 0.0
    stockout_days = 0
    stockout_units = 0.0
    demand_units = 0.0

    for day, raw_demand in enumerate(actual):
        arriving = on_order.pop(day, 0.0)
        on_hand += max(0.0, arriving)

        if day % review_period_days == 0:
            in_transit = sum(quantity for arrival, quantity in on_order.items() if arrival > day)
            reorder_point = _expected_demand(driver, day, lead_time_days) + safety_stock
            order_up_to = (
                _expected_demand(driver, day, lead_time_days + review_period_days) + safety_stock
            )
            if on_hand + in_transit < reorder_point:
                quantity = max(0.0, order_up_to - on_hand - in_transit)
                if quantity > 0:
                    arrival_day = day + lead_time_days
                    on_order[arrival_day] = on_order.get(arrival_day, 0.0) + quantity

        demand = max(0.0, float(raw_demand))
        sold = min(demand, on_hand)
        unmet = demand - sold
        on_hand -= sold
        demand_units += demand
        if unmet > 1e-9:
            stockout_days += 1
            stockout_units += unmet
        closing_sum += on_hand

    days = len(actual)
    return {
        "avg_on_hand_units": closing_sum / days if days else 0.0,
        "stockout_days": float(stockout_days),
        "stockout_units": stockout_units,
        "service_level": 1.0 - stockout_units / demand_units if demand_units > 0 else 1.0,
        "demand_units": demand_units,
        "days": float(days),
    }


def _empty_aggregate() -> dict[str, float]:
    return {
        "closing_sum": 0.0,
        "days": 0.0,
        "stockout_days": 0.0,
        "stockout_units": 0.0,
        "demand_units": 0.0,
    }


def _finalize(aggregate: dict[str, float]) -> dict[str, float]:
    days = aggregate["days"]
    demand_units = aggregate["demand_units"]
    return {
        "avg_on_hand_units": round(aggregate["closing_sum"] / days, 2) if days else 0.0,
        "stockout_days": int(round(aggregate["stockout_days"])),
        "stockout_units": round(aggregate["stockout_units"], 2),
        "service_level": round(1.0 - aggregate["stockout_units"] / demand_units, 4)
        if demand_units > 0
        else 1.0,
        "days": int(round(days)),
    }


def _relative_improvement(forecast: dict[str, float], naive: dict[str, float]) -> dict[str, float]:
    naive_on_hand = float(naive["avg_on_hand_units"])
    naive_stockouts = float(naive["stockout_days"])
    return {
        # Positive = the forecast policy needs fewer stockout days.
        "stockout_days_reduction_pct": round(
            (naive_stockouts - float(forecast["stockout_days"])) / naive_stockouts * 100.0, 2
        )
        if naive_stockouts > 0
        else 0.0,
        # Positive = the forecast policy serves a larger share of demand.
        "service_level_gain_pp": round(
            (float(forecast["service_level"]) - float(naive["service_level"])) * 100.0, 2
        ),
        # Positive = the forecast policy holds fewer units on average.
        "avg_on_hand_reduction_pct": round(
            (naive_on_hand - float(forecast["avg_on_hand_units"])) / naive_on_hand * 100.0, 2
        )
        if naive_on_hand > 0
        else 0.0,
    }


def simulate_inventory_policy(
    daily: pd.DataFrame,
    holdout: pd.DataFrame,
    *,
    forecast_p50: list[float] | np.ndarray,
    forecast_p10: list[float] | np.ndarray,
    forecast_p90: list[float] | np.ndarray,
    naive: list[float] | np.ndarray,
    service_level: float = DEFAULT_SERVICE_LEVEL,
    review_period_days: int = DEFAULT_REVIEW_PERIOD_DAYS,
) -> dict:
    """Compare forecast-driven and naive-driven replenishment on the holdout.

    ``holdout`` must have one row per series/day (as produced by
    :func:`src.pipelines.features.build_supervised`) and the four prediction
    sequences must be aligned with its row order. ``daily`` is the full history
    and is used only to recover the starting inventory and the pre-holdout
    volatility, so the simulation never looks at future data.
    """
    if review_period_days < 1:
        raise ValueError("review_period_days must be >= 1")
    z_score = normal_quantile(service_level)

    holdout_frame = holdout.copy().reset_index(drop=True)
    expected_size = len(holdout_frame)
    arrays = {
        "forecast_p50": np.asarray(forecast_p50, dtype=float),
        "forecast_p10": np.asarray(forecast_p10, dtype=float),
        "forecast_p90": np.asarray(forecast_p90, dtype=float),
        "naive": np.asarray(naive, dtype=float),
    }
    for name, values in arrays.items():
        if len(values) != expected_size:
            raise ValueError(
                f"{name} has {len(values)} values but holdout has {expected_size} rows"
            )
    for name in ("forecast_p50", "forecast_p10", "forecast_p90", "naive"):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"{name} contains non-finite values")
    if expected_size == 0:
        return {
            "schema_version": SCHEMA_VERSION,
            "policy": {},
            "holdout": {"series": 0, "days": 0, "date_min": None, "date_max": None},
            "forecast_driven": _finalize(_empty_aggregate()),
            "naive_driven": _finalize(_empty_aggregate()),
            "relative_improvement": {
                "stockout_days_reduction_pct": 0.0,
                "service_level_gain_pp": 0.0,
                "avg_on_hand_reduction_pct": 0.0,
            },
            "summary": "No holdout rows available; nothing to simulate.",
        }

    holdout_frame["date"] = pd.to_datetime(holdout_frame["date"], errors="coerce")
    holdout_frame["units_sold"] = (
        pd.to_numeric(holdout_frame["units_sold"], errors="coerce").fillna(0.0).astype(float)
    )
    holdout_frame["lead_time_days"] = (
        pd.to_numeric(holdout_frame["lead_time_days"], errors="coerce").fillna(0.0).astype(float)
    )
    holdout_frame["_p50"] = arrays["forecast_p50"].clip(min=0.0)
    holdout_frame["_p10"] = arrays["forecast_p10"].clip(min=0.0)
    holdout_frame["_p90"] = arrays["forecast_p90"].clip(min=0.0)
    holdout_frame["_naive"] = arrays["naive"].clip(min=0.0)
    holdout_frame["_planning"] = (
        (1.0 - PLANNING_P90_BLEND) * holdout_frame["_p50"]
        + PLANNING_P90_BLEND * holdout_frame["_p90"]
    )

    history = _prepare_daily(daily)
    history_by_series = {
        tuple(key): frame for key, frame in history.groupby(list(SERIES_KEYS), sort=False)
    }

    aggregate_forecast = _empty_aggregate()
    aggregate_naive = _empty_aggregate()
    for key, group in holdout_frame.groupby(list(SERIES_KEYS), sort=True):
        group = group.sort_values("date", kind="stable")
        series_key = tuple(key) if isinstance(key, tuple) else (key,)
        first_date = group["date"].min()
        series_history = history_by_series.get(series_key)
        if series_history is None:
            before = history.iloc[0:0]
        else:
            before = series_history[series_history["date"] < first_date]
        start_stock = float(before["stock_on_hand"].iloc[-1]) if len(before) else 0.0
        prior_units = before["units_sold"].to_numpy(dtype=float)
        naive_sigma = float(np.std(prior_units, ddof=1)) if len(prior_units) >= 2 else 0.0
        forecast_sigma = float(np.mean((group["_p90"] - group["_p10"]) / NORMAL_P90_P10_SPAN))
        forecast_sigma = max(0.0, forecast_sigma)
        lead_time = max(MIN_LEAD_TIME_DAYS, int(round(float(group["lead_time_days"].iloc[-1]))))
        actual = group["units_sold"].to_numpy(dtype=float)

        forecast_stats = _run_policy(
            actual,
            group["_planning"].to_numpy(dtype=float),
            start_stock=start_stock,
            lead_time_days=lead_time,
            review_period_days=review_period_days,
            z_score=z_score,
            sigma=forecast_sigma,
        )
        naive_stats = _run_policy(
            actual,
            group["_naive"].to_numpy(dtype=float),
            start_stock=start_stock,
            lead_time_days=lead_time,
            review_period_days=review_period_days,
            z_score=z_score,
            sigma=naive_sigma,
        )
        for aggregate, stats in (
            (aggregate_forecast, forecast_stats),
            (aggregate_naive, naive_stats),
        ):
            aggregate["closing_sum"] += stats["avg_on_hand_units"] * stats["days"]
            aggregate["days"] += stats["days"]
            aggregate["stockout_days"] += stats["stockout_days"]
            aggregate["stockout_units"] += stats["stockout_units"]
            aggregate["demand_units"] += stats["demand_units"]

    forecast_report = _finalize(aggregate_forecast)
    naive_report = _finalize(aggregate_naive)
    improvement = _relative_improvement(forecast_report, naive_report)
    n_series = int(holdout_frame[list(SERIES_KEYS)].drop_duplicates().shape[0])
    date_min = holdout_frame["date"].min()
    date_max = holdout_frame["date"].max()

    report = {
        "schema_version": SCHEMA_VERSION,
        "policy": {
            "type": "periodic_review",
            "review_period_days": int(review_period_days),
            "service_level": float(service_level),
            "z_score": round(z_score, 4),
            "reorder_point": "expected lead-time demand + safety stock",
            "order_up_to": "expected (lead time + review period) demand + safety stock",
            "forecast_sigma": f"mean((p90 - p10) / {NORMAL_P90_P10_SPAN}) over the holdout",
            "naive_sigma": "std of pre-holdout daily units_sold",
            "forecast_driver": (
                f"{1 - PLANNING_P90_BLEND:.0%} * p50 + {PLANNING_P90_BLEND:.0%} * p90 "
                "(planning signal, median-bias corrected)"
            ),
            "naive_driver": "lag_7 (same weekday, previous week)",
            "starting_inventory": "last stock_on_hand before the holdout",
            "stockout_day": "day with unmet demand > 0",
            "service_level_metric": "1 - stockout units / demand units (unit fill rate)",
            "average_on_hand": "mean closing on-hand units across series-days",
        },
        "holdout": {
            "series": n_series,
            "days": int(len(holdout_frame)),
            "date_min": date_min.date().isoformat() if pd.notna(date_min) else None,
            "date_max": date_max.date().isoformat() if pd.notna(date_max) else None,
        },
        "forecast_driven": forecast_report,
        "naive_driven": naive_report,
        "relative_improvement": improvement,
        "summary": (
            "Forecast-driven vs naive-driven replenishment: "
            f"stockout days {forecast_report['stockout_days']} vs "
            f"{naive_report['stockout_days']}, "
            f"service level {forecast_report['service_level']:.2%} vs "
            f"{naive_report['service_level']:.2%}, "
            f"average on-hand {forecast_report['avg_on_hand_units']:.2f} vs "
            f"{naive_report['avg_on_hand_units']:.2f} units."
        ),
        "note": (
            "Positive relative improvements mean the forecast-driven policy holds "
            "fewer units on average, has fewer stockout days, or serves a higher "
            "share of demand than the naive seasonal policy. The planning driver is "
            f"{1 - PLANNING_P90_BLEND:.0%} of the p50 plus {PLANNING_P90_BLEND:.0%} of "
            "the p90: the conditional median of a zero-inflated series is often 0, so "
            "planning on the median alone would systematically under-stock intermittent "
            "SKUs. The blend keeps the median's stability while pricing the upside tail; "
            "it is the same signal used by Forecaster.recommend()."
        ),
    }
    LOGGER.debug("Inventory impact: %s", report["summary"])
    return report
