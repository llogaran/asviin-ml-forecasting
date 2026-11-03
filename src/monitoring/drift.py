"""Distribution-drift monitoring for the canonical daily-demand dataset (SPEC 4.6).

A demand model degrades silently when the distribution of its inputs moves. This
module compares a *reference* window with a recent *current* window using two
complementary statistics:

* Population Stability Index (PSI) over 10 reference-quantile bins. It is easy to
  explain to operations staff and is robust to the small counts of a per-product
  series.
* Two-sample Kolmogorov-Smirnov test (``scipy.stats.ks_2samp``), which catches
  shape changes that coarse PSI bins can miss.

Design decisions where SPEC 4.6 leaves room (documented per SPEC 0):

* PSI bins are the deciles of the reference window. The outer bins are open-ended
  (``-inf``/``+inf``) so current values outside the reference range are counted
  instead of silently dropped. Bin shares are clipped to ``1e-6`` to keep the
  logarithm finite, which yields a large PSI when a bin empties or appears.
* Degenerate references are handled explicitly: near-duplicate quantile edges are
  collapsed (a constant flag otherwise yields a spurious edge), a two-valued
  reference falls back to a midpoint threshold, and a truly constant reference is
  compared on the share of values equal to that constant. Without this, flags such
  as ``promo`` or ``stockout_flag`` would always report PSI 0.
* ``ks_2samp`` uses the asymptotic method. Monitoring windows are large and the
  data has many ties (counts and flags), where the exact method is slow and can
  warn; the asymptotic p-value is deterministic.
* ``features=None`` checks the numeric defaults of SPEC 4.6. Identifier, date and
  categorical columns are never compared.
* The CLI defaults to a **seasonal split**: the last ``window_days`` days are
  compared with the same calendar window one year earlier. A plain 70/30
  chronological split across three years confounds seasonality and growth with
  drift and flags the reference dataset itself; year-over-year compares like
  with like. If the file is shorter than the required history, the CLI falls
  back to the chronological split and records ``window_mode`` in the report.
* The report timestamp is the only non-deterministic value and is emitted as an
  artifact field in UTC, as allowed by SPEC 6. No paths are embedded in reports.

Public API: :func:`detect_drift`, the CLI (:func:`main`), plus the reusable
:func:`population_stability_index` and :func:`chronological_split` helpers.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.data_contracts.validation import SchemaValidationError, load_csv

logger = logging.getLogger(__name__)

DEFAULT_FEATURES: tuple[str, ...] = (
    "units_sold",
    "revenue_mxn",
    "pending_units",
    "stock_on_hand",
    "unit_price_mxn",
    "promo",
    "stockout_flag",
)

PSI_BINS = 10
PSI_DRIFT_THRESHOLD = 0.2
PSI_WARNING_THRESHOLD = 0.1
KS_DRIFT_PVALUE = 0.01
KS_WARNING_PVALUE = 0.05
DRIFT_SHARE_THRESHOLD = 0.30
REFERENCE_FRACTION = 0.70

_EPSILON = 1e-6
_EDGE_RTOL = 1e-12
_SCHEMA_VERSION = "1.0.0"

THRESHOLDS: dict[str, float | int] = {
    "psi_drift": PSI_DRIFT_THRESHOLD,
    "psi_warning": PSI_WARNING_THRESHOLD,
    "ks_pvalue_drift": KS_DRIFT_PVALUE,
    "ks_pvalue_warning": KS_WARNING_PVALUE,
    "drift_share": DRIFT_SHARE_THRESHOLD,
    "psi_bins": PSI_BINS,
}


def _utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp (seconds resolution, ``Z`` suffix)."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _finite_values(column: pd.Series) -> np.ndarray:
    """Coerce a column to float and return only its finite values."""
    numeric = pd.to_numeric(column, errors="coerce").astype("float64").to_numpy()
    return numeric[np.isfinite(numeric)]


def _dedupe_edges(quantiles: np.ndarray, scale: float) -> np.ndarray:
    """Drop near-duplicate quantile edges that only add empty bins."""
    tolerance = _EDGE_RTOL * scale
    edges: list[float] = [float(quantiles[0])]
    for edge in quantiles[1:]:
        candidate = float(edge)
        if candidate - edges[-1] > tolerance:
            edges.append(candidate)
    return np.asarray(edges, dtype="float64")


def _reference_edges(reference: np.ndarray, bins: int) -> np.ndarray:
    """Build PSI bin edges from the reference distribution.

    Returns quantile boundaries (min and max included). A single returned value
    signals a constant reference, which the caller treats as a two-state indicator.
    """
    quantiles = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, bins + 1)))
    scale = float(np.max(np.abs(reference)))
    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0
    edges = _dedupe_edges(quantiles, scale)
    if edges.size >= 3:
        return edges

    # Collapsed deciles (discrete or heavily skewed feature): cut between the
    # observed reference values instead of returning a single, useless bin.
    support = np.unique(reference)
    if support.size < 2:
        return edges
    midpoints = (support[:-1] + support[1:]) / 2.0
    if midpoints.size > bins - 1:
        picks = np.rint(np.linspace(0, midpoints.size - 1, bins - 1)).astype(int)
        midpoints = midpoints[picks]
    return np.concatenate(([support[0]], midpoints, [support[-1]]))


def population_stability_index(
    reference: Sequence[float] | np.ndarray, current: Sequence[float] | np.ndarray,
    bins: int = PSI_BINS,
) -> float:
    """Compute PSI of ``current`` against 10 quantile bins learned on ``reference``.

    A value below 0.1 is commonly considered stable, 0.1-0.2 a warning and at or
    above 0.2 a material shift. Non-finite values are ignored; at least one finite
    value is required on each side.
    """
    ref = np.asarray(reference, dtype="float64")
    cur = np.asarray(current, dtype="float64")
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if bins < 2:
        raise ValueError("bins must be at least 2")
    if ref.size == 0 or cur.size == 0:
        raise ValueError("PSI requires at least one finite value in both samples")

    edges = _reference_edges(ref, bins)
    if edges.size < 2:
        constant = float(edges[0])
        equal_share = float(np.mean(cur == constant))
        ref_share = np.asarray([1.0, 0.0])
        cur_share = np.asarray([equal_share, 1.0 - equal_share])
    else:
        breaks = np.concatenate(([-np.inf], edges[1:-1], [np.inf]))
        ref_counts = np.histogram(ref, bins=breaks)[0].astype("float64")
        cur_counts = np.histogram(cur, bins=breaks)[0].astype("float64")
        ref_share = ref_counts / ref_counts.sum()
        cur_share = cur_counts / cur_counts.sum()

    ref_share = np.clip(ref_share, _EPSILON, None)
    cur_share = np.clip(cur_share, _EPSILON, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def _status_for(psi_value: float, ks_pvalue: float) -> str:
    """Map PSI and KS p-value to ``ok`` | ``warning`` | ``drift`` (SPEC 4.6)."""
    if psi_value >= PSI_DRIFT_THRESHOLD or ks_pvalue < KS_DRIFT_PVALUE:
        return "drift"
    if psi_value >= PSI_WARNING_THRESHOLD or ks_pvalue < KS_WARNING_PVALUE:
        return "warning"
    return "ok"


def _ro2(value: float, digits: int = 6) -> float:
    """Round a statistic for a stable, JSON-friendly report."""
    return round(float(value), digits)


def _compare_feature(reference: pd.Series, current: pd.Series, name: str) -> dict:
    """Compare one numeric feature and return its statistics and status."""
    ref_values = _finite_values(reference)
    cur_values = _finite_values(current)
    if ref_values.size == 0:
        raise ValueError(f"Feature '{name}' has no finite numeric values in the reference frame")
    if cur_values.size == 0:
        raise ValueError(f"Feature '{name}' has no finite numeric values in the current frame")

    psi_value = population_stability_index(ref_values, cur_values)
    ks_result = stats.ks_2samp(
        ref_values, cur_values, alternative="two-sided", method="asymp"
    )
    ks_pvalue = float(ks_result.pvalue)
    if not np.isfinite(ks_pvalue):
        ks_pvalue = 1.0

    return {
        "status": _status_for(psi_value, ks_pvalue),
        "psi": _ro2(psi_value),
        "ks_statistic": _ro2(float(ks_result.statistic)),
        "ks_pvalue": _ro2(ks_pvalue),
        "reference_count": int(ref_values.size),
        "current_count": int(cur_values.size),
        "reference_missing": int(len(reference) - ref_values.size),
        "current_missing": int(len(current) - cur_values.size),
        "reference_mean": _ro2(float(np.mean(ref_values))),
        "current_mean": _ro2(float(np.mean(cur_values))),
        "reference_std": _ro2(float(np.std(ref_values))),
        "current_std": _ro2(float(np.std(cur_values))),
    }


def _resolve_features(
    reference: pd.DataFrame, current: pd.DataFrame, features: Sequence[str] | str | None
) -> list[str]:
    """Validate the requested feature list against both frames."""
    if features is None:
        selected = list(DEFAULT_FEATURES)
    elif isinstance(features, str):
        selected = [features]
    else:
        selected = list(features)
    selected = list(dict.fromkeys(selected))
    if not selected:
        raise ValueError("features must contain at least one column name")

    missing_reference = [name for name in selected if name not in reference.columns]
    missing_current = [name for name in selected if name not in current.columns]
    if missing_reference or missing_current:
        problems = []
        if missing_reference:
            problems.append(f"reference missing {missing_reference}")
        if missing_current:
            problems.append(f"current missing {missing_current}")
        raise ValueError("missing required columns: " + "; ".join(problems))
    return selected


def _date_window(frame: pd.DataFrame) -> dict[str, str] | None:
    """Summarise the date span of a frame when a usable ``date`` column exists."""
    if "date" not in frame.columns:
        return None
    dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
    if dates.empty:
        return None
    return {
        "date_min": dates.min().date().isoformat(),
        "date_max": dates.max().date().isoformat(),
    }


def detect_drift(
    reference: pd.DataFrame, current: pd.DataFrame, features: list[str] | None = None
) -> dict:
    """Detect distribution drift between a reference and a current DataFrame.

    Args:
        reference: Earlier window of the daily-demand table (any DataFrame with the
            requested numeric columns; missing values are ignored per feature).
        current: More recent window with the same feature columns.
        features: Optional column subset. Defaults to the SPEC 4.6 numeric features.

    Returns:
        A JSON-serialisable report with the overall ``status`` (``drift`` when at
        least 30% of features drift, else ``warning`` when any feature warns or
        drifts, else ``ok``), ``drift_share``, thresholds, row counts, per-feature
        statistics and a UTC timestamp.

    Raises:
        TypeError: If an argument is not a DataFrame.
        ValueError: If a frame is empty or a requested feature is missing or has no
            finite numeric values.
    """
    if not isinstance(reference, pd.DataFrame) or not isinstance(current, pd.DataFrame):
        raise TypeError("reference and current must be pandas DataFrames")
    if reference.empty:
        raise ValueError("reference frame is empty; cannot compute drift")
    if current.empty:
        raise ValueError("current frame is empty; cannot compute drift")

    selected = _resolve_features(reference, current, features)
    details = {name: _compare_feature(reference[name], current[name], name) for name in selected}

    drifted = [name for name, detail in details.items() if detail["status"] == "drift"]
    warned = [name for name, detail in details.items() if detail["status"] == "warning"]
    drift_share = len(drifted) / len(selected)

    if drift_share >= DRIFT_SHARE_THRESHOLD:
        status = "drift"
    elif drifted or warned:
        status = "warning"
    else:
        status = "ok"

    report = {
        "schema_version": _SCHEMA_VERSION,
        "generated_at": _utc_timestamp(),
        "status": status,
        "drift_share": _ro2(drift_share),
        "n_features": len(selected),
        "features_checked": selected,
        "drifted_features": drifted,
        "warning_features": warned,
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "reference_window": _date_window(reference),
        "current_window": _date_window(current),
        "thresholds": dict(THRESHOLDS),
        "features": details,
    }
    logger.info(
        "drift check: status=%s drift_share=%.3f features=%d rows=%d/%d",
        status,
        drift_share,
        len(selected),
        len(reference),
        len(current),
    )
    return report


def chronological_split(
    frame: pd.DataFrame, reference_fraction: float = REFERENCE_FRACTION
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``frame`` by unique dates: oldest ``reference_fraction`` vs the rest.

    Splitting on distinct calendar days (instead of rows) guarantees that a single
    day never lands in both windows, even for unbalanced series.
    """
    if not 0.0 < reference_fraction < 1.0:
        raise ValueError("reference_fraction must be strictly between 0 and 1")
    if "date" not in frame.columns:
        raise ValueError("frame has no 'date' column required for a chronological split")

    parsed = pd.to_datetime(frame["date"], errors="coerce")
    invalid = int(parsed.isna().sum())
    if invalid:
        raise ValueError(f"{invalid} row(s) have an unparseable date")
    normalized = parsed.dt.normalize()
    unique_dates = np.sort(normalized.unique())
    if unique_dates.size < 2:
        raise ValueError(
            f"need at least 2 distinct dates to split, got {unique_dates.size}"
        )

    cut = int(unique_dates.size * reference_fraction)
    cut = min(max(cut, 1), unique_dates.size - 1)
    is_reference = normalized < unique_dates[cut]
    reference = frame.loc[is_reference].reset_index(drop=True)
    current = frame.loc[~is_reference].reset_index(drop=True)
    return reference, current


def seasonal_split(
    frame: pd.DataFrame, window_days: int = 28, lag_days: int = 364
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare the last ``window_days`` with the same window one year earlier.

    This is the default production comparison because it removes the annual
    seasonal cycle (December vs December, not January vs December) and therefore
    stops normal seasonality from masquerading as drift. It requires at least one
    full year of history plus the reference window; callers may fall back to
    :func:`chronological_split` for short files.
    """
    if window_days < 2:
        raise ValueError("window_days must be at least 2")
    if lag_days < window_days:
        raise ValueError("lag_days must be greater than or equal to window_days")
    if "date" not in frame.columns:
        raise ValueError("frame has no 'date' column required for a seasonal split")

    parsed = pd.to_datetime(frame["date"], errors="coerce")
    invalid = int(parsed.isna().sum())
    if invalid:
        raise ValueError(f"{invalid} row(s) have an unparseable date")
    normalized = parsed.dt.normalize()
    if normalized.dropna().empty:
        raise ValueError("frame has no valid dates")

    end = normalized.max()
    current_start = end - pd.Timedelta(days=window_days - 1)
    reference_end = end - pd.DateOffset(days=lag_days)
    reference_start = reference_end - pd.Timedelta(days=window_days - 1)

    reference = frame.loc[normalized.between(reference_start, reference_end)].reset_index(
        drop=True
    )
    current = frame.loc[normalized.between(current_start, end)].reset_index(drop=True)
    if reference.empty or current.empty:
        raise ValueError(
            "not enough history for a seasonal split; need at least "
            f"{lag_days + window_days} days, got "
            f"{(normalized.max() - normalized.min()).days + 1}"
        )
    return reference, current


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.monitoring.drift",
        description=(
            "Compare the last window with the same calendar window one year earlier "
            "(seasonal mode) or with the first 70% of dates (chronological mode)."
        ),
    )
    parser.add_argument("--data", required=True, help="Path to data/processed/demand_daily.csv")
    parser.add_argument("--out", required=True, help="Path of the JSON drift report to write")
    parser.add_argument(
        "--mode",
        choices=("seasonal", "chronological"),
        default="seasonal",
        help="Window strategy (default: seasonal year-over-year).",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=28,
        help="Length of the current window in days (default: 28).",
    )
    parser.add_argument(
        "--lag-days",
        type=int,
        default=364,
        help="Days between the end of the current and reference windows (default: 364).",
    )
    parser.add_argument(
        "--inject-drift",
        action="store_true",
        help="Multiply current-window units_sold by 1.6 to demonstrate detection",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code (0 on success)."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    data_path = Path(args.data)
    if not data_path.is_file():
        parser.error(
            f"data file not found: {data_path} (run `make data` to generate "
            "data/processed/demand_daily.csv)"
        )
    try:
        daily = load_csv(data_path, "demand_daily")
    except SchemaValidationError as exc:
        parser.error(f"invalid demand_daily file {data_path}: {exc}")

    missing = [name for name in DEFAULT_FEATURES if name not in daily.columns]
    if missing:
        parser.error(f"missing required columns: {missing}")

    window_mode = args.mode
    try:
        if args.mode == "seasonal":
            reference, current = seasonal_split(
                daily, window_days=args.window_days, lag_days=args.lag_days
            )
        else:
            reference, current = chronological_split(daily)
    except ValueError as exc:
        if args.mode == "seasonal":
            window_mode = "chronological_fallback"
            try:
                reference, current = chronological_split(daily)
            except ValueError as fallback_exc:
                parser.error(f"{exc}; fallback failed: {fallback_exc}")
        else:
            parser.error(str(exc))

    if args.inject_drift:
        current = current.copy()
        current["units_sold"] = pd.to_numeric(current["units_sold"], errors="coerce") * 1.6

    report = detect_drift(reference, current)
    report["inject_drift"] = bool(args.inject_drift)
    report["window_mode"] = window_mode
    report["window_days"] = int(args.window_days)
    report["lag_days"] = int(args.lag_days)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        f"drift: status={report['status']} drift_share={report['drift_share']:.3f} "
        f"drifted={len(report['drifted_features'])}/{report['n_features']} "
        f"mode={window_mode} "
        f"reference_rows={report['reference_rows']} current_rows={report['current_rows']} "
        f"inject_drift={str(report['inject_drift']).lower()} out={out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
