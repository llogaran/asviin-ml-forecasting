"""Global demand-model training, backtesting and artifact generation (SPEC 4.4).

The pipeline trains one global XGBoost model per quantile (``p10``/``p50``/``p90``)
on the panel of every branch/product series. A single global model shares
strength across the catalog and can forecast a series that is too short to train
on its own, which matters for slow-moving ingredients.

Backtest protocol:

* Holdout is the last 28 calendar days of each series (14 in ``--quick`` mode).
* Points are compared on the units scale (inverse ``log1p``); quantile
  predictions are made monotonic (``p10 <= p50 <= p90``) before scoring.
* Baselines are ``naive_seasonal`` (same weekday, previous week) and
  ``croston_sba`` (Croston with the Syntetos-Boylan approximation, alpha=0.1),
  both estimated deterministically from data before the holdout.
* ``mase`` is the model MAE divided by the naive-seasonal MAE on the same
  holdout (the SPEC's definition of the seasonal scale).

The written artifacts are the serving contract: ``models/xgboost_global.joblib``
holds the three estimators plus the feature/encoder metadata, and
``models/registry.json`` is the small JSON the API, agent and reports read.
Everything under ``reports/`` is derived from the same holdout predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.config import PROJECT_ROOT, get_settings
from src.data_contracts.validation import load_csv
from src.pipelines.features import (
    FEATURES,
    MIN_HISTORY_DAYS,
    SERIES_KEYS,
    TARGET_FEATURE,
    build_supervised,
)
from src.pipelines.impact import simulate_inventory_policy

LOGGER = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0.0"
MODEL_NAME = "xgboost_global"
MODEL_VERSION = "1.0.0"
ARTIFACT_FILENAME = "xgboost_global.joblib"
REGISTRY_FILENAME = "registry.json"
DATA_FILENAME = Path("processed") / "demand_daily.csv"

HOLDOUT_DAYS = 28
QUICK_HOLDOUT_DAYS = 14
QUANTILES: dict[str, float] = {"p10": 0.1, "p50": 0.5, "p90": 0.9}

QUICK_ESTIMATORS = 40
FULL_ESTIMATORS = 300
QUICK_MAX_DEPTH = 3
FULL_MAX_DEPTH = 6
QUICK_LEARNING_RATE = 0.10
FULL_LEARNING_RATE = 0.05

CROSTON_ALPHA = 0.1
PER_SERIES_SAMPLE_LIMIT = 10


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    """Prefer a repository-relative path so artifacts never embed a machine path."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            return path.as_posix()


def _random_state() -> int:
    """Fixed seed, overridable with ``MLFC_RANDOM_SEED`` for reproducibility."""
    return int(get_settings().random_seed)


def croston_sba(values: Iterable[float], alpha: float = CROSTON_ALPHA) -> float:
    """Croston forecast with the Syntetos-Boylan approximation.

    Only non-zero demands update the demand size ``z`` and the inter-demand
    interval ``p``::

        z_t = z_{t-1} + alpha * (demand_t - z_{t-1})
        p_t = p_{t-1} + alpha * (interval_t - p_{t-1})
        forecast = (1 - alpha / 2) * z_t / p_t

    The first non-zero demand initializes ``z`` and ``p`` from the observed
    demand and its interval. The result is a constant demand rate per day, which
    is the standard way to benchmark intermittent series.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    size: float | None = None
    interval: float | None = None
    since_last = 0
    forecast = 0.0
    for raw_value in values:
        value = max(0.0, float(raw_value))
        since_last += 1
        if value <= 0:
            continue
        if size is None or interval is None:
            size = value
            interval = float(since_last)
        else:
            size = size + alpha * (value - size)
            interval = interval + alpha * (float(since_last) - interval)
        since_last = 0
        forecast = max(0.0, (1.0 - alpha / 2.0) * size / max(interval, 1e-9))
    return forecast


def _load_daily(data_dir: Path | str) -> tuple[Path, pd.DataFrame]:
    path = Path(data_dir) / DATA_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}; build the dataset first (run `make data` or the ETL)."
        )
    return path, load_csv(path, "demand_daily")


def _make_model(alpha: float, quick: bool, random_state: int) -> XGBRegressor:
    """Quantile XGBoost regressor with deterministic single-threaded training."""
    return XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=alpha,
        tree_method="hist",
        random_state=random_state,
        n_estimators=QUICK_ESTIMATORS if quick else FULL_ESTIMATORS,
        max_depth=QUICK_MAX_DEPTH if quick else FULL_MAX_DEPTH,
        learning_rate=QUICK_LEARNING_RATE if quick else FULL_LEARNING_RATE,
        min_child_weight=1.0,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_lambda=1.0,
        n_jobs=1,
        verbosity=0,
    )


def _predict_quantiles(
    models: dict[str, XGBRegressor], matrix: np.ndarray
) -> dict[str, np.ndarray]:
    """Predict units for each quantile and enforce a non-crossing interval."""
    raw = {name: np.asarray(model.predict(matrix), dtype=float) for name, model in models.items()}
    lower = np.minimum(raw["p10"], raw["p90"])
    upper = np.maximum(raw["p10"], raw["p90"])
    middle = np.clip(raw["p50"], lower, upper)
    return {
        "p10": np.clip(np.expm1(lower), 0.0, None),
        "p50": np.clip(np.expm1(middle), 0.0, None),
        "p90": np.clip(np.expm1(upper), 0.0, None),
    }


def _error_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    errors = np.abs(actual - prediction)
    denominator = float(actual.sum())
    wape = float(errors.sum() / denominator) if denominator > 0 else 0.0
    return {"mae": _round(errors.mean()), "wape": _round(wape)}


def _metrics(
    actual: np.ndarray,
    prediction: np.ndarray,
    p10: np.ndarray,
    p90: np.ndarray,
    naive: np.ndarray,
    *,
    holdout_days: int,
    n_series: int,
) -> dict[str, float | int]:
    errors = np.abs(actual - prediction)
    denominator = float(actual.sum())
    wape = float(errors.sum() / denominator) if denominator > 0 else 0.0
    naive_mae = float(np.abs(actual - naive).mean())
    mae = float(errors.mean())
    mase = mae / naive_mae if naive_mae > 0 else 0.0
    coverage = float(np.mean((actual >= p10) & (actual <= p90)))
    return {
        "mae": _round(mae),
        "wape": _round(wape),
        "mase": _round(mase),
        "coverage_90": _round(coverage),
        "n_series": int(n_series),
        "holdout_days": int(holdout_days),
    }


def _baseline_frame(daily: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    """Add ``naive_seasonal`` and ``croston_sba`` predictions to the holdout."""
    frame = valid.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["naive_seasonal"] = pd.to_numeric(frame["lag_7"], errors="coerce").fillna(0.0)

    history = daily.copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    first_valid_date = frame.groupby(list(SERIES_KEYS), sort=False)["date"].min()
    croston_forecast: dict[tuple[str, str, str], float] = {}
    for key, group in history.groupby(list(SERIES_KEYS), sort=False):
        series_key = tuple(key) if isinstance(key, tuple) else (key,)
        cutoff = first_valid_date.get(series_key)
        past = group[group["date"] < cutoff] if cutoff is not None else group
        past = past.sort_values("date", kind="stable")
        croston_forecast[series_key] = croston_sba(past["units_sold"].to_numpy(dtype=float))
    keys = list(zip(frame["business_id"], frame["branch_id"], frame["product_id"], strict=False))
    frame["croston_sba"] = [croston_forecast.get(key, 0.0) for key in keys]
    return frame


def _per_series_sample(
    scored: pd.DataFrame, limit: int = PER_SERIES_SAMPLE_LIMIT
) -> list[dict[str, Any]]:
    """Small deterministic per-series slice for ``backtest.json``."""
    sample: list[dict[str, Any]] = []
    for key, group in scored.groupby(list(SERIES_KEYS), sort=True):
        actual = group["units_sold"].to_numpy(dtype=float)
        prediction = group["p50"].to_numpy(dtype=float)
        model_errors = np.abs(actual - prediction)
        naive_errors = np.abs(actual - group["naive_seasonal"].to_numpy(dtype=float))
        croston_errors = np.abs(actual - group["croston_sba"].to_numpy(dtype=float))
        denominator = float(actual.sum())
        sample.append(
            {
                "business_id": key[0],
                "branch_id": key[1],
                "product_id": key[2],
                "rows": int(len(group)),
                "mae": _round(model_errors.mean()),
                "wape": _round(float(model_errors.sum() / denominator) if denominator > 0 else 0.0),
                "naive_mae": _round(naive_errors.mean()),
                "croston_mae": _round(croston_errors.mean()),
                "coverage_90": _round(
                    float(
                        np.mean(
                            (actual >= group["p10"].to_numpy(dtype=float))
                            & (actual <= group["p90"].to_numpy(dtype=float))
                        )
                    )
                ),
            }
        )
        if len(sample) >= limit:
            break
    return sample


def _dataset_info(daily_path: Path, daily: pd.DataFrame) -> dict[str, Any]:
    dates = pd.to_datetime(daily["date"], errors="coerce")
    return {
        "path": _display_path(daily_path),
        "sha256": _sha256_file(daily_path),
        "rows": int(len(daily)),
        "series": int(daily[list(SERIES_KEYS)].drop_duplicates().shape[0]),
        "date_min": dates.min().date().isoformat() if pd.notna(dates.min()) else None,
        "date_max": dates.max().date().isoformat() if pd.notna(dates.max()) else None,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fmt(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def _write_backtest_markdown(path: Path, backtest: dict) -> None:
    metrics = backtest["metrics"]
    baselines = backtest["baselines"]
    dataset = backtest["dataset"]
    holdout = backtest["holdout"]
    lines = [
        f"# Backtest report — {backtest['model']}",
        "",
        f"Generated: {backtest['generated_at']}",
        "",
        f"- Dataset: `{dataset['path']}`",
        f"- SHA-256: `{dataset['sha256']}`",
        f"- Rows: {dataset['rows']} | Series: {dataset['series']} | "
        f"Dates: {dataset['date_min']} → {dataset['date_max']}",
        f"- Holdout: {holdout['date_min']} → {holdout['date_max']} "
        f"({holdout['series']} series, {holdout['rows']} rows, "
        f"holdout_days={metrics['holdout_days']})",
        "",
        "MASE is scaled by the naive-seasonal MAE on the same holdout.",
        "",
        "| metric | xgboost_global | naive_seasonal | croston_sba |",
        "| --- | ---: | ---: | ---: |",
        f"| MAE | {_fmt(metrics['mae'])} | {_fmt(baselines['naive_seasonal']['mae'])} "
        f"| {_fmt(baselines['croston_sba']['mae'])} |",
        f"| WAPE | {_fmt(metrics['wape'])} | {_fmt(baselines['naive_seasonal']['wape'])} "
        f"| {_fmt(baselines['croston_sba']['wape'])} |",
        f"| MASE | {_fmt(metrics['mase'])} | 1.0000 | "
        f"{_fmt(baselines['croston_sba']['mae'] / baselines['naive_seasonal']['mae']) if baselines['naive_seasonal']['mae'] else '—'} |",
        f"| coverage_90 | {_fmt(metrics['coverage_90'])} | — | — |",
        "",
        "## Per-series sample",
        "",
        "| business | branch | product | rows | mae | naive_mae | croston_mae | coverage_90 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in backtest["per_series_sample"]:
        lines.append(
            f"| {row['business_id']} | {row['branch_id']} | {row['product_id']} | "
            f"{row['rows']} | {_fmt(row['mae'])} | {_fmt(row['naive_mae'])} | "
            f"{_fmt(row['croston_mae'])} | {_fmt(row['coverage_90'])} |"
        )
    lines.extend(
        [
            "",
            f"Only the first {len(backtest['per_series_sample'])} of "
            f"{backtest['series_total']} series are shown; the full panel is scored in "
            "`backtest.json`.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def train(data_dir: Path | str, artifacts_dir: Path | str, quick: bool = False) -> dict[str, Any]:
    """Train the quantile models, evaluate them on the holdout and write artifacts."""
    data_path = Path(data_dir)
    artifacts = Path(artifacts_dir)
    daily_path, daily = _load_daily(data_path)
    holdout_days = QUICK_HOLDOUT_DAYS if quick else HOLDOUT_DAYS

    train_df, valid_df, meta = build_supervised(daily, holdout_days=holdout_days)
    if train_df.empty:
        raise ValueError(
            f"Not enough training history: {len(daily)} daily rows, holdout={holdout_days}, "
            f"minimum-history={MIN_HISTORY_DAYS}. Generate a longer dataset."
        )
    if valid_df.empty:
        raise ValueError("The holdout is empty; generate a dataset with more calendar days.")

    random_state = _random_state()
    matrix_train = train_df[FEATURES].to_numpy(dtype=np.float32)
    target_train = train_df[TARGET_FEATURE].to_numpy(dtype=np.float32)
    models: dict[str, XGBRegressor] = {}
    for name, alpha in QUANTILES.items():
        LOGGER.info("Training %s quantile model (alpha=%.1f)", name, alpha)
        model = _make_model(alpha, quick, random_state)
        model.fit(matrix_train, target_train)
        models[name] = model

    matrix_valid = valid_df[FEATURES].to_numpy(dtype=np.float32)
    predicted = _predict_quantiles(models, matrix_valid)
    trained_at = _utc_now()
    generated_at = trained_at

    scored = _baseline_frame(daily, valid_df)
    scored["p10"] = predicted["p10"]
    scored["p50"] = predicted["p50"]
    scored["p90"] = predicted["p90"]
    actual = scored["units_sold"].to_numpy(dtype=float)
    naive = scored["naive_seasonal"].to_numpy(dtype=float)
    n_series = int(scored[list(SERIES_KEYS)].drop_duplicates().shape[0])

    metrics = _metrics(
        actual,
        scored["p50"].to_numpy(dtype=float),
        predicted["p10"],
        predicted["p90"],
        naive,
        holdout_days=holdout_days,
        n_series=n_series,
    )
    baseline_metrics = {
        "naive_seasonal": _error_metrics(actual, naive),
        "croston_sba": _error_metrics(actual, scored["croston_sba"].to_numpy(dtype=float)),
    }

    dataset = _dataset_info(daily_path, daily)

    meta = dict(meta)
    meta.update(
        {
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "trained_at": trained_at,
            "random_state": random_state,
            "quick": bool(quick),
            "holdout_days": int(holdout_days),
            "target_feature": TARGET_FEATURE,
            "features": list(FEATURES),
        }
    )

    model_path = artifacts / "models" / ARTIFACT_FILENAME
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "meta": meta}, model_path)

    registry = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "champion": MODEL_NAME,
        "dataset": dataset,
        "models": {
            MODEL_NAME: {
                "artifact": ARTIFACT_FILENAME,
                "trained_at": trained_at,
                "holdout_days": int(holdout_days),
                "metrics": metrics,
                "baselines": baseline_metrics,
            }
        },
    }
    _write_json(artifacts / "models" / REGISTRY_FILENAME, registry)

    backtest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "quick": bool(quick),
        "dataset": dataset,
        "holdout_days": int(holdout_days),
        "holdout": {
            "date_min": scored["date"].min().date().isoformat(),
            "date_max": scored["date"].max().date().isoformat(),
            "rows": int(len(scored)),
            "series": n_series,
        },
        "metrics": metrics,
        "baselines": baseline_metrics,
        "series_total": n_series,
        "per_series_sample": _per_series_sample(scored),
        "notes": [
            "Metrics are computed on the units scale after inverse log1p.",
            "coverage_90 is the share of actual demand inside the [p10, p90] interval.",
            "mase is the model MAE divided by the naive-seasonal MAE on this holdout.",
            "croston_sba uses alpha=0.1 and is estimated from pre-holdout history only.",
        ],
    }
    _write_json(artifacts / "reports" / "backtest.json", backtest)
    _write_backtest_markdown(artifacts / "reports" / "backtest.md", backtest)

    impact_report = simulate_inventory_policy(
        daily,
        scored,
        forecast_p50=scored["p50"].to_numpy(dtype=float),
        forecast_p10=scored["p10"].to_numpy(dtype=float),
        forecast_p90=scored["p90"].to_numpy(dtype=float),
        naive=naive,
    )
    impact_report = {
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "generated_at": generated_at,
        **impact_report,
    }
    _write_json(artifacts / "reports" / "impact.json", impact_report)

    LOGGER.info(
        "Trained %s: wape=%.4f mase=%.4f coverage_90=%.4f (holdout %s days)",
        MODEL_NAME,
        metrics["wape"],
        metrics["mase"],
        metrics["coverage_90"],
        holdout_days,
    )
    return registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the global quantile demand model and write model/report artifacts."
    )
    parser.add_argument("--data", required=True, help="Dataset directory containing processed/.")
    parser.add_argument("--artifacts", required=True, help="Artifact output directory.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast smoke run: 14-day holdout and smaller models.",
    )
    args = parser.parse_args(argv)
    try:
        registry = train(args.data, args.artifacts, quick=args.quick)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1
    metrics = registry["models"][MODEL_NAME]["metrics"]
    print(
        f"Trained {MODEL_NAME} ({'quick' if args.quick else 'full'}): "
        f"wape={metrics['wape']:.4f} mase={metrics['mase']:.4f} "
        f"coverage_90={metrics['coverage_90']:.4f} "
        f"series={metrics['n_series']} holdout_days={metrics['holdout_days']}"
    )
    print(f"Artifacts written under {Path(args.artifacts)}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
