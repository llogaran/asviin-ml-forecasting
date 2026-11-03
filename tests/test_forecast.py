"""Tests for the recursive forecaster and recommendation math (SPEC 4.5)."""

from __future__ import annotations

import math
from pathlib import Path
from statistics import NormalDist
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from src.pipelines.etl import run_etl
from src.pipelines.forecast import Forecaster
from src.pipelines.synthetic_generator import generate_dataset
from src.pipelines.train import train


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    root = tmp_path_factory.mktemp("forecast")
    data_dir = root / "data"
    paths = generate_dataset(data_dir, days=120, n_products=6, n_branches=2, seed=23)
    run_etl(paths["raw_dir"], data_dir / "processed" / "demand_daily.csv")
    artifacts_dir = root / "artifacts"
    train(data_dir, artifacts_dir, quick=True)
    return SimpleNamespace(
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        forecaster=Forecaster.load(data_dir, artifacts_dir),
    )


def _pick_series(forecaster: Forecaster) -> tuple[str, str]:
    product_ids = [product["product_id"] for product in forecaster.products()]
    product_id = "P-003" if "P-003" in product_ids else product_ids[0]
    return product_id, forecaster.branches()[0]["branch_id"]


def _last_daily_row(data_dir: Path, product_id: str, branch_id: str) -> pd.Series:
    daily = pd.read_csv(data_dir / "processed" / "demand_daily.csv")
    rows = daily[
        (daily["product_id"] == product_id) & (daily["branch_id"] == branch_id)
    ].sort_values("date")
    return rows.iloc[-1]


def test_forecast_shape_and_quantiles(artifacts: SimpleNamespace) -> None:
    forecaster = artifacts.forecaster
    product_id, branch_id = _pick_series(forecaster)
    result = forecaster.forecast(product_id, branch_id, horizon_days=14)

    assert result["product_id"] == product_id
    assert result["branch_id"] == branch_id
    assert result["model"] == "xgboost_global"
    assert result["model_version"] == "1.0.0"
    assert result["generated_at"].endswith("Z")
    assert result["horizon_days"] == 14
    assert len(result["points"]) == 14

    previous = None
    for point in result["points"]:
        assert set(point) == {"date", "p50", "p10", "p90"}
        assert point["p10"] <= point["p50"] <= point["p90"]
        assert point["p10"] >= 0.0
        assert point["p50"] >= 0.0
        assert point["p90"] >= 0.0
        for key in ("p50", "p10", "p90"):
            assert point[key] == round(point[key], 2)
        current = pd.Timestamp(point["date"])
        if previous is not None:
            assert current - previous == pd.Timedelta(days=1)
        previous = current


def test_recommendation_math(artifacts: SimpleNamespace) -> None:
    forecaster = artifacts.forecaster
    product_id, branch_id = _pick_series(forecaster)
    horizon = 14
    forecast_result = forecaster.forecast(product_id, branch_id, horizon_days=horizon)
    recommendation = forecaster.recommend(
        product_id, branch_id, horizon_days=horizon, service_level=0.95
    )

    points = forecast_result["points"]
    planning = np.array([0.7 * point["p50"] + 0.3 * point["p90"] for point in points])
    daily_mean = float(np.mean(planning))
    sigma = float(np.mean([(point["p90"] - point["p10"]) / 2.562 for point in points]))
    z_score = NormalDist().inv_cdf(0.95)
    last_row = _last_daily_row(artifacts.data_dir, product_id, branch_id)
    lead_time = max(0, int(round(float(last_row["lead_time_days"]))))
    safety_stock = z_score * sigma * math.sqrt(lead_time)
    reorder_point = daily_mean * lead_time + safety_stock
    current_stock = float(last_row["stock_on_hand"])
    reorder_quantity = max(0.0, reorder_point - current_stock)

    assert recommendation["lead_time_days"] == lead_time
    assert recommendation["current_stock"] == pytest.approx(current_stock, abs=0.01)
    assert recommendation["daily_mean"] == pytest.approx(daily_mean, abs=0.05)
    assert recommendation["daily_mean_units"] == recommendation["daily_mean"]
    assert recommendation["sigma"] == pytest.approx(sigma, abs=0.02)
    assert recommendation["sigma_units"] == recommendation["sigma"]
    assert recommendation["z_score"] == pytest.approx(z_score, abs=1e-3)
    assert recommendation["safety_stock"] == pytest.approx(safety_stock, abs=0.1)
    assert recommendation["reorder_point"] == pytest.approx(reorder_point, abs=0.1)
    assert recommendation["reorder_quantity"] == pytest.approx(reorder_quantity, abs=0.1)
    assert recommendation["reorder_quantity"] == pytest.approx(
        max(0.0, recommendation["reorder_point"] - recommendation["current_stock"]), abs=0.01
    )
    if recommendation["daily_mean_units"] > 0:
        assert recommendation["days_of_cover"] == pytest.approx(
            recommendation["current_stock"] / recommendation["daily_mean_units"], abs=0.1
        )
    assert "reorder" in recommendation["rationale"].lower()
    assert recommendation["model"] == "xgboost_global"
    assert recommendation["model_version"] == "1.0.0"


def test_forecast_horizon_bounds(artifacts: SimpleNamespace) -> None:
    forecaster = artifacts.forecaster
    product_id, branch_id = _pick_series(forecaster)
    assert len(forecaster.forecast(product_id, branch_id, horizon_days=1)["points"]) == 1
    assert len(forecaster.forecast(product_id, branch_id, horizon_days=90)["points"]) == 90
    for invalid in (0, -1, 91):
        with pytest.raises(ValueError, match="horizon_days"):
            forecaster.forecast(product_id, branch_id, horizon_days=invalid)
    with pytest.raises(ValueError, match="horizon_days"):
        forecaster.forecast(product_id, branch_id, horizon_days=1.5)
    for invalid_level in (0.0, 1.0):
        with pytest.raises(ValueError, match="service_level"):
            forecaster.recommend(product_id, branch_id, service_level=invalid_level)


def test_unknown_product_and_branch(artifacts: SimpleNamespace) -> None:
    forecaster = artifacts.forecaster
    product_id, branch_id = _pick_series(forecaster)
    with pytest.raises(KeyError, match="Unknown product_id"):
        forecaster.forecast("P-999", branch_id)
    with pytest.raises(KeyError, match="Unknown branch_id"):
        forecaster.forecast(product_id, "B-99")
    with pytest.raises(KeyError, match="Unknown product_id"):
        forecaster.recommend("P-999", branch_id)


def test_catalog_and_model_info(artifacts: SimpleNamespace) -> None:
    forecaster = artifacts.forecaster
    products = forecaster.products()
    branches = forecaster.branches()
    assert len(products) == 6
    assert len(branches) == 2
    assert all(product["product_id"].startswith("P-") for product in products)
    assert all(product["name"] != product["product_id"] for product in products)
    assert [product["product_id"] for product in products] == sorted(
        product["product_id"] for product in products
    )
    assert all(branch["branch_id"].startswith("B-") for branch in branches)
    assert all(branch["name"] for branch in branches)

    info = forecaster.model_info()
    assert info["model"] == "xgboost_global"
    assert info["model_version"] == "1.0.0"
    assert info["holdout_days"] == 14
    assert set(info["metrics"]) >= {"mae", "wape", "mase", "coverage_90", "n_series"}
    assert info["features"]
    assert info["target_transform"] == "log1p"
    assert len(info["dataset"]["sha256"]) == 64
    assert info["n_series"] == 12


def test_missing_artifacts_raise_actionable_error(
    artifacts: SimpleNamespace, tmp_path: Path
) -> None:
    with pytest.raises(FileNotFoundError, match="make train"):
        Forecaster.load(artifacts.data_dir, tmp_path / "no-artifacts")
    with pytest.raises(FileNotFoundError, match="make data"):
        Forecaster.load(tmp_path / "no-data", artifacts.artifacts_dir)
