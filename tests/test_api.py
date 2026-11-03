"""Tests for the FastAPI serving layer (SPEC 4.7).

A tiny dataset (120 days x 4 products x 2 branches) is generated, ETLed and
trained in a temporary directory so the suite never depends on the committed
artifacts. All requests stay in-process through ``fastapi.testclient``.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from src.api.app import create_app
from src.api.service import ForecastService
from src.pipelines.etl import run_etl
from src.pipelines.synthetic_generator import generate_dataset
from src.pipelines.train import train

N_DAYS = 120
N_PRODUCTS = 4
N_BRANCHES = 2
N_SERIES = N_PRODUCTS * N_BRANCHES

FORECAST_FIELDS = {
    "product_id",
    "branch_id",
    "model",
    "model_version",
    "generated_at",
    "horizon_days",
    "points",
}
POINT_FIELDS = {"date", "p50", "p10", "p90"}
RECOMMENDATION_FIELDS = {
    "product_id",
    "branch_id",
    "model",
    "model_version",
    "generated_at",
    "horizon_days",
    "service_level",
    "lead_time_days",
    "daily_mean",
    "sigma",
    "daily_mean_units",
    "sigma_units",
    "z_score",
    "safety_stock",
    "reorder_point",
    "current_stock",
    "reorder_quantity",
    "days_of_cover",
    "rationale",
}
METRIC_FIELDS = {
    "requests_total",
    "forecast_requests",
    "recommendation_requests",
    "errors_total",
    "uptime_seconds",
}


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove ambient API/CORS settings so every app starts from the defaults."""
    for name in ("MLFC_API_KEY", "MLFC_CORS_ORIGINS"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    """Generate, ETL and quick-train a tiny deterministic dataset."""
    root = tmp_path_factory.mktemp("api")
    data_dir = root / "data"
    paths = generate_dataset(
        data_dir, days=N_DAYS, n_products=N_PRODUCTS, n_branches=N_BRANCHES, seed=31
    )
    run_etl(paths["raw_dir"], data_dir / "processed" / "demand_daily.csv")
    artifacts_dir = root / "artifacts"
    train(data_dir, artifacts_dir, quick=True)

    daily = pd.read_csv(data_dir / "processed" / "demand_daily.csv", usecols=["branch_id"])
    return SimpleNamespace(
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        branch_id=str(daily["branch_id"].iloc[0]),
    )


@pytest.fixture(scope="module")
def service(trained: SimpleNamespace) -> ForecastService:
    return ForecastService(trained.data_dir, trained.artifacts_dir)


@pytest.fixture(scope="module")
def sample(service: ForecastService, trained: SimpleNamespace) -> SimpleNamespace:
    products = service.list_products()
    assert len(products) == N_PRODUCTS
    return SimpleNamespace(product_id=products[0]["product_id"], branch_id=trained.branch_id)


@pytest.fixture()
def client(service: ForecastService) -> TestClient:
    """Fresh app per test: independent auth settings and metrics counters."""
    return TestClient(create_app(service))


def _forecast_payload(sample: SimpleNamespace, horizon_days: int = 7) -> dict[str, object]:
    return {
        "product_id": sample.product_id,
        "branch_id": sample.branch_id,
        "horizon_days": horizon_days,
    }


def _recommendation_payload(
    sample: SimpleNamespace, horizon_days: int = 14
) -> dict[str, object]:
    return {
        "product_id": sample.product_id,
        "branch_id": sample.branch_id,
        "horizon_days": horizon_days,
        "service_level": 0.95,
    }


def test_health_is_open_even_with_api_key(
    service: ForecastService, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLFC_API_KEY", "test-secret")
    client = TestClient(create_app(service))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_api_key_required_on_every_other_endpoint(
    service: ForecastService, sample: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLFC_API_KEY", "test-secret")
    client = TestClient(create_app(service))

    for path in ("/ready", "/model", "/products", "/metrics"):
        assert client.get(path).status_code == 401, path

    # Auth is enforced before body validation.
    assert client.post("/forecast", json={}).status_code == 401
    assert client.post("/recommendation", json={}).status_code == 401

    wrong = client.get("/model", headers={"X-API-Key": "wrong-key"})
    assert wrong.status_code == 401
    assert "X-API-Key" in wrong.json()["detail"]

    authorized = client.get("/model", headers={"X-API-Key": "test-secret"})
    assert authorized.status_code == 200
    forecast = client.post(
        "/forecast", json=_forecast_payload(sample), headers={"X-API-Key": "test-secret"}
    )
    assert forecast.status_code == 200

    # CORS preflight must not require the key.
    preflight = client.options(
        "/forecast",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-api-key",
        },
    )
    assert preflight.status_code == 200


def test_ready_and_model(client: TestClient) -> None:
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json() == {
        "status": "ready",
        "model": "xgboost_global",
        "model_version": "1.0.0",
    }

    response = client.get("/model")
    assert response.status_code == 200
    info = response.json()
    assert info["model"] == "xgboost_global"
    assert info["champion"] == "xgboost_global"
    assert info["schema_version"] == "1.0.0"
    assert info["model_version"] == "1.0.0"
    assert info["holdout_days"] == 14
    assert info["n_series"] == N_SERIES
    assert info["target_transform"] == "log1p"
    assert info["features"]
    assert set(info["metrics"]) >= {"mae", "wape", "mase", "coverage_90", "n_series"}
    assert len(info["dataset"]["sha256"]) == 64


def test_missing_model_returns_503(tmp_path: Path) -> None:
    empty_service = ForecastService(tmp_path / "data", tmp_path / "artifacts")
    client = TestClient(create_app(empty_service))

    # Health and metrics never need the model; readiness and inference do.
    assert client.get("/health").status_code == 200
    assert client.get("/metrics").status_code == 200

    ready = client.get("/ready")
    assert ready.status_code == 503
    detail = ready.json()["detail"]
    assert "make data" in detail
    assert "make train" in detail

    assert client.get("/model").status_code == 503
    assert client.get("/products").status_code == 503
    forecast = client.post(
        "/forecast",
        json={"product_id": "P-001", "branch_id": "B-01", "horizon_days": 7},
    )
    assert forecast.status_code == 503
    assert "not available" in forecast.json()["detail"]


def test_products_catalog_and_filter(client: TestClient, sample: SimpleNamespace) -> None:
    body = client.get("/products").json()
    assert body["query"] is None
    assert body["count"] == N_PRODUCTS
    assert len(body["products"]) == N_PRODUCTS
    assert set(body["products"][0]) == {
        "product_id",
        "name",
        "category",
        "unit",
        "price_mxn",
        "lead_time_days",
        "demand_pattern",
        "abc_class",
        "is_ingredient",
    }

    filtered = client.get("/products", params={"q": sample.product_id.lower()}).json()
    assert filtered["query"] == sample.product_id.lower()
    assert filtered["count"] == 1
    assert filtered["products"][0]["product_id"] == sample.product_id

    assert client.get("/products", params={"q": "zzz-no-such-product"}).json()["count"] == 0


def test_unknown_product_or_branch_returns_404(
    client: TestClient, sample: SimpleNamespace
) -> None:
    unknown_product = client.post(
        "/forecast",
        json={
            "product_id": "P-999",
            "branch_id": sample.branch_id,
            "horizon_days": 7,
        },
    )
    assert unknown_product.status_code == 404
    assert "Unknown product_id" in unknown_product.json()["detail"]

    unknown_branch = client.post(
        "/forecast",
        json={
            "product_id": sample.product_id,
            "branch_id": "B-99",
            "horizon_days": 7,
        },
    )
    assert unknown_branch.status_code == 404
    assert "Unknown branch_id" in unknown_branch.json()["detail"]

    unknown_recommendation = client.post(
        "/recommendation",
        json={
            "product_id": "P-999",
            "branch_id": sample.branch_id,
            "horizon_days": 14,
            "service_level": 0.95,
        },
    )
    assert unknown_recommendation.status_code == 404


@pytest.mark.parametrize("horizon", [0, 91, 1.5, "many"])
def test_invalid_horizon_returns_422(
    client: TestClient, sample: SimpleNamespace, horizon: object
) -> None:
    payload = _forecast_payload(sample)
    payload["horizon_days"] = horizon
    assert client.post("/forecast", json=payload).status_code == 422

    recommendation = _recommendation_payload(sample)
    recommendation["horizon_days"] = horizon
    assert client.post("/recommendation", json=recommendation).status_code == 422


@pytest.mark.parametrize(
    "changes",
    [
        {"service_level": 1.0},
        {"service_level": 0.0},
        {"horizon_days": 0},
        {"unexpected_field": True},
        {"product_id": ""},
    ],
)
def test_invalid_recommendation_payload_returns_422(
    client: TestClient, sample: SimpleNamespace, changes: dict[str, object]
) -> None:
    payload = _recommendation_payload(sample)
    payload.update(changes)
    assert client.post("/recommendation", json=payload).status_code == 422


def test_forecast_response_shape(client: TestClient, sample: SimpleNamespace) -> None:
    response = client.post("/forecast", json=_forecast_payload(sample, horizon_days=7))
    assert response.status_code == 200
    body = response.json()
    assert set(body) == FORECAST_FIELDS
    assert body["product_id"] == sample.product_id
    assert body["branch_id"] == sample.branch_id
    assert body["model"] == "xgboost_global"
    assert body["model_version"] == "1.0.0"
    assert body["generated_at"].endswith("Z")
    assert body["horizon_days"] == 7
    assert len(body["points"]) == 7

    first_day = date.fromisoformat(body["points"][0]["date"])
    for offset, point in enumerate(body["points"]):
        assert set(point) == POINT_FIELDS
        assert date.fromisoformat(point["date"]) == first_day + timedelta(days=offset)
        assert 0.0 <= point["p10"] <= point["p50"] <= point["p90"]
        for quantile in ("p50", "p10", "p90"):
            assert point[quantile] == round(point[quantile], 2)


def test_recommendation_response_fields(client: TestClient, sample: SimpleNamespace) -> None:
    response = client.post("/recommendation", json=_recommendation_payload(sample))
    assert response.status_code == 200
    body = response.json()
    assert set(body) == RECOMMENDATION_FIELDS
    assert body["product_id"] == sample.product_id
    assert body["branch_id"] == sample.branch_id
    assert body["model"] == "xgboost_global"
    assert body["model_version"] == "1.0.0"
    assert body["horizon_days"] == 14
    assert body["service_level"] == 0.95
    assert body["lead_time_days"] >= 0
    assert body["current_stock"] >= 0.0
    assert body["reorder_quantity"] >= 0.0
    assert body["daily_mean"] >= 0.0
    assert body["sigma"] >= 0.0
    assert body["daily_mean_units"] == body["daily_mean"]
    assert body["sigma_units"] == body["sigma"]
    # The forecaster derives the quantity from unrounded values; allow for the
    # rounding of daily_mean (x lead time), safety stock and current stock.
    assert body["reorder_quantity"] == pytest.approx(
        max(0.0, body["reorder_point"] - body["current_stock"]),
        abs=0.05 * body["lead_time_days"] + 0.05,
    )
    assert body["safety_stock"] == pytest.approx(
        body["z_score"] * body["sigma"] * math.sqrt(body["lead_time_days"]), abs=0.1
    )
    if body["daily_mean"] > 0:
        # The forecaster derives cover from unrounded values, then rounds both.
        assert body["days_of_cover"] == pytest.approx(
            body["current_stock"] / body["daily_mean"], rel=0.01, abs=0.02
        )
    assert "reorder" in body["rationale"].lower()


def test_metrics_counters_increment(client: TestClient, sample: SimpleNamespace) -> None:
    initial = client.get("/metrics").json()
    assert set(initial) == METRIC_FIELDS
    assert initial["requests_total"] == 0
    assert initial["forecast_requests"] == 0
    assert initial["recommendation_requests"] == 0
    assert initial["errors_total"] == 0
    assert initial["uptime_seconds"] >= 0.0

    assert client.post("/forecast", json=_forecast_payload(sample)).status_code == 200
    assert client.post("/recommendation", json=_recommendation_payload(sample)).status_code == 200
    assert (
        client.post(
            "/forecast",
            json={"product_id": "P-999", "branch_id": sample.branch_id, "horizon_days": 7},
        ).status_code
        == 404
    )
    assert client.get("/health").status_code == 200

    counters = client.get("/metrics").json()
    # 4 new requests plus the initial /metrics call, which is counted after its
    # own snapshot was taken.
    assert counters["requests_total"] == 5
    assert counters["forecast_requests"] == 2
    assert counters["recommendation_requests"] == 1
    assert counters["errors_total"] == 1
    assert counters["uptime_seconds"] >= initial["uptime_seconds"]


def test_cors_default_allows_any_origin(client: TestClient) -> None:
    response = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert response.headers["access-control-allow-origin"] == "*"
