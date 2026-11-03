"""Tests for quick-mode training, artifacts and the backtest reports (SPEC 4.4/5)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import pytest
from src.pipelines.etl import run_etl
from src.pipelines.features import FEATURES
from src.pipelines.synthetic_generator import generate_dataset
from src.pipelines.train import MODEL_NAME, croston_sba, main, train

N_DAYS = 120
N_PRODUCTS = 6
N_BRANCHES = 2
N_SERIES = N_PRODUCTS * N_BRANCHES


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    root = tmp_path_factory.mktemp("training")
    data_dir = root / "data"
    paths = generate_dataset(
        data_dir, days=N_DAYS, n_products=N_PRODUCTS, n_branches=N_BRANCHES, seed=11
    )
    run_etl(paths["raw_dir"], data_dir / "processed" / "demand_daily.csv")
    artifacts_dir = root / "artifacts"
    registry = train(data_dir, artifacts_dir, quick=True)
    return SimpleNamespace(root=root, data_dir=data_dir, artifacts=artifacts_dir, registry=registry)


def test_registry_contract(trained: SimpleNamespace) -> None:
    registry = trained.registry
    assert registry["schema_version"] == "1.0.0"
    assert registry["champion"] == MODEL_NAME
    assert registry["generated_at"].endswith("Z")

    dataset = registry["dataset"]
    assert dataset["path"].endswith("processed/demand_daily.csv")
    assert len(dataset["sha256"]) == 64
    assert dataset["rows"] == N_DAYS * N_PRODUCTS * N_BRANCHES
    assert dataset["series"] == N_SERIES
    assert dataset["date_min"] == "2023-01-01"
    assert dataset["date_max"] == "2023-04-30"

    entry = registry["models"][MODEL_NAME]
    assert entry["artifact"] == "xgboost_global.joblib"
    assert entry["trained_at"].endswith("Z")
    assert entry["holdout_days"] == 14
    metrics = entry["metrics"]
    for key in ("mae", "wape", "mase", "coverage_90", "n_series", "holdout_days"):
        assert key in metrics
    assert metrics["holdout_days"] == 14
    assert metrics["n_series"] == N_SERIES
    assert metrics["mae"] >= 0.0
    assert metrics["wape"] >= 0.0
    assert metrics["mase"] >= 0.0
    assert 0.0 <= metrics["coverage_90"] <= 1.0
    assert entry["baselines"]["naive_seasonal"].keys() >= {"mae", "wape"}
    assert entry["baselines"]["croston_sba"].keys() >= {"mae", "wape"}


def test_model_artifact_contract(trained: SimpleNamespace) -> None:
    path = trained.artifacts / "models" / "xgboost_global.joblib"
    assert path.exists()
    payload = joblib.load(path)
    assert set(payload) == {"models", "meta"}
    assert set(payload["models"]) == {"p10", "p50", "p90"}
    for model in payload["models"].values():
        assert hasattr(model, "predict")
    meta = payload["meta"]
    assert meta["features"] == FEATURES
    assert meta["target_transform"] == "log1p"
    assert meta["encoders"]["product_id"]
    assert meta["holdout_days"] == 14
    assert meta["quick"] is True
    assert meta["model_name"] == MODEL_NAME


def test_reports_are_written_and_consistent(trained: SimpleNamespace) -> None:
    reports = trained.artifacts / "reports"
    for name in ("backtest.json", "backtest.md", "impact.json"):
        assert (reports / name).exists(), name

    backtest = json.loads((reports / "backtest.json").read_text(encoding="utf-8"))
    entry = trained.registry["models"][MODEL_NAME]
    assert backtest["metrics"] == entry["metrics"]
    assert backtest["baselines"] == entry["baselines"]
    assert backtest["holdout_days"] == 14
    assert backtest["quick"] is True
    assert backtest["per_series_sample"]
    assert backtest["holdout"]["series"] == N_SERIES
    assert backtest["holdout"]["date_min"] <= backtest["holdout"]["date_max"]

    impact = json.loads((reports / "impact.json").read_text(encoding="utf-8"))
    assert impact["model"] == MODEL_NAME
    assert impact["generated_at"].endswith("Z")
    for side in ("forecast_driven", "naive_driven"):
        stats = impact[side]
        assert stats["avg_on_hand_units"] >= 0.0
        assert stats["stockout_days"] >= 0
        assert 0.0 <= stats["service_level"] <= 1.0
    for key in (
        "stockout_days_reduction_pct",
        "service_level_gain_pp",
        "avg_on_hand_reduction_pct",
    ):
        assert key in impact["relative_improvement"]
    assert "Forecast-driven" in impact["summary"]

    markdown = (reports / "backtest.md").read_text(encoding="utf-8")
    assert "# Backtest report" in markdown
    assert MODEL_NAME in markdown
    assert "coverage_90" in markdown
    assert "| metric |" in markdown


def test_cli_writes_artifacts(
    trained: SimpleNamespace, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cli_artifacts = tmp_path / "cli-artifacts"
    exit_code = main(
        ["--data", str(trained.data_dir), "--artifacts", str(cli_artifacts), "--quick"]
    )
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Trained xgboost_global" in output
    assert (cli_artifacts / "models" / "registry.json").exists()
    assert (cli_artifacts / "reports" / "impact.json").exists()


def test_missing_dataset_has_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make data"):
        train(tmp_path / "missing", tmp_path / "artifacts", quick=True)


def test_croston_sba_is_deterministic() -> None:
    assert croston_sba([0.0, 0.0, 0.0]) == 0.0
    constant = croston_sba([5.0] * 20)
    assert constant == pytest.approx(5.0 * (1.0 - 0.1 / 2.0))
    assert constant == croston_sba([5.0] * 20)
    assert croston_sba([0.0, 4.0, 0.0, 0.0, 6.0]) > 0.0
