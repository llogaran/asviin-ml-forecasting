"""Tests for the distribution-drift detector and its CLI (SPEC 4.6).

The fixture builds a stationary, weekday-seasonal frame that matches the
``demand_daily`` contract in memory, so the tests do not depend on the synthetic
generator or on committed data.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from src.data_contracts.schemas import CSV_COLUMNS
from src.monitoring.drift import (
    DEFAULT_FEATURES,
    PSI_BINS,
    chronological_split,
    detect_drift,
    population_stability_index,
    seasonal_split,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_COLUMNS = list(CSV_COLUMNS["demand_daily"])


def _demand_frame(days: int = 140) -> pd.DataFrame:
    """Build a deterministic, stationary daily-demand frame (whole weeks only)."""
    dates = pd.date_range("2023-01-01", periods=days, freq="D")
    records: list[dict] = []
    for product_index, product_id in enumerate(("P-001", "P-002")):
        for branch_index, branch_id in enumerate(("B-01", "B-02")):
            base_units = 4.0 + 2.0 * product_index + 1.5 * branch_index
            price = 12.0 + 5.0 * product_index
            cost = round(price * 0.6, 2)
            for day in dates:
                dow = int(day.dayofweek)
                weekly = 1.3 if dow >= 3 else 1.0
                noise = float((product_index * 3 + branch_index * 5 + dow) % 4) * 0.25
                units = round(base_units * weekly + noise, 2)
                stock = round(35.0 + (dow * 7 + product_index * 3 + branch_index * 5) % 30, 2)
                records.append(
                    {
                        "date": day.date(),
                        "business_id": "BIZ-01",
                        "branch_id": branch_id,
                        "product_id": product_id,
                        "category": "Producto Terminado",
                        "unit": "pza",
                        "units_sold": units,
                        "revenue_mxn": round(units * price, 2),
                        "cost_mxn": round(units * cost, 2),
                        "cancelled_units": 0.0,
                        "pending_units": round(units * 0.1, 2),
                        "stock_on_hand": stock,
                        "stockout_flag": 0,
                        "promo": int(dow == 2),
                        "unit_price_mxn": price,
                        "unit_cost_mxn": cost,
                        "lead_time_days": 7,
                        "demand_pattern": "smooth",
                        "abc_class": "A",
                    }
                )
    frame = pd.DataFrame.from_records(records, columns=CONTRACT_COLUMNS)
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    return frame


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the drift CLI with the interpreter used by the test session."""
    return subprocess.run(
        [sys.executable, "-m", "src.monitoring.drift", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_stable_chronological_split_is_not_drift() -> None:
    frame = _demand_frame()
    reference, current = chronological_split(frame)
    assert len(reference) > 0 and len(current) > 0
    assert len(reference) + len(current) == len(frame)

    report = detect_drift(reference, current)
    assert report["status"] != "drift"
    assert report["drifted_features"] == []
    assert report["drift_share"] < report["thresholds"]["drift_share"]


def test_report_structure_and_thresholds() -> None:
    frame = _demand_frame()
    reference, current = chronological_split(frame)
    report = detect_drift(reference, current)

    assert report["schema_version"] == "1.0.0"
    assert report["thresholds"] == {
        "psi_drift": 0.2,
        "psi_warning": 0.1,
        "ks_pvalue_drift": 0.01,
        "ks_pvalue_warning": 0.05,
        "drift_share": 0.30,
        "psi_bins": PSI_BINS,
    }
    assert report["reference_rows"] == len(reference)
    assert report["current_rows"] == len(current)
    assert report["n_features"] == len(DEFAULT_FEATURES)
    assert set(report["features"]) == set(DEFAULT_FEATURES)
    assert 0.0 <= report["drift_share"] <= 1.0

    stamp = report["generated_at"]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", stamp)
    parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    assert parsed.utcoffset() == timedelta(0)

    for name, detail in report["features"].items():
        assert detail["status"] in {"ok", "warning", "drift"}, name
        assert isinstance(detail["psi"], float)
        assert isinstance(detail["ks_pvalue"], float)
        assert detail["reference_count"] == len(reference)
        assert detail["current_count"] == len(current)

    # A constant flag exercises the degenerate-reference PSI path; identical
    # constant samples must stay stable instead of raising or warning.
    stockout = report["features"]["stockout_flag"]
    assert stockout["status"] == "ok"
    assert stockout["psi"] == 0.0


def test_injected_shift_is_detected() -> None:
    frame = _demand_frame()
    reference, current = chronological_split(frame)
    shifted = current.copy()
    shifted["units_sold"] = shifted["units_sold"] * 1.6

    report = detect_drift(reference, shifted)
    detail = report["features"]["units_sold"]
    assert detail["status"] == "drift"
    assert detail["psi"] >= 0.2 or detail["ks_pvalue"] < 0.01
    assert report["drift_share"] > 0.0
    assert report["status"] in {"drift", "warning"}
    assert "units_sold" in report["drifted_features"]


def test_features_argument_limits_checked_columns() -> None:
    frame = _demand_frame()
    reference, current = chronological_split(frame)
    report = detect_drift(reference, current, features=["units_sold", "revenue_mxn"])

    assert report["features_checked"] == ["units_sold", "revenue_mxn"]
    assert set(report["features"]) == {"units_sold", "revenue_mxn"}
    assert report["n_features"] == 2


def test_missing_feature_raises_clear_error() -> None:
    frame = _demand_frame()
    reference, current = chronological_split(frame)
    with pytest.raises(ValueError, match="stock_on_hand"):
        detect_drift(reference.drop(columns=["stock_on_hand"]), current)


def test_psi_handles_constant_reference() -> None:
    assert population_stability_index(np.zeros(50), np.zeros(50)) == 0.0
    assert population_stability_index(np.zeros(50), np.ones(50)) >= 0.2


def test_seasonal_split_uses_the_same_calendar_window_one_year_earlier() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=800, freq="D"),
            "units_sold": np.arange(800, dtype=float),
        }
    )
    reference, current = seasonal_split(frame, window_days=28, lag_days=364)

    assert len(reference) == len(current) == 28
    reference_end = pd.to_datetime(reference["date"]).max()
    current_end = pd.to_datetime(current["date"]).max()
    assert (current_end - reference_end).days == 364
    assert reference_end < current_end


def test_cli_smoke_stable_and_injected(tmp_path: Path) -> None:
    data_path = tmp_path / "demand_daily.csv"
    _demand_frame().to_csv(data_path, index=False)

    stable_out = tmp_path / "stable.json"
    stable = _run_cli("--data", str(data_path), "--out", str(stable_out))
    assert stable.returncode == 0, stable.stderr
    summary_lines = stable.stdout.strip().splitlines()
    assert len(summary_lines) == 1
    assert summary_lines[0].startswith("drift: status=")
    stable_report = json.loads(stable_out.read_text(encoding="utf-8"))
    assert stable_report["inject_drift"] is False
    assert stable_report["status"] != "drift"

    injected_out = tmp_path / "injected.json"
    injected = _run_cli(
        "--data", str(data_path), "--out", str(injected_out), "--inject-drift"
    )
    assert injected.returncode == 0, injected.stderr
    injected_report = json.loads(injected_out.read_text(encoding="utf-8"))
    assert injected_report["inject_drift"] is True
    assert injected_report["features"]["units_sold"]["status"] == "drift"
    assert injected_report["drift_share"] > 0.0


def test_cli_fails_clearly_for_missing_file(tmp_path: Path) -> None:
    result = _run_cli("--data", str(tmp_path / "absent.csv"), "--out", str(tmp_path / "o.json"))
    assert result.returncode != 0
    assert "not found" in result.stderr.lower()


def test_cli_fails_clearly_for_missing_columns(tmp_path: Path) -> None:
    data_path = tmp_path / "broken.csv"
    _demand_frame().drop(columns=["stock_on_hand"]).to_csv(data_path, index=False)
    result = _run_cli("--data", str(data_path), "--out", str(tmp_path / "o.json"))
    assert result.returncode != 0
    assert "stock_on_hand" in result.stderr
