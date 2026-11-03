"""Tests for the raw-to-daily ETL (SPEC 3.7 and 4.2)."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pandas as pd
import pytest
from src.data_contracts.schemas import CSV_COLUMNS
from src.data_contracts.validation import SchemaValidationError, load_csv
from src.pipelines.etl import build_daily_demand, main, run_etl
from src.pipelines.synthetic_generator import generate_dataset

KEY = ["date", "business_id", "branch_id", "product_id"]
SERIES = ["business_id", "branch_id", "product_id"]
REQUIRED_QUALITY_KEYS = {
    "rows",
    "series",
    "date_min",
    "date_max",
    "zero_sale_share",
    "negative_stock_rows",
    "missing_product_refs",
    "duplicate_keys",
    "cancelled_units_total",
    "pending_units_total",
}
START = "2023-01-01"
END = "2023-04-30"
N_DAYS = 120


def _local_dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True).dt.tz_convert("America/Mexico_City").dt.date


@pytest.fixture(scope="module")
def raw_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("etl-raw")
    paths = generate_dataset(out, days=N_DAYS, n_products=8, n_branches=2, seed=13)
    return paths["raw_dir"]


@pytest.fixture(scope="module")
def daily(raw_dir: Path) -> pd.DataFrame:
    return build_daily_demand(raw_dir)


def _sales(raw_dir: Path) -> pd.DataFrame:
    events = pd.read_csv(raw_dir / "economic_events.csv")
    sales = events[events["event_type"] == "venta"].copy()
    sales["date"] = _local_dates(sales["event_ts"]).astype(str)
    return sales


def _copy_raw(raw_dir: Path, target: Path) -> Path:
    destination = target / "raw"
    shutil.copytree(raw_dir, destination)
    return destination


def _read_text_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def test_daily_contract_columns_and_types(daily: pd.DataFrame) -> None:
    assert list(daily.columns) == list(CSV_COLUMNS["demand_daily"])
    assert daily["date"].str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()
    assert daily["stockout_flag"].isin([0, 1]).all()
    assert daily["promo"].isin([0, 1]).all()
    assert (daily["units_sold"] >= 0).all()
    assert (daily["stock_on_hand"] >= 0).all()
    assert daily["unit_price_mxn"].gt(0).all()
    assert daily["unit_cost_mxn"].gt(0).all()


def test_daily_grain_dates_and_zero_fill(daily: pd.DataFrame, raw_dir: Path) -> None:
    assert not daily.duplicated(subset=KEY).any()
    dates = pd.to_datetime(daily["date"]).dt.date
    assert dates.min().isoformat() == START
    assert dates.max().isoformat() == END
    assert len(daily) == daily[SERIES].drop_duplicates().shape[0] * N_DAYS
    assert daily.groupby(SERIES).size().eq(N_DAYS).all()

    products = pd.read_csv(raw_dir / "products.csv")
    branches = pd.read_csv(raw_dir / "branches.csv")
    expected_series = branches[["business_id", "branch_id"]].merge(
        products[["product_id"]], how="cross"
    )
    actual_series = daily[SERIES].drop_duplicates()
    assert len(actual_series) == len(products) * len(branches)
    assert set(map(tuple, actual_series.to_numpy())) == set(
        map(tuple, expected_series[SERIES].to_numpy())
    )
    assert (daily["units_sold"] == 0).any()


def test_cancelled_lines_are_excluded_from_sales(raw_dir: Path, daily: pd.DataFrame) -> None:
    sales = _sales(raw_dir)
    active = sales[sales["is_cancelled"] == 0]
    cancelled = sales[sales["is_cancelled"] == 1]
    assert abs(active["quantity"].sum() - daily["units_sold"].sum()) < 0.01
    assert abs(active["amount_mxn"].sum() - daily["revenue_mxn"].sum()) < 0.02
    assert abs((active["quantity"] * active["unit_cost_mxn"]).sum() - daily["cost_mxn"].sum()) < 0.5
    assert abs(cancelled["quantity"].sum() - daily["cancelled_units"].sum()) < 0.01
    assert daily["cancelled_units"].sum() > 0
    cancelled_only = daily[(daily["cancelled_units"] > 0) & (daily["units_sold"] == 0)]
    assert not cancelled_only.empty


def test_pending_units_and_status_filter(raw_dir: Path, daily: pd.DataFrame) -> None:
    pending = pd.read_csv(raw_dir / "pending_sales.csv")
    pending["date"] = _local_dates(pending["created_ts"]).astype(str)
    active = pending[pending["status"] != "cancelada"]
    expected = active.groupby(KEY)["quantity"].sum().rename("expected").reset_index()
    merged = daily.merge(expected, on=KEY, how="left")
    merged["expected"] = merged["expected"].fillna(0.0)
    assert (merged["expected"] - merged["pending_units"]).abs().max() < 0.01
    assert daily["pending_units"].sum() > 0
    assert daily["pending_units"].ge(0).all()


def test_stock_forward_fill_and_stockout_flag(raw_dir: Path, daily: pd.DataFrame) -> None:
    movements = pd.read_csv(raw_dir / "stock_movements.csv")
    movements["date"] = _local_dates(movements["movement_ts"]).astype(str)
    movements = movements.sort_values(["movement_ts", "movement_id"], kind="stable")
    last = (
        movements.groupby(KEY)["stock_after"]
        .last()
        .rename("expected_stock")
        .reset_index()
    )
    check = daily.merge(last, on=KEY, how="left")
    check = check.sort_values(SERIES + ["date"], kind="stable")
    check["expected_stock"] = (
        check.groupby(SERIES, sort=False)["expected_stock"].ffill().fillna(0.0)
    )
    assert ((check["stock_on_hand"] - check["expected_stock"]).abs() < 1e-9).all()
    assert (daily["stock_on_hand"] >= 0).all()
    assert ((daily["stockout_flag"] == 1) == (daily["stock_on_hand"] <= 0)).all()
    assert daily["stockout_flag"].isin([0, 1]).all()


def test_weighted_prices_promo_and_fallback(raw_dir: Path, daily: pd.DataFrame) -> None:
    products = pd.read_csv(raw_dir / "products.csv").set_index("product_id")
    master_price = daily["product_id"].map(products["price_mxn"])

    sale_rows = daily[daily["units_sold"] > 0]
    assert not sale_rows.empty
    assert (sale_rows["unit_price_mxn"] < master_price[sale_rows.index]).any()

    promo_rows = daily[(daily["promo"] == 1) & (daily["units_sold"] > 0)]
    assert not promo_rows.empty
    assert (
        promo_rows["unit_price_mxn"]
        < 0.85 * promo_rows["product_id"].map(products["price_mxn"])
    ).all()

    no_sale = daily[daily["units_sold"] == 0]
    assert not no_sale.empty
    assert (no_sale["unit_price_mxn"].to_numpy() == no_sale["product_id"].map(products["price_mxn"]).to_numpy()).all()
    assert (no_sale["unit_cost_mxn"].to_numpy() == no_sale["product_id"].map(products["cost_mxn"]).to_numpy()).all()

    sales = _sales(raw_dir)
    active = sales[sales["is_cancelled"] == 0]
    lines = active.groupby(KEY).agg(
        price=("unit_price_mxn", "first"), cost=("unit_cost_mxn", "first")
    ).reset_index()
    merged = daily.merge(lines, on=KEY, how="left")
    matched = merged["units_sold"] > 0
    assert (merged.loc[matched, "unit_price_mxn"] - merged.loc[matched, "price"]).abs().max() < 0.01
    assert (merged.loc[matched, "unit_cost_mxn"] - merged.loc[matched, "cost"]).abs().max() < 0.01


def test_quality_report_and_written_csv(tmp_path: Path, raw_dir: Path) -> None:
    out = tmp_path / "processed" / "demand_daily.csv"
    frame = run_etl(raw_dir, out)
    report = json.loads((out.parent / "data_quality.json").read_text(encoding="utf-8"))
    assert REQUIRED_QUALITY_KEYS <= set(report)
    assert report["rows"] == len(frame)
    products = pd.read_csv(raw_dir / "products.csv")
    branches = pd.read_csv(raw_dir / "branches.csv")
    assert report["series"] == len(products) * len(branches)
    assert report["series"] == frame[SERIES].drop_duplicates().shape[0]
    assert report["date_min"] == START
    assert report["date_max"] == END
    assert report["negative_stock_rows"] == 0
    assert report["missing_product_refs"] == 0
    assert report["duplicate_keys"] == 0
    assert report["cancelled_units_total"] == pytest.approx(frame["cancelled_units"].sum(), abs=0.01)
    assert report["pending_units_total"] == pytest.approx(frame["pending_units"].sum(), abs=0.01)
    assert report["zero_sale_share"] == pytest.approx(
        float((frame["units_sold"] == 0).mean()), abs=0.0001
    )
    assert isinstance(report["stockout_rows"], int)

    loaded = load_csv(out, "demand_daily")
    assert len(loaded) == len(frame)
    text = out.read_text(encoding="utf-8")
    assert ",".join(CSV_COLUMNS["demand_daily"]) in text.splitlines()[0]
    assert not re.search(r",-?\d+\.\d{3,}(,|$)", text)
    assert re.search(r",\d+\.\d{2},", text)


def test_etl_is_byte_identical_across_runs(tmp_path: Path, raw_dir: Path) -> None:
    first = tmp_path / "a" / "demand_daily.csv"
    second = tmp_path / "b" / "demand_daily.csv"
    run_etl(raw_dir, first)
    run_etl(raw_dir, second)
    assert first.read_bytes() == second.read_bytes()
    assert (first.parent / "data_quality.json").read_bytes() == (
        second.parent / "data_quality.json"
    ).read_bytes()


def test_invalid_enum_is_rejected(tmp_path: Path, raw_dir: Path) -> None:
    raw = _copy_raw(raw_dir, tmp_path)
    events = _read_text_frame(raw / "economic_events.csv")
    events.loc[events["event_type"] == "venta", "event_type"] = "devolucion"
    events.to_csv(raw / "economic_events.csv", index=False)
    with pytest.raises(SchemaValidationError, match="event_type"):
        build_daily_demand(raw)


def test_broken_foreign_key_is_rejected(tmp_path: Path, raw_dir: Path) -> None:
    raw = _copy_raw(raw_dir, tmp_path)
    events = _read_text_frame(raw / "economic_events.csv")
    index = events.index[events["event_type"] == "venta"][0]
    events.loc[index, "product_id"] = "P-999"
    events.to_csv(raw / "economic_events.csv", index=False)
    with pytest.raises(SchemaValidationError, match="not in products"):
        build_daily_demand(raw)


def test_missing_column_is_rejected(tmp_path: Path, raw_dir: Path) -> None:
    raw = _copy_raw(raw_dir, tmp_path)
    events = _read_text_frame(raw / "economic_events.csv")
    events.drop(columns=["unit_cost_mxn"]).to_csv(raw / "economic_events.csv", index=False)
    with pytest.raises(SchemaValidationError, match="missing required columns"):
        build_daily_demand(raw)


def test_sale_line_without_product_is_rejected(tmp_path: Path, raw_dir: Path) -> None:
    raw = _copy_raw(raw_dir, tmp_path)
    events = _read_text_frame(raw / "economic_events.csv")
    index = events.index[events["event_type"] == "venta"][0]
    events.loc[index, "product_id"] = ""
    events.to_csv(raw / "economic_events.csv", index=False)
    with pytest.raises(SchemaValidationError, match="invalid 'venta' line"):
        build_daily_demand(raw)


def test_file_without_sales_is_rejected(tmp_path: Path, raw_dir: Path) -> None:
    raw = _copy_raw(raw_dir, tmp_path)
    events = _read_text_frame(raw / "economic_events.csv")
    events[events["event_type"] != "venta"].to_csv(raw / "economic_events.csv", index=False)
    with pytest.raises(SchemaValidationError, match="no 'venta' lines"):
        build_daily_demand(raw)


def test_missing_file_is_rejected(tmp_path: Path, raw_dir: Path) -> None:
    raw = _copy_raw(raw_dir, tmp_path)
    (raw / "pending_sales.csv").unlink()
    with pytest.raises(SchemaValidationError, match="file not found"):
        build_daily_demand(raw)


def test_cli_success_and_failure(
    tmp_path: Path, raw_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "cli" / "demand_daily.csv"
    assert main(["--raw", str(raw_dir), "--out", str(out)]) == 0
    assert out.exists()
    assert (out.parent / "data_quality.json").exists()
    assert "Wrote" in capsys.readouterr().out

    empty = tmp_path / "empty"
    empty.mkdir()
    code = main(["--raw", str(empty), "--out", str(tmp_path / "missing.csv")])
    assert code == 1
    assert "ETL failed" in capsys.readouterr().err
