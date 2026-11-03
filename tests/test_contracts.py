"""Contract tests for the canonical CSV data contracts (coordinator-owned)."""

from __future__ import annotations

import pandas as pd
import pytest
from src.data_contracts import (
    SCHEMA_VERSION,
    SchemaValidationError,
    assert_valid_csv,
    load_csv,
    validate_csv,
    validate_referential_integrity,
)
from src.data_contracts.schemas import CSV_COLUMNS


def _write_csv(path, rows: list[dict], columns: list[str]) -> None:
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def _valid_raw(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()

    _write_csv(
        raw / "products.csv",
        [
            {
                "product_id": "P-001",
                "name": "PET clear flakes",
                "category": "Materia Prima",
                "unit": "kg",
                "price_mxn": 18.5,
                "cost_mxn": 12.0,
                "is_ingredient": 0,
                "has_recipe": 0,
                "safety_stock": 10.0,
                "lead_time_days": 7,
                "demand_pattern": "smooth",
                "abc_class": "A",
            },
            {
                "product_id": "P-002",
                "name": "HDPE pellets",
                "category": "Materia Prima",
                "unit": "kg",
                "price_mxn": 24.0,
                "cost_mxn": 16.0,
                "is_ingredient": 0,
                "has_recipe": 0,
                "safety_stock": 5.0,
                "lead_time_days": 10,
                "demand_pattern": "intermittent",
                "abc_class": "B",
            },
        ],
        list(CSV_COLUMNS["products"]),
    )
    _write_csv(
        raw / "branches.csv",
        [{"branch_id": "B-01", "business_id": "biz_demo", "name": "Matriz", "region": "Centro"}],
        list(CSV_COLUMNS["branches"]),
    )
    _write_csv(
        raw / "recipes.csv",
        [
            {
                "recipe_id": "R-001",
                "business_id": "biz_demo",
                "output_product_id": "P-001",
                "input_product_id": "P-002",
                "quantity_per_unit": 0.5,
            }
        ],
        list(CSV_COLUMNS["recipes"]),
    )
    _write_csv(
        raw / "economic_events.csv",
        [
            {
                "event_id": "EVT-0001",
                "business_id": "biz_demo",
                "branch_id": "B-01",
                "event_type": "venta",
                "event_ts": "2024-01-01T10:00:00-06:00",
                "amount_mxn": 100.0,
                "payment_method": "efectivo",
                "customer_id": "",
                "product_id": "P-001",
                "product_name": "PET clear flakes",
                "quantity": 2.0,
                "unit_price_mxn": 18.5,
                "unit_cost_mxn": 12.0,
                "is_cancelled": 0,
                "note": "",
            },
            {
                "event_id": "EVT-0002",
                "business_id": "biz_demo",
                "branch_id": "B-01",
                "event_type": "gasto",
                "event_ts": "2024-01-01T12:00:00-06:00",
                "amount_mxn": 250.0,
                "payment_method": "",
                "customer_id": "",
                "product_id": "",
                "product_name": "",
                "quantity": "",
                "unit_price_mxn": "",
                "unit_cost_mxn": "",
                "is_cancelled": 0,
                "note": "electricidad",
            },
        ],
        list(CSV_COLUMNS["economic_events"]),
    )
    _write_csv(
        raw / "stock_movements.csv",
        [
            {
                "movement_id": "MOV-0001",
                "business_id": "biz_demo",
                "branch_id": "B-01",
                "product_id": "P-001",
                "movement_ts": "2024-01-01T10:00:00-06:00",
                "direction": "salida",
                "reason": "venta_manual",
                "quantity": 2.0,
                "unit_cost_mxn": 12.0,
                "stock_after": 8.0,
                "note": "",
            }
        ],
        list(CSV_COLUMNS["stock_movements"]),
    )
    _write_csv(
        raw / "pending_sales.csv",
        [
            {
                "pending_sale_id": "PS-0001",
                "business_id": "biz_demo",
                "branch_id": "B-01",
                "product_id": "P-001",
                "created_ts": "2024-01-01T09:00:00-06:00",
                "promised_ts": "2024-01-05T09:00:00-06:00",
                "quantity": 3.0,
                "unit_price_mxn": 18.5,
                "status": "convertida",
                "converted_event_id": "EVT-0001",
            }
        ],
        list(CSV_COLUMNS["pending_sales"]),
    )
    _write_csv(
        raw / "demand_daily.csv",
        [
            {
                "date": "2024-01-01",
                "business_id": "biz_demo",
                "branch_id": "B-01",
                "product_id": "P-001",
                "category": "Materia Prima",
                "unit": "kg",
                "units_sold": 2.0,
                "revenue_mxn": 37.0,
                "cost_mxn": 24.0,
                "cancelled_units": 0.0,
                "pending_units": 3.0,
                "stock_on_hand": 8.0,
                "stockout_flag": 0,
                "promo": 0,
                "unit_price_mxn": 18.5,
                "unit_cost_mxn": 12.0,
                "lead_time_days": 7,
                "demand_pattern": "smooth",
                "abc_class": "A",
            }
        ],
        list(CSV_COLUMNS["demand_daily"]),
    )
    return raw


def test_schema_version_is_semver_like():
    parts = SCHEMA_VERSION.split(".")
    assert len(parts) == 3 and all(part.isdigit() for part in parts)


@pytest.mark.parametrize("schema_name", sorted(CSV_COLUMNS))
def test_valid_csvs_pass_contract(tmp_path, schema_name):
    raw = _valid_raw(tmp_path)
    path = raw / f"{schema_name}.csv" if schema_name != "demand_daily" else raw / "demand_daily.csv"
    assert validate_csv(path, schema_name) == []


def test_foreign_keys_are_consistent(tmp_path):
    raw = _valid_raw(tmp_path)
    assert validate_referential_integrity(raw) == []


def test_missing_columns_are_reported(tmp_path):
    raw = _valid_raw(tmp_path)
    df = pd.read_csv(raw / "products.csv").drop(columns=["lead_time_days"])
    df.to_csv(raw / "products.csv", index=False)
    errors = validate_csv(raw / "products.csv", "products")
    assert any("missing required columns" in error for error in errors)


def test_unexpected_columns_are_reported(tmp_path):
    raw = _valid_raw(tmp_path)
    df = pd.read_csv(raw / "products.csv")
    df["surprise"] = 1
    df.to_csv(raw / "products.csv", index=False)
    errors = validate_csv(raw / "products.csv", "products")
    assert any("unexpected columns" in error for error in errors)


def test_negative_values_are_rejected(tmp_path):
    raw = _valid_raw(tmp_path)
    df = pd.read_csv(raw / "demand_daily.csv")
    df.loc[0, "units_sold"] = -5
    df.to_csv(raw / "demand_daily.csv", index=False)
    errors = validate_csv(raw / "demand_daily.csv", "demand_daily")
    assert any("negative value" in error for error in errors)
    with pytest.raises(SchemaValidationError):
        assert_valid_csv(raw / "demand_daily.csv", "demand_daily")


def test_invalid_enum_values_are_rejected(tmp_path):
    raw = _valid_raw(tmp_path)
    df = pd.read_csv(raw / "products.csv")
    df.loc[0, "unit"] = "tonelada"
    df.to_csv(raw / "products.csv", index=False)
    errors = validate_csv(raw / "products.csv", "products")
    assert any("invalid values" in error for error in errors)


def test_null_required_values_are_rejected(tmp_path):
    raw = _valid_raw(tmp_path)
    df = pd.read_csv(raw / "products.csv")
    df.loc[0, "name"] = None
    df.to_csv(raw / "products.csv", index=False)
    errors = validate_csv(raw / "products.csv", "products")
    assert any("null value" in error for error in errors)


def test_referential_integrity_reports_bad_product(tmp_path):
    raw = _valid_raw(tmp_path)
    df = pd.read_csv(raw / "economic_events.csv")
    df.loc[0, "product_id"] = "P-999"
    df.to_csv(raw / "economic_events.csv", index=False)
    errors = validate_referential_integrity(raw)
    assert any("economic_events.product_id not in products" in error for error in errors)


def test_load_csv_parses_date_columns(tmp_path):
    raw = _valid_raw(tmp_path)
    daily = load_csv(raw / "demand_daily.csv", "demand_daily")
    assert daily["date"].iloc[0].isoformat() == "2024-01-01"

    events = load_csv(raw / "economic_events.csv", "economic_events")
    assert str(events["event_ts"].dt.tz) == "UTC"

    pending = load_csv(raw / "pending_sales.csv", "pending_sales")
    assert pending["promised_ts"].notna().all()
