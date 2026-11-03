"""Validation helpers for the canonical CSV data contracts.

These checks are intentionally dependency-light (pandas only) so they can run in
CI and inside the ETL without pulling extra libraries.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schemas import (
    ABC_CLASSES,
    CSV_COLUMNS,
    DEMAND_PATTERNS,
    UNITS,
    EventType,
    MovementDirection,
    MovementReason,
    PendingSaleStatus,
)

DATE_COLUMNS: dict[str, tuple[str, ...]] = {
    "economic_events": ("event_ts",),
    "stock_movements": ("movement_ts",),
    "pending_sales": ("created_ts", "promised_ts"),
    "demand_daily": ("date",),
}

OPTIONAL_COLUMNS: dict[str, set[str]] = {
    "products": set(),
    "branches": set(),
    "recipes": set(),
    "economic_events": {
        "payment_method",
        "customer_id",
        "product_id",
        "product_name",
        "quantity",
        "unit_price_mxn",
        "unit_cost_mxn",
        "note",
    },
    "stock_movements": {"note"},
    "pending_sales": {"promised_ts", "converted_event_id"},
    "demand_daily": set(),
}

ENUM_COLUMNS: dict[str, dict[str, tuple[str, ...]]] = {
    "products": {
        "unit": UNITS,
        "demand_pattern": DEMAND_PATTERNS,
        "abc_class": ABC_CLASSES,
    },
    "economic_events": {"event_type": tuple(e.value for e in EventType)},
    "stock_movements": {
        "direction": tuple(e.value for e in MovementDirection),
        "reason": tuple(e.value for e in MovementReason),
    },
    "pending_sales": {"status": tuple(e.value for e in PendingSaleStatus)},
    "demand_daily": {
        "unit": UNITS,
        "demand_pattern": DEMAND_PATTERNS,
        "abc_class": ABC_CLASSES,
    },
}

NON_NEGATIVE: dict[str, tuple[str, ...]] = {
    "products": ("cost_mxn", "safety_stock", "lead_time_days"),
    "economic_events": ("amount_mxn", "quantity", "unit_price_mxn", "unit_cost_mxn"),
    "stock_movements": ("quantity", "unit_cost_mxn", "stock_after"),
    "pending_sales": ("quantity", "unit_price_mxn"),
    "demand_daily": (
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
    ),
}

BOOLEAN_COLUMNS: dict[str, tuple[str, ...]] = {
    "products": ("is_ingredient", "has_recipe"),
    "economic_events": ("is_cancelled",),
}


class SchemaValidationError(ValueError):
    """Raised when a CSV does not honour the canonical data contract."""


def _is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value)) or value is pd.NA


def validate_csv(path: str | Path, schema_name: str) -> list[str]:
    """Return a list of human-readable contract violations (empty when valid)."""
    if schema_name not in CSV_COLUMNS:
        raise ValueError(f"Unknown schema '{schema_name}'. Known: {sorted(CSV_COLUMNS)}")

    path = Path(path)
    if not path.exists():
        return [f"{path}: file not found"]

    df = pd.read_csv(path)
    expected = CSV_COLUMNS[schema_name]
    errors: list[str] = []

    missing = [column for column in expected if column not in df.columns]
    if missing:
        errors.append(f"{path.name}: missing required columns {missing}")
    unexpected = [column for column in df.columns if column not in expected]
    if unexpected:
        errors.append(f"{path.name}: unexpected columns {unexpected}")
    if errors:
        return errors

    optional = OPTIONAL_COLUMNS.get(schema_name, set())
    for column in expected:
        if column in optional:
            continue
        nulls = int(df[column].isna().sum())
        if nulls:
            errors.append(f"{path.name}: {nulls} null value(s) in required column '{column}'")

    for column in DATE_COLUMNS.get(schema_name, ()):
        if column in optional and df[column].isna().all():
            continue
        parsed = pd.to_datetime(df[column], errors="coerce", utc=True, format="ISO8601")
        invalid = int(parsed.isna().sum())
        if invalid:
            errors.append(f"{path.name}: {invalid} unparseable value(s) in '{column}'")

    for column, allowed in ENUM_COLUMNS.get(schema_name, {}).items():
        values = set(df[column].dropna().astype(str).str.strip())
        invalid_values = sorted(values - set(allowed))
        if invalid_values:
            errors.append(f"{path.name}: invalid values in '{column}': {invalid_values[:5]}")

    for column in NON_NEGATIVE.get(schema_name, ()):
        numeric = pd.to_numeric(df[column], errors="coerce")
        invalid = int((numeric < 0).sum())
        if invalid:
            errors.append(f"{path.name}: {invalid} negative value(s) in '{column}'")

    for column in BOOLEAN_COLUMNS.get(schema_name, ()):
        values = set(df[column].dropna().unique().tolist())
        bad = {value for value in values if value not in (0, 1, True, False)}
        if bad:
            errors.append(f"{path.name}: non-boolean values in '{column}': {sorted(bad)[:5]}")

    if schema_name == "products":
        price = pd.to_numeric(df["price_mxn"], errors="coerce")
        invalid = int((price <= 0).sum())
        if invalid:
            errors.append(f"{path.name}: {invalid} non-positive value(s) in 'price_mxn'")
        duplicated = int(df["product_id"].duplicated().sum())
        if duplicated:
            errors.append(f"{path.name}: {duplicated} duplicated product_id value(s)")

    if schema_name == "branches":
        duplicated = int(df["branch_id"].duplicated().sum())
        if duplicated:
            errors.append(f"{path.name}: {duplicated} duplicated branch_id value(s)")

    return errors


def assert_valid_csv(path: str | Path, schema_name: str) -> None:
    """Raise :class:`SchemaValidationError` if the CSV violates the contract."""
    errors = validate_csv(path, schema_name)
    if errors:
        raise SchemaValidationError(f"Invalid {schema_name} CSV:\n- " + "\n- ".join(errors))


def load_csv(path: str | Path, schema_name: str) -> pd.DataFrame:
    """Validate and load a CSV with the canonical dtypes for date columns."""
    assert_valid_csv(path, schema_name)
    df = pd.read_csv(path)
    for column in DATE_COLUMNS.get(schema_name, ()):
        if schema_name == "demand_daily" and column == "date":
            df[column] = pd.to_datetime(df[column]).dt.date
        else:
            df[column] = pd.to_datetime(df[column], utc=True, format="ISO8601")
    return df


def validate_referential_integrity(raw_dir: str | Path) -> list[str]:
    """Check foreign keys between the raw CSV files."""
    raw = Path(raw_dir)
    errors: list[str] = []

    products = pd.read_csv(raw / "products.csv")
    branches = pd.read_csv(raw / "branches.csv")
    product_ids = set(products["product_id"].astype(str))
    branch_ids = set(branches["branch_id"].astype(str))

    events = pd.read_csv(raw / "economic_events.csv")
    bad_products = set(events.loc[events["product_id"].notna(), "product_id"].astype(str)) - product_ids
    if bad_products:
        errors.append(f"economic_events.product_id not in products: {sorted(bad_products)[:5]}")
    bad_branches = set(events["branch_id"].astype(str)) - branch_ids
    if bad_branches:
        errors.append(f"economic_events.branch_id not in branches: {sorted(bad_branches)[:5]}")

    movements = pd.read_csv(raw / "stock_movements.csv")
    bad_products = set(movements["product_id"].astype(str)) - product_ids
    if bad_products:
        errors.append(f"stock_movements.product_id not in products: {sorted(bad_products)[:5]}")
    bad_branches = set(movements["branch_id"].astype(str)) - branch_ids
    if bad_branches:
        errors.append(f"stock_movements.branch_id not in branches: {sorted(bad_branches)[:5]}")

    pending = pd.read_csv(raw / "pending_sales.csv")
    bad_products = set(pending["product_id"].astype(str)) - product_ids
    if bad_products:
        errors.append(f"pending_sales.product_id not in products: {sorted(bad_products)[:5]}")
    bad_branches = set(pending["branch_id"].astype(str)) - branch_ids
    if bad_branches:
        errors.append(f"pending_sales.branch_id not in branches: {sorted(bad_branches)[:5]}")

    recipes = pd.read_csv(raw / "recipes.csv")
    for column in ("output_product_id", "input_product_id"):
        bad = set(recipes[column].astype(str)) - product_ids
        if bad:
            errors.append(f"recipes.{column} not in products: {sorted(bad)[:5]}")

    return errors
