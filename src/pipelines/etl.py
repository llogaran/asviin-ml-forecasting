"""ETL from the raw Asviin contract CSVs to the daily demand modeling table.

The raw files are validated with :mod:`src.data_contracts` before anything is
aggregated, so downstream code never consumes a broken export. The resulting
``demand_daily.csv`` has one row per ``(date, business, branch, product)`` for
every calendar day in the raw range and is zero-filled when a series has no
sale that day.

Semantics implemented from the SPEC (section 3.7):

* ``units_sold`` / ``revenue_mxn`` / ``cost_mxn`` only count non-cancelled sale
  lines; ``cancelled_units`` sums the cancelled quantities separately.
* ``pending_units`` is the quantity of pending sales created that day whose
  status is not ``cancelada``.
* ``stock_on_hand`` is the last ``stock_after`` of the day, forward-filled, and
  0 before the first movement. ``stockout_flag`` is 1 when it is <= 0.
* ``promo`` is 1 when any sale line that day priced below 85% of the product
  master price; ``unit_price_mxn`` / ``unit_cost_mxn`` are quantity-weighted
  sale-line averages with the product master as fallback.

The grain is the full cartesian product of every master product, every master
branch and every calendar day in the raw range. A catalog item without an
invoiced sale still gets a zero row with its physical stock instead of
disappearing, so downstream code can safely ask for any known product/branch
pair without special-casing the catalog.

The quality report is written next to ``out_path`` as ``data_quality.json`` and
contains no timestamps, so repeated runs over the same raw data are
byte-identical.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import timedelta, timezone, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

from src.data_contracts.schemas import CSV_COLUMNS
from src.data_contracts.validation import (
    SchemaValidationError,
    load_csv,
    validate_referential_integrity,
)

LOGGER = logging.getLogger(__name__)

TIMEZONE_NAME = "America/Mexico_City"
KEY_COLUMNS = ["date", "business_id", "branch_id", "product_id"]
SERIES_COLUMNS = ["business_id", "branch_id", "product_id"]
RAW_FILES = (
    "products",
    "branches",
    "recipes",
    "economic_events",
    "stock_movements",
    "pending_sales",
)
SUM_COLUMNS = (
    "units_sold",
    "revenue_mxn",
    "cost_mxn",
    "cancelled_units",
    "pending_units",
    "stock_on_hand",
)
PRICE_COLUMNS = ("unit_price_mxn", "unit_cost_mxn")
FLOAT_COLUMNS = SUM_COLUMNS + PRICE_COLUMNS


def _local_timezone() -> tzinfo:
    """America/Mexico_City with a fixed UTC-6 fallback (no DST since 2022)."""
    try:
        return ZoneInfo(TIMEZONE_NAME)
    except ZoneInfoNotFoundError:  # pragma: no cover - exercised only without tzdata
        return timezone(timedelta(hours=-6))


def _local_dates(values: pd.Series) -> pd.Series:
    """Convert UTC timestamps to Mexico City calendar dates."""
    return values.dt.tz_convert(_local_timezone()).dt.tz_localize(None).dt.normalize().dt.date


def _load_validated(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Load every raw contract file and enforce schema + referential integrity."""
    frames = {name: load_csv(raw_dir / f"{name}.csv", name) for name in RAW_FILES}
    integrity_errors = validate_referential_integrity(raw_dir)
    if integrity_errors:
        raise SchemaValidationError(
            "Invalid raw dataset referential integrity:\n- " + "\n- ".join(integrity_errors)
        )
    return frames


def _blank(values: pd.Series) -> pd.Series:
    return values.isna() | (values.astype(str).str.strip() == "")


def _extract_sales(events: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Filter sale lines and enrich them with master price/cost fallbacks."""
    sales = events[events["event_type"].astype(str).str.strip() == "venta"].copy()
    if sales.empty:
        raise SchemaValidationError("economic_events.csv contains no 'venta' lines")

    missing_product = _blank(sales["product_id"])
    missing_branch = _blank(sales["branch_id"])
    missing_quantity = _blank(sales["quantity"])
    non_positive = pd.to_numeric(sales["quantity"], errors="coerce").fillna(0) <= 0
    invalid = missing_product | missing_branch | missing_quantity | non_positive
    if bool(invalid.any()):
        raise SchemaValidationError(
            "economic_events.csv has "
            f"{int(invalid.sum())} invalid 'venta' line(s): product_id, branch_id and a "
            "positive quantity are required"
        )

    sales["quantity"] = pd.to_numeric(sales["quantity"], errors="coerce").astype(float)
    price_map = products.set_index("product_id")["price_mxn"]
    cost_map = products.set_index("product_id")["cost_mxn"]
    sales["unit_price_mxn"] = pd.to_numeric(sales["unit_price_mxn"], errors="coerce").fillna(
        sales["product_id"].map(price_map)
    )
    sales["unit_cost_mxn"] = pd.to_numeric(sales["unit_cost_mxn"], errors="coerce").fillna(
        sales["product_id"].map(cost_map)
    )
    sales["amount_mxn"] = pd.to_numeric(sales["amount_mxn"], errors="coerce").astype(float)
    sales["is_cancelled"] = sales["is_cancelled"].astype(int).astype(bool)
    sales["date"] = _local_dates(sales["event_ts"])
    sales["line_cost"] = sales["quantity"] * sales["unit_cost_mxn"]
    sales["price_times_quantity"] = sales["quantity"] * sales["unit_price_mxn"]
    sales["promo_line"] = sales["unit_price_mxn"] < 0.85 * sales["product_id"].map(price_map)
    return sales


def _aggregate_sales(sales: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sale lines to one row per date/business/branch/product.

    Cancelled-only keys are kept on purpose: ``cancelled_units`` must survive
    even when the ticket line was the only event of that series-day.
    """
    active = ~sales["is_cancelled"]
    working = sales.copy()
    working["sold_quantity"] = working["quantity"].where(active, 0.0)
    working["sold_revenue"] = working["amount_mxn"].where(active, 0.0)
    working["sold_cost"] = working["line_cost"].where(active, 0.0)
    working["cancelled_quantity"] = working["quantity"].where(~active, 0.0)
    working["price_times_sold"] = working["price_times_quantity"].where(active, 0.0)

    aggregated = (
        working.groupby(KEY_COLUMNS, sort=False)
        .agg(
            units_sold=("sold_quantity", "sum"),
            revenue_mxn=("sold_revenue", "sum"),
            cost_mxn=("sold_cost", "sum"),
            cancelled_units=("cancelled_quantity", "sum"),
            promo=("promo_line", "max"),
            _quantity_total=("sold_quantity", "sum"),
            _price_total=("price_times_sold", "sum"),
        )
        .reset_index()
    )
    quantity_total = aggregated["_quantity_total"].where(aggregated["_quantity_total"] > 0)
    aggregated["unit_price_mxn"] = aggregated["_price_total"] / quantity_total
    aggregated["unit_cost_mxn"] = aggregated["cost_mxn"] / quantity_total
    return aggregated.drop(columns=["_quantity_total", "_price_total"])


def _aggregate_pending(pending: pd.DataFrame) -> pd.DataFrame:
    pending = pending.copy()
    pending["date"] = _local_dates(pending["created_ts"])
    active = pending[pending["status"].astype(str).str.strip() != "cancelada"]
    pending_units = (
        active.groupby(KEY_COLUMNS, sort=False)["quantity"]
        .sum()
        .rename("pending_units")
        .reset_index()
    )
    return pending_units


def _aggregate_stock(movements: pd.DataFrame) -> pd.DataFrame:
    movements = movements.copy()
    movements["date"] = _local_dates(movements["movement_ts"])
    movements = movements.sort_values(["movement_ts", "movement_id"], kind="stable")
    stock = (
        movements.groupby(KEY_COLUMNS, sort=False)["stock_after"]
        .last()
        .rename("stock_on_hand")
        .reset_index()
    )
    return stock


def _missing_product_refs(
    events: pd.DataFrame,
    movements: pd.DataFrame,
    pending: pd.DataFrame,
    products: pd.DataFrame,
) -> int:
    product_ids = set(products["product_id"].astype(str))
    missing = 0
    for frame in (events, movements, pending):
        referenced = frame.loc[~_blank(frame["product_id"]), "product_id"].astype(str)
        missing += int((~referenced.isin(product_ids)).sum())
    return missing


def _build(raw_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    frames = _load_validated(raw_dir)
    products = frames["products"]
    branches = frames["branches"]
    events = frames["economic_events"]
    movements = frames["stock_movements"]
    pending = frames["pending_sales"]

    sales = _extract_sales(events, products)
    aggregated = _aggregate_sales(sales)
    pending_units = _aggregate_pending(pending)
    stock = _aggregate_stock(movements)

    series = (
        branches[["business_id", "branch_id"]]
        .merge(products[["product_id"]], how="cross")
        .loc[:, SERIES_COLUMNS]
        .reset_index(drop=True)
    )
    date_values = pd.concat([sales["date"], stock["date"], pending_units["date"]]).dropna()
    if date_values.empty:
        raise SchemaValidationError("raw data contains no usable timestamps")
    date_min = date_values.min()
    date_max = date_values.max()
    calendar = pd.DataFrame({"date": pd.date_range(date_min, date_max, freq="D").date})

    daily = series.merge(calendar, how="cross")
    daily = daily.merge(aggregated, on=KEY_COLUMNS, how="left")
    daily = daily.merge(pending_units, on=KEY_COLUMNS, how="left")
    daily = daily.merge(stock, on=KEY_COLUMNS, how="left")
    daily = daily.sort_values(SERIES_COLUMNS + ["date"], kind="stable").reset_index(drop=True)

    daily["stock_on_hand"] = daily.groupby(SERIES_COLUMNS, sort=False)["stock_on_hand"].ffill()
    daily["stock_on_hand"] = pd.to_numeric(
        daily["stock_on_hand"], errors="coerce"
    ).fillna(0.0)

    for column in SUM_COLUMNS:
        daily[column] = pd.to_numeric(daily[column], errors="coerce").fillna(0.0)
    for column in PRICE_COLUMNS:
        daily[column] = pd.to_numeric(daily[column], errors="coerce")
    daily["promo"] = pd.to_numeric(daily["promo"], errors="coerce").fillna(0).astype(int)

    metadata = products[
        [
            "product_id",
            "category",
            "unit",
            "lead_time_days",
            "demand_pattern",
            "abc_class",
            "price_mxn",
            "cost_mxn",
        ]
    ].rename(columns={"price_mxn": "master_price_mxn", "cost_mxn": "master_cost_mxn"})
    daily = daily.merge(metadata, on="product_id", how="left", validate="many_to_one")
    if daily["category"].isna().any():
        raise SchemaValidationError("some sale lines reference products outside products.csv")

    daily["unit_price_mxn"] = daily["unit_price_mxn"].fillna(daily["master_price_mxn"])
    daily["unit_cost_mxn"] = daily["unit_cost_mxn"].fillna(daily["master_cost_mxn"])
    daily["stockout_flag"] = (daily["stock_on_hand"] <= 0).astype(int)

    daily["date"] = daily["date"].astype(str)
    for column in FLOAT_COLUMNS:
        daily[column] = daily[column].round(2)

    duplicate_keys = int(daily.duplicated(subset=KEY_COLUMNS).sum())
    if duplicate_keys:
        raise SchemaValidationError(
            f"daily grain is not unique: {duplicate_keys} duplicated (date, branch, product) rows"
        )

    daily = daily.reindex(columns=list(CSV_COLUMNS["demand_daily"]))
    daily = daily.sort_values(KEY_COLUMNS, kind="stable").reset_index(drop=True)

    quality: dict[str, object] = {
        "rows": int(len(daily)),
        "series": int(daily[SERIES_COLUMNS].drop_duplicates().shape[0]),
        "date_min": str(daily["date"].min()),
        "date_max": str(daily["date"].max()),
        "zero_sale_share": round(float((daily["units_sold"] == 0).mean()), 4),
        "negative_stock_rows": int((daily["stock_on_hand"] < 0).sum()),
        "missing_product_refs": _missing_product_refs(events, movements, pending, products),
        "duplicate_keys": duplicate_keys,
        "cancelled_units_total": round(float(daily["cancelled_units"].sum()), 2),
        "pending_units_total": round(float(daily["pending_units"].sum()), 2),
        "stockout_rows": int(daily["stockout_flag"].sum()),
        "promo_rows": int(daily["promo"].sum()),
    }
    return daily, quality


def build_daily_demand(raw_dir: Path | str) -> pd.DataFrame:
    """Build the daily demand table from the raw contract CSVs (no files written)."""
    daily, _ = _build(Path(raw_dir))
    return daily


def run_etl(raw_dir: Path | str, out_path: Path | str) -> pd.DataFrame:
    """Build the daily demand table, write it and its quality report, return it."""
    raw = Path(raw_dir)
    out = Path(out_path)
    daily, quality = _build(raw)
    out.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(
        out,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        float_format="%.2f",
    )
    quality_path = out.parent / "data_quality.json"
    quality_path.write_text(
        json.dumps(quality, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOGGER.info(
        "Wrote %s rows / %s series to %s (quality: %s)",
        quality["rows"],
        quality["series"],
        out,
        quality_path,
    )
    return daily


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build data/processed/demand_daily.csv.")
    parser.add_argument("--raw", required=True, help="Directory containing the raw contract CSVs.")
    parser.add_argument("--out", required=True, help="Destination path for demand_daily.csv.")
    args = parser.parse_args(argv)
    try:
        daily = run_etl(args.raw, args.out)
    except (SchemaValidationError, FileNotFoundError) as exc:
        print(f"ETL failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"Wrote {len(daily)} rows for "
        f"{daily[['business_id', 'branch_id', 'product_id']].drop_duplicates().shape[0]} "
        f"series to {args.out}"
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
