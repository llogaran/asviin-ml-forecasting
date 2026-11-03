"""Tests for the deterministic synthetic ERP generator (SPEC 3.1-3.6, 4.1)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest
from src.data_contracts.schemas import CSV_COLUMNS
from src.data_contracts.validation import validate_csv, validate_referential_integrity
from src.pipelines.synthetic_generator import generate_dataset, main

RAW_NAMES = (
    "products",
    "branches",
    "recipes",
    "economic_events",
    "stock_movements",
    "pending_sales",
)
SERIES_KEY = ["branch_id", "product_id"]


def _local_dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True).dt.tz_convert("America/Mexico_City").dt.date


def _sales(paths: dict[str, Path]) -> pd.DataFrame:
    events = pd.read_csv(paths["economic_events"])
    sales = events[events["event_type"] == "venta"].copy()
    sales["date"] = _local_dates(sales["event_ts"])
    return sales


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    out = tmp_path_factory.mktemp("generator-small")
    return generate_dataset(out, days=120, n_products=8, n_branches=2, seed=11)


@pytest.fixture(scope="module")
def full_dataset(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    out = tmp_path_factory.mktemp("generator-full-catalog")
    return generate_dataset(out, days=365, n_products=20, n_branches=3, seed=7)


def test_files_match_contract_and_referential_integrity(dataset: dict[str, Path]) -> None:
    for name in RAW_NAMES:
        header = dataset[name].read_text(encoding="utf-8").splitlines()[0]
        assert header == ",".join(CSV_COLUMNS[name])
        assert validate_csv(dataset[name], name) == []
    assert validate_referential_integrity(dataset["raw_dir"]) == []


def test_product_and_recipe_realism(dataset: dict[str, Path]) -> None:
    products = pd.read_csv(dataset["products"])
    assert list(products["product_id"]) == [f"P-{index:03d}" for index in range(1, 9)]
    assert int(products["is_ingredient"].sum()) == 2
    assert int(products["has_recipe"].sum()) == 2
    assert not (products["is_ingredient"].astype(bool) & products["has_recipe"].astype(bool)).any()
    cost_ratio = products["cost_mxn"] / products["price_mxn"]
    assert cost_ratio.between(0.55, 0.80).all()
    assert products["price_mxn"].gt(0).all()
    assert products["lead_time_days"].between(3, 21).all()
    assert products["safety_stock"].ge(0).all()
    assert set(products["demand_pattern"]) <= {"smooth", "intermittent", "lumpy"}
    assert set(products["abc_class"]) <= {"A", "B", "C"}

    recipes = pd.read_csv(dataset["recipes"])
    ingredient_ids = set(products.loc[products["is_ingredient"] == 1, "product_id"])
    recipe_outputs = set(products.loc[products["has_recipe"] == 1, "product_id"])
    assert not recipes.empty
    assert set(recipes["input_product_id"]) <= ingredient_ids
    assert set(recipes["output_product_id"]) <= recipe_outputs
    assert recipes["quantity_per_unit"].gt(0).all()


def test_full_catalog_shape_and_branches(full_dataset: dict[str, Path]) -> None:
    products = pd.read_csv(full_dataset["products"])
    branches = pd.read_csv(full_dataset["branches"])
    assert len(products) == 20
    assert list(products["product_id"]) == [f"P-{index:03d}" for index in range(1, 21)]
    assert int(products["is_ingredient"].sum()) == 6
    assert int(products["has_recipe"].sum()) == 5
    assert list(branches["branch_id"]) == ["B-01", "B-02", "B-03"]
    assert list(branches["name"]) == ["Matriz Centro", "Sucursal Norte", "Bodega Sur"]
    abc = products.loc[products["is_ingredient"] == 0, "abc_class"].value_counts()
    assert abc.get("A", 0) == 3 and abc.get("B", 0) == 4 and abc.get("C", 0) == 7

    sales = _sales(full_dataset)
    active = sales[sales["is_cancelled"] == 0]
    assert set(active["product_id"]) == set(products["product_id"])
    ingredient_ids = set(products.loc[products["is_ingredient"] == 1, "product_id"])
    ingredient_share = (
        active.loc[active["product_id"].isin(ingredient_ids), "quantity"].sum()
        / active["quantity"].sum()
    )
    assert 0.01 < float(ingredient_share) < 0.20


def test_branch_volume_multipliers(full_dataset: dict[str, Path]) -> None:
    sales = _sales(full_dataset)
    units = sales.loc[sales["is_cancelled"] == 0].groupby("branch_id")["quantity"].sum()
    ratio_north = units["B-02"] / units["B-01"]
    ratio_south = units["B-03"] / units["B-01"]
    assert ratio_north == pytest.approx(0.62, abs=0.06)
    assert ratio_south == pytest.approx(0.38, abs=0.06)


def test_dates_and_timestamps(dataset: dict[str, Path]) -> None:
    sales = _sales(dataset)
    assert sales["date"].min().isoformat() == "2023-01-01"
    assert sales["date"].max().isoformat() == "2023-04-30"
    assert set(sales["business_id"]) == {"BIZ-001"}

    timestamp_pattern = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}")
    for name in ("economic_events", "stock_movements", "pending_sales"):
        timestamps = timestamp_pattern.findall(dataset[name].read_text(encoding="utf-8"))
        assert timestamps
        assert all(stamp.endswith("-06:00") for stamp in timestamps)


def test_seed_json_is_deterministic_metadata(dataset: dict[str, Path]) -> None:
    payload = json.loads(dataset["seed"].read_text(encoding="utf-8"))
    assert payload["seed"] == 11
    assert payload["days"] == 120
    assert payload["n_products"] == 8
    assert payload["n_branches"] == 2
    assert payload["start_date"] == "2023-01-01"
    assert payload["end_date"] == "2023-04-30"
    assert payload["business_id"] == "BIZ-001"
    text = dataset["seed"].read_text(encoding="utf-8")
    assert not re.search(r"\d{4}-\d{2}-\d{2}T", text)
    assert "generated_at" not in text.lower()
    assert payload["row_counts"]["products"] == 8


def test_generation_is_byte_identical_for_same_seed(tmp_path: Path) -> None:
    first = generate_dataset(tmp_path / "a", days=45, n_products=6, n_branches=2, seed=3)
    second = generate_dataset(tmp_path / "b", days=45, n_products=6, n_branches=2, seed=3)
    for name in (*RAW_NAMES, "seed"):
        assert first[name].read_bytes() == second[name].read_bytes()
    third = generate_dataset(tmp_path / "c", days=45, n_products=6, n_branches=2, seed=4)
    assert first["economic_events"].read_bytes() != third["economic_events"].read_bytes()


def test_no_negative_stock_and_balanced_running_stock(dataset: dict[str, Path]) -> None:
    movements = pd.read_csv(dataset["stock_movements"])
    assert (movements["stock_after"] >= 0).all()
    assert (movements["quantity"] > 0).all()
    signed = movements["quantity"].where(movements["direction"] == "entrada", -movements["quantity"])
    ordered = movements.assign(signed=signed).sort_values(["movement_ts", "movement_id"])
    ordered["running"] = ordered.groupby(SERIES_KEY)["signed"].cumsum()
    assert ((ordered["running"] - ordered["stock_after"]).abs() <= 0.011).all()
    assert set(movements["product_id"]) <= set(pd.read_csv(dataset["products"])["product_id"])


def test_identifiers_ticket_grouping_and_non_sale_fields(dataset: dict[str, Path]) -> None:
    events = pd.read_csv(dataset["economic_events"])
    movements = pd.read_csv(dataset["stock_movements"])
    pending = pd.read_csv(dataset["pending_sales"])
    assert movements["movement_id"].is_unique
    assert pending["pending_sale_id"].is_unique

    sales = events[events["event_type"] == "venta"]
    ticket_sizes = sales.groupby("event_id").size()
    assert ticket_sizes.max() <= 3
    assert (ticket_sizes > 1).any()
    non_sales = events[events["event_type"] != "venta"]
    assert non_sales["event_id"].is_unique
    assert non_sales["payment_method"].isna().all()
    assert non_sales["customer_id"].isna().all()
    assert (sales["is_cancelled"].isin([0, 1])).all()


def test_cancelled_promo_and_pending_shares(dataset: dict[str, Path]) -> None:
    sales = _sales(dataset)
    products = pd.read_csv(dataset["products"]).set_index("product_id")
    master_price = sales["product_id"].map(products["price_mxn"])
    discount = 1.0 - sales["unit_price_mxn"] / master_price
    promo_line = discount > 0.001

    assert 0.005 <= float(sales["is_cancelled"].mean()) <= 0.05
    assert 0.03 <= float(promo_line.mean()) <= 0.13
    assert (discount[promo_line] >= 0.09).all()
    assert (discount[promo_line] <= 0.26).all()
    assert (sales.loc[promo_line, "unit_price_mxn"] < 0.85 * master_price[promo_line]).all()
    promo_day_share = sales.loc[promo_line, "date"].nunique() / sales["date"].nunique()
    assert 0.03 < float(promo_day_share) < 0.15

    pending = pd.read_csv(dataset["pending_sales"])
    assert set(pending["status"]) <= {"pendiente", "convertida", "cancelada"}
    assert pending["promised_ts"].notna().all()
    sold_units = sales.loc[sales["is_cancelled"] == 0, "quantity"].sum()
    active_pending = pending.loc[pending["status"] != "cancelada", "quantity"].sum()
    assert 0.04 <= float(active_pending / sold_units) <= 0.18
    converted = pending[pending["status"] == "convertida"]
    assert not converted.empty
    assert converted["converted_event_id"].notna().all()
    event_ids = set(pd.read_csv(dataset["economic_events"])["event_id"])
    assert set(converted["converted_event_id"]) <= event_ids
    assert pending.loc[pending["status"] != "convertida", "converted_event_id"].isna().all()


def test_zero_sale_share_and_intermittency(dataset: dict[str, Path]) -> None:
    sales = _sales(dataset)
    active = sales[sales["is_cancelled"] == 0]
    grouped = active.groupby(["date", "branch_id", "product_id"])["quantity"].sum()
    dates = sorted(sales["date"].unique())
    branches = sorted(sales["branch_id"].unique())
    product_ids = sorted(sales["product_id"].unique())
    index = pd.MultiIndex.from_product(
        [dates, branches, product_ids], names=["date", "branch_id", "product_id"]
    )
    series = grouped.reindex(index, fill_value=0.0)
    zero_share = float((series == 0).mean())
    assert 0.10 < zero_share < 0.85
    per_series = (series == 0).groupby(level=SERIES_KEY).mean()
    assert per_series.min() < 0.30
    assert per_series.max() > 0.40
    assert (series.groupby(level=SERIES_KEY).sum() > 0).all()


def test_weekly_and_annual_seasonality(full_dataset: dict[str, Path]) -> None:
    sales = _sales(full_dataset)
    active = sales[sales["is_cancelled"] == 0]
    daily = active.groupby("date")["quantity"].sum()
    frame = pd.DataFrame({"date": daily.index, "units": daily.values})
    frame["date"] = pd.to_datetime(frame["date"])
    frame["month"] = frame["date"].dt.month
    frame["dow"] = frame["date"].dt.dayofweek

    monthly = frame.groupby("month")["units"].mean()
    relative = monthly / monthly.mean()
    assert 1.15 < relative.loc[12] < 1.65
    assert 0.65 < relative.loc[5] < 0.95
    march_april = frame.loc[frame["month"].isin([3, 4]), "units"].mean()
    assert march_april > monthly.loc[5] * 1.08
    january_february = frame.loc[frame["month"].isin([1, 2]), "units"].mean()
    assert march_april > january_february * 1.03

    weekday = frame.groupby("dow")["units"].mean()
    assert weekday.loc[3:5].mean() > weekday.loc[0:2].mean() * 1.08


def test_utf8_names_and_two_decimal_formatting(dataset: dict[str, Path]) -> None:
    products_text = dataset["products"].read_text(encoding="utf-8")
    assert "plástico" in products_text
    for name in RAW_NAMES:
        text = dataset[name].read_text(encoding="utf-8")
        assert not re.search(r"\d+\.\d{3,}", text)


def test_invalid_arguments_are_rejected(tmp_path: Path) -> None:
    invalid = (
        {"n_products": 1},
        {"n_products": 21},
        {"n_branches": 0},
        {"n_branches": 4},
        {"days": 0},
    )
    for kwargs in invalid:
        with pytest.raises(ValueError):
            generate_dataset(tmp_path / "bad", **kwargs)
    with pytest.raises(ValueError):
        generate_dataset(tmp_path / "bad-start", start="not-a-date")


def test_cli_entry_point(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "--out",
            str(tmp_path / "cli"),
            "--days",
            "30",
            "--products",
            "6",
            "--branches",
            "2",
            "--seed",
            "5",
        ]
    )
    assert code == 0
    assert (tmp_path / "cli" / "raw" / "products.csv").exists()
    assert (tmp_path / "cli" / "raw" / "seed.json").exists()
    assert "Generated" in capsys.readouterr().out
