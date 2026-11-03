"""Tests for feature engineering, holdout split and train/serve consistency (SPEC 4.3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.pipelines.features import (
    FEATURES,
    SERIES_KEYS,
    build_supervised,
    features_for_next_day,
)

START = pd.Timestamp("2024-01-01")


def _make_daily(
    days: int = 96,
    products: tuple[str, ...] = ("P-001", "P-002"),
    branches: tuple[str, ...] = ("B-01", "B-02"),
) -> pd.DataFrame:
    """Small deterministic panel with one intermittent series and one smooth one."""
    rows = []
    for product_index, product_id in enumerate(products):
        for branch_id in branches:
            for day_index in range(days):
                intermittent = product_id == "P-002" and day_index % 5 == 0
                units = 0.0 if intermittent else float((day_index + 3 * product_index) % 7 + 1)
                rows.append(
                    {
                        "date": START + pd.Timedelta(days=day_index),
                        "business_id": "BIZ-001",
                        "branch_id": branch_id,
                        "product_id": product_id,
                        "category": "Producto Terminado" if product_index else "Materia Prima",
                        "unit": "pza" if product_index else "kg",
                        "units_sold": units,
                        "revenue_mxn": units * 3.0,
                        "cost_mxn": units * 2.0,
                        "cancelled_units": 0.0,
                        "pending_units": 1.0 if day_index % 11 == 0 else 0.0,
                        "stock_on_hand": max(0.0, 120.0 - day_index),
                        "stockout_flag": int(day_index >= 119),
                        "promo": int(day_index % 13 == 0),
                        "unit_price_mxn": 3.0,
                        "unit_cost_mxn": 2.0,
                        "lead_time_days": 5,
                        "demand_pattern": "intermittent" if product_index else "smooth",
                        "abc_class": "A" if product_index == 0 else "C",
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def daily() -> pd.DataFrame:
    return _make_daily()


@pytest.fixture(scope="module")
def supervised(daily: pd.DataFrame):
    return build_supervised(daily, holdout_days=14)


def test_feature_contract_and_target(supervised) -> None:
    train_df, valid_df, meta = supervised
    assert meta["features"] == FEATURES
    assert meta["target_transform"] == "log1p"
    assert set(meta["encoders"]) == {
        "product_id",
        "branch_id",
        "category",
        "unit",
        "demand_pattern",
    }
    assert meta["encoders"]["product_id"] == {"P-001": 0, "P-002": 1}
    for frame in (train_df, valid_df):
        assert list(frame.columns) == [
            "business_id",
            "branch_id",
            "product_id",
            "date",
            "y",
            "units_sold",
            *FEATURES,
        ]
        assert frame[FEATURES].notna().all().all()
        assert np.isfinite(frame[FEATURES].to_numpy(dtype=float)).all()
        assert np.allclose(frame["y"], np.log1p(frame["units_sold"]))
        for feature in FEATURES:
            assert pd.api.types.is_numeric_dtype(frame[feature]), feature
    assert len(train_df) > 0
    assert len(valid_df) == 14 * 4


def test_holdout_is_disjoint_and_skips_history(supervised, daily: pd.DataFrame) -> None:
    train_df, valid_df, _ = supervised
    series_start = daily.groupby(list(SERIES_KEYS))["date"].min()
    for key, holdout_group in valid_df.groupby(list(SERIES_KEYS), sort=True):
        train_group = train_df[
            (train_df["business_id"] == key[0])
            & (train_df["branch_id"] == key[1])
            & (train_df["product_id"] == key[2])
        ]
        assert len(holdout_group) == 14
        assert holdout_group["date"].max() == START + pd.Timedelta(days=95)
        assert train_group["date"].max() < holdout_group["date"].min()
        assert not set(train_group["date"]).intersection(set(holdout_group["date"]))
        assert (train_group["date"].min() - series_start.loc[key]).days >= 28


def test_target_is_not_used_as_a_same_day_feature(supervised, daily: pd.DataFrame) -> None:
    """Changing demand on day D must not alter that day's features, only later ones."""
    train_df, _, _ = supervised
    changed_date = START + pd.Timedelta(days=60)
    next_date = changed_date + pd.Timedelta(days=1)
    mutated = daily.copy()
    mask = mutated["date"] == changed_date
    mutated.loc[mask, "units_sold"] = mutated.loc[mask, "units_sold"] + 100.0
    mutated_train, _, _ = build_supervised(mutated, holdout_days=14)

    def _rows(frame: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
        return (
            frame[frame["date"] == date]
            .sort_values(["branch_id", "product_id"])
            .reset_index(drop=True)
        )

    original_day = _rows(train_df, changed_date)[FEATURES]
    mutated_day = _rows(mutated_train, changed_date)[FEATURES]
    pd.testing.assert_frame_equal(original_day, mutated_day)

    # The shifted features for the following day must see the new demand.
    original_next = _rows(train_df, next_date)
    mutated_next = _rows(mutated_train, next_date)
    assert not np.allclose(original_next["lag_1"], mutated_next["lag_1"])
    assert np.allclose(mutated_next["lag_1"], original_next["lag_1"] + 100.0)


def test_lag_1_equals_previous_day_demand(supervised, daily: pd.DataFrame) -> None:
    train_df, _, _ = supervised
    day = START + pd.Timedelta(days=70)
    previous = daily[daily["date"] == day - pd.Timedelta(days=1)]
    current = train_df[train_df["date"] == day].sort_values(["branch_id", "product_id"])
    expected = previous.sort_values(["branch_id", "product_id"])["units_sold"].to_numpy()
    assert np.allclose(current["lag_1"].to_numpy(), expected)


def test_features_for_next_day_matches_training_row(supervised, daily: pd.DataFrame) -> None:
    """Inference features must be identical to the row the model trained on."""
    _, valid_df, meta = supervised
    for key, holdout_group in valid_df.groupby(list(SERIES_KEYS), sort=True):
        first_date = holdout_group["date"].min()
        history = daily[
            (daily["business_id"] == key[0])
            & (daily["branch_id"] == key[1])
            & (daily["product_id"] == key[2])
            & (daily["date"] < first_date)
        ]
        expected = holdout_group.sort_values("date").iloc[0][FEATURES].astype(float)
        produced = features_for_next_day(history, meta)
        assert list(produced.index) == FEATURES
        assert np.allclose(produced.to_numpy(dtype=float), expected.to_numpy(dtype=float))


def test_short_and_empty_history_are_tolerated(daily: pd.DataFrame) -> None:
    _, _, meta = build_supervised(daily, holdout_days=14)
    empty = features_for_next_day(daily.iloc[0:0], meta)
    assert list(empty.index) == FEATURES
    assert (empty == 0.0).all()

    short_history = daily[
        (daily["product_id"] == "P-001")
        & (daily["branch_id"] == "B-01")
        & (daily["date"] < START + pd.Timedelta(days=3))
    ]
    row = features_for_next_day(short_history, meta)
    assert row.notna().all()
    assert np.isfinite(row.to_numpy(dtype=float)).all()
    assert row["lag_7"] == 0.0
    assert row["lag_1"] == short_history.sort_values("date").iloc[-1]["units_sold"]
    assert row["product_code"] == meta["encoders"]["product_id"]["P-001"]

    short_train, short_valid, _ = build_supervised(_make_daily(days=20), holdout_days=14)
    assert short_train.empty
    assert len(short_valid) == 14 * 4
    assert short_valid[FEATURES].notna().all().all()


def test_unknown_category_encodes_to_minus_one(daily: pd.DataFrame) -> None:
    _, _, meta = build_supervised(daily, holdout_days=14)
    history = (
        daily[(daily["product_id"] == "P-001") & (daily["branch_id"] == "B-01")].head(5).copy()
    )
    history["product_id"] = "P-999"
    row = features_for_next_day(history, meta)
    assert row["product_code"] == -1.0
