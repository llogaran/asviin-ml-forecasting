"""Feature engineering for the daily demand model (SPEC 4.3).

The modeling grain is one row per ``(business, branch, product, date)`` and the
target is ``y = log1p(units_sold)``. Features are split into four groups:

* **Demand history** -- ``lag_*``/``roll_*``/``days_since_last_sale``. Every one
  of them is computed from data strictly before the target day, so the trained
  model never sees the value it is trying to predict.
* **Calendar** -- `dow`, `month`, `is_weekend`, `sin_doy`, `cos_doy`. These are
  known in advance for any future date, so they are not lagged.
* **Inventory covariates** -- `promo_lag1`, `stockout_flag_lag1`, `pending_units`,
  `unit_price_mxn`, `unit_cost_mxn`, `margin_pct`, `stock_to_demand`. The ETL
  writes these for the target day but they are only known after that day closes
  (pending sales are created intraday and average prices depend on the sales of
  the day), so they are also shifted by one day. Keeping them same-day would
  leak the target.
* **Static product/branch attributes** -- `lead_time_days` and the encoded
  categorical ids. They are constant per series and safe to use unshifted.

Fresh series (shorter than ``MIN_HISTORY_DAYS``) still get a complete feature
row: missing history is filled with zeros, which matches the ETL's zero-filled
semantics. This keeps training, validation and recursive serving on the exact
same code path.

``meta["encoders"]`` maps each raw categorical column to a ``value -> int`` map
(nested dict). Unseen categories encode to ``-1`` at inference time.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

SERIES_KEYS: tuple[str, ...] = ("business_id", "branch_id", "product_id")
DATE_COLUMN = "date"
TARGET_COLUMN = "units_sold"
TARGET_TRANSFORM = "log1p"
TARGET_FEATURE = "y"

FEATURES: list[str] = [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "roll_mean_7",
    "roll_mean_14",
    "roll_mean_28",
    "roll_std_7",
    "days_since_last_sale",
    "dow",
    "month",
    "is_weekend",
    "sin_doy",
    "cos_doy",
    "promo_lag1",
    "stockout_flag_lag1",
    "pending_units",
    "unit_price_mxn",
    "unit_cost_mxn",
    "margin_pct",
    "lead_time_days",
    "stock_to_demand",
    "product_code",
    "branch_code",
    "category_code",
    "unit_code",
    "demand_pattern_code",
]

LAG_DAYS: tuple[int, ...] = (1, 7, 14, 28)
ROLLING_WINDOWS: tuple[int, ...] = (7, 14, 28)
MIN_HISTORY_DAYS = 28

#: No sale seen in the lookback window encodes as this many days.
MAX_DAYS_SINCE_LAST_SALE = 60.0
#: ``stock_on_hand / mean_daily_demand`` is clipped to this value for tree safety.
STOCK_TO_DEMAND_CAP = 90.0

#: Categorical source column -> encoded feature column.
ENCODER_COLUMNS: dict[str, str] = {
    "product_id": "product_code",
    "branch_id": "branch_code",
    "category": "category_code",
    "unit": "unit_code",
    "demand_pattern": "demand_pattern_code",
}

_NUMERIC_COLUMNS: tuple[str, ...] = (
    "units_sold",
    "pending_units",
    "stock_on_hand",
    "promo",
    "stockout_flag",
    "unit_price_mxn",
    "unit_cost_mxn",
    "lead_time_days",
)

_REQUIRED_COLUMNS: tuple[str, ...] = (
    *SERIES_KEYS,
    DATE_COLUMN,
    "category",
    "unit",
    "demand_pattern",
    *_NUMERIC_COLUMNS,
)

_DEFAULT_ROW_COLUMNS: tuple[str, ...] = (*SERIES_KEYS, DATE_COLUMN, TARGET_FEATURE, TARGET_COLUMN)


def _empty_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=[*_DEFAULT_ROW_COLUMNS, *FEATURES])


def _check_columns(daily: pd.DataFrame) -> None:
    missing = [column for column in _REQUIRED_COLUMNS if column not in daily.columns]
    if missing:
        raise ValueError(f"daily demand frame is missing required columns: {missing}")


def _prepare(daily: pd.DataFrame) -> pd.DataFrame:
    """Normalize dtypes and sort by series + date without mutating the input."""
    frame = daily.copy()
    frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN], errors="coerce")
    if frame[DATE_COLUMN].isna().any():
        raise ValueError("daily demand frame contains unparseable dates")
    for column in _NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype(float)
    for column in ("business_id", "branch_id", "product_id", "category", "unit", "demand_pattern"):
        frame[column] = frame[column].fillna("").astype(str)
    frame = frame.sort_values([*SERIES_KEYS, DATE_COLUMN], kind="stable").reset_index(drop=True)
    return frame


def _shift(daily: pd.DataFrame, column: str, periods: int = 1) -> pd.Series:
    """``column`` shifted inside each series (never across series boundaries)."""
    return daily.groupby(list(SERIES_KEYS), sort=False)[column].shift(periods)


def _shifted_rolling(daily: pd.DataFrame, column: str, window: int, statistic: str) -> pd.Series:
    """Rolling ``statistic`` over the shifted series, using only past values."""
    shifted = _shift(daily, column)
    grouper = [daily[key] for key in SERIES_KEYS]
    rolling = shifted.groupby(grouper).rolling(window, min_periods=1)
    rolled = rolling.mean() if statistic == "mean" else rolling.std()
    aligned = rolled.reset_index(level=list(range(len(SERIES_KEYS))), drop=True)
    return aligned.reindex(daily.index)


def _days_since_last_sale(values: pd.Series) -> pd.Series:
    """Days since the latest positive demand, excluding the current day."""
    array = values.to_numpy(dtype=float)
    result = np.full(len(array), MAX_DAYS_SINCE_LAST_SALE, dtype=float)
    last_sale = -1
    for position, value in enumerate(array):
        if last_sale >= 0:
            result[position] = float(position - last_sale)
        if value > 0:
            last_sale = position
    return pd.Series(result, index=values.index, dtype=float)


def _stock_to_demand(stock_on_hand: pd.Series, roll_mean_7: pd.Series) -> np.ndarray:
    ratio = stock_on_hand.to_numpy(dtype=float) / (roll_mean_7.to_numpy(dtype=float) + 1.0)
    return np.clip(np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0), 0.0, STOCK_TO_DEMAND_CAP)


def _build_encoders(daily: pd.DataFrame) -> dict[str, dict[str, int]]:
    encoders: dict[str, dict[str, int]] = {}
    for column in ENCODER_COLUMNS:
        values = sorted(daily[column].astype(str).unique())
        encoders[column] = {value: index for index, value in enumerate(values)}
    return encoders


def _feature_meta(encoders: dict[str, dict[str, int]], holdout_days: int) -> dict:
    return {
        "features": list(FEATURES),
        "encoders": encoders,
        "target_transform": TARGET_TRANSFORM,
        "holdout_days": int(holdout_days),
        "min_history_days": MIN_HISTORY_DAYS,
        "series_keys": list(SERIES_KEYS),
        "encoder_columns": dict(ENCODER_COLUMNS),
    }


def build_supervised(
    daily: pd.DataFrame, holdout_days: int = 28
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build the training/validation frames and the encoder metadata.

    Returns ``(train_df, valid_df, meta)``. ``valid_df`` holds the last
    ``holdout_days`` calendar days of every series (roughly one forecast cycle);
    ``train_df`` excludes both the holdout and the first
    ``MIN_HISTORY_DAYS`` days used only as lag history. Both frames contain the
    series keys, ``date``, ``y`` (``log1p(units_sold)``), the raw target and the
    exact ``FEATURES`` columns, with no NaNs in the features.
    """
    if holdout_days < 1:
        raise ValueError("holdout_days must be >= 1")

    if daily is None or len(daily) == 0:
        LOGGER.warning("build_supervised received an empty daily frame")
        meta = _feature_meta({column: {} for column in ENCODER_COLUMNS}, holdout_days)
        return _empty_feature_frame(), _empty_feature_frame(), meta

    _check_columns(daily)
    frame = _prepare(daily)
    encoders = _build_encoders(frame)

    frame[TARGET_FEATURE] = np.log1p(frame[TARGET_COLUMN].clip(lower=0.0))
    for lag in LAG_DAYS:
        frame[f"lag_{lag}"] = _shift(frame, TARGET_COLUMN, lag)
    for window in ROLLING_WINDOWS:
        frame[f"roll_mean_{window}"] = _shifted_rolling(frame, TARGET_COLUMN, window, "mean")
    frame["roll_std_7"] = _shifted_rolling(frame, TARGET_COLUMN, 7, "std").fillna(0.0)
    frame["days_since_last_sale"] = frame.groupby(list(SERIES_KEYS), sort=False)[
        TARGET_COLUMN
    ].transform(_days_since_last_sale)

    dates = frame[DATE_COLUMN]
    day_of_week = dates.dt.dayofweek.astype(float)
    day_of_year = dates.dt.dayofyear.astype(float)
    frame["dow"] = day_of_week
    frame["month"] = dates.dt.month.astype(float)
    frame["is_weekend"] = (day_of_week >= 5).astype(float)
    frame["sin_doy"] = np.sin(2.0 * np.pi * day_of_year / 365.25)
    frame["cos_doy"] = np.cos(2.0 * np.pi * day_of_year / 365.25)

    frame["promo_lag1"] = _shift(frame, "promo")
    frame["stockout_flag_lag1"] = _shift(frame, "stockout_flag")
    frame["pending_units"] = _shift(frame, "pending_units")
    frame["unit_price_mxn"] = _shift(frame, "unit_price_mxn")
    frame["unit_cost_mxn"] = _shift(frame, "unit_cost_mxn")
    price = frame["unit_price_mxn"].to_numpy(dtype=float)
    cost = frame["unit_cost_mxn"].to_numpy(dtype=float)
    frame["margin_pct"] = np.where(price > 0, (price - cost) / np.where(price > 0, price, 1.0), 0.0)
    frame["lead_time_days"] = frame["lead_time_days"].astype(float)
    frame["stock_to_demand"] = _stock_to_demand(
        _shift(frame, "stock_on_hand"), frame["roll_mean_7"]
    )

    for column, code in ENCODER_COLUMNS.items():
        frame[code] = frame[column].astype(str).map(encoders[column]).fillna(-1).astype(float)

    feature_block = frame[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    frame[FEATURES] = feature_block.astype(float)

    position = frame.groupby(list(SERIES_KEYS), sort=False).cumcount()
    series_end = frame.groupby(list(SERIES_KEYS), sort=False)[DATE_COLUMN].transform("max")
    holdout_start = series_end - pd.Timedelta(days=holdout_days - 1)
    is_holdout = frame[DATE_COLUMN] >= holdout_start
    is_train = (~is_holdout) & (position >= MIN_HISTORY_DAYS)

    columns = [*_DEFAULT_ROW_COLUMNS, *FEATURES]
    train_df = frame.loc[is_train, columns].reset_index(drop=True)
    valid_df = frame.loc[is_holdout, columns].reset_index(drop=True)
    meta = _feature_meta(encoders, holdout_days)
    LOGGER.debug(
        "build_supervised: %s train rows, %s holdout rows, %s features",
        len(train_df),
        len(valid_df),
        len(FEATURES),
    )
    return train_df, valid_df, meta


def _empty_feature_series(features: list[str]) -> pd.Series:
    return pd.Series(0.0, index=list(features), dtype=float)


def _window_mean(array: np.ndarray, window: int) -> float:
    values = array[-window:]
    return float(values.mean()) if len(values) else 0.0


def _window_std(array: np.ndarray, window: int) -> float:
    values = array[-window:]
    return float(np.std(values, ddof=1)) if len(values) >= 2 else 0.0


def features_for_next_day(history: pd.DataFrame, meta: dict) -> pd.Series:
    """Feature row for the day after the last row of a single-series ``history``.

    Mirrors :func:`build_supervised` exactly (see ``tests/test_features.py`` for
    the train/serve consistency check) but only scans the tail of the history.
    An empty or zero-history series returns an all-zero row, which is what the
    recursive forecaster needs for a brand-new series.
    """
    features = list(meta.get("features") or FEATURES)
    encoders = meta.get("encoders") or {}
    if history is None or len(history) == 0:
        return _empty_feature_series(features)
    _check_columns(history)

    frame = history.copy()
    frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN], errors="coerce")
    if frame[DATE_COLUMN].isna().any():
        raise ValueError("history contains unparseable dates")
    for column in _NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype(float)
    for column in ("business_id", "branch_id", "product_id", "category", "unit", "demand_pattern"):
        frame[column] = frame[column].fillna("").astype(str)
    frame = frame.sort_values(DATE_COLUMN, kind="stable").reset_index(drop=True)

    if SERIES_KEYS and frame[list(SERIES_KEYS)].drop_duplicates().shape[0] > 1:
        raise ValueError(
            "features_for_next_day expects a single series; "
            f"got {frame[list(SERIES_KEYS)].drop_duplicates().shape[0]} series"
        )

    last = frame.iloc[-1]
    next_date = pd.Timestamp(frame[DATE_COLUMN].iloc[-1]) + pd.Timedelta(days=1)
    units = frame[TARGET_COLUMN].to_numpy(dtype=float)
    day_of_year = float(next_date.dayofyear)
    day_of_week = float(next_date.dayofweek)

    values: dict[str, float] = {}
    for lag in LAG_DAYS:
        values[f"lag_{lag}"] = float(units[-lag]) if len(units) >= lag else 0.0
    for window in ROLLING_WINDOWS:
        values[f"roll_mean_{window}"] = _window_mean(units, window)
    values["roll_std_7"] = _window_std(units, 7)

    days_since = MAX_DAYS_SINCE_LAST_SALE
    for offset, value in enumerate(reversed(units), start=1):
        if value > 0:
            days_since = float(offset)
            break
    values["days_since_last_sale"] = days_since
    values["dow"] = day_of_week
    values["month"] = float(next_date.month)
    values["is_weekend"] = float(day_of_week >= 5)
    values["sin_doy"] = float(np.sin(2.0 * np.pi * day_of_year / 365.25))
    values["cos_doy"] = float(np.cos(2.0 * np.pi * day_of_year / 365.25))
    values["promo_lag1"] = float(last["promo"])
    values["stockout_flag_lag1"] = float(last["stockout_flag"])

    price = float(last["unit_price_mxn"])
    cost = float(last["unit_cost_mxn"])
    values["pending_units"] = float(last["pending_units"])
    values["unit_price_mxn"] = price
    values["unit_cost_mxn"] = cost
    values["margin_pct"] = (price - cost) / price if price > 0 else 0.0
    values["lead_time_days"] = float(last["lead_time_days"])
    values["stock_to_demand"] = float(
        _stock_to_demand(
            pd.Series([float(last["stock_on_hand"])]), pd.Series([values["roll_mean_7"]])
        )[0]
    )

    for column, code in ENCODER_COLUMNS.items():
        mapping = encoders.get(column, {})
        values[code] = float(mapping.get(str(last[column]), -1))

    result = pd.Series({feature: values.get(feature, 0.0) for feature in features}, dtype=float)
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)
