"""Shared feature engineering for the household-month dataset.

Used by Model 1 (forecaster), Model 2 (anomaly detector, which builds on
Model 1's predictions), and Model 3 (disaggregator). Models 1/2 use a sliding
3-month lag window per household: given the previous 3 months of billed units
plus static household profile and the target month's known climate, predict
units_consumed_wh for the target month. See ml/MODELS.md for why the target
is units, never amount_paise, directly. Each example also carries the target
month's `is_anomaly` / `anomaly_reason` ground truth (unused by Model 1's
`encode_features`, which only selects its own named feature columns; used by
Model 2 as labels).

Model 3 (disaggregator) is a direct per-row mapping (no history/lag needed —
a month's appliance-category breakdown depends only on that month's own
context), so its feature builder is simpler than the lag-window one above."""

import numpy as np
import pandas as pd
from data.generate_synthetic import APPLIANCE_CATEGORIES

LAG_MONTHS = 3

CATEGORICAL_COLUMNS = ["zone", "tariff_name"]
NUMERIC_FEATURE_COLUMNS = [
    "lag_1_units_wh",
    "lag_2_units_wh",
    "lag_3_units_wh",
    "rolling_mean_3_units_wh",
    "rolling_std_3_units_wh",
    "family_size",
    "sanctioned_load_kw",
    "target_month_temp_c",
    "target_month_sin",
    "target_month_cos",
]


def build_forecast_examples(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (household, target month) with `LAG_MONTHS` of prior
    history as features. The first `LAG_MONTHS` months of each household are
    lag-only context and never appear as a prediction target."""
    examples: list[dict[str, object]] = []

    for household_id, group in df.groupby("household_id"):
        group = group.sort_values("month_index").reset_index(drop=True)
        units = group["units_consumed_wh"].to_numpy(dtype=float)

        for target_idx in range(LAG_MONTHS, len(group)):
            lags = units[target_idx - LAG_MONTHS : target_idx][::-1]  # most recent first
            row = group.iloc[target_idx]
            examples.append(
                {
                    "household_id": household_id,
                    "target_month_index": int(row["month_index"]),
                    "lag_1_units_wh": float(lags[0]),
                    "lag_2_units_wh": float(lags[1]),
                    "lag_3_units_wh": float(lags[2]),
                    "rolling_mean_3_units_wh": float(np.mean(lags)),
                    "rolling_std_3_units_wh": float(np.std(lags)),
                    "family_size": int(row["family_size"]),
                    "sanctioned_load_kw": float(row["sanctioned_load_kw"]),
                    "zone": str(row["zone"]),
                    "tariff_name": str(row["tariff_name"]),
                    "target_month_temp_c": float(row["climate_temp_c"]),
                    "target_month_sin": float(np.sin(2 * np.pi * row["month_index"] / 12)),
                    "target_month_cos": float(np.cos(2 * np.pi * row["month_index"] / 12)),
                    "target_units_wh": float(row["units_consumed_wh"]),
                    "is_anomaly": bool(row["is_anomaly"]),
                    "anomaly_reason": str(row["anomaly_reason"]),
                }
            )

    return pd.DataFrame(examples)


def household_train_test_split(
    examples: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits by household_id so no household appears on both sides — a
    row-level split would leak a household's own consumption pattern from
    train into test via the lag features."""
    household_ids = examples["household_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(household_ids)

    n_test = int(len(household_ids) * test_size)
    test_ids = set(household_ids[:n_test])

    is_test = examples["household_id"].isin(test_ids)
    train = examples.loc[~is_test].reset_index(drop=True)
    test = examples.loc[is_test].reset_index(drop=True)
    return train, test


def encode_features(
    examples: pd.DataFrame, categories: dict[str, list[str]] | None = None
) -> tuple[pd.DataFrame, pd.Series, dict[str, list[str]]]:
    """One-hot encodes the categorical columns. `categories` fixes the encoding
    to a known category order (used at inference time, and when encoding a
    test split against categories learned from the train split)."""
    df = examples.copy()
    if categories is None:
        categories = {col: sorted(df[col].unique().tolist()) for col in CATEGORICAL_COLUMNS}

    for col in CATEGORICAL_COLUMNS:
        for category in categories[col]:
            df[f"{col}_{category}"] = (df[col] == category).astype(float)

    feature_columns = NUMERIC_FEATURE_COLUMNS + [
        f"{col}_{category}" for col in CATEGORICAL_COLUMNS for category in categories[col]
    ]
    features = df[feature_columns]
    target = df["target_units_wh"]
    return features, target, categories


# --- Model 3 (disaggregator) ---

DISAGGREGATION_CATEGORICAL_COLUMNS = ["zone", "tariff_name"]
DISAGGREGATION_CONTEXT_COLUMNS = [
    "total_units_wh",
    "family_size",
    "sanctioned_load_kw",
    "climate_temp_c",
    "month_sin",
    "month_cos",
]
# Appliance-inventory columns — exactly what the product's onboarding wizard
# collects. Present in the full feature set; deliberately excluded from the
# "ablated" feature set used to measure how much of Model 3's accuracy is
# structural (having the inventory as input at all) vs. from generic context
# alone. See ml/MODELS.md's Model 3 section.
DISAGGREGATION_APPLIANCE_COLUMNS = [
    "fridge_star",
    "owns_ac",
    "ac_star",
    "owns_geyser",
    "geyser_star",
    "num_fans",
    "fan_star",
    "num_bulbs",
    "owns_washing_machine",
    "owns_tv",
]
SHARE_COLUMNS = [f"{category}_share" for category in APPLIANCE_CATEGORIES]


def build_disaggregation_examples(df: pd.DataFrame) -> pd.DataFrame:
    """One row per household-month — no lag window, since a month's
    appliance-category breakdown depends only on that month's own context.

    Target: `{category}_share` for each of the 8 categories, always summing
    to 1.0 — computed as `category_kwh / sum(all category_kwh)`, i.e. the
    *true* (pre-anomaly) appliance breakdown, not a share of the possibly
    anomaly-inflated `units_consumed_wh`. During an anomalous month the input
    feature `total_units_wh` (the actual metered reading) and the true
    category shares can therefore disagree somewhat — which is realistic: an
    anomalous bill doesn't necessarily change a household's appliance mix."""
    result = df.copy()
    category_kwh_columns = [f"{category}_kwh" for category in APPLIANCE_CATEGORIES]
    true_total_kwh = result[category_kwh_columns].sum(axis=1)

    for category in APPLIANCE_CATEGORIES:
        result[f"{category}_share"] = result[f"{category}_kwh"] / true_total_kwh

    result["total_units_wh"] = result["units_consumed_wh"].astype(float)
    result["month_sin"] = np.sin(2 * np.pi * result["month_index"] / 12)
    result["month_cos"] = np.cos(2 * np.pi * result["month_index"] / 12)
    for column in ("owns_ac", "owns_geyser", "owns_washing_machine", "owns_tv"):
        result[column] = result[column].astype(float)

    return result


def encode_disaggregation_features(
    examples: pd.DataFrame,
    categories: dict[str, list[str]] | None = None,
    include_appliance_inventory: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    """Mirrors `encode_features`'s one-hot pattern. `include_appliance_inventory=False`
    builds the ablated feature set (see `DISAGGREGATION_APPLIANCE_COLUMNS`)."""
    df = examples.copy()
    if categories is None:
        categories = {
            col: sorted(df[col].unique().tolist()) for col in DISAGGREGATION_CATEGORICAL_COLUMNS
        }

    for col in DISAGGREGATION_CATEGORICAL_COLUMNS:
        for category in categories[col]:
            df[f"{col}_{category}"] = (df[col] == category).astype(float)

    appliance_columns = DISAGGREGATION_APPLIANCE_COLUMNS if include_appliance_inventory else []
    feature_columns = (
        DISAGGREGATION_CONTEXT_COLUMNS
        + appliance_columns
        + [
            f"{col}_{category}"
            for col in DISAGGREGATION_CATEGORICAL_COLUMNS
            for category in categories[col]
        ]
    )
    features = df[feature_columns]
    targets = df[SHARE_COLUMNS]
    return features, targets, categories


# --- Model 4 (recommender) ---


def build_recommender_examples(df: pd.DataFrame) -> pd.DataFrame:
    """One row per household-month (months 4-12 only, since it needs Model
    1/2's lag-window context), combining `build_forecast_examples`'s lag
    features (for running the forecaster/anomaly detector at serving time)
    with `build_disaggregation_examples`'s true category shares and
    appliance inventory (for running the disaggregator, and for computing
    the ground-truth "achievable savings" ceiling Model 4 is evaluated
    against). Merged on `(household_id, month_index)` — the same
    household-month, described by both feature sets."""
    forecast_examples = build_forecast_examples(df)
    disagg_examples = build_disaggregation_examples(df)
    return forecast_examples.merge(
        disagg_examples,
        left_on=["household_id", "target_month_index"],
        right_on=["household_id", "month_index"],
        suffixes=("", "_disagg"),
    )
