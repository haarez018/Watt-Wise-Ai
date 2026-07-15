import pandas as pd
from data.generate_synthetic import generate_dataset
from features.engineering import (
    LAG_MONTHS,
    build_forecast_examples,
    encode_features,
    household_train_test_split,
)

N_HOUSEHOLDS = 15


def _small_dataset() -> pd.DataFrame:
    return generate_dataset(n_households=N_HOUSEHOLDS, seed=99)


def test_build_forecast_examples_shape() -> None:
    examples = build_forecast_examples(_small_dataset())
    months_per_household = 12 - LAG_MONTHS
    assert len(examples) == N_HOUSEHOLDS * months_per_household


def test_lag_features_match_actual_prior_months() -> None:
    df = _small_dataset()
    examples = build_forecast_examples(df)

    household_id = df["household_id"].iloc[0]
    household_df = df[df["household_id"] == household_id].sort_values("month_index")
    household_examples = examples[examples["household_id"] == household_id].sort_values(
        "target_month_index"
    )

    units_by_month = dict(
        zip(household_df["month_index"], household_df["units_consumed_wh"], strict=False)
    )

    for _, example in household_examples.iterrows():
        target_month = int(example["target_month_index"])
        assert example["lag_1_units_wh"] == units_by_month[target_month - 1]
        assert example["lag_2_units_wh"] == units_by_month[target_month - 2]
        assert example["lag_3_units_wh"] == units_by_month[target_month - 3]


def test_no_household_predicted_from_its_own_first_lag_months() -> None:
    examples = build_forecast_examples(_small_dataset())
    assert examples["target_month_index"].min() == LAG_MONTHS + 1


def test_train_test_split_has_no_household_overlap() -> None:
    examples = build_forecast_examples(_small_dataset())
    train, test = household_train_test_split(examples, test_size=0.4, seed=1)

    train_ids = set(train["household_id"])
    test_ids = set(test["household_id"])
    assert train_ids.isdisjoint(test_ids)
    assert len(train_ids) + len(test_ids) == N_HOUSEHOLDS


def test_train_test_split_is_deterministic_for_a_given_seed() -> None:
    examples = build_forecast_examples(_small_dataset())
    train_a, test_a = household_train_test_split(examples, seed=5)
    train_b, test_b = household_train_test_split(examples, seed=5)
    assert set(train_a["household_id"]) == set(train_b["household_id"])
    assert set(test_a["household_id"]) == set(test_b["household_id"])


def test_encode_features_applies_same_categories_to_train_and_test() -> None:
    examples = build_forecast_examples(_small_dataset())
    train, test = household_train_test_split(examples, test_size=0.3, seed=2)

    X_train, y_train, categories = encode_features(train)
    X_test, y_test, _ = encode_features(test, categories=categories)

    assert list(X_train.columns) == list(X_test.columns)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_encode_features_one_hot_columns_are_mutually_exclusive() -> None:
    examples = build_forecast_examples(_small_dataset())
    features, _target, categories = encode_features(examples)

    for column in categories["zone"]:
        assert features[f"zone_{column}"].isin([0.0, 1.0]).all()
    zone_columns = [f"zone_{c}" for c in categories["zone"]]
    assert (features[zone_columns].sum(axis=1) == 1.0).all()
