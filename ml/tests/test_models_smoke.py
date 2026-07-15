import tempfile
from pathlib import Path

import pandas as pd
import xgboost as xgb
from data.generate_synthetic import generate_dataset
from features.engineering import build_forecast_examples, encode_features
from models.forecaster import load_artifact, save_artifact, train


def test_forecaster_trains_on_a_small_dataset() -> None:
    df = generate_dataset(n_households=200, seed=11)
    boosters, metrics, metadata = train(df, seed=11)

    assert set(boosters) == {"point", "lower", "upper"}
    assert all(isinstance(b, xgb.Booster) for b in boosters.values())
    assert metrics["n_train_examples"] > 0
    assert metrics["n_test_examples"] > 0
    assert metadata["feature_columns"]


def test_forecaster_predicts_on_a_known_input_with_lower_le_point_le_upper() -> None:
    df = generate_dataset(n_households=200, seed=12)
    boosters, _metrics, metadata = train(df, seed=12)
    feature_columns = metadata["feature_columns"]

    example = pd.DataFrame([dict.fromkeys(feature_columns, 0.0)])
    example["lag_1_units_wh"] = 150_000.0
    example["lag_2_units_wh"] = 145_000.0
    example["lag_3_units_wh"] = 155_000.0
    example["rolling_mean_3_units_wh"] = 150_000.0
    example["rolling_std_3_units_wh"] = 4_000.0
    example["family_size"] = 4
    example["sanctioned_load_kw"] = 4.0
    example["target_month_temp_c"] = 30.0
    zone_column = f"zone_{metadata['categories']['zone'][0]}"
    tariff_column = f"tariff_name_{metadata['categories']['tariff_name'][0]}"
    example[zone_column] = 1.0
    example[tariff_column] = 1.0
    example = example[feature_columns]

    dmatrix = xgb.DMatrix(example)
    point = boosters["point"].predict(dmatrix)[0]
    lower = boosters["lower"].predict(dmatrix)[0]
    upper = boosters["upper"].predict(dmatrix)[0]

    assert lower <= point <= upper
    assert point > 0


def test_encode_features_output_matches_saved_feature_columns() -> None:
    df = generate_dataset(n_households=200, seed=13)
    _boosters, _metrics, metadata = train(df, seed=13)

    examples = build_forecast_examples(df)
    features, _target, _categories = encode_features(examples, categories=metadata["categories"])
    assert list(features.columns) == metadata["feature_columns"]


def test_save_and_load_artifact_round_trip_predicts_identically() -> None:
    df = generate_dataset(n_households=200, seed=14)
    boosters, _metrics, metadata = train(df, seed=14)

    examples = build_forecast_examples(df)
    features, _target, _categories = encode_features(examples, categories=metadata["categories"])
    dmatrix = xgb.DMatrix(features.head(10))
    original_predictions = boosters["point"].predict(dmatrix)

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = save_artifact(boosters, metadata, output_dir=Path(tmp_dir))
        loaded_boosters, loaded_metadata = load_artifact(path)

    assert loaded_metadata["feature_columns"] == metadata["feature_columns"]
    assert loaded_metadata["categories"] == metadata["categories"]

    loaded_predictions = loaded_boosters["point"].predict(dmatrix)
    assert list(original_predictions) == list(loaded_predictions)
