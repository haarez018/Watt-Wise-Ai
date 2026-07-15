"""Proves the Phase 2 model artifacts load on the backend side using only
backend dependencies (xgboost + stdlib json) — no import from the `ml`
package at all. This is the serialization contract itself: `ml/models/*.py`'s
`save_artifact` functions are the reference implementation, not something
the backend depends on directly. `app.core.model_registry` is the real
production loader this test exercises — if this ever needs an `ml` import
to pass, the contract has been broken.
"""

import json

import xgboost as xgb

from app.core.model_registry import (
    MODELS_DIR,
    load_anomaly,
    load_disaggregator,
    load_forecaster,
    load_recommender,
)

MODEL_PATH = MODELS_DIR / "forecaster_v1.json"
ANOMALY_MODEL_PATH = MODELS_DIR / "anomaly_v1.json"
DISAGGREGATOR_MODEL_PATH = MODELS_DIR / "disaggregator_v1.json"
RECOMMENDER_MODEL_PATH = MODELS_DIR / "recommender_v1.json"


def test_model_artifact_exists() -> None:
    assert MODEL_PATH.exists(), f"expected a trained model at {MODEL_PATH}"


def test_forecaster_loads_and_predicts_without_importing_ml() -> None:
    forecaster = load_forecaster(MODEL_PATH)

    assert set(forecaster.boosters) == {"point", "lower", "upper"}
    assert forecaster.metadata["model_version"] == "forecaster_v1"
    feature_columns = forecaster.metadata["feature_columns"]
    assert isinstance(feature_columns, list) and len(feature_columns) > 0

    example = {column: 0.0 for column in feature_columns}
    example["lag_1_units_wh"] = 150_000.0
    example["lag_2_units_wh"] = 145_000.0
    example["lag_3_units_wh"] = 155_000.0
    example["rolling_mean_3_units_wh"] = 150_000.0
    example["rolling_std_3_units_wh"] = 4_000.0
    example["family_size"] = 4
    example["sanctioned_load_kw"] = 4.0
    example["target_month_temp_c"] = 30.0
    categories = forecaster.metadata["categories"]
    example[f"zone_{categories['zone'][0]}"] = 1.0
    example[f"tariff_name_{categories['tariff_name'][0]}"] = 1.0

    dmatrix = xgb.DMatrix([[example[c] for c in feature_columns]], feature_names=feature_columns)

    point = forecaster.boosters["point"].predict(dmatrix)[0]
    lower = forecaster.boosters["lower"].predict(dmatrix)[0]
    upper = forecaster.boosters["upper"].predict(dmatrix)[0]

    assert lower <= point <= upper
    assert point > 0


def test_anomaly_artifact_exists() -> None:
    assert ANOMALY_MODEL_PATH.exists(), f"expected a trained model at {ANOMALY_MODEL_PATH}"


def test_anomaly_artifact_loads_without_importing_ml() -> None:
    """Model 2's artifact has no booster at all — it's tuned scalars — so
    loading it is just json.loads. The point of this test is confirming the
    expected keys are there and it cross-references the forecaster version it
    was tuned against, not exercising any special deserialization logic."""
    anomaly = load_anomaly(ANOMALY_MODEL_PATH)

    assert anomaly.state["model_version"] == "anomaly_v1"
    assert anomaly.state["mad_residual_ratio"] > 0
    assert isinstance(anomaly.state["z_threshold"], int | float)
    assert isinstance(anomaly.state["severity_bands"], dict)

    forecaster_payload = json.loads(MODEL_PATH.read_text())
    assert anomaly.state["forecaster_version"] == forecaster_payload["model_version"]


def test_disaggregator_artifact_exists() -> None:
    assert (
        DISAGGREGATOR_MODEL_PATH.exists()
    ), f"expected a trained model at {DISAGGREGATOR_MODEL_PATH}"


def test_disaggregator_loads_and_predicts_without_importing_ml() -> None:
    disaggregator = load_disaggregator(DISAGGREGATOR_MODEL_PATH)

    assert disaggregator.metadata["model_version"] == "disaggregator_v1"
    share_columns = disaggregator.metadata["share_columns"]
    assert isinstance(share_columns, list) and len(share_columns) == 8
    assert disaggregator.metadata["categorical_columns"] == ["zone", "tariff_name"]
    assert set(disaggregator.boosters) == {c.removesuffix("_share") for c in share_columns}

    context_columns = [
        "total_units_wh",
        "family_size",
        "sanctioned_load_kw",
        "climate_temp_c",
        "month_sin",
        "month_cos",
    ]
    appliance_columns = [
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
    zone_categories = disaggregator.metadata["categories"]["zone"]
    tariff_categories = disaggregator.metadata["categories"]["tariff_name"]
    one_hot_columns = [f"zone_{z}" for z in zone_categories] + [
        f"tariff_name_{t}" for t in tariff_categories
    ]
    feature_columns = context_columns + appliance_columns + one_hot_columns

    example = {column: 0.0 for column in feature_columns}
    example["total_units_wh"] = 250_000.0
    example["family_size"] = 4
    example["sanctioned_load_kw"] = 4.0
    example["climate_temp_c"] = 30.0
    example["fridge_star"] = 3
    example["owns_ac"] = 1.0
    example["ac_star"] = 4
    example[f"zone_{zone_categories[0]}"] = 1.0
    example[f"tariff_name_{tariff_categories[0]}"] = 1.0

    dmatrix = xgb.DMatrix([[example[c] for c in feature_columns]], feature_names=feature_columns)

    raw_shares = {
        category: max(0.0, booster.predict(dmatrix)[0])
        for category, booster in disaggregator.boosters.items()
    }
    total = sum(raw_shares.values())
    normalized = {category: share / total for category, share in raw_shares.items()}

    assert abs(sum(normalized.values()) - 1.0) < 1e-6
    assert all(share >= 0 for share in normalized.values())


def test_recommender_artifact_exists() -> None:
    assert RECOMMENDER_MODEL_PATH.exists(), f"expected a trained model at {RECOMMENDER_MODEL_PATH}"


def test_recommender_loads_and_scores_without_importing_ml() -> None:
    recommender = load_recommender(RECOMMENDER_MODEL_PATH)

    assert recommender.metadata["model_version"] == "recommender_v1"
    feature_columns = recommender.metadata["feature_columns"]
    assert isinstance(feature_columns, list) and len(feature_columns) > 0
    assert recommender.metadata["forecaster_version"] == "forecaster_v1"
    assert recommender.metadata["anomaly_version"] == "anomaly_v1"
    assert recommender.metadata["disaggregator_version"] == "disaggregator_v1"

    example = {column: 0.0 for column in feature_columns}
    example["is_rule_star_upgrade_ac"] = 1.0
    example["predicted_savings_paise"] = 15000.0
    example["family_size"] = 4.0
    example["sanctioned_load_kw"] = 4.0
    example[f"zone_{recommender.metadata['zone_categories'][0]}"] = 1.0
    example[f"tariff_name_{recommender.metadata['tariff_categories'][0]}"] = 1.0

    dmatrix = xgb.DMatrix([[example[c] for c in feature_columns]], feature_names=feature_columns)
    score = recommender.booster.predict(dmatrix)[0]

    # not NaN — the ranker score is a raw predicted-savings regression, not clipped to [0, inf)
    assert score == score


def test_model_registry_manifest_exists() -> None:
    manifest_path = MODELS_DIR / "models_manifest.json"
    assert manifest_path.exists(), f"expected a manifest at {manifest_path}"
    manifest = json.loads(manifest_path.read_text())
    assert set(manifest["models"]) == {"forecaster", "anomaly", "disaggregator", "recommender"}
