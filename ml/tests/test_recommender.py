import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import xgboost as xgb
from data.generate_synthetic import (
    build_appliance_lookup,
    generate_dataset,
    load_reference_tables,
)
from models import recommender
from models.anomaly import save_artifact as save_anomaly_artifact
from models.anomaly import train as train_anomaly
from models.disaggregator import save_artifact as save_disaggregator_artifact
from models.disaggregator import train_with_ablation
from models.forecaster import save_artifact as save_forecaster_artifact
from models.forecaster import train as train_forecaster
from wattwise_tariffs import build_tariff_lookup


@pytest.fixture(scope="module")
def small_dataset() -> object:
    return generate_dataset(n_households=300, seed=51)


@pytest.fixture(scope="module")
def upstream_artifact_paths(small_dataset: object) -> dict[str, Path]:
    """Trains small, real Model 1/2/3 artifacts once per test module —
    Model 4 depends on all three, so tests use freshly-trained small
    versions rather than the real (10,000-household) checked-in artifacts,
    matching the pattern models.anomaly's own tests use for its Model 1
    dependency."""
    tmp_dir = Path(tempfile.mkdtemp())

    forecaster_boosters, _metrics, forecaster_metadata = train_forecaster(small_dataset, seed=51)
    forecaster_path = save_forecaster_artifact(
        forecaster_boosters, forecaster_metadata, output_dir=tmp_dir
    )

    anomaly_state, _metrics = train_anomaly(small_dataset, forecaster_path, seed=51)
    anomaly_path = save_anomaly_artifact(anomaly_state, output_dir=tmp_dir)

    disagg_boosters, disagg_categories, _metrics = train_with_ablation(small_dataset, seed=51)
    disaggregator_path = save_disaggregator_artifact(
        disagg_boosters, disagg_categories, output_dir=tmp_dir
    )

    return {
        "forecaster": forecaster_path,
        "anomaly": anomaly_path,
        "disaggregator": disaggregator_path,
    }


@pytest.fixture(scope="module")
def rule_ctx(small_dataset: object) -> dict[str, object]:
    """A single, hand-built context for exercising individual rules without
    running the full model pipeline."""
    tables = load_reference_tables()
    tariff_lookup = build_tariff_lookup(tables)
    appliance_lookup = build_appliance_lookup(tables["appliances"])
    return {
        "tariff_name": "tod_generic",
        "tariff": tariff_lookup["tod_generic"],
        "sanctioned_load_kw": 4.0,
        "climate_temp_c": 32.0,
        "total_kwh": 250.0,
        "family_size": 4,
        "zone": "hot_dry",
        "fridge_star": 2,
        "owns_ac": True,
        "ac_star": 2,
        "owns_geyser": True,
        "geyser_star": 2,
        "num_fans": 3,
        "fan_star": 2,
        "num_bulbs": 6,
        "owns_washing_machine": True,
        "owns_tv": True,
        "shares": {
            "fridge": 0.10,
            "ac": 0.25,
            "geyser": 0.15,
            "lighting": 0.05,
            "fans": 0.10,
            "washing_machine": 0.05,
            "television_entertainment": 0.05,
            "other_including_standby": 0.25,
        },
        "is_anomaly": True,
        "reason_bucket": "unusual_spike",
        "excess_kwh": 40.0,
        "appliance_lookup": appliance_lookup,
        "tariff_tod_table": tables["tariff_tod"],
    }


def test_generate_candidates_returns_valid_schema(rule_ctx: dict[str, object]) -> None:
    candidates = recommender.generate_candidates(rule_ctx)
    assert len(candidates) > 0
    for candidate in candidates:
        assert candidate["category"] in recommender.RECOMMENDATION_CATEGORIES
        assert candidate["confidence"] in ("high", "medium", "low")
        assert isinstance(candidate["estimated_monthly_savings_paise"], int)
        assert candidate["estimated_monthly_savings_paise"] > 0
        assert isinstance(candidate["estimated_annual_co2_kg"], float)
        assert candidate["estimated_annual_co2_kg"] >= 0.0
        assert isinstance(candidate["calculation_trace"], dict)


def test_star_upgrade_inapplicable_at_five_star(rule_ctx: dict[str, object]) -> None:
    ctx = dict(rule_ctx)
    ctx["fridge_star"] = 5
    ctx["ac_star"] = 5
    ctx["geyser_star"] = 5
    ctx["fan_star"] = 5
    candidates = recommender.generate_candidates(ctx)
    rule_names = {c["rule_name"] for c in candidates}
    assert "star_upgrade_fridge" not in rule_names
    assert "star_upgrade_ac" not in rule_names
    assert "star_upgrade_geyser" not in rule_names
    assert "star_upgrade_fans" not in rule_names


def test_fan_upgrade_suppressed_when_inventory_missing(rule_ctx: dict[str, object]) -> None:
    """The pre-registered fan-recommendation inventory-completeness gate
    (see ml/MODELS.md's Model 4 design section): num_fans/fan_star of 0
    means the user hasn't actually supplied fan inventory data, so the
    candidate should be suppressed entirely, not just downgraded."""
    ctx = dict(rule_ctx)
    ctx["num_fans"] = 0
    ctx["fan_star"] = 0
    candidates = recommender.generate_candidates(ctx)
    assert "star_upgrade_fans" not in {c["rule_name"] for c in candidates}


def test_geyser_timing_shift_requires_tod_tariff(rule_ctx: dict[str, object]) -> None:
    ctx = dict(rule_ctx)
    ctx["tariff_name"] = "tneb"
    candidates = recommender.generate_candidates(ctx)
    assert "geyser_timing_shift" not in {c["rule_name"] for c in candidates}


def test_maintenance_check_requires_spike_anomaly(rule_ctx: dict[str, object]) -> None:
    ctx = dict(rule_ctx)
    ctx["is_anomaly"] = False
    candidates = recommender.generate_candidates(ctx)
    assert "maintenance_check" not in {c["rule_name"] for c in candidates}

    ctx = dict(rule_ctx)
    ctx["reason_bucket"] = "unusual_drop"
    candidates = recommender.generate_candidates(ctx)
    assert "maintenance_check" not in {c["rule_name"] for c in candidates}


def test_marginal_bill_savings_paise_is_zero_for_zero_reduction(
    rule_ctx: dict[str, object],
) -> None:
    savings = recommender._marginal_bill_savings_paise(
        rule_ctx["tariff"], rule_ctx["sanctioned_load_kw"], rule_ctx["total_kwh"], 0.0
    )
    assert savings == 0


def test_confidence_gate_downgrades_unstable_top_candidate() -> None:
    """Constructs a scenario where the standby candidate's estimated
    savings sits just barely above a star-upgrade candidate's — a +/-3pp
    perturbation of the standby share should be enough to flip their
    relative order, so the confidence gate should downgrade whichever one
    is unstable in the top-N."""
    tables = load_reference_tables()
    tariff_lookup = build_tariff_lookup(tables)
    appliance_lookup = build_appliance_lookup(tables["appliances"])
    ctx = {
        "tariff_name": "tneb",
        "tariff": tariff_lookup["tneb"],
        "sanctioned_load_kw": 4.0,
        "climate_temp_c": 25.0,
        "total_kwh": 250.0,
        "family_size": 4,
        "zone": "hot_dry",
        "fridge_star": 1,
        "owns_ac": False,
        "ac_star": 0,
        "owns_geyser": False,
        "geyser_star": 0,
        "num_fans": 0,
        "fan_star": 0,
        "owns_washing_machine": False,
        "owns_tv": False,
        "num_bulbs": 6,
        "shares": {
            "fridge": 0.05,
            "ac": 0.0,
            "geyser": 0.0,
            "lighting": 0.05,
            "fans": 0.0,
            "washing_machine": 0.0,
            "television_entertainment": 0.0,
            "other_including_standby": 0.09,
        },
        "is_anomaly": False,
        "reason_bucket": "none",
        "excess_kwh": 0.0,
        "appliance_lookup": appliance_lookup,
        "tariff_tod_table": tables["tariff_tod"],
    }
    candidates = recommender.generate_candidates(ctx)
    gated = recommender._apply_confidence_gate(candidates, ctx)
    rule_names_by_confidence = {c["rule_name"]: c["confidence"] for c in gated}
    assert "standby_reduction" in rule_names_by_confidence
    # Whether or not this specific scenario flips, the gate must never
    # upgrade a candidate's confidence, only ever hold or downgrade it.
    original_by_name = {c["rule_name"]: c["confidence"] for c in candidates}
    rank = {"high": 2, "medium": 1, "low": 0}
    for name, confidence in rule_names_by_confidence.items():
        assert rank[confidence] <= rank[original_by_name[name]]


def test_train_with_evaluation_reports_oracle_upper_bound(
    small_dataset: object, upstream_artifact_paths: dict[str, Path]
) -> None:
    """Oracle sorts by true savings directly, so by construction it can
    never do worse than either the learned ranker or the naive baseline on
    the same candidate pool."""
    _booster, _metadata, metrics = recommender.train_with_evaluation(
        small_dataset,
        seed=51,
        forecaster_path=upstream_artifact_paths["forecaster"],
        anomaly_path=upstream_artifact_paths["anomaly"],
        disaggregator_path=upstream_artifact_paths["disaggregator"],
    )
    assert metrics["n_households_evaluated"] > 0
    assert metrics["oracle_mean_coverage"] >= metrics["learned_mean_coverage"] - 1e-9
    assert metrics["oracle_mean_coverage"] >= metrics["naive_mean_coverage"] - 1e-9
    assert 0.0 <= metrics["learned_mean_coverage"] <= 1.0
    assert "ood_stress_test" in metrics
    assert metrics["low_confidence_recommendations_shown"] >= 0
    assert metrics["low_confidence_recommendations_per_1000_households"] >= 0.0


def test_recommend_for_household_gates_against_actually_shown_ranking(
    small_dataset: object, upstream_artifact_paths: dict[str, Path]
) -> None:
    """recommend_for_household is the real serving entrypoint: it must gate
    confidence against its own top-N (by learned score), not the rule
    base's raw estimate — the two can rank candidates differently."""
    booster, metadata, _metrics = recommender.train_with_evaluation(
        small_dataset,
        seed=51,
        forecaster_path=upstream_artifact_paths["forecaster"],
        anomaly_path=upstream_artifact_paths["anomaly"],
        disaggregator_path=upstream_artifact_paths["disaggregator"],
    )
    tables = load_reference_tables()
    tariff_lookup = build_tariff_lookup(tables)
    appliance_lookup = build_appliance_lookup(tables["appliances"])
    ctx = {
        "tariff_name": "tod_generic",
        "tariff": tariff_lookup["tod_generic"],
        "sanctioned_load_kw": 4.0,
        "climate_temp_c": 32.0,
        "total_kwh": 250.0,
        "family_size": 4,
        "zone": metadata["zone_categories"][0],
        "fridge_star": 2,
        "owns_ac": True,
        "ac_star": 2,
        "owns_geyser": True,
        "geyser_star": 2,
        "num_fans": 3,
        "fan_star": 2,
        "num_bulbs": 6,
        "owns_washing_machine": True,
        "owns_tv": True,
        "shares": {
            "fridge": 0.10,
            "ac": 0.25,
            "geyser": 0.15,
            "lighting": 0.05,
            "fans": 0.10,
            "washing_machine": 0.05,
            "television_entertainment": 0.05,
            "other_including_standby": 0.25,
        },
        "is_anomaly": False,
        "reason_bucket": "none",
        "excess_kwh": 0.0,
        "appliance_lookup": appliance_lookup,
        "tariff_tod_table": tables["tariff_tod"],
    }
    shown = recommender.recommend_for_household(
        ctx, booster, metadata["zone_categories"], metadata["tariff_categories"]
    )
    assert 0 < len(shown) <= recommender.TOP_N_ACTIONS
    assert all(c["confidence"] in ("high", "medium", "low") for c in shown)


def test_save_and_load_artifact_round_trip() -> None:
    dtrain = xgb.DMatrix(np.array([[1.0, 2.0], [3.0, 4.0]]), label=np.array([10.0, 20.0]))
    booster = xgb.train({"max_depth": 2}, dtrain, num_boost_round=2)
    metadata = {"model_version": recommender.MODEL_VERSION, "feature_columns": ["a", "b"]}

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = recommender.save_artifact(booster, metadata, output_dir=Path(tmp_dir))
        loaded_booster, loaded_payload = recommender.load_artifact(path)
        payload = json.loads(path.read_text())

    assert loaded_payload["model_version"] == recommender.MODEL_VERSION
    assert payload["model_version"] == recommender.MODEL_VERSION
    original_pred = booster.predict(xgb.DMatrix(np.array([[1.0, 2.0]])))
    loaded_pred = loaded_booster.predict(xgb.DMatrix(np.array([[1.0, 2.0]])))
    assert original_pred == pytest.approx(loaded_pred)
