import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from data.generate_synthetic import APPLIANCE_CATEGORIES, generate_dataset
from features.engineering import (
    DISAGGREGATION_APPLIANCE_COLUMNS,
    build_disaggregation_examples,
    encode_disaggregation_features,
)
from models import disaggregator


@pytest.fixture(scope="module")
def small_dataset() -> object:
    return generate_dataset(n_households=150, seed=31)


def test_build_disaggregation_examples_shares_sum_to_one(small_dataset: object) -> None:
    examples = build_disaggregation_examples(small_dataset)
    share_columns = [f"{category}_share" for category in APPLIANCE_CATEGORIES]
    totals = examples[share_columns].sum(axis=1)
    assert np.allclose(totals, 1.0, atol=1e-9)


def test_build_disaggregation_examples_shares_are_non_negative(small_dataset: object) -> None:
    examples = build_disaggregation_examples(small_dataset)
    share_columns = [f"{category}_share" for category in APPLIANCE_CATEGORIES]
    assert (examples[share_columns] >= 0).all().all()


def test_encode_disaggregation_features_full_includes_appliance_columns(
    small_dataset: object,
) -> None:
    examples = build_disaggregation_examples(small_dataset)
    features, targets, _categories = encode_disaggregation_features(examples)

    for column in DISAGGREGATION_APPLIANCE_COLUMNS:
        assert column in features.columns
    assert len(targets.columns) == len(APPLIANCE_CATEGORIES)


def test_encode_disaggregation_features_ablated_excludes_appliance_columns(
    small_dataset: object,
) -> None:
    examples = build_disaggregation_examples(small_dataset)
    features, _targets, _categories = encode_disaggregation_features(
        examples, include_appliance_inventory=False
    )

    for column in DISAGGREGATION_APPLIANCE_COLUMNS:
        assert column not in features.columns
    # Context features are still present.
    assert "total_units_wh" in features.columns
    assert "family_size" in features.columns


def test_predict_shares_output_sums_to_one_and_non_negative(small_dataset: object) -> None:
    boosters, _categories, _metrics = disaggregator.train_with_ablation(small_dataset, seed=31)
    examples = build_disaggregation_examples(small_dataset)
    features, _targets, categories = encode_disaggregation_features(examples)
    predicted = disaggregator._predict_shares(boosters, features)

    assert (predicted >= 0).all().all()
    assert np.allclose(predicted.sum(axis=1), 1.0, atol=1e-6)


def test_train_with_ablation_reports_all_three_variants(small_dataset: object) -> None:
    _boosters, _categories, metrics = disaggregator.train_with_ablation(small_dataset, seed=32)

    variant_keys = (
        "full_model",
        "ablated_model_no_appliance_inventory",
        "naive_population_mean_baseline",
    )
    for key in variant_keys:
        assert key in metrics
        assert "mae_per_category_pp" in metrics[key]
        assert set(metrics[key]["mae_per_category_pp"]) == set(APPLIANCE_CATEGORIES)


def test_full_model_signed_error_is_not_systematically_biased(small_dataset: object) -> None:
    """Checks that clip-and-renormalize post-processing (used instead of
    softmax) doesn't consistently push any category's predictions up or down
    — signed error should be small relative to MAE, not close in magnitude
    to it (which would indicate a directional bias)."""
    _boosters, _categories, metrics = disaggregator.train_with_ablation(small_dataset, seed=36)

    full_model = metrics["full_model"]
    signed_error = full_model["signed_error_per_category_pp"]
    mae = full_model["mae_per_category_pp"]

    assert set(signed_error) == set(APPLIANCE_CATEGORIES)
    for category in APPLIANCE_CATEGORIES:
        assert abs(signed_error[category]) < max(mae[category], 0.5)


def test_full_model_beats_ablated_which_beats_naive(small_dataset: object) -> None:
    """The whole point of the ablation: appliance inventory should help (full
    < ablated), and any real signal should beat a population-mean floor
    (ablated < naive). This is the honesty check the report depends on."""
    _boosters, _categories, metrics = disaggregator.train_with_ablation(small_dataset, seed=33)

    full_mae = metrics["full_model"]["mean_mae_pp"]
    ablated_mae = metrics["ablated_model_no_appliance_inventory"]["mean_mae_pp"]
    naive_mae = metrics["naive_population_mean_baseline"]["mean_mae_pp"]

    assert full_mae < ablated_mae
    assert ablated_mae < naive_mae
    assert metrics["synthetic_construction_advantage_pp"] == pytest.approx(ablated_mae - full_mae)


def test_save_and_load_artifact_round_trip(small_dataset: object) -> None:
    boosters, categories, _metrics = disaggregator.train_with_ablation(small_dataset, seed=34)
    examples = build_disaggregation_examples(small_dataset)
    features, _targets, _cats = encode_disaggregation_features(examples, categories=categories)
    original_predictions = disaggregator._predict_shares(boosters, features.head(10))

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = disaggregator.save_artifact(boosters, categories, output_dir=Path(tmp_dir))
        loaded_boosters, loaded_payload = disaggregator.load_artifact(path)

    assert loaded_payload["model_version"] == disaggregator.MODEL_VERSION
    assert loaded_payload["categories"] == categories

    loaded_predictions = disaggregator._predict_shares(loaded_boosters, features.head(10))
    assert np.allclose(original_predictions.to_numpy(), loaded_predictions.to_numpy())


def test_save_artifact_is_plain_json(small_dataset: object) -> None:
    boosters, categories, _metrics = disaggregator.train_with_ablation(small_dataset, seed=35)
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = disaggregator.save_artifact(boosters, categories, output_dir=Path(tmp_dir))
        # No special decoding needed — proof there's no pickled object here.
        payload = json.loads(path.read_text())
    assert payload["model_version"] == disaggregator.MODEL_VERSION
