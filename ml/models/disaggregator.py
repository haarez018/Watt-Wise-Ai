"""Model 3 — Appliance Disaggregator.

Predicts each household-month's % share across the 8 appliance categories
from (total units, household profile, climate, tariff, month) — a direct
per-row mapping, no lag window needed (unlike Models 1/2).

This model has a structural risk the other two don't: the synthetic
generator computes each category's kWh via a deterministic formula (appliance
ownership + star rating + climate temperature + fixed daily-hours
assumptions, see ml/DATA.md), with zero per-appliance stochastic noise beyond
the shared "other/standby" jitter. A model given the same appliance-inventory
inputs the generator used can, in principle, recover the breakdown almost
exactly — which would look like an excellent model but would really be
measuring how deterministic the *dataset* is, not how good disaggregation
from real (noisy) bills would be. `train_with_ablation` below runs three
variants specifically to quantify this — see ml/MODELS.md for the results and
what they mean for how much to trust this model's numbers in production.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from data.generate_synthetic import APPLIANCE_CATEGORIES, generate_dataset
from features.engineering import (
    DISAGGREGATION_CATEGORICAL_COLUMNS,
    SHARE_COLUMNS,
    build_disaggregation_examples,
    encode_disaggregation_features,
    household_train_test_split,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACT_DIR = REPO_ROOT / "backend" / "models"
REPORT_DIR = Path(__file__).resolve().parent.parent / "evaluation" / "reports"

MODEL_VERSION = "disaggregator_v1"
MAE_THRESHOLD_PERCENTAGE_POINTS = 5.0

_NUM_BOOST_ROUND = 200
_BASE_PARAMS: dict[str, object] = {"max_depth": 4, "eta": 0.1, "objective": "reg:squarederror"}


def _train_category_boosters(
    features: pd.DataFrame, targets: pd.DataFrame, seed: int
) -> dict[str, xgb.Booster]:
    boosters = {}
    for category in APPLIANCE_CATEGORIES:
        dtrain = xgb.DMatrix(features, label=targets[f"{category}_share"])
        params = {**_BASE_PARAMS, "seed": seed}
        boosters[category] = xgb.train(params, dtrain, num_boost_round=_NUM_BOOST_ROUND)
    return boosters


def _predict_shares(boosters: dict[str, xgb.Booster], features: pd.DataFrame) -> pd.DataFrame:
    """Predicts a raw share per category, then clips to non-negative and
    renormalizes so every row sums to exactly 1.0 — the "ensure a valid
    distribution" post-processing step (in place of softmax; see
    ml/MODELS.md for why direct-share regression + renormalization was used
    instead of the brief's suggested softmax-of-logits approach)."""
    dmatrix = xgb.DMatrix(features)
    raw = pd.DataFrame(
        {
            category: np.clip(boosters[category].predict(dmatrix), 0.0, None)
            for category in APPLIANCE_CATEGORIES
        }
    )
    row_sums = raw.sum(axis=1).replace(0.0, 1.0)  # guard: never divide by zero
    return raw.div(row_sums, axis=0)


def _mae_per_category_pp(predicted: pd.DataFrame, true_shares: pd.DataFrame) -> dict[str, float]:
    """Mean absolute error per category, in percentage points (0-100 scale)."""
    result = {}
    for category in APPLIANCE_CATEGORIES:
        true_col = true_shares[f"{category}_share"].to_numpy()
        pred_col = predicted[category].to_numpy()
        result[category] = float(np.mean(np.abs(pred_col - true_col)) * 100)
    return result


def _signed_error_per_category_pp(
    predicted: pd.DataFrame, true_shares: pd.DataFrame
) -> dict[str, float]:
    """Mean signed error (predicted - true) per category, in percentage
    points. Checks whether renormalization (clip-negative-then-rescale, used
    instead of softmax — see ml/MODELS.md) systematically biases specific
    categories up or down rather than just adding unbiased noise. A near-zero
    signed error alongside a larger MAE means the errors are unbiased in
    direction; a signed error close in magnitude to the MAE means the errors
    skew consistently one way for that category."""
    result = {}
    for category in APPLIANCE_CATEGORIES:
        true_col = true_shares[f"{category}_share"].to_numpy()
        pred_col = predicted[category].to_numpy()
        result[category] = float(np.mean(pred_col - true_col) * 100)
    return result


def _naive_baseline_predictions(train_targets: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    """Predicts every row with the training set's population-mean share per
    category — no household-specific information at all. The floor any real
    model should clear by a wide margin; how much either model beats this by
    is itself informative about how much genuine per-household signal there
    is to learn versus how uniform category shares are across the population."""
    means = {
        category: train_targets[f"{category}_share"].mean() for category in APPLIANCE_CATEGORIES
    }
    total = sum(means.values())
    normalized = {category: means[category] / total for category in APPLIANCE_CATEGORIES}
    return pd.DataFrame(
        {category: [normalized[category]] * n_rows for category in APPLIANCE_CATEGORIES}
    )


def train_with_ablation(
    df: pd.DataFrame, seed: int = 42
) -> tuple[dict[str, xgb.Booster], dict[str, list[str]], dict[str, object]]:
    """Trains the real (full-feature) model, plus two comparison variants
    purely for the honesty report: an ablated model with appliance-inventory
    features removed, and a naive population-mean baseline. Only the
    full-feature model's boosters are returned for saving — the other two
    exist only to measure the size of the synthetic-construction advantage."""
    examples = build_disaggregation_examples(df)
    train_examples, test_examples = household_train_test_split(examples, seed=seed)

    train_features_full, train_targets, categories = encode_disaggregation_features(train_examples)
    test_features_full, test_targets, _ = encode_disaggregation_features(
        test_examples, categories=categories
    )
    train_features_ablated, _, _ = encode_disaggregation_features(
        train_examples, categories=categories, include_appliance_inventory=False
    )
    test_features_ablated, _, _ = encode_disaggregation_features(
        test_examples, categories=categories, include_appliance_inventory=False
    )

    full_boosters = _train_category_boosters(train_features_full, train_targets, seed)
    ablated_boosters = _train_category_boosters(train_features_ablated, train_targets, seed)

    full_predictions = _predict_shares(full_boosters, test_features_full)
    ablated_predictions = _predict_shares(ablated_boosters, test_features_ablated)
    naive_predictions = _naive_baseline_predictions(train_targets, len(test_targets))

    full_mae = _mae_per_category_pp(full_predictions, test_targets)
    ablated_mae = _mae_per_category_pp(ablated_predictions, test_targets)
    naive_mae = _mae_per_category_pp(naive_predictions, test_targets)
    full_signed_error = _signed_error_per_category_pp(full_predictions, test_targets)

    full_mean_mae = float(np.mean(list(full_mae.values())))
    ablated_mean_mae = float(np.mean(list(ablated_mae.values())))
    naive_mean_mae = float(np.mean(list(naive_mae.values())))

    metrics: dict[str, object] = {
        "mae_threshold_percentage_points": MAE_THRESHOLD_PERCENTAGE_POINTS,
        "full_model": {
            "mae_per_category_pp": full_mae,
            "mean_mae_pp": full_mean_mae,
            "signed_error_per_category_pp": full_signed_error,
            "all_categories_pass": all(
                v <= MAE_THRESHOLD_PERCENTAGE_POINTS for v in full_mae.values()
            ),
        },
        "ablated_model_no_appliance_inventory": {
            "mae_per_category_pp": ablated_mae,
            "mean_mae_pp": ablated_mean_mae,
            "all_categories_pass": all(
                v <= MAE_THRESHOLD_PERCENTAGE_POINTS for v in ablated_mae.values()
            ),
        },
        "naive_population_mean_baseline": {
            "mae_per_category_pp": naive_mae,
            "mean_mae_pp": naive_mean_mae,
        },
        "synthetic_construction_advantage_pp": ablated_mean_mae - full_mean_mae,
        "n_train_examples": int(len(train_examples)),
        "n_test_examples": int(len(test_examples)),
    }

    return full_boosters, categories, metrics


def save_artifact(
    boosters: dict[str, xgb.Booster],
    categories: dict[str, list[str]],
    output_dir: Path = ARTIFACT_DIR,
) -> Path:
    payload: dict[str, object] = {
        "model_version": MODEL_VERSION,
        "categories": categories,
        "categorical_columns": DISAGGREGATION_CATEGORICAL_COLUMNS,
        "share_columns": SHARE_COLUMNS,
    }
    for category, booster in boosters.items():
        payload[f"{category}_model_json"] = booster.save_raw(raw_format="json").decode("utf-8")

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{MODEL_VERSION}.json"
    path.write_text(json.dumps(payload))
    return path


def load_artifact(path: Path) -> tuple[dict[str, xgb.Booster], dict[str, object]]:
    payload = json.loads(path.read_text())
    boosters = {}
    for category in APPLIANCE_CATEGORIES:
        booster = xgb.Booster()
        booster.load_model(bytearray(payload.pop(f"{category}_model_json").encode("utf-8")))
        boosters[category] = booster
    return boosters, payload


def save_metrics_report(metrics: dict[str, object], report_dir: Path = REPORT_DIR) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"{MODEL_VERSION}_metrics.json"
    path.write_text(json.dumps(metrics, indent=2))
    return path


def main() -> None:
    df = generate_dataset()
    boosters, categories, metrics = train_with_ablation(df)
    print(json.dumps(metrics, indent=2))

    artifact_path = save_artifact(boosters, categories)
    report_path = save_metrics_report(metrics)
    print(f"Saved model to {artifact_path}")
    print(f"Saved metrics report to {report_path}")

    full_model_metrics = metrics["full_model"]
    assert isinstance(full_model_metrics, dict)
    if not full_model_metrics["all_categories_pass"]:
        raise SystemExit(
            f"Some categories exceed {MAE_THRESHOLD_PERCENTAGE_POINTS}pp MAE: "
            f"{full_model_metrics['mae_per_category_pp']}"
        )


if __name__ == "__main__":
    main()
