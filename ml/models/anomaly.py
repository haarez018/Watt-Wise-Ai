"""Model 2 — Anomaly Detector.

Flags an anomalous household-month using seasonal residual + robust z-score:
Model 1's point forecast is the "expected" units for a month, and how far the
ACTUAL reading deviates from that expectation — in robust z-score terms,
using median/MAD rather than mean/std so a handful of genuine anomalies in
the training data can't skew the very statistic used to detect them — is the
anomaly signal. See ml/MODELS.md for why this was chosen over Isolation
Forest, and for an important, honest limitation on the `reason` field: three
of the five synthetic anomaly reasons are not statistically separable from
monthly total units alone, by construction of the Step 1 generator.
"""

import json
from pathlib import Path
from typing import cast

import pandas as pd
import xgboost as xgb
from data.generate_synthetic import generate_dataset
from features.engineering import (
    build_forecast_examples,
    encode_features,
    household_train_test_split,
)

from models.forecaster import ARTIFACT_DIR as FORECASTER_ARTIFACT_DIR
from models.forecaster import load_artifact as load_forecaster_artifact

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACT_DIR = REPO_ROOT / "backend" / "models"
REPORT_DIR = Path(__file__).resolve().parent.parent / "evaluation" / "reports"

MODEL_VERSION = "anomaly_v1"
PRECISION_THRESHOLD = 0.8
RECALL_THRESHOLD = 0.6

# Selecting the smallest threshold that *just barely* clears
# PRECISION_THRESHOLD on the validation split reproduces well on validation
# but is fragile on test: validation and test are different held-out
# households, so precision at a given threshold naturally varies a percentage
# point or two between them. A threshold chosen to clear 0.80 on validation
# by a hair landed at 0.8025 on test in an earlier run of this pipeline —
# a real pass, but by less margin than the selection process should aim for.
# Requiring a safety margin above the bar on validation (not on test — this
# is a property of the selection procedure, applied identically regardless
# of what test happens to show) fixes that without touching the actual
# 0.8/0.6 thresholds themselves.
_VALIDATION_SAFETY_MARGIN = 0.05

# Candidate z-thresholds for the validation-set grid search below. A
# traditional z=3 "outlier" cutoff turned out to flag ~13% of all
# household-months (vs. the true ~4% anomaly rate) at precision 0.31 — the
# residual distribution has heavier tails than MAD assumes, because ~4% of
# it is a genuinely different (anomaly-injected) distribution mixed into an
# otherwise well-forecast (~6% MAPE) signal. The threshold is tuned, not
# hardcoded, precisely because "z=3" turned out to be the wrong intuition
# here — see ml/MODELS.md. 0.5-step resolution rather than whole numbers, so
# the safety margin above has room to actually pick a better point, not just
# the same coarse candidates that produced the fragile result.
_Z_THRESHOLD_CANDIDATES = [2.5 + 0.5 * i for i in range(36)]  # 2.5, 3.0, ..., 20.0

# night_load_surge and sustained_high share unusual_spike's exact synthetic
# multiplier range (1.35x-1.9x, see generate_synthetic.py's
# maybe_inject_anomaly) — from monthly total units alone there is no signal
# that distinguishes them. They're bucketed into "unusual_spike" for
# reason-labeling. This does not affect the primary is_anomaly detection
# metric, only the secondary (reported, not gated) reason-bucket accuracy.
_SPIKE_LIKE_REASONS = frozenset({"unusual_spike", "night_load_surge", "sustained_high"})


def _severity(z_above_threshold: float) -> str:
    if z_above_threshold >= 2.0:
        return "high"
    if z_above_threshold >= 1.0:
        return "medium"
    return "low"


def _reason_bucket(residual_ratio: float, seasonal_cutoff: float) -> str:
    if residual_ratio < 0:
        return "unusual_drop"
    if residual_ratio >= seasonal_cutoff:
        return "unusual_spike"
    return "seasonal_deviation"


def _bucket_true_reason(true_reason: str) -> str:
    return "unusual_spike" if true_reason in _SPIKE_LIKE_REASONS else true_reason


def _predict_units(
    examples: pd.DataFrame,
    booster: xgb.Booster,
    feature_columns: list[str],
    categories: dict[str, list[str]],
) -> pd.DataFrame:
    features, _target, _cats = encode_features(examples, categories=categories)
    dmatrix = xgb.DMatrix(features[feature_columns])
    result = examples.copy()
    result["predicted_units_wh"] = booster.predict(dmatrix)
    result["residual_ratio"] = (result["target_units_wh"] - result["predicted_units_wh"]) / result[
        "predicted_units_wh"
    ]
    return result


def _precision_recall(
    flagged: pd.Series, is_anomaly: pd.Series
) -> tuple[float, float, dict[str, int]]:
    true_positive = int((flagged & is_anomaly).sum())
    false_positive = int((flagged & ~is_anomaly).sum())
    false_negative = int((~flagged & is_anomaly).sum())
    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive)
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative)
        else 0.0
    )
    counts = {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }
    return precision, recall, counts


def _choose_z_threshold(validation: pd.DataFrame) -> float:
    """Picks the smallest candidate threshold that clears
    PRECISION_THRESHOLD + _VALIDATION_SAFETY_MARGIN on the validation split —
    smallest, because precision only increases and recall only decreases as
    the threshold rises, so the smallest threshold that clears the (margined)
    precision bar is the one with the best recall among those that qualify.
    Falls back to the highest-precision candidate if none clear the bar (and
    `train`'s caller will see that as a failed run)."""
    required_precision = PRECISION_THRESHOLD + _VALIDATION_SAFETY_MARGIN
    best_meets_bar: tuple[float, float] | None = None  # (threshold, recall)
    best_overall: tuple[float, float] = (_Z_THRESHOLD_CANDIDATES[0], -1.0)  # (threshold, precision)

    for candidate in _Z_THRESHOLD_CANDIDATES:
        flagged = validation["robust_z"].abs() > candidate
        precision, recall, _counts = _precision_recall(flagged, validation["is_anomaly"])
        if precision > best_overall[1]:
            best_overall = (candidate, precision)
        if precision >= required_precision and (
            best_meets_bar is None or recall > best_meets_bar[1]
        ):
            best_meets_bar = (candidate, recall)

    return best_meets_bar[0] if best_meets_bar is not None else best_overall[0]


def train(
    df: pd.DataFrame, forecaster_path: Path, seed: int = 42
) -> tuple[dict[str, object], dict[str, object]]:
    boosters, forecaster_metadata = load_forecaster_artifact(forecaster_path)
    feature_columns = cast(list[str], forecaster_metadata["feature_columns"])
    categories = cast(dict[str, list[str]], forecaster_metadata["categories"])

    examples = build_forecast_examples(df)
    fit_and_validation, test_examples = household_train_test_split(
        examples, test_size=0.2, seed=seed
    )
    fit_examples, validation_examples = household_train_test_split(
        fit_and_validation, test_size=0.25, seed=seed
    )  # 0.25 of the remaining 80% = 20% overall, giving a 60/20/20 fit/validation/test split

    fit_with_preds = _predict_units(fit_examples, boosters["point"], feature_columns, categories)
    validation_with_preds = _predict_units(
        validation_examples, boosters["point"], feature_columns, categories
    )
    test_with_preds = _predict_units(test_examples, boosters["point"], feature_columns, categories)

    median = float(fit_with_preds["residual_ratio"].median())
    mad = float((fit_with_preds["residual_ratio"] - median).abs().median())
    mad = max(mad, 1e-6)  # guards against a degenerate all-identical residual set

    def _add_robust_z(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        frame["robust_z"] = 0.6745 * (frame["residual_ratio"] - median) / mad
        return frame

    validation_with_preds = _add_robust_z(validation_with_preds)
    test_with_preds = _add_robust_z(test_with_preds)

    z_threshold = _choose_z_threshold(validation_with_preds)

    test_with_preds["predicted_is_anomaly"] = test_with_preds["robust_z"].abs() > z_threshold
    precision, recall, counts = _precision_recall(
        test_with_preds["predicted_is_anomaly"], test_with_preds["is_anomaly"]
    )

    positive_residuals = fit_with_preds.loc[fit_with_preds["residual_ratio"] > 0, "residual_ratio"]
    seasonal_cutoff = float(positive_residuals.quantile(0.6)) if len(positive_residuals) else 0.3

    detected = test_with_preds[
        test_with_preds["predicted_is_anomaly"] & test_with_preds["is_anomaly"]
    ]
    reason_correct = sum(
        1
        for _, row in detected.iterrows()
        if _reason_bucket(row["residual_ratio"], seasonal_cutoff)
        == _bucket_true_reason(row["anomaly_reason"])
    )
    reason_bucket_accuracy = reason_correct / len(detected) if len(detected) else 0.0

    metrics: dict[str, object] = {
        "precision": precision,
        "precision_threshold": PRECISION_THRESHOLD,
        "precision_pass": precision >= PRECISION_THRESHOLD,
        "recall": recall,
        "recall_threshold": RECALL_THRESHOLD,
        "recall_pass": recall >= RECALL_THRESHOLD,
        "reason_bucket_accuracy_on_detected_anomalies": reason_bucket_accuracy,
        "z_threshold_chosen_on_validation": z_threshold,
        "n_fit_examples": int(len(fit_with_preds)),
        "n_validation_examples": int(len(validation_with_preds)),
        "n_test_examples": int(len(test_with_preds)),
        "n_test_anomalies": int(test_with_preds["is_anomaly"].sum()),
        **counts,
    }

    model_state: dict[str, object] = {
        "model_version": MODEL_VERSION,
        "median_residual_ratio": median,
        "mad_residual_ratio": mad,
        "z_threshold": z_threshold,
        "seasonal_cutoff": seasonal_cutoff,
        "severity_bands": {"medium": 1.0, "high": 2.0},
        "forecaster_version": forecaster_metadata["model_version"],
    }
    return model_state, metrics


def save_artifact(model_state: dict[str, object], output_dir: Path = ARTIFACT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{MODEL_VERSION}.json"
    path.write_text(json.dumps(model_state, indent=2))
    return path


def save_metrics_report(metrics: dict[str, object], report_dir: Path = REPORT_DIR) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"{MODEL_VERSION}_metrics.json"
    path.write_text(json.dumps(metrics, indent=2))
    return path


def main() -> None:
    forecaster_path = FORECASTER_ARTIFACT_DIR / "forecaster_v1.json"
    if not forecaster_path.exists():
        raise SystemExit(
            f"{forecaster_path} not found — train Model 1 first (python -m models.forecaster)"
        )

    df = generate_dataset()
    model_state, metrics = train(df, forecaster_path)
    print(json.dumps(metrics, indent=2))

    artifact_path = save_artifact(model_state)
    report_path = save_metrics_report(metrics)
    print(f"Saved model to {artifact_path}")
    print(f"Saved metrics report to {report_path}")

    if not metrics["precision_pass"]:
        raise SystemExit(
            f"Precision {metrics['precision']:.3f} below threshold {PRECISION_THRESHOLD}"
        )
    if not metrics["recall_pass"]:
        raise SystemExit(f"Recall {metrics['recall']:.3f} below threshold {RECALL_THRESHOLD}")


if __name__ == "__main__":
    main()
