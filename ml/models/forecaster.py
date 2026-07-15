"""Model 1 — Bill Forecaster.

Predicts next-month units_consumed_wh with an 80% prediction interval. Never
predicts amount_paise directly — see ml/MODELS.md for why. Three XGBoost
regressors share one feature set: a point estimate (reg:squarederror) and a
10th/90th percentile pair (reg:quantileerror) that together form the interval.

Serialization contract (see ml/MODELS.md): the saved artifact is a single
plain-JSON file — each booster's native JSON export, plus feature/category
metadata as plain dicts/lists. No custom Python class is pickled. Loading
requires only `xgboost` and the stdlib `json` module, on either side of the
ml/backend boundary — there is no shared class whose import path has to stay
stable between train-time and serve-time.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from data.generate_synthetic import generate_dataset
from features.engineering import (
    build_forecast_examples,
    encode_features,
    household_train_test_split,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACT_DIR = REPO_ROOT / "backend" / "models"
REPORT_DIR = Path(__file__).resolve().parent.parent / "evaluation" / "reports"

MODEL_VERSION = "forecaster_v1"
QUANTILE_LOW = 0.1
QUANTILE_HIGH = 0.9
MAPE_THRESHOLD_PERCENT = 12.0
PI80_COVERAGE_BAND = (0.75, 0.85)

_NUM_BOOST_ROUND = 300
_BASE_PARAMS: dict[str, object] = {
    "max_depth": 5,
    "eta": 0.05,
}


def _train_one(params: dict[str, object], dtrain: xgb.DMatrix, seed: int) -> xgb.Booster:
    full_params = {**_BASE_PARAMS, **params, "seed": seed}
    return xgb.train(full_params, dtrain, num_boost_round=_NUM_BOOST_ROUND)


def train(
    df: pd.DataFrame, seed: int = 42
) -> tuple[dict[str, xgb.Booster], dict[str, object], dict[str, object]]:
    """Returns (boosters, metrics, metadata). `boosters` has keys
    "point"/"lower"/"upper". `metadata` has "feature_columns" and
    "categories" — everything `save_artifact` needs, and nothing that
    requires a custom class to deserialize.

    Uses XGBoost's native train()/Booster/DMatrix API rather than the
    scikit-learn-compatible wrapper, so this pipeline doesn't need
    scikit-learn installed at all — one fewer dependency on both the
    training and serving side."""
    examples = build_forecast_examples(df)
    train_examples, test_examples = household_train_test_split(examples, seed=seed)

    X_train, y_train, categories = encode_features(train_examples)
    X_test, y_test, _ = encode_features(test_examples, categories=categories)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    point_model = _train_one({"objective": "reg:squarederror"}, dtrain, seed)
    lower_model = _train_one(
        {"objective": "reg:quantileerror", "quantile_alpha": QUANTILE_LOW}, dtrain, seed
    )
    upper_model = _train_one(
        {"objective": "reg:quantileerror", "quantile_alpha": QUANTILE_HIGH}, dtrain, seed
    )

    point_preds = point_model.predict(dtest)
    lower_preds = lower_model.predict(dtest)
    upper_preds = upper_model.predict(dtest)

    mape = float(np.mean(np.abs((y_test.to_numpy() - point_preds) / y_test.to_numpy())) * 100)
    coverage = float(
        np.mean((y_test.to_numpy() >= lower_preds) & (y_test.to_numpy() <= upper_preds))
    )

    metrics: dict[str, object] = {
        "mape_percent": mape,
        "mape_threshold_percent": MAPE_THRESHOLD_PERCENT,
        "mape_pass": mape <= MAPE_THRESHOLD_PERCENT,
        "pi80_coverage": coverage,
        "pi80_coverage_band": list(PI80_COVERAGE_BAND),
        "pi80_pass": PI80_COVERAGE_BAND[0] <= coverage <= PI80_COVERAGE_BAND[1],
        "n_train_examples": int(len(X_train)),
        "n_test_examples": int(len(X_test)),
        "n_train_households": int(train_examples["household_id"].nunique()),
        "n_test_households": int(test_examples["household_id"].nunique()),
    }

    boosters = {"point": point_model, "lower": lower_model, "upper": upper_model}
    metadata: dict[str, object] = {
        "model_version": MODEL_VERSION,
        "feature_columns": list(X_train.columns),
        "categories": categories,
        "quantile_low": QUANTILE_LOW,
        "quantile_high": QUANTILE_HIGH,
    }
    return boosters, metrics, metadata


def save_artifact(
    boosters: dict[str, xgb.Booster],
    metadata: dict[str, object],
    output_dir: Path = ARTIFACT_DIR,
) -> Path:
    """Writes one plain-JSON file: each booster's native JSON export as a
    string value, plus metadata. No pickling, no custom classes."""
    payload = dict(metadata)
    for name, booster in boosters.items():
        payload[f"{name}_model_json"] = booster.save_raw(raw_format="json").decode("utf-8")

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{MODEL_VERSION}.json"
    path.write_text(json.dumps(payload))
    return path


def load_artifact(path: Path) -> tuple[dict[str, xgb.Booster], dict[str, object]]:
    """Reference loader, used by ml-side tests. Deliberately trivial — the
    whole point of this serialization contract is that the backend can
    reimplement these ~6 lines itself with zero dependency on `ml`."""
    payload = json.loads(path.read_text())
    boosters = {}
    for name in ("point", "lower", "upper"):
        booster = xgb.Booster()
        booster.load_model(bytearray(payload.pop(f"{name}_model_json").encode("utf-8")))
        boosters[name] = booster
    return boosters, payload


def save_metrics_report(metrics: dict[str, object], report_dir: Path = REPORT_DIR) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"{MODEL_VERSION}_metrics.json"
    path.write_text(json.dumps(metrics, indent=2))
    return path


def main() -> None:
    df = generate_dataset()
    boosters, metrics, metadata = train(df)
    print(json.dumps(metrics, indent=2))

    artifact_path = save_artifact(boosters, metadata)
    report_path = save_metrics_report(metrics)
    print(f"Saved model to {artifact_path}")
    print(f"Saved metrics report to {report_path}")

    if not metrics["mape_pass"]:
        raise SystemExit(
            f"MAPE {metrics['mape_percent']:.2f}% exceeds threshold {MAPE_THRESHOLD_PERCENT}%"
        )
    if not metrics["pi80_pass"]:
        raise SystemExit(
            f"PI80 coverage {metrics['pi80_coverage']:.3f} outside band {PI80_COVERAGE_BAND}"
        )


if __name__ == "__main__":
    main()
