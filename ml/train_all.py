"""Trains all four Phase 2 models from scratch, in dependency order, and
regenerates `backend/models/models_manifest.json` (see
`evaluation/generate_manifest.py` and `backend/app/core/model_registry.py`).

Idempotent: rerunning overwrites all four artifacts and the manifest with a
fresh run using the same default seed (42 throughout), so the same command
reproduces the same result (see each model's own module docstring for the
byte-identical-metrics guarantee this relies on). The dataset (10,000
households x 12 months) is generated **once** and reused across all four
models' training — each model's own `main()` regenerates it independently
when run standalone, which is fine for training one model in isolation but
wasteful here.

Run from `ml/` with `ml/.venv` activated:

    python train_all.py

See `ml/MODELS.md`'s "Retraining" note for the last measured wall-clock
runtime on a laptop.
"""

import time
from typing import cast

from data.generate_synthetic import generate_dataset
from evaluation import generate_manifest
from models import anomaly, disaggregator, forecaster, recommender


def main() -> None:
    start = time.monotonic()

    print("Generating dataset (10,000 households x 12 months)...")
    df = generate_dataset()

    print("\n=== Model 1: Bill Forecaster ===")
    forecaster_boosters, forecaster_metrics, forecaster_metadata = forecaster.train(df)
    if not (forecaster_metrics["mape_pass"] and forecaster_metrics["pi80_pass"]):
        raise SystemExit(f"Model 1 (forecaster) failed thresholds: {forecaster_metrics}")
    forecaster_path = forecaster.save_artifact(forecaster_boosters, forecaster_metadata)
    forecaster.save_metrics_report(forecaster_metrics)
    print(f"MAPE={forecaster_metrics['mape_percent']:.2f}%, saved to {forecaster_path}")

    print("\n=== Model 2: Anomaly Detector ===")
    anomaly_state, anomaly_metrics = anomaly.train(df, forecaster_path)
    if not (anomaly_metrics["precision_pass"] and anomaly_metrics["recall_pass"]):
        raise SystemExit(f"Model 2 (anomaly) failed thresholds: {anomaly_metrics}")
    anomaly_path = anomaly.save_artifact(anomaly_state)
    anomaly.save_metrics_report(anomaly_metrics)
    print(f"precision={anomaly_metrics['precision']:.3f}, saved to {anomaly_path}")

    print("\n=== Model 3: Appliance Disaggregator ===")
    disagg_boosters, disagg_categories, disagg_metrics = disaggregator.train_with_ablation(df)
    full_model_metrics = cast(dict[str, object], disagg_metrics["full_model"])
    if not full_model_metrics["all_categories_pass"]:
        raise SystemExit(f"Model 3 (disaggregator) failed thresholds: {full_model_metrics}")
    disaggregator_path = disaggregator.save_artifact(disagg_boosters, disagg_categories)
    disaggregator.save_metrics_report(disagg_metrics)
    print(f"mean_mae_pp={full_model_metrics['mean_mae_pp']:.3f}, saved to {disaggregator_path}")

    print("\n=== Model 4: Recommendation Ranker ===")
    recommender_booster, recommender_metadata, recommender_metrics = (
        recommender.train_with_evaluation(
            df,
            forecaster_path=forecaster_path,
            anomaly_path=anomaly_path,
            disaggregator_path=disaggregator_path,
        )
    )
    if not recommender_metrics["learned_coverage_pass"]:
        raise SystemExit(f"Model 4 (recommender) failed thresholds: {recommender_metrics}")
    recommender_path = recommender.save_artifact(recommender_booster, recommender_metadata)
    recommender.save_metrics_report(recommender_metrics)
    print(
        f"learned_mean_coverage={recommender_metrics['learned_mean_coverage']:.3f}, "
        f"saved to {recommender_path}"
    )

    print("\n=== Manifest ===")
    manifest_path = generate_manifest.main_and_return_path()
    print(f"Saved manifest to {manifest_path}")

    elapsed = time.monotonic() - start
    print(
        f"\nAll four models trained, validated, and saved in {elapsed:.1f}s ({elapsed / 60:.1f}m)."
    )


if __name__ == "__main__":
    main()
