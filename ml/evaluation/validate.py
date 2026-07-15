"""CI validation gate: checks that the *currently committed* model artifacts
and their metrics reports still meet Phase 2's acceptance thresholds, and
that `models_manifest.json`'s declared version AND content hash match what's
actually in `backend/models/` — the hash check is what catches a corrupted,
truncated, or hand-edited artifact that still happens to carry the right
`model_version` string (Phase 2 audit, Check 5); the same check
`ModelRegistry._check_manifest_entry` runs at backend startup, using the
same `sha256_of` this module reuses from `generate_manifest` rather than a
third independent hash implementation.

Deliberately does **not** retrain — retraining is `train_all.py`'s job, run
manually/on-demand, not on every CI run. A multi-minute retrain on every
push would make CI slow, and worse, would make "green CI" mean "a fresh
training run happened to pass" rather than "the artifacts actually in the
repo are the ones that were validated." This is the last line of defense
against committing a stale manifest, a partially-updated set of artifacts,
or a metrics report that's quietly out of sync with what's on disk.

Run from `ml/` with `ml/.venv` activated:

    python -m evaluation.validate
"""

import json
from pathlib import Path
from typing import cast

from evaluation.generate_manifest import sha256_of

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACT_DIR = REPO_ROOT / "backend" / "models"
REPORT_DIR = Path(__file__).resolve().parent / "reports"


def _load(path: Path) -> dict[str, object]:
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")
    return cast(dict[str, object], json.loads(path.read_text()))


def _manifest_entry(manifest: dict[str, object], key: str) -> dict[str, object]:
    models = cast(dict[str, object], manifest.get("models", {}))
    return cast(dict[str, object], models.get(key, {}))


def _check_manifest_consistency(
    manifest: dict[str, object],
    key: str,
    label: str,
    artifact: dict[str, object],
    artifact_path: Path,
    failures: list[str],
) -> None:
    entry = _manifest_entry(manifest, key)
    if entry.get("version") != artifact["model_version"]:
        failures.append(f"{label}: manifest version does not match artifact")
    if entry.get("sha256") != sha256_of(artifact_path):
        failures.append(f"{label}: manifest sha256 does not match artifact on disk")


def validate() -> list[str]:
    failures: list[str] = []
    manifest = _load(ARTIFACT_DIR / "models_manifest.json")

    forecaster_path = ARTIFACT_DIR / "forecaster_v1.json"
    forecaster_artifact = _load(forecaster_path)
    forecaster_metrics = _load(REPORT_DIR / "forecaster_v1_metrics.json")
    if not forecaster_metrics["mape_pass"]:
        failures.append("Model 1 (forecaster): MAPE threshold failed")
    if not forecaster_metrics["pi80_pass"]:
        failures.append("Model 1 (forecaster): PI80 coverage threshold failed")
    _check_manifest_consistency(
        manifest,
        "forecaster",
        "Model 1 (forecaster)",
        forecaster_artifact,
        forecaster_path,
        failures,
    )

    anomaly_path = ARTIFACT_DIR / "anomaly_v1.json"
    anomaly_artifact = _load(anomaly_path)
    anomaly_metrics = _load(REPORT_DIR / "anomaly_v1_metrics.json")
    if not anomaly_metrics["precision_pass"]:
        failures.append("Model 2 (anomaly): precision threshold failed")
    if not anomaly_metrics["recall_pass"]:
        failures.append("Model 2 (anomaly): recall threshold failed")
    _check_manifest_consistency(
        manifest, "anomaly", "Model 2 (anomaly)", anomaly_artifact, anomaly_path, failures
    )

    disaggregator_path = ARTIFACT_DIR / "disaggregator_v1.json"
    disaggregator_artifact = _load(disaggregator_path)
    disaggregator_metrics = _load(REPORT_DIR / "disaggregator_v1_metrics.json")
    full_model_metrics = cast(dict[str, object], disaggregator_metrics["full_model"])
    if not full_model_metrics["all_categories_pass"]:
        failures.append("Model 3 (disaggregator): full-model category MAE threshold failed")
    _check_manifest_consistency(
        manifest,
        "disaggregator",
        "Model 3 (disaggregator)",
        disaggregator_artifact,
        disaggregator_path,
        failures,
    )

    recommender_path = ARTIFACT_DIR / "recommender_v1.json"
    recommender_artifact = _load(recommender_path)
    recommender_metrics = _load(REPORT_DIR / "recommender_v1_metrics.json")
    if not recommender_metrics["learned_coverage_pass"]:
        failures.append("Model 4 (recommender): learned-ranker coverage threshold failed")
    _check_manifest_consistency(
        manifest,
        "recommender",
        "Model 4 (recommender)",
        recommender_artifact,
        recommender_path,
        failures,
    )

    return failures


def main() -> None:
    failures = validate()
    if failures:
        print("VALIDATION FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("All four models pass their thresholds and match models_manifest.json.")


if __name__ == "__main__":
    main()
