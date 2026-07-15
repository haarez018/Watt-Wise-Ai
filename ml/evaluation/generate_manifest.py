"""Generates `backend/models/models_manifest.json` — the version/hash/metric
snapshot `backend/app/core/model_registry.py` checks each artifact against
on load, so a version mismatch OR a corrupted/hand-edited/truncated artifact
fails readiness immediately instead of serving predictions from an
unexpected or damaged model.

Run after (re)training all four models — `train_all.py` calls this
automatically; standalone use: `python -m evaluation.generate_manifest`
(from `ml/`, with `ml/.venv` activated).

Content integrity (Phase 2 audit, Check 5): each model's entry carries a
SHA-256 of the artifact file's actual bytes, not just the `model_version`
string declared inside it — a hand-edited or corrupted artifact that still
happens to carry the right `model_version` value is caught by the hash,
where a version-string-only check would miss it. `trained_from_commit` is
the git SHA of the commit `train_all.py` was run against, giving the
manifest real provenance back to a specific point in history — deliberately
not a fresh UUID minted per manifest regeneration (two runs against
byte-identical artifacts from the same commit now produce a manifest that's
byte-identical except `generated_at`, rather than two unrelated-looking
run IDs).

Deliberately NOT included: signing, HMAC, or external attestation (e.g.
Sigstore) — a file hash plus git provenance closes the actual gaps found in
the audit; anything beyond that is Phase 4+ scope, not proportionate to a
single-machine training pipeline whose artifacts are committed to git
alongside the code that produced them.
"""

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACT_DIR = REPO_ROOT / "backend" / "models"
REPORT_DIR = Path(__file__).resolve().parent / "reports"

MODELS: dict[str, dict[str, str]] = {
    "forecaster": {"artifact": "forecaster_v1.json", "report": "forecaster_v1_metrics.json"},
    "anomaly": {"artifact": "anomaly_v1.json", "report": "anomaly_v1_metrics.json"},
    "disaggregator": {
        "artifact": "disaggregator_v1.json",
        "report": "disaggregator_v1_metrics.json",
    },
    "recommender": {"artifact": "recommender_v1.json", "report": "recommender_v1_metrics.json"},
}


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha() -> str:
    """The commit HEAD was at when the manifest was generated. Falls back to
    "unknown" rather than failing the whole training run if git isn't
    available or this isn't a git checkout (e.g. a stripped-down CI cache) —
    provenance is valuable but not worth blocking training over."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def generate_manifest() -> dict[str, object]:
    models: dict[str, object] = {}
    for key, paths in MODELS.items():
        artifact_path = ARTIFACT_DIR / paths["artifact"]
        report_path = REPORT_DIR / paths["report"]
        if not artifact_path.exists():
            raise SystemExit(f"{artifact_path} not found — train Model ({key}) first")
        if not report_path.exists():
            raise SystemExit(f"{report_path} not found — train Model ({key}) first")

        artifact_payload = json.loads(artifact_path.read_text())
        metrics = json.loads(report_path.read_text())
        models[key] = {
            "version": artifact_payload["model_version"],
            "sha256": sha256_of(artifact_path),
            "metrics": metrics,
        }

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "trained_from_commit": _git_sha(),
        "models": models,
    }


def main_and_return_path() -> Path:
    manifest = generate_manifest()
    path = ARTIFACT_DIR / "models_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def main() -> None:
    path = main_and_return_path()
    print(f"Saved manifest to {path}")


if __name__ == "__main__":
    main()
