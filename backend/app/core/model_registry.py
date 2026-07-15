"""Loads all four Phase 2 model artifacts from `backend/models/` once, kept
in memory for in-process prediction — no external AI API call is ever on the
request path.

Deliberately does not import anything from `ml` — this is the serialization
contract itself: `ml/models/*.py`'s `save_artifact` functions are the
reference implementation of this same plain-JSON format, not something this
module depends on directly. `backend/tests/test_model_loading.py` exercises
these same loader functions and would fail first if this contract were ever
broken by importing from `ml`.

Loaded eagerly at module import time (not deferred into FastAPI's lifespan
event) so it behaves the same whether or not the ASGI server actually drives
the lifespan protocol (test clients don't always) — in a real deployment,
module import happens once per worker process at boot, which is what "loaded
at startup" means in practice. `app/main.py`'s lifespan still logs the
already-loaded state for visibility. Loading errors fail `/readyz`
(readiness), never `/healthz` (liveness) — see `app/api/routes/system.py`.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import structlog
import xgboost as xgb

logger = structlog.get_logger("app.model_registry")

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"


@dataclass
class ForecasterModel:
    boosters: dict[str, xgb.Booster]
    metadata: dict[str, object]


def load_forecaster(path: Path) -> ForecasterModel:
    payload = json.loads(path.read_text())
    boosters = {}
    for name in ("point", "lower", "upper"):
        booster = xgb.Booster()
        booster.load_model(bytearray(payload.pop(f"{name}_model_json").encode("utf-8")))
        boosters[name] = booster
    return ForecasterModel(boosters=boosters, metadata=payload)


@dataclass
class AnomalyModel:
    state: dict[str, object]


def load_anomaly(path: Path) -> AnomalyModel:
    return AnomalyModel(state=json.loads(path.read_text()))


@dataclass
class DisaggregatorModel:
    boosters: dict[str, xgb.Booster]
    metadata: dict[str, object]


def load_disaggregator(path: Path) -> DisaggregatorModel:
    payload = json.loads(path.read_text())
    share_columns = payload["share_columns"]
    boosters = {}
    for column in share_columns:
        category = column.removesuffix("_share")
        booster = xgb.Booster()
        booster.load_model(bytearray(payload.pop(f"{category}_model_json").encode("utf-8")))
        boosters[category] = booster
    return DisaggregatorModel(boosters=boosters, metadata=payload)


@dataclass
class RecommenderModel:
    booster: xgb.Booster
    metadata: dict[str, object]


def load_recommender(path: Path) -> RecommenderModel:
    payload = json.loads(path.read_text())
    booster = xgb.Booster()
    booster.load_model(bytearray(payload.pop("ranker_model_json").encode("utf-8")))
    return RecommenderModel(booster=booster, metadata=payload)


def _sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_manifest_entry(
    manifest: dict[str, object], key: str, actual_version: object, artifact_path: Path
) -> None:
    """Checks both that the manifest's declared version matches the loaded
    artifact's own `model_version`, AND that the manifest's declared SHA-256
    matches the artifact file's actual bytes on disk — the version check
    alone can't catch a corrupted, truncated, or hand-edited artifact that
    still happens to carry the right `model_version` string (Phase 2 audit,
    Check 5)."""
    models = manifest.get("models")
    entry: dict[str, object] = {}
    if isinstance(models, dict):
        maybe_entry = models.get(key)
        if isinstance(maybe_entry, dict):
            entry = maybe_entry

    declared_version = entry.get("version")
    if declared_version != actual_version:
        raise ValueError(
            f"models_manifest.json declares {key!r} version {declared_version!r} but the "
            f"loaded artifact reports {actual_version!r} — retrain or regenerate the "
            "manifest (python -m evaluation.generate_manifest from ml/)."
        )

    declared_sha256 = entry.get("sha256")
    actual_sha256 = _sha256_of(artifact_path)
    if declared_sha256 != actual_sha256:
        raise ValueError(
            f"models_manifest.json declares {key!r} sha256 {declared_sha256!r} but the "
            f"artifact on disk hashes to {actual_sha256!r} — the file may be corrupted, "
            "truncated, or hand-edited. Retrain or regenerate the manifest "
            "(python -m evaluation.generate_manifest from ml/)."
        )


@dataclass
class ModelRegistry:
    forecaster: ForecasterModel | None = None
    anomaly: AnomalyModel | None = None
    disaggregator: DisaggregatorModel | None = None
    recommender: RecommenderModel | None = None
    manifest: dict[str, object] | None = None
    load_error: str | None = None

    @property
    def is_ready(self) -> bool:
        return self.load_error is None and self.forecaster is not None

    def load(self, models_dir: Path) -> None:
        """Loads every artifact fresh and only replaces this instance's
        state if all four succeed and pass their manifest version+hash
        check — a partial/broken load never leaves the registry
        half-updated."""
        try:
            manifest = json.loads((models_dir / "models_manifest.json").read_text())

            forecaster_path = models_dir / "forecaster_v1.json"
            anomaly_path = models_dir / "anomaly_v1.json"
            disaggregator_path = models_dir / "disaggregator_v1.json"
            recommender_path = models_dir / "recommender_v1.json"

            forecaster = load_forecaster(forecaster_path)
            anomaly = load_anomaly(anomaly_path)
            disaggregator = load_disaggregator(disaggregator_path)
            recommender = load_recommender(recommender_path)

            _check_manifest_entry(
                manifest, "forecaster", forecaster.metadata["model_version"], forecaster_path
            )
            _check_manifest_entry(manifest, "anomaly", anomaly.state["model_version"], anomaly_path)
            _check_manifest_entry(
                manifest,
                "disaggregator",
                disaggregator.metadata["model_version"],
                disaggregator_path,
            )
            _check_manifest_entry(
                manifest, "recommender", recommender.metadata["model_version"], recommender_path
            )
        except Exception as exc:
            self.load_error = str(exc)
            logger.error("model_registry_load_failed", error=str(exc))
            return

        self.forecaster = forecaster
        self.anomaly = anomaly
        self.disaggregator = disaggregator
        self.recommender = recommender
        self.manifest = manifest
        self.load_error = None
        logger.info(
            "model_registry_loaded", trained_from_commit=manifest.get("trained_from_commit")
        )


model_registry = ModelRegistry()
model_registry.load(MODELS_DIR)
