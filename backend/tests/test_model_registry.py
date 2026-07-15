import json
import tempfile
from pathlib import Path

from app.core.model_registry import MODELS_DIR, ModelRegistry, model_registry


def test_module_level_registry_is_ready_with_real_artifacts() -> None:
    """The module-level singleton loads eagerly at import time — this test
    only passes if that already succeeded against the real, committed
    artifacts in backend/models/."""
    assert model_registry.is_ready
    assert model_registry.load_error is None
    assert model_registry.forecaster is not None
    assert model_registry.manifest is not None


def test_load_fails_readiness_for_missing_directory() -> None:
    registry = ModelRegistry()
    registry.load(Path(tempfile.mkdtemp()))  # empty dir, no manifest or artifacts
    assert not registry.is_ready
    assert registry.load_error is not None
    assert registry.forecaster is None


def test_load_fails_readiness_on_manifest_version_mismatch() -> None:
    """Copies the real artifacts into a temp dir with a manifest that
    declares the wrong forecaster version — the registry must refuse to
    treat itself as ready rather than silently serving mismatched models."""
    tmp_dir = Path(tempfile.mkdtemp())
    for name in (
        "forecaster_v1.json",
        "anomaly_v1.json",
        "disaggregator_v1.json",
        "recommender_v1.json",
    ):
        (tmp_dir / name).write_text((MODELS_DIR / name).read_text())

    real_manifest = json.loads((MODELS_DIR / "models_manifest.json").read_text())
    broken_manifest = json.loads(json.dumps(real_manifest))  # deep copy via round-trip
    broken_manifest["models"]["forecaster"]["version"] = "forecaster_v0_wrong"
    (tmp_dir / "models_manifest.json").write_text(json.dumps(broken_manifest))

    registry = ModelRegistry()
    registry.load(tmp_dir)
    assert not registry.is_ready
    assert registry.load_error is not None
    assert "forecaster" in registry.load_error


def test_load_fails_readiness_on_manifest_sha256_mismatch() -> None:
    """Same as the version-mismatch test, but the version string is correct
    and only the declared hash is wrong — the exact "hand-edited artifact
    with the right model_version" class of bug Phase 2 audit Check 5 asked
    for, which a version-only check can't catch."""
    tmp_dir = Path(tempfile.mkdtemp())
    for name in (
        "forecaster_v1.json",
        "anomaly_v1.json",
        "disaggregator_v1.json",
        "recommender_v1.json",
    ):
        (tmp_dir / name).write_text((MODELS_DIR / name).read_text())

    real_manifest = json.loads((MODELS_DIR / "models_manifest.json").read_text())
    broken_manifest = json.loads(json.dumps(real_manifest))  # deep copy via round-trip
    broken_manifest["models"]["anomaly"]["sha256"] = "0" * 64
    (tmp_dir / "models_manifest.json").write_text(json.dumps(broken_manifest))

    registry = ModelRegistry()
    registry.load(tmp_dir)
    assert not registry.is_ready
    assert registry.load_error is not None
    assert "anomaly" in registry.load_error
    assert "sha256" in registry.load_error
