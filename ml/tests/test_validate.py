import json
import tempfile
from pathlib import Path

from evaluation import validate
from evaluation.generate_manifest import sha256_of


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _write_artifacts_and_reports(artifact_dir: Path, report_dir: Path) -> dict[str, str]:
    """Writes the 4 mock artifacts + reports and returns each one's real
    sha256, so tests can build a manifest that's either consistent (passes)
    or deliberately wrong (one specific check fails)."""
    _write(artifact_dir / "forecaster_v1.json", {"model_version": "forecaster_v1"})
    _write(report_dir / "forecaster_v1_metrics.json", {"mape_pass": True, "pi80_pass": True})
    _write(artifact_dir / "anomaly_v1.json", {"model_version": "anomaly_v1"})
    _write(report_dir / "anomaly_v1_metrics.json", {"precision_pass": True, "recall_pass": True})
    _write(artifact_dir / "disaggregator_v1.json", {"model_version": "disaggregator_v1"})
    _write(
        report_dir / "disaggregator_v1_metrics.json",
        {"full_model": {"all_categories_pass": True}},
    )
    _write(artifact_dir / "recommender_v1.json", {"model_version": "recommender_v1"})
    _write(report_dir / "recommender_v1_metrics.json", {"learned_coverage_pass": True})

    return {
        key: sha256_of(artifact_dir / f"{key}_v1.json")
        for key in ("forecaster", "anomaly", "disaggregator", "recommender")
    }


def _swap_dirs(artifact_dir: Path, report_dir: Path) -> tuple[Path, Path]:
    original = (validate.ARTIFACT_DIR, validate.REPORT_DIR)
    validate.ARTIFACT_DIR = artifact_dir
    validate.REPORT_DIR = report_dir
    return original


def _restore_dirs(original: tuple[Path, Path]) -> None:
    validate.ARTIFACT_DIR, validate.REPORT_DIR = original


def test_validate_passes_against_real_committed_artifacts() -> None:
    """The real, currently-committed artifacts and reports must validate
    clean — this is the same check CI runs on every push."""
    failures = validate.validate()
    assert failures == []


def test_validate_catches_version_mismatch() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        artifact_dir = tmp_path / "backend" / "models"
        report_dir = tmp_path / "ml" / "evaluation" / "reports"
        hashes = _write_artifacts_and_reports(artifact_dir, report_dir)

        _write(
            artifact_dir / "models_manifest.json",
            {
                "models": {
                    "forecaster": {
                        "version": "forecaster_v0_STALE",  # deliberate mismatch
                        "sha256": hashes["forecaster"],
                    },
                    "anomaly": {"version": "anomaly_v1", "sha256": hashes["anomaly"]},
                    "disaggregator": {
                        "version": "disaggregator_v1",
                        "sha256": hashes["disaggregator"],
                    },
                    "recommender": {"version": "recommender_v1", "sha256": hashes["recommender"]},
                }
            },
        )

        original = _swap_dirs(artifact_dir, report_dir)
        try:
            failures = validate.validate()
        finally:
            _restore_dirs(original)

        assert len(failures) == 1
        assert "forecaster" in failures[0]
        assert "version" in failures[0]


def test_validate_catches_sha256_mismatch() -> None:
    """The check the audit specifically asked for: a manifest whose version
    string is correct but whose declared hash doesn't match the artifact's
    actual bytes must fail — this is exactly the "hand-edited artifact"
    class of bug a version-string-only check can't catch."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        artifact_dir = tmp_path / "backend" / "models"
        report_dir = tmp_path / "ml" / "evaluation" / "reports"
        hashes = _write_artifacts_and_reports(artifact_dir, report_dir)

        _write(
            artifact_dir / "models_manifest.json",
            {
                "models": {
                    "forecaster": {
                        "version": "forecaster_v1",
                        "sha256": "0" * 64,  # deliberate mismatch — real version, wrong hash
                    },
                    "anomaly": {"version": "anomaly_v1", "sha256": hashes["anomaly"]},
                    "disaggregator": {
                        "version": "disaggregator_v1",
                        "sha256": hashes["disaggregator"],
                    },
                    "recommender": {"version": "recommender_v1", "sha256": hashes["recommender"]},
                }
            },
        )

        original = _swap_dirs(artifact_dir, report_dir)
        try:
            failures = validate.validate()
        finally:
            _restore_dirs(original)

        assert len(failures) == 1
        assert "forecaster" in failures[0]
        assert "sha256" in failures[0]


def test_validate_catches_failed_threshold() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        artifact_dir = tmp_path / "backend" / "models"
        report_dir = tmp_path / "ml" / "evaluation" / "reports"
        _write(artifact_dir / "forecaster_v1.json", {"model_version": "forecaster_v1"})
        _write(report_dir / "forecaster_v1_metrics.json", {"mape_pass": False, "pi80_pass": True})
        _write(artifact_dir / "anomaly_v1.json", {"model_version": "anomaly_v1"})
        _write(
            report_dir / "anomaly_v1_metrics.json", {"precision_pass": True, "recall_pass": True}
        )
        _write(artifact_dir / "disaggregator_v1.json", {"model_version": "disaggregator_v1"})
        _write(
            report_dir / "disaggregator_v1_metrics.json",
            {"full_model": {"all_categories_pass": True}},
        )
        _write(artifact_dir / "recommender_v1.json", {"model_version": "recommender_v1"})
        _write(report_dir / "recommender_v1_metrics.json", {"learned_coverage_pass": True})
        hashes = {
            key: sha256_of(artifact_dir / f"{key}_v1.json")
            for key in ("forecaster", "anomaly", "disaggregator", "recommender")
        }
        _write(
            artifact_dir / "models_manifest.json",
            {
                "models": {
                    "forecaster": {"version": "forecaster_v1", "sha256": hashes["forecaster"]},
                    "anomaly": {"version": "anomaly_v1", "sha256": hashes["anomaly"]},
                    "disaggregator": {
                        "version": "disaggregator_v1",
                        "sha256": hashes["disaggregator"],
                    },
                    "recommender": {"version": "recommender_v1", "sha256": hashes["recommender"]},
                }
            },
        )

        original = _swap_dirs(artifact_dir, report_dir)
        try:
            failures = validate.validate()
        finally:
            _restore_dirs(original)

        assert len(failures) == 1
        assert "MAPE" in failures[0]


def test_validate_missing_file_raises_system_exit() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        original_artifact_dir = validate.ARTIFACT_DIR
        validate.ARTIFACT_DIR = Path(tmp_dir) / "nonexistent"
        try:
            try:
                validate.validate()
                raised = False
            except SystemExit:
                raised = True
        finally:
            validate.ARTIFACT_DIR = original_artifact_dir
        assert raised
