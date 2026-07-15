import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from data.generate_synthetic import generate_dataset
from models import anomaly
from models.forecaster import save_artifact as save_forecaster_artifact
from models.forecaster import train as train_forecaster


@pytest.fixture(scope="module")
def forecaster_path() -> Path:
    """Trains a small real forecaster once per test module — Model 2 depends
    on Model 1's artifact existing, so this is the fixture every test here
    needs rather than a fake/stubbed one."""
    df = generate_dataset(n_households=300, seed=21)
    boosters, _metrics, metadata = train_forecaster(df, seed=21)
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = save_forecaster_artifact(boosters, metadata, output_dir=Path(tmp_dir))
        # Copy out of the temp dir before it's cleaned up.
        persisted = Path(tempfile.mkdtemp()) / path.name
        persisted.write_text(path.read_text())
        yield persisted


def test_precision_recall_on_toy_data() -> None:
    flagged = pd.Series([True, True, False, True, False])
    is_anomaly = pd.Series([True, False, False, True, True])
    precision, recall, counts = anomaly._precision_recall(flagged, is_anomaly)

    assert counts == {"true_positive": 2, "false_positive": 1, "false_negative": 1}
    assert precision == pytest.approx(2 / 3)
    assert recall == pytest.approx(2 / 3)


def test_choose_z_threshold_prefers_smallest_that_clears_margin() -> None:
    # High |z| = anomalous (matching real usage). The 5 true anomalies sit at
    # z=6..10; z=1..5 are normal points with a couple that overlap into the
    # low end of the "high" range, so low thresholds pick up false positives
    # and precision only reaches 1.0 once the threshold clears z=5.
    robust_z = pd.Series([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    is_anomaly = pd.Series([True, True, True, True, True, False, False, False, False, False])
    validation = pd.DataFrame({"robust_z": robust_z, "is_anomaly": is_anomaly})

    chosen = anomaly._choose_z_threshold(validation)
    flagged = validation["robust_z"].abs() > chosen
    precision, recall, _counts = anomaly._precision_recall(flagged, validation["is_anomaly"])

    assert precision >= anomaly.PRECISION_THRESHOLD + anomaly._VALIDATION_SAFETY_MARGIN
    assert recall == pytest.approx(1.0)  # the smallest qualifying threshold keeps every anomaly
    assert chosen == pytest.approx(5.0)


def test_reason_bucket_direction_and_cutoff() -> None:
    assert anomaly._reason_bucket(-0.5, seasonal_cutoff=0.3) == "unusual_drop"
    assert anomaly._reason_bucket(0.1, seasonal_cutoff=0.3) == "seasonal_deviation"
    assert anomaly._reason_bucket(0.5, seasonal_cutoff=0.3) == "unusual_spike"


def test_bucket_true_reason_folds_spike_like_reasons() -> None:
    assert anomaly._bucket_true_reason("night_load_surge") == "unusual_spike"
    assert anomaly._bucket_true_reason("sustained_high") == "unusual_spike"
    assert anomaly._bucket_true_reason("unusual_spike") == "unusual_spike"
    assert anomaly._bucket_true_reason("unusual_drop") == "unusual_drop"
    assert anomaly._bucket_true_reason("seasonal_deviation") == "seasonal_deviation"


def test_severity_bands() -> None:
    assert anomaly._severity(0.5) == "low"
    assert anomaly._severity(1.5) == "medium"
    assert anomaly._severity(3.0) == "high"


def test_train_produces_valid_metrics_shape(forecaster_path: Path) -> None:
    df = generate_dataset(n_households=300, seed=22)
    model_state, metrics = anomaly.train(df, forecaster_path, seed=22)

    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["reason_bucket_accuracy_on_detected_anomalies"] <= 1.0
    assert model_state["model_version"] == anomaly.MODEL_VERSION
    assert model_state["z_threshold"] in anomaly._Z_THRESHOLD_CANDIDATES
    assert model_state["mad_residual_ratio"] > 0


def test_save_artifact_is_plain_json_no_booster(forecaster_path: Path) -> None:
    df = generate_dataset(n_households=300, seed=23)
    model_state, _metrics = anomaly.train(df, forecaster_path, seed=23)

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = anomaly.save_artifact(model_state, output_dir=Path(tmp_dir))
        # Round-trips through plain json.load with zero special handling —
        # proof there's no pickled object in here.
        reloaded = json.loads(path.read_text())

    assert reloaded == model_state
