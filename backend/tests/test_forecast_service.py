"""Unit tests for app/services/forecast.py's lag-window derivation —
Phase 2 audit Check 2's fix: MIN_BILLS_REQUIRED must never be an independent
literal that can drift from Model 1's actual trained lag window. These tests
use a synthetic ForecasterModel with a *different* lag count than the real
5-column model to prove the derivation is genuinely dynamic, not just
coincidentally correct against today's 3-lag artifact.
"""

from app.core.model_registry import ForecasterModel
from app.services.forecast import _lag_columns, _min_bills_required


def _fake_forecaster(feature_columns: list[str]) -> ForecasterModel:
    return ForecasterModel(boosters={}, metadata={"feature_columns": feature_columns})


def test_lag_columns_extracts_and_orders_lag_features() -> None:
    columns = ["family_size", "lag_2_units_wh", "lag_1_units_wh", "sanctioned_load_kw"]
    assert _lag_columns(columns) == ["lag_1_units_wh", "lag_2_units_wh"]


def test_lag_columns_ignores_non_matching_columns() -> None:
    columns = ["rolling_mean_3_units_wh", "zone_hot_dry", "target_month_sin"]
    assert _lag_columns(columns) == []


def test_min_bills_required_matches_todays_three_lag_model() -> None:
    forecaster = _fake_forecaster(
        ["lag_1_units_wh", "lag_2_units_wh", "lag_3_units_wh", "family_size"]
    )
    assert _min_bills_required(forecaster) == 3


def test_min_bills_required_generalizes_to_a_different_lag_window() -> None:
    """The regression case: if Model 1 were retrained with a 6-month lag
    window, this must return 6 without any code change here — proving the
    threshold is genuinely derived, not hardcoded to match today's model."""
    columns = [f"lag_{n}_units_wh" for n in range(1, 7)] + ["family_size"]
    forecaster = _fake_forecaster(columns)
    assert _min_bills_required(forecaster) == 6


def test_min_bills_required_handles_no_lag_columns() -> None:
    forecaster = _fake_forecaster(["family_size", "sanctioned_load_kw"])
    assert _min_bills_required(forecaster) == 0
