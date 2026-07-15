import pandas as pd
import pytest
from data.generate_synthetic import (
    N_MONTHS,
    ac_daily_run_hours,
    generate_dataset,
    geyser_daily_run_hours,
)

SMALL_N = 50


@pytest.fixture(scope="module")
def small_dataset() -> pd.DataFrame:
    return generate_dataset(n_households=SMALL_N, seed=123)


def test_shape_is_exactly_households_times_months(small_dataset: pd.DataFrame) -> None:
    assert len(small_dataset) == SMALL_N * N_MONTHS


def test_every_household_has_exactly_twelve_months(small_dataset: pd.DataFrame) -> None:
    counts = small_dataset.groupby("household_id").size()
    assert (counts == N_MONTHS).all()
    assert counts.index.nunique() == SMALL_N


def test_every_household_has_each_month_index_once(small_dataset: pd.DataFrame) -> None:
    for _, group in small_dataset.groupby("household_id"):
        assert sorted(group["month_index"]) == list(range(1, N_MONTHS + 1))


def test_no_negative_units_or_amounts(small_dataset: pd.DataFrame) -> None:
    assert (small_dataset["units_consumed_wh"] > 0).all()
    assert (small_dataset["amount_paise"] >= 0).all()


def test_appliance_categories_are_non_negative(small_dataset: pd.DataFrame) -> None:
    category_cols = [c for c in small_dataset.columns if c.endswith("_kwh")]
    assert len(category_cols) == 8
    for col in category_cols:
        assert (small_dataset[col] >= 0).all(), col


def test_anomaly_rate_is_near_configured_probability(
    small_dataset: pd.DataFrame,
) -> None:
    # SMALL_N is small enough that we allow a generous band around the 4% target.
    rate = small_dataset["is_anomaly"].mean()
    assert 0.0 <= rate <= 0.15


def test_anomaly_reason_present_iff_anomalous(small_dataset: pd.DataFrame) -> None:
    anomalous = small_dataset[small_dataset["is_anomaly"]]
    non_anomalous = small_dataset[~small_dataset["is_anomaly"]]
    assert (anomalous["anomaly_reason"] != "").all()
    assert (non_anomalous["anomaly_reason"] == "").all()


def test_same_seed_is_deterministic() -> None:
    first = generate_dataset(n_households=20, seed=7)
    second = generate_dataset(n_households=20, seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_different_seeds_differ() -> None:
    first = generate_dataset(n_households=20, seed=7)
    second = generate_dataset(n_households=20, seed=8)
    assert not first["units_consumed_wh"].equals(second["units_consumed_wh"])


def test_ac_run_hours_zero_below_threshold() -> None:
    assert ac_daily_run_hours(20.0) == 0.0
    assert ac_daily_run_hours(24.0) == 0.0


def test_ac_run_hours_increase_with_temperature() -> None:
    assert ac_daily_run_hours(28.0) < ac_daily_run_hours(34.0)
    assert ac_daily_run_hours(40.0) <= 8.0  # capped


def test_geyser_run_hours_decrease_with_temperature() -> None:
    assert geyser_daily_run_hours(10.0) > geyser_daily_run_hours(29.0)
    assert geyser_daily_run_hours(35.0) >= 0.15  # floor


# Tariff-calculator tests (TariffModel/build_tariff_lookup/compute_bill_amount_paise)
# now live in libs/wattwise_tariffs/tests/test_tariffs.py — that package is the
# single source of truth, used by both this generator and backend/'s serving
# code, so its own test suite covers it rather than duplicating tests here.
