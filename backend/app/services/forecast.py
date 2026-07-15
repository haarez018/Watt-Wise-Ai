"""Turns a household's bill history + profile into a Model 1 forecast.

Feature-building here must exactly mirror `ml/features/engineering.py`'s
`build_forecast_examples`/`encode_features` — see `ml/MODELS.md`'s Model 1
section for the feature list and `ml/data/generate_synthetic.py` for the
lag-window convention (most-recent-first). Backend never imports `ml`
directly (the plain-JSON serialization contract — see
`app/core/model_registry.py`), so this is a from-scratch reimplementation
against the `feature_columns`/`categories` the trained artifact itself
declares, not a shared function. `backend/tests/test_forecast_endpoint.py`
and the model-registry version check are what would catch drift between
this and the training side.
"""

import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from typing import cast

import xgboost as xgb
from wattwise_climate import city_to_zone, zone_month_avg_temp
from wattwise_tariffs import TariffModel, build_tariff_lookup, compute_bill_amount_paise
from wattwise_tariffs import load_tariff_reference_tables as load_tariffs

from app.core.model_registry import ForecasterModel
from app.models.bill import Bill
from app.models.household import Household

# No independent lag-window constant here — see ml/features/engineering.py's
# LAG_MONTHS. This used to be a hardcoded `MIN_BILLS_REQUIRED = 3` that could
# silently drift from Model 1's actual training-time lag window (Phase 2
# audit, Check 2). It's now derived per-request from the loaded artifact's
# own `feature_columns` (`_lag_columns`/`_min_bills_required` below), the
# single source of truth for how many lag_N_units_wh features the model
# actually expects.
_LAG_COLUMN_PATTERN = re.compile(r"^lag_(\d+)_units_wh$")

# Household.discom (product/onboarding vocabulary, 6 values) -> the 3 tariff
# structures Model 1 was trained on. Discoms without a dedicated modeled
# tariff fall back to "tod_generic" — the same generic-ToD fallback the
# training data's own generator documents for "other DISCOMs" (see
# ml/DATA.md's "Tariff structures" section).
DISCOM_TO_TARIFF_NAME: dict[str, str] = {
    "TNEB": "tneb",
    "BESCOM": "bescom",
}
DEFAULT_TARIFF_NAME = "tod_generic"

# Household.occupants / sanctioned_load_kw are nullable — there's no
# onboarding wizard yet (Phase 4) requiring them at household-creation time.
# Fall back to population-typical values consistent with the training data's
# own household-generation distribution (see
# ml/data/generate_synthetic.py's generate_households), rather than 0 or
# None, which would be a nonsensical input to a model trained on realistic
# household ranges. ASSUMPTION, not a citation.
DEFAULT_FAMILY_SIZE = 4
DEFAULT_SANCTIONED_LOAD_KW = 4.0


class InsufficientHistoryError(Exception):
    def __init__(self, n_bills: int, min_bills_required: int) -> None:
        self.n_bills = n_bills
        self.min_bills_required = min_bills_required
        super().__init__(
            f"At least {min_bills_required} billing months of history are required to "
            f"generate a forecast; found {n_bills}."
        )


@dataclass
class ForecastResult:
    predicted_units_wh: int
    predicted_amount_paise: int
    prediction_interval_low_wh: int
    prediction_interval_high_wh: int
    model_version: str
    generated_at: datetime


@lru_cache(maxsize=1)
def _tariff_lookup() -> dict[str, TariffModel]:
    # cast: mypy doesn't fully resolve wattwise_tariffs' py.typed marker
    # through this editable install, so it infers Any here despite the
    # source's own explicit return annotation — see docs/RUNBOOK.md.
    return cast(dict[str, TariffModel], build_tariff_lookup(load_tariffs()))


def _tariff_name_for(household: Household) -> str:
    return DISCOM_TO_TARIFF_NAME.get(household.discom, DEFAULT_TARIFF_NAME)


def _target_month_index(most_recent_bill: Bill) -> int:
    """The calendar month immediately after the most recent bill's period —
    1-12, wrapping December to January."""
    month = most_recent_bill.billing_period_end.month
    return month + 1 if month < 12 else 1


def _lag_columns(feature_columns: list[str]) -> list[str]:
    """The artifact's own `lag_N_units_wh` feature names, in order (`lag_1`
    first, i.e. most-recent-first — matching `ml/features/engineering.py`'s
    convention). The count of these **is** the model's actual lag-window
    size — the single source of truth this module derives
    `_min_bills_required` from, instead of an independent literal."""
    numbered = []
    for column in feature_columns:
        match = _LAG_COLUMN_PATTERN.fullmatch(column)
        if match:
            numbered.append((int(match.group(1)), column))
    numbered.sort()
    return [column for _, column in numbered]


def _min_bills_required(forecaster: ForecasterModel) -> int:
    feature_columns = cast(list[str], forecaster.metadata["feature_columns"])
    return len(_lag_columns(feature_columns))


def _build_feature_vector(
    household: Household, recent_bills: list[Bill], forecaster: ForecasterModel
) -> list[float]:
    """`recent_bills` must already be at least `_min_bills_required(forecaster)`
    bills, sorted most-recent-first (`recent_bills[0]` is the latest)."""
    feature_columns = cast(list[str], forecaster.metadata["feature_columns"])
    lag_columns = _lag_columns(feature_columns)
    lags = [float(b.units_consumed_wh) for b in recent_bills[: len(lag_columns)]]
    rolling_mean = sum(lags) / len(lags)
    rolling_std = math.sqrt(sum((x - rolling_mean) ** 2 for x in lags) / len(lags))

    zone = city_to_zone(household.city)
    tariff_name = _tariff_name_for(household)
    target_month_index = _target_month_index(recent_bills[0])
    target_month_temp_c = zone_month_avg_temp(zone, target_month_index)

    family_size = household.occupants or DEFAULT_FAMILY_SIZE
    sanctioned_load_kw = household.sanctioned_load_kw or DEFAULT_SANCTIONED_LOAD_KW

    values: dict[str, float] = dict.fromkeys(feature_columns, 0.0)
    for lag_column, lag_value in zip(lag_columns, lags, strict=True):
        values[lag_column] = lag_value
    values["rolling_mean_3_units_wh"] = rolling_mean
    values["rolling_std_3_units_wh"] = rolling_std
    values["family_size"] = float(family_size)
    values["sanctioned_load_kw"] = float(sanctioned_load_kw)
    values["target_month_temp_c"] = target_month_temp_c
    values["target_month_sin"] = math.sin(2 * math.pi * target_month_index / 12)
    values["target_month_cos"] = math.cos(2 * math.pi * target_month_index / 12)

    zone_column = f"zone_{zone}"
    if zone_column in values:
        values[zone_column] = 1.0
    tariff_column = f"tariff_name_{tariff_name}"
    if tariff_column in values:
        values[tariff_column] = 1.0

    return [values[column] for column in feature_columns]


def generate_forecast(
    household: Household, bills: list[Bill], forecaster: ForecasterModel
) -> ForecastResult:
    sorted_bills = sorted(bills, key=lambda b: b.billing_period_end, reverse=True)
    min_bills_required = _min_bills_required(forecaster)
    if len(sorted_bills) < min_bills_required:
        raise InsufficientHistoryError(len(sorted_bills), min_bills_required)

    feature_columns = cast(list[str], forecaster.metadata["feature_columns"])
    feature_values = _build_feature_vector(household, sorted_bills, forecaster)
    dmatrix = xgb.DMatrix([feature_values], feature_names=feature_columns)

    point = float(forecaster.boosters["point"].predict(dmatrix)[0])
    lower = float(forecaster.boosters["lower"].predict(dmatrix)[0])
    upper = float(forecaster.boosters["upper"].predict(dmatrix)[0])

    predicted_units_wh = max(0, round(point))
    sanctioned_load_kw = household.sanctioned_load_kw or DEFAULT_SANCTIONED_LOAD_KW
    predicted_amount_paise = compute_bill_amount_paise(
        predicted_units_wh / 1000.0,
        _tariff_lookup()[_tariff_name_for(household)],
        sanctioned_load_kw,
    )

    return ForecastResult(
        predicted_units_wh=predicted_units_wh,
        predicted_amount_paise=predicted_amount_paise,
        prediction_interval_low_wh=max(0, round(lower)),
        prediction_interval_high_wh=max(0, round(upper)),
        model_version=cast(str, forecaster.metadata["model_version"]),
        generated_at=datetime.now(UTC),
    )
