"""Model 4 — Recommendation Ranker.

Input: Model 3's disaggregated category shares, Model 2's anomaly signal
(via Model 1's forecast), household appliance inventory, tariff, and current
month. Output: a ranked list of actions, each with a title, plain-language
body, `estimated_monthly_savings_paise`, `estimated_annual_co2_kg`, a
`category` (behavior_change | timing_shift | replacement | maintenance), a
`confidence`, and a `calculation_trace` (dev-only, shows the exact formula
and inputs so every rupee/CO2 figure is independently checkable).

Approach: a physics-grounded rule base (`generate_candidates`) proposes
candidate actions; a learned XGBoost regressor scores each candidate by
predicted *true* savings, and ranking is done off that score, not the rule
base's own raw point estimate. See ml/MODELS.md's Model 4 sections
(pre-registered design + this file's actual results) for why that
distinction matters and how the "achievable savings" ceiling used for
evaluation is computed.

Confidence gate (pre-registered in ml/MODELS.md before this file was
written): Model 3's shares are assumed accurate to ~3pp (not its own gating
0.31pp), and any top-ranked candidate whose ranking would flip under a 3pp
share perturbation is downgraded to "low" confidence — see
`_apply_confidence_gate`.
"""

import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import xgboost as xgb
from data.generate_synthetic import (
    APPLIANCE_CATEGORIES,
    DAYS_PER_MONTH,
    ApplianceLookup,
    ac_daily_run_hours,
    build_appliance_lookup,
    fan_usage_multiplier,
    generate_dataset,
    geyser_daily_run_hours,
    load_reference_tables,
)
from features.engineering import (
    build_recommender_examples,
    encode_disaggregation_features,
    encode_features,
    household_train_test_split,
)
from wattwise_tariffs import (
    TOD_BASE_RATE_RUPEES_PER_UNIT,
    TariffModel,
    build_tariff_lookup,
    compute_bill_amount_paise,
)

from models import anomaly as anomaly_module
from models import disaggregator as disaggregator_module
from models.disaggregator import ARTIFACT_DIR as DISAGGREGATOR_ARTIFACT_DIR
from models.disaggregator import load_artifact as load_disaggregator_artifact
from models.forecaster import ARTIFACT_DIR as FORECASTER_ARTIFACT_DIR
from models.forecaster import load_artifact as load_forecaster_artifact

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACT_DIR = REPO_ROOT / "backend" / "models"
REPORT_DIR = Path(__file__).resolve().parent.parent / "evaluation" / "reports"
ANOMALY_ARTIFACT_PATH = ARTIFACT_DIR / "anomaly_v1.json"
FORECASTER_ARTIFACT_PATH = FORECASTER_ARTIFACT_DIR / "forecaster_v1.json"
DISAGGREGATOR_ARTIFACT_PATH = DISAGGREGATOR_ARTIFACT_DIR / "disaggregator_v1.json"

MODEL_VERSION = "recommender_v1"

# CEA, see ml/data/reference/grid_emission_factor.md — this is the exact
# application point referenced there.
GRID_EMISSION_FACTOR_KG_PER_KWH = 0.716

RECOMMENDATION_CATEGORIES = ("behavior_change", "timing_shift", "replacement", "maintenance")
TOP_N_ACTIONS = 3
COVERAGE_THRESHOLD_PERCENT = 70.0

# Model 3's ablated-model-equivalent working error bar (~2.94pp, rounded to
# 3pp) — not its gating 0.31pp. Pre-registered in ml/MODELS.md's Model 4
# design section before this file was written.
DISAGGREGATION_WORKING_MAE_PP = 3.0

# --- Rule-base assumptions, each labeled as such (see ml/DATA.md convention) ---
STANDBY_SHARE_ACTIONABLE_THRESHOLD = 0.06  # ASSUMPTION
STANDBY_REDUCIBLE_FRACTION = 0.30  # ASSUMPTION
AC_SHARE_ACTIONABLE_THRESHOLD = 0.10  # ASSUMPTION
AC_SAVINGS_FRACTION_PER_DEGREE = (
    0.06  # ASSUMPTION — general HVAC literature range 3-8%/degree, midpoint used
)
AC_SETPOINT_SHIFT_DEGREES = 2.0

_NUM_BOOST_ROUND = 150
_BASE_PARAMS: dict[str, object] = {"max_depth": 4, "eta": 0.1, "objective": "reg:squarederror"}

Candidate = dict[str, object]


# --- Rule base ---


def _marginal_bill_savings_paise(
    tariff: TariffModel, sanctioned_load_kw: float, total_kwh: float, savings_kwh_month: float
) -> int:
    """Prices any kWh reduction by re-running the exact same tariff
    calculator used to generate the synthetic ground-truth bills
    (`compute_bill_amount_paise`), rather than a flat rupee-per-unit
    assumption — this correctly handles telescoping slabs (the marginal
    rate depends on where in the slab structure the household currently
    sits) and ToD's blended rate alike, and it's the same function a
    reviewer can independently re-run to check any candidate's number."""
    current = compute_bill_amount_paise(total_kwh, tariff, sanctioned_load_kw)
    reduced = compute_bill_amount_paise(
        max(0.0, total_kwh - savings_kwh_month), tariff, sanctioned_load_kw
    )
    return max(0, current - reduced)


_STAR_UPGRADE_APPLIANCE_KEYS = {"fridge": "fridge", "ac": "ac_1.5ton", "geyser": "geyser_15l"}
_STAR_UPGRADE_OWNERSHIP_KEYS = {"fridge": None, "ac": "owns_ac", "geyser": "owns_geyser"}


def _rule_star_upgrade(category: str, ctx: dict[str, object]) -> Candidate | None:
    """Star-rating upgrade ROI for fridge/AC/geyser. Depends only on
    directly-known appliance inventory (never on Model 3's predicted
    shares), so its savings estimate is identical whether computed from
    predicted or true context — there is no oracle/predicted discrepancy to
    speak of for this rule."""
    ownership_key = _STAR_UPGRADE_OWNERSHIP_KEYS[category]
    if ownership_key is not None and not ctx[ownership_key]:
        return None
    current_star = cast(int, ctx[f"{category}_star"])
    if current_star <= 0 or current_star >= 5:
        return None

    appliance_key = _STAR_UPGRADE_APPLIANCE_KEYS[category]
    lookup = cast(ApplianceLookup, ctx["appliance_lookup"])
    current_per_unit = lookup[(appliance_key, current_star)]
    five_star_per_unit = lookup[(appliance_key, 5)]

    if category == "fridge":
        current_kwh_month = current_per_unit * DAYS_PER_MONTH
        five_star_kwh_month = five_star_per_unit * DAYS_PER_MONTH
    else:
        run_hours_fn = ac_daily_run_hours if category == "ac" else geyser_daily_run_hours
        run_hours = run_hours_fn(cast(float, ctx["climate_temp_c"]))
        current_kwh_month = current_per_unit * run_hours * DAYS_PER_MONTH
        five_star_kwh_month = five_star_per_unit * run_hours * DAYS_PER_MONTH

    savings_kwh_month = current_kwh_month - five_star_kwh_month
    if savings_kwh_month <= 0:
        return None

    savings_paise = _marginal_bill_savings_paise(
        cast(TariffModel, ctx["tariff"]),
        cast(float, ctx["sanctioned_load_kw"]),
        cast(float, ctx["total_kwh"]),
        savings_kwh_month,
    )
    return {
        "rule_name": f"star_upgrade_{category}",
        "category": "replacement",
        "title": f"Upgrade your {current_star}-star {category} to 5-star",
        "body": (
            f"Your {category} is rated {current_star}-star. A 5-star model of "
            "the same size uses meaningfully less electricity for the same "
            "job — see the calculation trace for the exact BEE-cited wattage "
            "figures used."
        ),
        "estimated_monthly_savings_paise": savings_paise,
        "estimated_annual_co2_kg": savings_kwh_month * 12 * GRID_EMISSION_FACTOR_KG_PER_KWH,
        "confidence": "high",
        "calculation_trace": {
            "formula": (
                "(current_star_kwh_month - five_star_kwh_month), priced via "
                "marginal bill difference (compute_bill_amount_paise)"
            ),
            "current_star": current_star,
            "current_kwh_month": current_kwh_month,
            "five_star_kwh_month": five_star_kwh_month,
            "savings_kwh_month": savings_kwh_month,
        },
        "involved_share_category": None,
    }


def _rule_fan_upgrade(ctx: dict[str, object]) -> Candidate | None:
    """Star-rating upgrade for fans. Gated on the user having actually
    supplied fan-count/star-rating inventory data — see ml/MODELS.md's Model
    4 design section: `fans` is the disaggregator's least-reliable category
    without real inventory, so this candidate is suppressed entirely (not
    just downgraded) when that data looks missing/defaulted, i.e. num_fans
    <= 0 or fan_star <= 0."""
    num_fans = cast(int, ctx["num_fans"])
    fan_star = cast(int, ctx["fan_star"])
    if num_fans <= 0 or fan_star <= 0 or fan_star >= 5:
        return None

    lookup = cast(ApplianceLookup, ctx["appliance_lookup"])
    run_hours_multiplier = fan_usage_multiplier(cast(float, ctx["climate_temp_c"]))
    current_kwh_month = (
        lookup[("ceiling_fan", fan_star)] * num_fans * run_hours_multiplier * DAYS_PER_MONTH
    )
    five_star_kwh_month = (
        lookup[("ceiling_fan", 5)] * num_fans * run_hours_multiplier * DAYS_PER_MONTH
    )
    savings_kwh_month = current_kwh_month - five_star_kwh_month
    if savings_kwh_month <= 0:
        return None

    savings_paise = _marginal_bill_savings_paise(
        cast(TariffModel, ctx["tariff"]),
        cast(float, ctx["sanctioned_load_kw"]),
        cast(float, ctx["total_kwh"]),
        savings_kwh_month,
    )
    return {
        "rule_name": "star_upgrade_fans",
        "category": "replacement",
        "title": f"Upgrade your {fan_star}-star fans to 5-star",
        "body": (
            f"Your {num_fans} fan(s) are rated {fan_star}-star. 5-star fans "
            "use meaningfully less power for the same airflow."
        ),
        "estimated_monthly_savings_paise": savings_paise,
        "estimated_annual_co2_kg": savings_kwh_month * 12 * GRID_EMISSION_FACTOR_KG_PER_KWH,
        # Capped below "high" even with valid inventory data — fans is the
        # disaggregator's least-reliable category (Model 3's ablation).
        "confidence": "medium",
        "calculation_trace": {
            "formula": (
                "(current_star_kwh_month - five_star_kwh_month) across all fans, "
                "priced via marginal bill difference"
            ),
            "num_fans": num_fans,
            "current_star": fan_star,
            "current_kwh_month": current_kwh_month,
            "five_star_kwh_month": five_star_kwh_month,
            "savings_kwh_month": savings_kwh_month,
        },
        "involved_share_category": None,
    }


def _rule_geyser_timing_shift(ctx: dict[str, object]) -> Candidate | None:
    """Shifts the peak-block share of geyser use to the cheapest ToD block.

    Honest caveat (see ml/MODELS.md): the synthetic generator bills
    tod_generic tariffs at one blended rate regardless of when electricity
    is actually used (see `build_tariff_lookup`/`compute_bill_amount_paise`),
    so this candidate's savings estimate is NOT cross-validated against the
    generator's own bill computation the way the other rules implicitly are
    — it's computed directly from `tariff_tod.csv`'s real per-block rate
    structure instead, which is what a genuine ToD tariff prices on in
    reality. Capped at "medium" confidence for this reason, not because the
    physics is doubted, but because this project's own synthetic ground
    truth can't confirm it."""
    if not ctx["owns_geyser"] or ctx["tariff_name"] != "tod_generic":
        return None

    shares = cast(dict[str, float], ctx["shares"])
    geyser_kwh_month = shares["geyser"] * cast(float, ctx["total_kwh"])
    if geyser_kwh_month <= 0:
        return None

    tod = cast(pd.DataFrame, ctx["tariff_tod_table"])
    peak = tod[tod["block_name"] == "peak"].iloc[0]
    offpeak = tod[tod["block_name"] == "solar_offpeak"].iloc[0]
    peak_load_share = float(peak["assumed_load_share"])
    shiftable_kwh_month = geyser_kwh_month * peak_load_share
    rate_delta = (
        float(peak["rate_multiplier"]) - float(offpeak["rate_multiplier"])
    ) * TOD_BASE_RATE_RUPEES_PER_UNIT
    savings_rupees = shiftable_kwh_month * rate_delta
    if savings_rupees <= 0:
        return None

    return {
        "rule_name": "geyser_timing_shift",
        "category": "timing_shift",
        "title": "Shift geyser use to off-peak hours",
        "body": (
            "Your tariff charges a lower per-unit rate during solar/off-peak "
            "hours (9am-5pm) than during the evening peak block (6-10pm). "
            "Running your geyser in the off-peak window instead can lower "
            "your bill without using less electricity overall."
        ),
        "estimated_monthly_savings_paise": round(savings_rupees * 100),
        # Pure timing shift: total kWh consumed doesn't change, so there is
        # no CO2 impact — reported honestly as 0, not omitted.
        "estimated_annual_co2_kg": 0.0,
        "confidence": "medium",
        "calculation_trace": {
            "formula": (
                "shiftable_kwh_month * (peak_rate_multiplier - offpeak_rate_multiplier) "
                "* TOD_BASE_RATE_RUPEES_PER_UNIT"
            ),
            "geyser_kwh_month": geyser_kwh_month,
            "peak_load_share_assumed": peak_load_share,
            "peak_rate_multiplier": float(peak["rate_multiplier"]),
            "offpeak_rate_multiplier": float(offpeak["rate_multiplier"]),
            "tod_base_rate_rupees_per_unit": TOD_BASE_RATE_RUPEES_PER_UNIT,
        },
        "involved_share_category": "geyser",
    }


def _rule_standby_reduction(ctx: dict[str, object]) -> Candidate | None:
    """Behavior-change candidate targeting phantom/standby load. See
    ml/MODELS.md: `other_including_standby` is a residual sink whose share
    aggregates slack from the other 7 categories rather than measuring an
    independent signal — capped at "medium" confidence for that reason."""
    shares = cast(dict[str, float], ctx["shares"])
    share = shares["other_including_standby"]
    if share <= STANDBY_SHARE_ACTIONABLE_THRESHOLD:
        return None

    total_kwh = cast(float, ctx["total_kwh"])
    standby_kwh_month = share * total_kwh
    savings_kwh_month = standby_kwh_month * STANDBY_REDUCIBLE_FRACTION
    if savings_kwh_month <= 0:
        return None

    savings_paise = _marginal_bill_savings_paise(
        cast(TariffModel, ctx["tariff"]),
        cast(float, ctx["sanctioned_load_kw"]),
        total_kwh,
        savings_kwh_month,
    )
    return {
        "rule_name": "standby_reduction",
        "category": "behavior_change",
        "title": "Cut phantom/standby load",
        "body": (
            "A meaningful share of your bill looks like standby draw — "
            "chargers, set-top boxes, routers, and similar devices left "
            "plugged in. Using a power strip to fully cut power to these "
            "when not in active use is a low-effort way to reduce this."
        ),
        "estimated_monthly_savings_paise": savings_paise,
        "estimated_annual_co2_kg": savings_kwh_month * 12 * GRID_EMISSION_FACTOR_KG_PER_KWH,
        "confidence": "medium",
        "calculation_trace": {
            "formula": (
                "other_including_standby_share * total_kwh * STANDBY_REDUCIBLE_FRACTION, "
                "priced via marginal bill difference"
            ),
            "share": share,
            "standby_kwh_month": standby_kwh_month,
            "reducible_fraction_assumption": STANDBY_REDUCIBLE_FRACTION,
            "savings_kwh_month": savings_kwh_month,
        },
        "involved_share_category": "other_including_standby",
    }


def _rule_ac_setpoint(ctx: dict[str, object]) -> Candidate | None:
    if not ctx["owns_ac"]:
        return None
    shares = cast(dict[str, float], ctx["shares"])
    share = shares["ac"]
    if share <= AC_SHARE_ACTIONABLE_THRESHOLD:
        return None

    total_kwh = cast(float, ctx["total_kwh"])
    ac_kwh_month = share * total_kwh
    savings_fraction = AC_SAVINGS_FRACTION_PER_DEGREE * AC_SETPOINT_SHIFT_DEGREES
    savings_kwh_month = ac_kwh_month * savings_fraction
    if savings_kwh_month <= 0:
        return None

    savings_paise = _marginal_bill_savings_paise(
        cast(TariffModel, ctx["tariff"]),
        cast(float, ctx["sanctioned_load_kw"]),
        total_kwh,
        savings_kwh_month,
    )
    return {
        "rule_name": "ac_setpoint_adjustment",
        "category": "behavior_change",
        "title": f"Raise your AC setpoint by {AC_SETPOINT_SHIFT_DEGREES:g}°C",
        "body": (
            f"Raising your AC's thermostat setpoint by "
            f"{AC_SETPOINT_SHIFT_DEGREES:g}°C reduces compressor "
            "run-time for a small comfort tradeoff. This is an assumption "
            "based on general HVAC literature, not a measured figure for "
            "your specific AC — see the calculation trace."
        ),
        "estimated_monthly_savings_paise": savings_paise,
        "estimated_annual_co2_kg": savings_kwh_month * 12 * GRID_EMISSION_FACTOR_KG_PER_KWH,
        "confidence": "medium",
        "calculation_trace": {
            "formula": (
                "ac_share * total_kwh * (AC_SAVINGS_FRACTION_PER_DEGREE * "
                "AC_SETPOINT_SHIFT_DEGREES), priced via marginal bill difference"
            ),
            "ac_kwh_month": ac_kwh_month,
            "savings_fraction_assumption": savings_fraction,
            "savings_kwh_month": savings_kwh_month,
        },
        "involved_share_category": "ac",
    }


def _rule_maintenance_check(ctx: dict[str, object]) -> Candidate | None:
    """Anomaly-driven candidate: reuses Model 1's forecast as "what normal
    costs" (same idea as Model 2's residual), flagging the excess above
    normal as a fixable fault rather than a genuine usage change — only for
    the reason bucket that survives Model 2's honest reason-bucket
    limitation (see ml/MODELS.md's Model 2 section)."""
    if not ctx["is_anomaly"] or ctx["reason_bucket"] != "unusual_spike":
        return None
    excess_kwh_month = cast(float, ctx["excess_kwh"])
    if excess_kwh_month <= 0:
        return None

    savings_paise = _marginal_bill_savings_paise(
        cast(TariffModel, ctx["tariff"]),
        cast(float, ctx["sanctioned_load_kw"]),
        cast(float, ctx["total_kwh"]),
        excess_kwh_month,
    )
    return {
        "rule_name": "maintenance_check",
        "category": "maintenance",
        "title": "Get your AC/geyser checked for a fault",
        "body": (
            "This month's usage is well above your normal pattern. A sudden, "
            "sustained jump like this is often caused by a fault (e.g. a "
            "failing thermostat or a compressor running continuously) rather "
            "than a genuine change in behavior. A maintenance check can catch "
            "it before it recurs every month."
        ),
        "estimated_monthly_savings_paise": savings_paise,
        "estimated_annual_co2_kg": excess_kwh_month * 12 * GRID_EMISSION_FACTOR_KG_PER_KWH,
        "confidence": "medium",
        "calculation_trace": {
            "formula": (
                "(actual_total_kwh - expected_normal_kwh), priced via marginal bill difference"
            ),
            "excess_kwh_month": excess_kwh_month,
        },
        "involved_share_category": None,
    }


_RULES = (
    lambda ctx: _rule_star_upgrade("fridge", ctx),
    lambda ctx: _rule_star_upgrade("ac", ctx),
    lambda ctx: _rule_star_upgrade("geyser", ctx),
    _rule_fan_upgrade,
    _rule_geyser_timing_shift,
    _rule_standby_reduction,
    _rule_ac_setpoint,
    _rule_maintenance_check,
)


def generate_candidates(ctx: dict[str, object]) -> list[Candidate]:
    """Runs every rule against one household-month's context, returning
    only the applicable candidates (each rule returns None if inapplicable
    or if it would produce non-positive savings)."""
    candidates = []
    for rule in _RULES:
        candidate = rule(ctx)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _perturb_shares(shares: dict[str, float], category: str, delta_pp: float) -> dict[str, float]:
    perturbed = dict(shares)
    perturbed[category] = max(0.0, perturbed[category] + delta_pp / 100.0)
    total = sum(perturbed.values())
    return {k: v / total for k, v in perturbed.items()}


def _apply_confidence_gate(
    candidates: list[Candidate],
    ctx: dict[str, object],
    top_rule_names: set[object] | None = None,
) -> list[Candidate]:
    """Downgrades a candidate in `top_rule_names` to "low" confidence if a
    +/-`DISAGGREGATION_WORKING_MAE_PP` perturbation of its involved
    disaggregation share would knock it out of the top `TOP_N_ACTIONS` by the
    rule base's own raw savings estimate. `top_rule_names` defaults to the
    rule base's own raw-savings top N when not given (used by standalone
    unit tests); `recommend_for_household` passes the *actually shown*
    top N (by the learned ranker's score) instead — the gate only means
    anything when it's checked against what a user would actually see, not
    an intermediate ranking nobody is shown. Pre-registered in ml/MODELS.md
    before this file was written."""
    if top_rule_names is None:
        ranked = sorted(
            candidates, key=lambda c: cast(int, c["estimated_monthly_savings_paise"]), reverse=True
        )
        top_rule_names = {c["rule_name"] for c in ranked[:TOP_N_ACTIONS]}

    result = []
    for candidate in candidates:
        updated = dict(candidate)
        share_category = candidate["involved_share_category"]
        if share_category is not None and candidate["rule_name"] in top_rule_names:
            unstable = False
            for delta_pp in (DISAGGREGATION_WORKING_MAE_PP, -DISAGGREGATION_WORKING_MAE_PP):
                perturbed_ctx = dict(ctx)
                perturbed_ctx["shares"] = _perturb_shares(
                    cast(dict[str, float], ctx["shares"]), cast(str, share_category), delta_pp
                )
                perturbed_candidates = generate_candidates(perturbed_ctx)
                perturbed_top = sorted(
                    perturbed_candidates,
                    key=lambda c: cast(int, c["estimated_monthly_savings_paise"]),
                    reverse=True,
                )[:TOP_N_ACTIONS]
                if candidate["rule_name"] not in {c["rule_name"] for c in perturbed_top}:
                    unstable = True
                    break
            if unstable:
                updated["confidence"] = "low"
        result.append(updated)
    return result


# --- Context building ---

_RULE_NAMES = (
    "star_upgrade_fridge",
    "star_upgrade_ac",
    "star_upgrade_geyser",
    "star_upgrade_fans",
    "geyser_timing_shift",
    "standby_reduction",
    "ac_setpoint_adjustment",
    "maintenance_check",
)


def _shared_ctx_fields(
    row: pd.Series,
    tariff_lookup: dict[str, TariffModel],
    appliance_lookup: ApplianceLookup,
    tariff_tod_table: pd.DataFrame,
) -> dict[str, object]:
    """Fields that are identical between the predicted and true context for
    a household-month: directly-known appliance inventory, tariff, and the
    actual metered total — none of these are model outputs."""
    tariff_name = str(row["tariff_name"])
    return {
        "tariff_name": tariff_name,
        "tariff": tariff_lookup[tariff_name],
        "sanctioned_load_kw": float(row["sanctioned_load_kw"]),
        "climate_temp_c": float(row["climate_temp_c"]),
        "total_kwh": float(row["target_units_wh"]) / 1000.0,
        "family_size": int(row["family_size"]),
        "zone": str(row["zone"]),
        "fridge_star": int(row["fridge_star"]),
        "owns_ac": bool(row["owns_ac"]),
        "ac_star": int(row["ac_star"]),
        "owns_geyser": bool(row["owns_geyser"]),
        "geyser_star": int(row["geyser_star"]),
        "num_fans": int(row["num_fans"]),
        "fan_star": int(row["fan_star"]),
        "num_bulbs": int(row["num_bulbs"]),
        "owns_washing_machine": bool(row["owns_washing_machine"]),
        "owns_tv": bool(row["owns_tv"]),
        "appliance_lookup": appliance_lookup,
        "tariff_tod_table": tariff_tod_table,
    }


def _predicted_context(row: pd.Series, shared: dict[str, object]) -> dict[str, object]:
    """The context Model 4 actually has at serving time: Model 3's predicted
    shares, Model 1/2-derived anomaly signal."""
    ctx = dict(shared)
    ctx["shares"] = {
        category: float(row[f"predicted_{category}_share"]) for category in APPLIANCE_CATEGORIES
    }
    ctx["is_anomaly"] = bool(row["predicted_is_anomaly"])
    ctx["reason_bucket"] = str(row["predicted_reason_bucket"])
    ctx["excess_kwh"] = float(row["predicted_excess_kwh"])
    return ctx


def _true_context(row: pd.Series, shared: dict[str, object]) -> dict[str, object]:
    """The oracle context, used only for evaluation: true category shares
    and true anomaly ground truth (never available at serving time) —
    defines the "achievable savings" ceiling Model 4 is scored against."""
    ctx = dict(shared)
    ctx["shares"] = {category: float(row[f"{category}_share"]) for category in APPLIANCE_CATEGORIES}
    ctx["is_anomaly"] = bool(row["is_anomaly"])
    ctx["reason_bucket"] = anomaly_module._bucket_true_reason(str(row["anomaly_reason"]))
    base_kwh = sum(float(row[f"{category}_kwh"]) for category in APPLIANCE_CATEGORIES)
    ctx["excess_kwh"] = max(0.0, float(cast(float, ctx["total_kwh"])) - base_kwh)
    return ctx


def _add_predictions(
    examples: pd.DataFrame,
    forecaster_boosters: dict[str, xgb.Booster],
    forecaster_metadata: dict[str, object],
    anomaly_state: dict[str, object],
    disaggregator_boosters: dict[str, xgb.Booster],
    disaggregator_categories: dict[str, list[str]],
) -> pd.DataFrame:
    """Runs Models 1, 2, and 3 over every household-month once, up front, so
    candidate generation itself is pure Python row logic with no repeated
    model inference inside the loop."""
    result = examples.copy()

    feature_columns = cast(list[str], forecaster_metadata["feature_columns"])
    forecast_categories = cast(dict[str, list[str]], forecaster_metadata["categories"])
    forecast_features, _target, _ = encode_features(result, categories=forecast_categories)
    dmatrix = xgb.DMatrix(forecast_features[feature_columns])
    result["predicted_units_wh"] = forecaster_boosters["point"].predict(dmatrix)

    residual_ratio = (result["target_units_wh"] - result["predicted_units_wh"]) / result[
        "predicted_units_wh"
    ]
    median = cast(float, anomaly_state["median_residual_ratio"])
    mad = cast(float, anomaly_state["mad_residual_ratio"])
    z_threshold = cast(float, anomaly_state["z_threshold"])
    seasonal_cutoff = cast(float, anomaly_state["seasonal_cutoff"])
    robust_z = 0.6745 * (residual_ratio - median) / mad

    result["predicted_is_anomaly"] = robust_z.abs() > z_threshold
    result["predicted_reason_bucket"] = [
        anomaly_module._reason_bucket(ratio, seasonal_cutoff) if flagged else "none"
        for ratio, flagged in zip(residual_ratio, result["predicted_is_anomaly"], strict=True)
    ]
    result["predicted_excess_kwh"] = (
        (result["target_units_wh"] - result["predicted_units_wh"]) / 1000.0
    ).clip(lower=0.0)

    disagg_features, _targets, _cats = encode_disaggregation_features(
        result, categories=disaggregator_categories
    )
    predicted_shares = disaggregator_module._predict_shares(disaggregator_boosters, disagg_features)
    for category in APPLIANCE_CATEGORIES:
        result[f"predicted_{category}_share"] = predicted_shares[category].to_numpy()

    return result


def _candidate_feature_row(
    candidate: Candidate,
    ctx: dict[str, object],
    zone_categories: list[str],
    tariff_categories: list[str],
) -> dict[str, float]:
    shares = cast(dict[str, float], ctx["shares"])
    features: dict[str, float] = {
        f"is_rule_{name}": (1.0 if candidate["rule_name"] == name else 0.0) for name in _RULE_NAMES
    }
    features["predicted_savings_paise"] = float(
        cast(int, candidate["estimated_monthly_savings_paise"])
    )
    features["family_size"] = float(cast(int, ctx["family_size"]))
    features["sanctioned_load_kw"] = float(cast(float, ctx["sanctioned_load_kw"]))
    features["climate_temp_c"] = float(cast(float, ctx["climate_temp_c"]))
    features["predicted_is_anomaly"] = 1.0 if ctx.get("is_anomaly") else 0.0
    for category in APPLIANCE_CATEGORIES:
        features[f"share_{category}"] = shares[category]
    for zone in zone_categories:
        features[f"zone_{zone}"] = 1.0 if ctx["zone"] == zone else 0.0
    for tariff_name in tariff_categories:
        features[f"tariff_name_{tariff_name}"] = 1.0 if ctx["tariff_name"] == tariff_name else 0.0
    return features


def recommend_for_household(
    ctx: dict[str, object],
    ranker_booster: xgb.Booster,
    zone_categories: list[str],
    tariff_categories: list[str],
) -> list[Candidate]:
    """The actual serving-time entrypoint: generates candidates, scores them
    with the learned ranker, takes the top `TOP_N_ACTIONS`, and applies the
    confidence gate against that *exact shown* ranking — not the rule base's
    raw estimate, which is what `_apply_confidence_gate`'s default behavior
    checks instead. Reference implementation for whatever Phase 3
    recommendation endpoint eventually calls this; also what `_evaluate`
    below uses to compute the confidence-gate's suppression rate."""
    candidates = generate_candidates(ctx)
    if not candidates:
        return []

    feature_df = pd.DataFrame(
        [_candidate_feature_row(c, ctx, zone_categories, tariff_categories) for c in candidates]
    )
    scores = ranker_booster.predict(xgb.DMatrix(feature_df))
    ranked = [
        c
        for c, _score in sorted(
            zip(candidates, scores, strict=True), key=lambda p: p[1], reverse=True
        )
    ]
    shown = ranked[:TOP_N_ACTIONS]
    shown_rule_names = {c["rule_name"] for c in shown}

    gated_by_name = {
        c["rule_name"]: c for c in _apply_confidence_gate(candidates, ctx, shown_rule_names)
    }
    return [gated_by_name[c["rule_name"]] for c in shown]


# --- Training and evaluation ---


def _build_ranker_training_data(
    examples: pd.DataFrame,
    tariff_lookup: dict[str, TariffModel],
    appliance_lookup: ApplianceLookup,
    tariff_tod_table: pd.DataFrame,
    zone_categories: list[str],
    tariff_categories: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """One row per (household-month, predicted-applicable-candidate) pair.
    Target: that same candidate's savings under the TRUE context (0 if the
    candidate isn't applicable there) — the ranker learns to predict a
    candidate's *real* value from only serving-time-available features."""
    feature_rows = []
    targets = []

    for _, row in examples.iterrows():
        shared = _shared_ctx_fields(row, tariff_lookup, appliance_lookup, tariff_tod_table)
        predicted_ctx = _predicted_context(row, shared)
        true_ctx = _true_context(row, shared)

        predicted_candidates = generate_candidates(predicted_ctx)
        true_by_name = {c["rule_name"]: c for c in generate_candidates(true_ctx)}

        for candidate in predicted_candidates:
            oracle_candidate = true_by_name.get(cast(str, candidate["rule_name"]))
            oracle_savings = (
                float(cast(int, oracle_candidate["estimated_monthly_savings_paise"]))
                if oracle_candidate is not None
                else 0.0
            )
            feature_rows.append(
                _candidate_feature_row(candidate, predicted_ctx, zone_categories, tariff_categories)
            )
            targets.append(oracle_savings)

    features_df = pd.DataFrame(feature_rows)
    targets_series = pd.Series(targets, name="oracle_savings_paise")
    return features_df, targets_series


def _true_savings_of(candidate: Candidate, true_by_name: dict[object, Candidate]) -> int:
    matched = true_by_name.get(candidate["rule_name"])
    return cast(int, matched["estimated_monthly_savings_paise"]) if matched else 0


def _evaluate(
    ranker_booster: xgb.Booster,
    examples: pd.DataFrame,
    tariff_lookup: dict[str, TariffModel],
    appliance_lookup: ApplianceLookup,
    tariff_tod_table: pd.DataFrame,
    zone_categories: list[str],
    tariff_categories: list[str],
) -> dict[str, object]:
    """For each household-month, compares three ways of picking the top-N
    candidates: oracle (sorts by true savings directly — the achievable
    upper bound), naive (sorts by the rule base's own raw point estimate —
    the ablation baseline), and learned (sorts by the ranker's predicted
    true savings — what's actually shipped). Coverage is realized (true)
    savings of the chosen top-N divided by the true savings of every
    applicable candidate for that household-month."""
    oracle_coverages = []
    learned_coverages = []
    naive_coverages = []
    n_low_confidence_shown = 0

    for _, row in examples.iterrows():
        shared = _shared_ctx_fields(row, tariff_lookup, appliance_lookup, tariff_tod_table)
        predicted_ctx = _predicted_context(row, shared)
        true_ctx = _true_context(row, shared)

        predicted_candidates = generate_candidates(predicted_ctx)
        true_candidates = generate_candidates(true_ctx)
        true_by_name = {c["rule_name"]: c for c in true_candidates}
        total_achievable = sum(
            cast(int, c["estimated_monthly_savings_paise"]) for c in true_candidates
        )
        if total_achievable <= 0 or not predicted_candidates:
            continue

        oracle_top = sorted(
            predicted_candidates,
            key=lambda c: _true_savings_of(c, true_by_name),
            reverse=True,
        )[:TOP_N_ACTIONS]
        oracle_coverages.append(
            sum(_true_savings_of(c, true_by_name) for c in oracle_top) / total_achievable
        )

        naive_top = sorted(
            predicted_candidates,
            key=lambda c: cast(int, c["estimated_monthly_savings_paise"]),
            reverse=True,
        )[:TOP_N_ACTIONS]
        naive_coverages.append(
            sum(_true_savings_of(c, true_by_name) for c in naive_top) / total_achievable
        )

        feature_df = pd.DataFrame(
            [
                _candidate_feature_row(c, predicted_ctx, zone_categories, tariff_categories)
                for c in predicted_candidates
            ]
        )
        scores = ranker_booster.predict(xgb.DMatrix(feature_df))
        learned_top = [
            c
            for c, _score in sorted(
                zip(predicted_candidates, scores, strict=True),
                key=lambda pair: pair[1],
                reverse=True,
            )[:TOP_N_ACTIONS]
        ]
        learned_coverages.append(
            sum(_true_savings_of(c, true_by_name) for c in learned_top) / total_achievable
        )

        # Confidence-gate suppression rate: gate the candidates that are
        # ACTUALLY shown (the learned ranker's top-N), not the rule base's
        # raw estimate — see `recommend_for_household`'s docstring for why
        # that distinction matters. This measures work the coverage metric
        # above can't see: whether the gate is catching genuinely unstable
        # recommendations before they're shown, not whether they're the
        # highest-value ones.
        shown_rule_names = {c["rule_name"] for c in learned_top}
        gated_shown = _apply_confidence_gate(predicted_candidates, predicted_ctx, shown_rule_names)
        gated_by_name = {c["rule_name"]: c for c in gated_shown}
        n_low_confidence_shown += sum(
            1 for c in learned_top if gated_by_name[c["rule_name"]]["confidence"] == "low"
        )

    n = len(learned_coverages)
    learned_mean = float(np.mean(learned_coverages)) if n else 0.0
    return {
        "n_households_evaluated": n,
        "oracle_mean_coverage": float(np.mean(oracle_coverages)) if n else 0.0,
        "learned_mean_coverage": learned_mean,
        "naive_mean_coverage": float(np.mean(naive_coverages)) if n else 0.0,
        "learned_coverage_pass": learned_mean * 100 >= COVERAGE_THRESHOLD_PERCENT,
        "low_confidence_recommendations_shown": n_low_confidence_shown,
        "low_confidence_recommendations_per_1000_households": (
            (n_low_confidence_shown / n) * 1000 if n else 0.0
        ),
    }


def train_with_evaluation(
    df: pd.DataFrame,
    seed: int = 42,
    forecaster_path: Path = FORECASTER_ARTIFACT_PATH,
    anomaly_path: Path = ANOMALY_ARTIFACT_PATH,
    disaggregator_path: Path = DISAGGREGATOR_ARTIFACT_PATH,
) -> tuple[xgb.Booster, dict[str, object], dict[str, object]]:
    """`forecaster_path`/`anomaly_path`/`disaggregator_path` default to the
    real checked-in artifacts but are overridable — tests pass small
    freshly-trained artifacts here rather than depending on
    `backend/models/`'s real (10,000-household) versions, matching the
    pattern `models.anomaly`'s tests use for its Model 1 dependency."""
    if not forecaster_path.exists():
        raise SystemExit(f"{forecaster_path} not found — train Model 1 first")
    if not anomaly_path.exists():
        raise SystemExit(f"{anomaly_path} not found — train Model 2 first")
    if not disaggregator_path.exists():
        raise SystemExit(f"{disaggregator_path} not found — train Model 3 first")

    forecaster_boosters, forecaster_metadata = load_forecaster_artifact(forecaster_path)
    anomaly_state = json.loads(anomaly_path.read_text())
    disaggregator_boosters, disaggregator_metadata = load_disaggregator_artifact(disaggregator_path)
    disaggregator_categories = cast(dict[str, list[str]], disaggregator_metadata["categories"])

    tables = load_reference_tables()
    tariff_lookup = build_tariff_lookup(tables)
    appliance_lookup = build_appliance_lookup(tables["appliances"])
    tariff_tod_table = tables["tariff_tod"]

    examples = build_recommender_examples(df)
    examples = _add_predictions(
        examples,
        forecaster_boosters,
        forecaster_metadata,
        anomaly_state,
        disaggregator_boosters,
        disaggregator_categories,
    )

    train_examples, test_examples = household_train_test_split(examples, seed=seed)

    zone_categories = cast(list[str], disaggregator_categories["zone"])
    tariff_categories = cast(list[str], disaggregator_categories["tariff_name"])

    train_features, train_targets = _build_ranker_training_data(
        train_examples,
        tariff_lookup,
        appliance_lookup,
        tariff_tod_table,
        zone_categories,
        tariff_categories,
    )

    dtrain = xgb.DMatrix(train_features, label=train_targets)
    params = {**_BASE_PARAMS, "seed": seed}
    ranker_booster = xgb.train(params, dtrain, num_boost_round=_NUM_BOOST_ROUND)

    metrics = _evaluate(
        ranker_booster,
        test_examples,
        tariff_lookup,
        appliance_lookup,
        tariff_tod_table,
        zone_categories,
        tariff_categories,
    )
    metrics["coverage_threshold_percent"] = COVERAGE_THRESHOLD_PERCENT

    # Out-of-distribution stress test (diagnostic, not gating): the zone
    # with the fewest TRAIN households, evaluated the same way — see
    # ml/MODELS.md's Model 4 design section for why this was pre-registered
    # before training (both the rule base and the ranker are shaped by the
    # same synthetic population; the in-distribution coverage number alone
    # is too easy a bar).
    train_zones = train_examples.drop_duplicates("household_id")[["household_id", "zone"]]
    zone_counts = train_zones["zone"].value_counts()
    ood_zone = str(zone_counts.idxmin())
    ood_test_examples = test_examples[test_examples["zone"] == ood_zone]
    if len(ood_test_examples) > 0:
        ood_metrics = _evaluate(
            ranker_booster,
            ood_test_examples,
            tariff_lookup,
            appliance_lookup,
            tariff_tod_table,
            zone_categories,
            tariff_categories,
        )
    else:
        ood_metrics = {"note": f"no test households in OOD zone {ood_zone}"}
    metrics["ood_stress_test"] = {
        "zone": ood_zone,
        "n_train_households_in_zone": int(zone_counts.min()),
        **ood_metrics,
    }

    metadata: dict[str, object] = {
        "model_version": MODEL_VERSION,
        "feature_columns": list(train_features.columns),
        "zone_categories": zone_categories,
        "tariff_categories": tariff_categories,
        "forecaster_version": forecaster_metadata["model_version"],
        "anomaly_version": anomaly_state["model_version"],
        "disaggregator_version": disaggregator_metadata["model_version"],
    }
    return ranker_booster, metadata, metrics


def save_artifact(
    booster: xgb.Booster, metadata: dict[str, object], output_dir: Path = ARTIFACT_DIR
) -> Path:
    payload = dict(metadata)
    payload["ranker_model_json"] = booster.save_raw(raw_format="json").decode("utf-8")

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{MODEL_VERSION}.json"
    path.write_text(json.dumps(payload))
    return path


def load_artifact(path: Path) -> tuple[xgb.Booster, dict[str, object]]:
    payload = json.loads(path.read_text())
    booster = xgb.Booster()
    booster.load_model(bytearray(payload.pop("ranker_model_json").encode("utf-8")))
    return booster, payload


def save_metrics_report(metrics: dict[str, object], report_dir: Path = REPORT_DIR) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"{MODEL_VERSION}_metrics.json"
    path.write_text(json.dumps(metrics, indent=2))
    return path


def main() -> None:
    df = generate_dataset()
    booster, metadata, metrics = train_with_evaluation(df)
    print(json.dumps(metrics, indent=2))

    artifact_path = save_artifact(booster, metadata)
    report_path = save_metrics_report(metrics)
    print(f"Saved model to {artifact_path}")
    print(f"Saved metrics report to {report_path}")

    if not metrics["learned_coverage_pass"]:
        raise SystemExit(
            f"Learned-ranker top-{TOP_N_ACTIONS} coverage "
            f"{cast(float, metrics['learned_mean_coverage']):.3f} below "
            f"{COVERAGE_THRESHOLD_PERCENT}% threshold"
        )


if __name__ == "__main__":
    main()
