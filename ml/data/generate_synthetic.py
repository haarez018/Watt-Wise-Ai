"""Synthetic Indian household electricity dataset generator.

Produces exactly 10,000 households x 12 months = 120,000 household-months.
Every reference number this reads from `ml/data/reference/` is cited or
explicitly labeled as a modeling assumption in `ml/DATA.md` — read that file
before trusting any number produced here.

This is deliberately NOT a physics simulation. It assigns each household a
fixed appliance profile, scales AC and geyser usage by a simple
temperature-dependent run-hours curve, and sums per-category energy draw. See
`ml/DATA.md` -> "What this dataset does NOT model" for the explicit list of
things left out on purpose (humidity, occupancy schedules, appliance duty
cycles, etc.).
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from wattwise_climate import MONTH_COLUMNS, load_climate_reference_tables
from wattwise_tariffs import (
    build_tariff_lookup,
    compute_bill_amount_paise,
    load_tariff_reference_tables,
)

REFERENCE_DIR = Path(__file__).resolve().parent / "reference"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "processed" / "household_months.csv"

N_HOUSEHOLDS_DEFAULT = 10_000
N_MONTHS = 12
DAYS_PER_MONTH = 30  # deliberate simplification — see DATA.md
ANOMALY_PROBABILITY = 0.04
ANOMALY_REASONS = (
    "unusual_spike",
    "unusual_drop",
    "night_load_surge",
    "seasonal_deviation",
    "sustained_high",
)
STAR_RATINGS = (1, 2, 3, 4, 5)

APPLIANCE_CATEGORIES = (
    "fridge",
    "ac",
    "geyser",
    "lighting",
    "fans",
    "washing_machine",
    "television_entertainment",
    "other_including_standby",
)


def load_reference_tables() -> dict[str, pd.DataFrame]:
    tables = {"appliances": pd.read_csv(REFERENCE_DIR / "appliance_wattages.csv")}
    tables.update(load_climate_reference_tables())
    tables.update(load_tariff_reference_tables())
    return tables


ApplianceLookup = dict[tuple[str, int | None], float]


def build_appliance_lookup(appliances: pd.DataFrame) -> ApplianceLookup:
    """Precomputes a plain-dict lookup so the per-household-month loop never
    pays pandas' boolean-mask overhead — this is called 120,000+ times."""
    lookup: ApplianceLookup = {}
    for _, row in appliances.iterrows():
        star = row["star_rating"]
        key_star = int(star) if pd.notna(star) else None
        lookup[(row["category"], key_star)] = float(row["kwh_value"])
    return lookup


@dataclass
class Household:
    household_id: int
    city: str
    zone: str
    family_size: int
    tariff_name: str
    sanctioned_load_kw: float
    fridge_star: int
    owns_ac: bool
    ac_star: int | None
    owns_geyser: bool
    geyser_star: int | None
    num_fans: int
    fan_star: int
    num_bulbs: int
    owns_washing_machine: bool
    owns_tv: bool


def generate_households(n: int, rng: np.random.Generator, cities: pd.DataFrame) -> list[Household]:
    tariff_choices = np.array(["tneb", "bescom", "tod_generic"])
    fridge_star_probs = [0.05, 0.15, 0.4, 0.25, 0.15]
    ac_star_probs = [0.05, 0.15, 0.4, 0.25, 0.15]
    geyser_star_probs = [0.1, 0.2, 0.4, 0.2, 0.1]
    fan_star_probs = [0.15, 0.2, 0.35, 0.2, 0.1]

    households = []
    for i in range(n):
        city_row = cities.iloc[rng.integers(0, len(cities))]
        family_size = int(
            rng.choice(
                [1, 2, 3, 4, 5, 6, 7, 8],
                p=[0.08, 0.18, 0.22, 0.24, 0.14, 0.08, 0.04, 0.02],
            )
        )
        tariff_name = str(rng.choice(tariff_choices, p=[0.4, 0.4, 0.2]))
        num_fans = max(1, family_size // 2 + int(rng.integers(0, 2)))
        num_bulbs = max(3, family_size + int(rng.integers(0, 4)))
        owns_ac = bool(rng.random() < (0.20 + 0.05 * family_size))
        owns_geyser = bool(rng.random() < 0.55)
        owns_washing_machine = bool(rng.random() < 0.5)
        owns_tv = bool(rng.random() < 0.92)
        sanctioned_load_kw = round(1.0 + 0.5 * family_size + (2.0 if owns_ac else 0.0), 1)

        households.append(
            Household(
                household_id=i,
                city=str(city_row["city"]),
                zone=str(city_row["zone"]),
                family_size=family_size,
                tariff_name=tariff_name,
                sanctioned_load_kw=sanctioned_load_kw,
                fridge_star=int(rng.choice(STAR_RATINGS, p=fridge_star_probs)),
                owns_ac=owns_ac,
                ac_star=(int(rng.choice(STAR_RATINGS, p=ac_star_probs)) if owns_ac else None),
                owns_geyser=owns_geyser,
                geyser_star=(
                    int(rng.choice(STAR_RATINGS, p=geyser_star_probs)) if owns_geyser else None
                ),
                num_fans=num_fans,
                fan_star=int(rng.choice(STAR_RATINGS, p=fan_star_probs)),
                num_bulbs=num_bulbs,
                owns_washing_machine=owns_washing_machine,
                owns_tv=owns_tv,
            )
        )
    return households


def ac_daily_run_hours(temp_c: float) -> float:
    """No AC use below 24C, ramping to a capped 8 hrs/day by the high 30s.

    Public (not `_`-prefixed): this is also the AC-usage physics Model 4's
    rule base reuses to estimate a setpoint-adjustment's savings (see
    ml/models/recommender.py) — it's this project's own temperature-vs-usage
    model, not an independently validated physical law, so treat it as a
    shared modeling assumption, not a citation, wherever it's reused."""
    if temp_c <= 24:
        return 0.0
    return min(8.0, (temp_c - 24) * 0.55)


def geyser_daily_run_hours(temp_c: float) -> float:
    """More geyser use in cooler months; a small floor even in peak summer.

    Public for the same reason as `ac_daily_run_hours` above — reused by
    Model 4's rule base."""
    if temp_c >= 30:
        return 0.15
    return max(0.15, 1.0 - (temp_c / 30))


def fan_usage_multiplier(temp_c: float) -> float:
    """Fans run more as it gets hotter; capped so cold months don't hit zero.

    Public for the same reason as `ac_daily_run_hours` above — reused by
    Model 4's rule base's fan star-rating upgrade candidate."""
    return 0.5 + min(1.0, temp_c / 35)


def simulate_month(
    household: Household,
    temp_c: float,
    appliances: ApplianceLookup,
    rng: np.random.Generator,
) -> dict[str, float]:
    fridge_kwh = appliances[("fridge", household.fridge_star)] * DAYS_PER_MONTH

    ac_kwh = 0.0
    if household.owns_ac:
        per_hour = appliances[("ac_1.5ton", household.ac_star)]
        ac_kwh = per_hour * ac_daily_run_hours(temp_c) * DAYS_PER_MONTH

    geyser_kwh = 0.0
    if household.owns_geyser:
        per_hour = appliances[("geyser_15l", household.geyser_star)]
        geyser_kwh = per_hour * geyser_daily_run_hours(temp_c) * DAYS_PER_MONTH

    lighting_kwh = appliances[("led_bulb", None)] * household.num_bulbs * DAYS_PER_MONTH

    fans_kwh = (
        appliances[("ceiling_fan", household.fan_star)]
        * household.num_fans
        * fan_usage_multiplier(temp_c)
        * DAYS_PER_MONTH
    )

    washing_machine_kwh = 0.0
    if household.owns_washing_machine:
        washing_machine_kwh = appliances[("washing_machine", None)] * DAYS_PER_MONTH

    television_kwh = 0.0
    if household.owns_tv:
        television_kwh = appliances[("television", None)] * DAYS_PER_MONTH

    metered_kwh = (
        fridge_kwh
        + ac_kwh
        + geyser_kwh
        + lighting_kwh
        + fans_kwh
        + washing_machine_kwh
        + television_kwh
    )
    # Standby/misc is a percentage add-on, not a per-unit figure — see DATA.md.
    standby_share = rng.uniform(0.05, 0.08)
    other_kwh = metered_kwh * standby_share / (1 - standby_share)

    return {
        "fridge": fridge_kwh,
        "ac": ac_kwh,
        "geyser": geyser_kwh,
        "lighting": lighting_kwh,
        "fans": fans_kwh,
        "washing_machine": washing_machine_kwh,
        "television_entertainment": television_kwh,
        "other_including_standby": other_kwh,
    }


def maybe_inject_anomaly(
    base_kwh: float, rng: np.random.Generator
) -> tuple[float, bool, str | None]:
    if rng.random() >= ANOMALY_PROBABILITY:
        return base_kwh, False, None

    reason = str(rng.choice(ANOMALY_REASONS))
    if reason in ("unusual_spike", "night_load_surge", "sustained_high"):
        multiplier = rng.uniform(1.35, 1.9)
    elif reason == "unusual_drop":
        multiplier = rng.uniform(0.4, 0.65)
    else:  # seasonal_deviation
        multiplier = rng.uniform(1.2, 1.5)
    return base_kwh * multiplier, True, reason


def generate_dataset(n_households: int = N_HOUSEHOLDS_DEFAULT, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tables = load_reference_tables()
    households = generate_households(n_households, rng, tables["cities"])

    appliance_lookup = build_appliance_lookup(tables["appliances"])
    tariff_lookup = build_tariff_lookup(tables)
    zone_temps: dict[str, list[float]] = {
        str(row["zone"]): [float(row[m]) for m in MONTH_COLUMNS]
        for _, row in tables["zone_temps"].iterrows()
    }

    rows: list[dict[str, object]] = []
    for household in households:
        monthly_temps = zone_temps[household.zone]
        tariff = tariff_lookup[household.tariff_name]
        for month_index in range(1, N_MONTHS + 1):
            temp_c = monthly_temps[month_index - 1]
            breakdown = simulate_month(household, temp_c, appliance_lookup, rng)
            base_total_kwh = sum(breakdown.values())

            final_kwh, is_anomaly, anomaly_reason = maybe_inject_anomaly(base_total_kwh, rng)
            amount_paise = compute_bill_amount_paise(
                final_kwh, tariff, household.sanctioned_load_kw
            )

            row: dict[str, object] = {
                "household_id": household.household_id,
                "month_index": month_index,
                "city": household.city,
                "zone": household.zone,
                "family_size": household.family_size,
                "tariff_name": household.tariff_name,
                "sanctioned_load_kw": household.sanctioned_load_kw,
                "climate_temp_c": temp_c,
                "units_consumed_wh": round(final_kwh * 1000),
                "amount_paise": amount_paise,
                "is_anomaly": is_anomaly,
                "anomaly_reason": anomaly_reason or "",
                # Appliance inventory — matches what the product's onboarding
                # wizard actually collects (Phase 1's "~60 seconds of
                # checkboxes with age/star rating"), so Model 3 (disaggregator)
                # can use it as a real, available-at-inference-time input
                # rather than something only the generator knows.
                "fridge_star": household.fridge_star,
                "owns_ac": household.owns_ac,
                "ac_star": household.ac_star or 0,
                "owns_geyser": household.owns_geyser,
                "geyser_star": household.geyser_star or 0,
                "num_fans": household.num_fans,
                "fan_star": household.fan_star,
                "num_bulbs": household.num_bulbs,
                "owns_washing_machine": household.owns_washing_machine,
                "owns_tv": household.owns_tv,
            }
            for category in APPLIANCE_CATEGORIES:
                row[f"{category}_kwh"] = breakdown[category]
            rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--households", type=int, default=N_HOUSEHOLDS_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    df = generate_dataset(n_households=args.households, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df):,} household-months ({args.households:,} households) to {args.output}")


if __name__ == "__main__":
    main()
