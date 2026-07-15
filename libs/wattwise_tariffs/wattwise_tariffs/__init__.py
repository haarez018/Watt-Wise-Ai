"""WattWise AI's Indian electricity tariff calculator.

The single source of truth for how kWh consumption becomes a rupee bill —
used by `ml/data/generate_synthetic.py` to compute the synthetic dataset's
ground-truth bills, by `ml/models/recommender.py` to price every
recommendation candidate, and by the backend's forecast endpoint to convert
`predicted_units_wh` into `predicted_amount_paise`. All three call the exact
same code, not independent reimplementations of the same slab/ToD logic —
see `docs/RUNBOOK.md`'s "known operational quirks" for why that duplication
would be a silent time bomb (a tariff schedule update made in one place and
missed in another).

Reference tables (tariff schedules, sourced/cited in `ml/DATA.md`'s "Tariff
structures" section) live in `reference/`, a sibling of this package
directory — resolved relative to `__file__`, so this only works with an
editable install (`pip install -e`), which is how both `ml/` and `backend/`
install this package in this monorepo.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"

# ₹/unit before Time-of-Day multipliers apply. Assumption, not a published
# per-DISCOM figure — see ml/DATA.md.
TOD_BASE_RATE_RUPEES_PER_UNIT = 7.50


@dataclass
class TariffModel:
    is_tod: bool
    slabs: list[tuple[int, int | None, float]]  # (min_units, max_units_or_None, rate_per_unit)
    tod_rate_per_unit: float
    fixed_rate_per_kw: float


def load_tariff_reference_tables() -> dict[str, pd.DataFrame]:
    return {
        "tariff_slabs": pd.read_csv(REFERENCE_DIR / "tariff_slabs.csv"),
        "tariff_fixed": pd.read_csv(REFERENCE_DIR / "tariff_fixed_charges.csv"),
        "tariff_tod": pd.read_csv(REFERENCE_DIR / "tariff_tod.csv"),
    }


def build_tariff_lookup(tables: dict[str, pd.DataFrame]) -> dict[str, TariffModel]:
    """Precomputes plain-Python tariff structures once, so a per-household-
    month loop (training) or a per-request lookup (serving) never re-filters
    the reference DataFrames."""
    tod = tables["tariff_tod"]
    blended_multiplier = float((tod["rate_multiplier"] * tod["assumed_load_share"]).sum())
    tod_rate = TOD_BASE_RATE_RUPEES_PER_UNIT * blended_multiplier

    fixed = tables["tariff_fixed"].set_index("tariff_name")["fixed_charge_rupees_per_kw_sanctioned"]

    lookup: dict[str, TariffModel] = {}
    slab_table = tables["tariff_slabs"]
    for tariff_name in fixed.index:
        if tariff_name == "tod_generic":
            lookup[tariff_name] = TariffModel(
                is_tod=True,
                slabs=[],
                tod_rate_per_unit=tod_rate,
                fixed_rate_per_kw=float(fixed[tariff_name]),
            )
            continue

        rows = slab_table[slab_table["tariff_name"] == tariff_name].sort_values("slab_order")
        slabs = [
            (
                int(r["min_units"]),
                int(r["max_units"]) if pd.notna(r["max_units"]) else None,
                float(r["rate_rupees_per_unit"]),
            )
            for _, r in rows.iterrows()
        ]
        lookup[tariff_name] = TariffModel(
            is_tod=False,
            slabs=slabs,
            tod_rate_per_unit=0.0,
            fixed_rate_per_kw=float(fixed[tariff_name]),
        )
    return lookup


def compute_bill_amount_paise(
    units_kwh: float, tariff: TariffModel, sanctioned_load_kw: float
) -> int:
    whole_units = round(units_kwh)

    if tariff.is_tod:
        amount_rupees = whole_units * tariff.tod_rate_per_unit
    else:
        amount_rupees = 0.0
        remaining = whole_units
        for min_units, max_units, rate in tariff.slabs:
            slab_size = (max_units - min_units + 1) if max_units is not None else remaining
            units_in_slab = max(0, min(remaining, slab_size))
            amount_rupees += units_in_slab * rate
            remaining -= units_in_slab
            if remaining <= 0:
                break

    amount_rupees += tariff.fixed_rate_per_kw * sanctioned_load_kw
    return round(amount_rupees * 100)
