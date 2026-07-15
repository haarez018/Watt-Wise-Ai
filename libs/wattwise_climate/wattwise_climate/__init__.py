"""WattWise AI's city-to-climate-zone and zone-to-monthly-temperature lookup.

Used two ways:

- **Bulk** (`load_climate_reference_tables`): `ml/data/generate_synthetic.py`
  needs the full city and zone-temperature tables to randomly assign
  synthetic households to cities and simulate their monthly climate.
- **Single-lookup** (`city_to_zone`, `zone_month_avg_temp`): the backend's
  forecast endpoint needs one household's zone and one target month's
  average temperature per request — real households only store `city`
  (free text, ~50 cities cited in `ml/DATA.md`), never `zone` directly, so
  this lookup is the only way to get the `zone`/`target_month_temp_c`
  features Model 1 was trained on without asking a user to name their own
  climate zone.

Reference tables (city→zone assignment and zone monthly average
temperatures, cited in `ml/DATA.md`) live in `reference/`, a sibling of this
package directory — resolved relative to `__file__`, so this only works with
an editable install (`pip install -e`), which is how both `ml/` and
`backend/` install this package in this monorepo. Same pattern as the
`wattwise_tariffs` package.
"""

from functools import lru_cache
from pathlib import Path

import pandas as pd

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"

MONTH_COLUMNS = [
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
]

# Used when a household's `city` isn't in the reference table (free-text
# field, no onboarding validation against the ~50-city list yet — see
# ml/DATA.md) or is missing entirely. "composite" is the ECBC climate zone
# with the least extreme seasonal swing among the 6 zones, so it's the least
# wrong default when the real city is unknown — an assumption, not a
# citation.
DEFAULT_ZONE = "composite"


def load_climate_reference_tables() -> dict[str, pd.DataFrame]:
    return {
        "cities": pd.read_csv(REFERENCE_DIR / "city_climate_zones.csv"),
        "zone_temps": pd.read_csv(REFERENCE_DIR / "zone_monthly_temps.csv"),
    }


@lru_cache(maxsize=1)
def _cities_table() -> pd.DataFrame:
    return pd.read_csv(REFERENCE_DIR / "city_climate_zones.csv")


@lru_cache(maxsize=1)
def _zone_temps_table() -> pd.DataFrame:
    return pd.read_csv(REFERENCE_DIR / "zone_monthly_temps.csv")


def city_to_zone(city: str | None) -> str:
    """Case-insensitive, whitespace-trimmed match against the reference
    table's `city` column. Falls back to `DEFAULT_ZONE` for an unmatched or
    missing city rather than raising — a forecast should still be possible
    for a household whose city isn't in the ~50-city reference list."""
    if not city:
        return DEFAULT_ZONE
    normalized = city.strip().casefold()
    cities = _cities_table()
    matches = cities[cities["city"].str.strip().str.casefold() == normalized]
    if matches.empty:
        return DEFAULT_ZONE
    return str(matches.iloc[0]["zone"])


def zone_month_avg_temp(zone: str, month_index: int) -> float:
    """`month_index` is 1-12 (January=1). Falls back to `DEFAULT_ZONE`'s
    temperature for that month if `zone` itself isn't recognized (defensive
    — `city_to_zone` should never return an unrecognized zone, but this
    keeps the function safe to call independently)."""
    month_column = MONTH_COLUMNS[month_index - 1]
    zone_temps = _zone_temps_table()
    matches = zone_temps[zone_temps["zone"] == zone]
    if matches.empty:
        matches = zone_temps[zone_temps["zone"] == DEFAULT_ZONE]
    return float(matches.iloc[0][month_column])
