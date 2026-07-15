# Dataset construction

## Status

Step 1 of Phase 2. This document describes the synthetic dataset generator in
`ml/data/generate_synthetic.py` and the reference tables it reads from
`ml/data/reference/`. Nothing here has been used to train a model yet — that's
steps 2-5.

## What this is, in one paragraph

10,000 synthetic Indian households, each with a fixed appliance/tariff profile,
simulated across 12 consecutive months. Each household-month gets a per-appliance
energy breakdown (fridge, AC, geyser, lighting, fans, washing machine, TV, other),
a total units-consumed figure, a computed bill amount under one of three tariff
structures, and — for ~4% of rows — an injected anomaly. This is **not** a
physics simulation. It's a rules-based generator whose every rule and constant
is either cited from a real source or explicitly labeled as a modeling
assumption below.

## Scope (deliberately fixed, per the Phase 2 brief)

- **10,000 households × 12 months = 120,000 household-months.** Exactly this,
  not more.
- **6 climate zones.** hot_dry, warm_humid, composite, temperate, cold,
  hot_humid_coastal.
- **8 appliance categories.** fridge, ac, geyser, lighting, fans,
  washing_machine, television_entertainment, other_including_standby.
- **3 tariff structures.** tneb (telescopic slab), bescom (near-flat with a
  free block), tod_generic (illustrative Time-of-Day).

## Climate zones

### We use 6 zones, built on top of ECBC's 5 — not a claim that ECBC defines 6

The canonical Indian standard (ECBC — India's Energy Conservation Building
Code) defines **5** zones: Hot-Dry, Warm-Humid, Composite, Temperate, Cold.
That's the classification you'll find cited everywhere (BEE, ECBC 2020, GBPN),
and it's the authority we start from.

**We split ECBC's Warm-Humid zone into two for modeling reasons, not
climatological ones**: `warm_humid` (a moderate coastal-west profile,
Mumbai-like) and `hot_humid_coastal` (a hotter, more consistently humid
east-coast/south-coastal profile, Chennai/Visakhapatnam-like). The reason is
appliance load, not weather classification: Chennai and Mumbai sit in the same
ECBC zone but have meaningfully different AC-driven consumption profiles —
Chennai runs hotter and more consistently across the year, which we expect
(and will check, once Model 3 is trained) to show up as a different
appliance-mix signature than Mumbai's milder, more monsoon-buffered profile.
Keeping them in one zone would blur a distinction the disaggregation model
needs to learn; splitting them gives Model 3 the ability to learn separate
AC-share profiles for each.

That said, this split is **our own adaptation of ECBC for this dataset's
training needs, not a sixth zone ECBC itself recognizes**. It's loosely
consistent with a district-level climate reclassification study found during
research that independently splits the humid zone into "Composite-Humid" and
"Hot-Humid" sub-categories via clustering on real station data — similar
shape of split, different names, chosen to match this product's vocabulary —
but that correspondence is supporting evidence for the idea, not the authority
the split rests on. The authority it rests on is the modeling argument above.
If asked "which classification is this," the honest answer is: "ECBC's 5
zones, with Warm-Humid split in two because it improves what Model 3 can
learn about appliance load" — not "a 6-zone version of ECBC."

### Monthly average temperatures

`libs/wattwise_climate/reference/zone_monthly_temps.csv` (moved out of
`ml/data/reference/` alongside the tariff tables — see the "Tariff
structures" section above for why: this table is now shared between this
generator and the backend's forecast endpoint via the `wattwise_climate`
package) gives one representative monthly mean temperature (°C) per zone,
for all 12 months. These are **approximate
values consistent with the general shape of IMD long-period (1981-2010)
climatological normals**, synthesized from published climate summaries for
representative cities in each zone — not a verbatim export from IMD's
Climate Data Services Platform (which requires direct portal access we didn't
have during this research pass). If a real IMD normals CSV becomes available,
swap it in here; the generator only depends on the column shape (zone + 12
month columns), not on these specific values being exact.

### City → zone mapping

`libs/wattwise_climate/reference/city_climate_zones.csv` (also moved, same
reason as above) maps 50 Indian cities to one of the
6 zones, based on well-established geography (state/region climate character),
consistent with the ECBC zone descriptions cited above. Each household is
randomly assigned to one of these 50 cities; its climate comes from the city's
zone, not the city individually — i.e., all cities in a zone share the same
monthly temperature curve. This is a simplification (real cities within a zone
do vary), acceptable at our stated scope.

## Appliance energy figures

`ml/data/reference/appliance_wattages.csv` — every row has a `basis` column
citing either a published figure (BEE star-rating data, typical Indian
appliance wattage guides) or explicitly stating "Assumption" when no citation
was found during this research pass. Read that column, not this summary, for
the authoritative per-row sourcing. Highlights:

- **Fridge**: BEE-published annual consumption ~250 kWh/yr (3-star) and
  ~190 kWh/yr (5-star) for a standard single-door refrigerator; 1/2/4-star
  linearly interpolated/extrapolated from those two anchors.
- **AC (1.5-ton)**: published typical hourly draw ~1.5 kWh/hr (3-star) and
  ~1.2 kWh/hr (5-star); this is a **per-hour-of-active-use** figure, not a
  daily total — the generator multiplies it by a temperature-dependent
  daily run-hours curve (see below).
- **Geyser (15L)**: published typical rated power ~2000W, i.e. ~2 kWh per hour
  of active heating; also a **per-hour-of-active-use** figure, temperature-
  dependent (more use in cold months) via the same mechanism as AC.
- **Ceiling fan**: published typical draw ~70W (standard) / ~45W (5-star);
  this one IS a per-day figure, at an assumed 8 hrs/day baseline, further
  scaled by a temperature-dependent usage multiplier.
- **LED bulb, washing machine, television**: lighting is cited (BEE/EESL
  typical LED wattage); washing machine and television are labeled
  **assumption** — no dedicated citation was retrieved for either during this
  research pass. Revisit before treating Model 3's per-category breakdown as
  authoritative for those two categories specifically.
- **Other/standby**: not a per-unit wattage at all — modeled as 5-8% of a
  household's total metered load, reflecting general energy-audit literature
  on miscellaneous/standby draw (phone chargers, routers, standby power) as a
  proportion of total consumption, not a specific cited study.

### Why AC and geyser scale with temperature but the rest don't

Fridge runs continuously regardless of season (near-constant daily draw).
Fans get *somewhat* more use in heat (a modest multiplier). AC and geyser are
the two categories where real Indian household seasonality is dominated by a
single driver — cooling load in summer, water-heating load in winter — so
those two get an explicit temperature-driven daily-run-hours function
(`ac_daily_run_hours`, `geyser_daily_run_hours` in `generate_synthetic.py` —
public, not `_`-prefixed, since Model 4's rule base reuses them directly; see
`ml/MODELS.md`'s Model 4 section), capped at physically sensible bounds (AC
maxes at 8 hrs/day; geyser never drops below a 0.15 hr/day floor even in peak
summer, since some hot water use persists year-round).

## Tariff structures

`libs/wattwise_tariffs/reference/tariff_slabs.csv`, `tariff_fixed_charges.csv`,
`tariff_tod.csv`. These moved out of `ml/data/reference/` into the shared
`wattwise_tariffs` package during Model 4 (Recommendation Ranker) — the
tariff calculator itself (`TariffModel`, `build_tariff_lookup`,
`compute_bill_amount_paise`) is now the single implementation used by this
generator, by Model 4's rule base, and by the backend's serving code (see
`libs/wattwise_tariffs/wattwise_tariffs/__init__.py`'s docstring and
`docs/RUNBOOK.md`'s "known operational quirks" for why duplicating it would
be a silent time bomb). The citations and modeling decisions below are
unchanged by that move — only the file location is different.

- **TNEB (Tamil Nadu)**: real TNEB/TANGEDCO billing is bi-monthly with a
  telescopic slab structure; current (July 2024 revision) rates cited during
  research were ₹4.70/unit (101-400 slab) up to ₹11.55/unit (>1000 units,
  bi-monthly), with 100-200 free units depending on the household's
  consumption band. **We convert this to a monthly-equivalent approximation**
  (halving the bi-monthly unit thresholds) so it's consistent with this
  dataset's monthly granularity — real TNEB bills are bi-monthly, ours are
  monthly-equivalent. TNEB's real bill also includes a small fixed/service
  charge not captured in the sources found during this research pass; we set
  it to ₹0 here rather than fabricate a figure — see the "not modeled" section.
- **BESCOM (Karnataka)**: KERC abolished progressive slabs in 2025; current
  structure cited is a **near-flat rate** (~₹6.82/unit including surcharges)
  plus a fixed charge (~₹145/kW of sanctioned load), with 200 free units/month
  for eligible households under the Gruha Jyothi scheme. We model exactly
  this shape: 0-200 free, 201+ flat at ₹6.82/unit, plus the ₹145/kW fixed
  charge. **Important training-data consequence**: a household using under
  200 units/month on this tariff gets a bill that's *entirely* the fixed
  charge — flat every month regardless of usage variation. This is real
  BESCOM behavior for lower-consumption households, not a generator defect;
  it does mean Model 1 should treat `units_consumed_wh` as the primary,
  always-informative target and derive `amount_paise` from it via the tariff
  function, rather than forecasting amount directly as an independent target.
- **ToD (generic, illustrative)**: real residential Time-of-Day tariffs in
  India are only just rolling out nationally (Ministry of Power amendment,
  effective for most residential consumers from April 2025), so a clean,
  widely-published per-DISCOM residential ToD slab table wasn't available
  during this research pass. We construct an **illustrative generic ToD**
  instead: a base rate (₹7.50/unit — an assumption representing a typical
  national-average domestic per-unit rate, not a specific DISCOM's published
  figure) with three time blocks (peak 18:00-22:00 at 1.3×, solar/off-peak
  09:00-17:00 at 0.85×, normal the rest at 1.0×), each with an assumed share
  of a household's total load. Since this dataset is monthly-granularity
  (no hourly load curve), we don't actually split a household's consumption
  by hour — we compute one **blended rate** from the block multipliers and
  assumed load shares, and apply that blended rate to the household's total
  monthly units. This is a real simplification: true ToD billing depends on
  *when* a household actually consumes, which we aren't modeling at that
  resolution. Label this construct as illustrative, not as a specific real
  DISCOM's ToD tariff.

## Anomaly injection

~4% of household-months get an injected anomaly, uniformly across 5 reasons
(`unusual_spike`, `unusual_drop`, `night_load_surge`, `sustained_high`,
`seasonal_deviation`), applied as a multiplier (1.2×-1.9× for the "high" family
of reasons, 0.4×-0.65× for `unusual_drop`) on that month's total energy
*after* the normal appliance simulation. The per-category breakdown (used as
Model 3's ground truth) reflects the **pre-anomaly** normal appliance draw —
i.e., during an anomalous month, the disaggregation ground truth still shows
"what appliances normally draw," while the total bill reflects the anomalous
spike. This is intentional: it models "something happened that isn't
explained by the household's normal appliance behavior" rather than
attributing the anomaly to a specific appliance category, which would be a
much stronger (and unjustified) claim.

## Household profile generation

Family size (1-8, weighted toward 3-4), tariff assignment (40% TNEB / 40%
BESCOM / 20% ToD), appliance ownership (AC probability scales with family
size as a rough proxy for household size/affluence; geyser ~55%, washing
machine ~50%, TV ~92%), and star ratings (weighted toward 3-star, the most
common BEE rating in the Indian market) are all **assumptions**, not sourced
from a specific ownership survey. If a real appliance-ownership survey
(e.g. NSSO or a similar government household survey) becomes available and
you want to ground these probabilities in it, ask before pulling it in per
the working agreement — external data has licensing implications.

## What this dataset does NOT model

Being explicit about this, as required:

- **No humidity modeling.** Only a single monthly mean temperature per zone.
  Real cooling/heating load depends on humidity too (a "warm-humid" day feels
  different from a "hot-dry" day at the same temperature) — we approximate
  this only through the zone's temperature *curve shape*, not a separate
  humidity variable.
- **No occupancy schedules.** A household's appliance usage is a fixed
  daily-hours assumption per category, not modeled as varying by whether
  anyone is home, weekday vs weekend, or time of day (beyond the ToD tariff's
  load-share assumption, which is a billing construct, not a usage model).
- **No appliance duty cycles or thermostat behavior.** AC/geyser "run-hours"
  are a smooth function of monthly average temperature, not a simulation of
  actual compressor cycling, thermostat setpoints, or door-opening behavior.
- **No per-city temperature variation within a zone.** All cities sharing a
  zone get an identical monthly temperature curve.
- **No real historical bill data.** Every household-month in this dataset is
  synthetic. If a real (anonymized) Indian household consumption dataset
  becomes available, we'd want to validate model performance against it
  before trusting synthetic-only metrics for a production launch — but per
  the working agreement, pulling in external data needs a decision first.
- **No multi-year trends** (tariff inflation, appliance efficiency
  improvements over time, degradation). Each household's 12 months are
  treated as one representative year.
- **No intra-month or hourly resolution**, which means anomaly reasons
  `unusual_spike`, `night_load_surge`, and `sustained_high` are not
  distinguishable at this dataset's monthly-aggregate resolution (they share
  the same injected multiplier range in `maybe_inject_anomaly` — see
  `ml/MODELS.md`'s Model 2 section for how this is handled). Real anomalies
  of these three kinds *do* have distinguishing signals in principle — a
  night-load surge is a time-of-day pattern, a sustained-high is a duration
  pattern, a spike is a magnitude pattern — but a monthly-total-only row
  can't carry any of that information no matter how good the model is. This
  is a gap in what this dataset resolves, not a Model 2 weakness. Adding
  intra-month or hourly features is a real future enhancement if the product
  ever needs to distinguish these three reasons for real, not something to
  attempt within the current monthly-granularity design.

## Reproducibility

`python -m ml.data.generate_synthetic` (from the repo root, with `ml/.venv`
activated) regenerates the dataset from scratch, seeded (`--seed`, default 42)
for determinism. Runtime: ~4 seconds for the full 10,000-household run on a
laptop (after an initial pandas-filtering-in-a-hot-loop performance bug was
found and fixed during this step — see the generator's `build_appliance_lookup`
/ `build_tariff_lookup` functions, which precompute plain-dict lookups instead
of re-filtering reference DataFrames 120,000+ times). Output is gitignored
(`ml/data/processed/`) — it's a build artifact, not a tracked file.
