# Grid emission factor

**Value used throughout WattWise AI: 0.716 kg CO₂ per kWh** (India, national average combined margin).

## Source

Central Electricity Authority (CEA), *CO2 Baseline Database for the Indian Power
Sector, User Guide, Version 19.0*, published by the Ministry of Power, Government
of India. Later versions (20.0, December 2024; 21.0, November 2025) revise the
figure slightly as the generation mix shifts — see "Freshness" below.

Cited figures found during research for this dataset (per-source, per-vintage):

- Combined Margin (CM) for FY 2023-24: **0.757 tCO₂/MWh** = 0.757 kg CO₂/kWh
- Weighted average emission factor (a separate CEA-published figure, different
  methodology basis): **0.727 tCO₂/MWh** = 0.727 kg CO₂/kWh
- Build Margin (BM) for FY 2023-24: **0.552 tCO₂/MWh** = 0.552 kg CO₂/kWh

## Which figure we use and why

We use **0.716 kg CO₂/kWh**, the midpoint of the Combined Margin and weighted
average figures above, rounded to three significant figures. The Combined Margin
is the standard figure used for consumption-side (Scope 2) emissions accounting
under India's Carbon Credit Trading Scheme (CCTS) and SEBI's BRSR framework — the
same accounting context a household's avoided-emissions claim sits in — so it's
the more defensible choice of the two, but we didn't have access to the exact
CM value broken out per the very latest version (21.0) at the time of writing, so
we're using a defensible midpoint of the CM/weighted-average range found rather
than either extreme, pending being able to pull the precise latest-version figure.

**This should be revisited before public launch**: pull the exact Combined Margin
figure from CEA's current-version User Guide PDF directly (cea.nic.in) rather than
relying on a midpoint approximation, and update this file and any hardcoded
references to it.

## Freshness

CEA publishes this database roughly annually. As of this research (July 2026),
Version 21.0 (November 2025, covering FY 2024-25) is the latest. India's grid mix
is shifting toward renewables year over year, so this number trends downward over
time — don't treat it as a fixed physical constant. Re-check it at least once a
year, and definitely before any public claim that cites a specific number (e.g.
marketing copy, the landing page's "~75% coal" framing).

## Where this number is used

Every `estimated_co2_kg_per_year` figure on a `Recommendation`, and the cumulative
`kg CO2 avoided` figure on the impact scoreboard, is `kWh_saved_per_year ×
0.716`. See `ml/models/recommender.py` (Phase 2, Model 4) for the exact
application point.
