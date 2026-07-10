# ML

## Status: not yet built (Phase 2)

Nothing in this document is implemented yet. `/ml`, `/data`, and `/backend/models/`
exist as empty scaffolding directories as of Phase 1. This document records the design
so Phase 2 has a concrete target and reviewers can see the plan before the code lands.
When Phase 2 starts, this file should be rewritten with what was actually built,
including real evaluation numbers — not left as this aspirational draft.

## Non-negotiable constraint

**No external AI API calls at inference time, ever.** All four models below are trained
offline with scikit-learn / XGBoost / statsmodels, saved to `backend/models/` as
versioned artifacts, and loaded once at FastAPI startup. Prediction happens in-process.
External APIs (e.g. a weather API for training features, or an LLM used only to help
generate synthetic training scenarios) are permitted during offline training and data
prep, never on the request path.

## Dataset plan

Target: 10,000+ synthetic + real-hybrid Indian household-months, spanning:

- **Climate zones:** IMD's climate zone classification (hot-dry, warm-humid,
  composite, temperate, cold), which drives cooling/heating load assumptions.
- **Family sizes and dwelling types:** 1–2 person to joint-family households, 1BHK to
  independent houses.
- **Appliance mixes:** built from published appliance wattage tables and BEE
  (Bureau of Energy Efficiency) star-rating energy consumption data, so a "5-star
  1.5-ton AC, 8 years old" and a "3-star 1-ton AC, 2 years old" have different,
  traceable simulated draw.
- **Seasons:** monthly seasonality (summer AC load spike, monsoon humidity, winter
  geyser load).
- **Tariff structures:** slab-based billing and Time-of-Day (ToD) rates for TNEB,
  BESCOM, Adani, TATA Power, and MSEDCL, with a generic slab fallback for other DISCOMs.

Every data source will be cited in `/ml/DATA.md` with an explicit note on what's
synthetic vs. grounded in real published tables, and why — the CO₂ math and appliance
disaggregation both depend on those assumptions being inspectable, not hidden.

## Model 1 — Bill Forecaster

- **Approach:** SARIMAX (statsmodels) or XGBoost with lag features (previous 1–3 months),
  seasonal indicators, and weather-proxy features (climate zone × month).
- **Target:** next-month total bill amount, with a confidence interval derived from
  residual variance on the hold-out set.
- **Acceptance threshold:** MAPE ≤ 12% on hold-out. CI fails the build if a retrained
  model regresses past this.

## Model 2 — Anomaly Detector

- **Approach:** Isolation Forest, or a seasonal-residual + robust z-score baseline
  (whichever clears the precision bar with less complexity — decided empirically once
  the dataset exists, not pre-committed).
- **Target:** flag a billing month as anomalous, with a plain-language explanation
  (e.g. "this month is 40% above your typical usage for this season").
- **Acceptance threshold:** precision ≥ 0.8 on synthetic anomalies injected into the
  hold-out set (known ground truth by construction).

## Model 3 — Appliance Disaggregator

- **Approach:** regression estimating % consumption share per appliance category
  (fridge, AC, geyser, lighting, fans, washing machine, other, standby) from
  `(total units, household profile, season, tariff)`. This is **not** hardware NILM —
  it's a software-only estimate trained on the synthetic set where ground truth is
  known exactly because the synthetic generator assigns real wattage draws per
  appliance and sums them.
- **Acceptance threshold:** MAE per category, tracked and gated in CI once implemented.

## Model 4 — Recommendation Ranker

- **Approach:** hybrid of a rule base (physics-grounded actions — geyser timing shifts,
  ToD load shifting, star-rating upgrade ROI, standby load elimination) and learned
  prioritization using the disaggregated context from Model 3.
- **Every output carries:**
  - `estimated_savings_paise_per_month` (integer, paise)
  - `estimated_co2_kg_per_year` (float, kg)
  - `calculation_method` (text — the literal formula and constants used, so a user or
    reviewer can verify the number by hand)

## Training pipeline

`/ml/train_all.py` will reproduce every artifact from scratch: dataset generation →
feature engineering → train/validate/save for all four models → write version-tagged
artifacts to `/backend/models/`. CI will run a validation script (`ml/validate_models.py`,
referenced but disabled in `.github/workflows/ci.yml` until Phase 2 lands) that asserts
the MAPE/precision/MAE thresholds above and fails the build on regression.

## Sustainability constant

The CO₂ figure on every recommendation depends on a grid emission factor (kg CO₂ per
kWh). Phase 2 will source this from the Central Electricity Authority's CO2 Baseline
Database, applied per-DISCOM where available and a national average fallback otherwise,
with the exact figure and vintage cited in `/ml/DATA.md`.
