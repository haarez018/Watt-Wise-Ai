# Architecture

## Status

This document describes the system as built through **Phase 1 (production scaffolding)**
and **Phase 2 (the ML pipeline, plus one endpoint — `GET /households/{id}/forecast` —
wired end-to-end to prove the serving pattern before Phase 3's full API surface lands)**.
Sections describing the full API surface and dashboard UI describe the target design for
Phases 3–4 and are marked accordingly. See the "ML pipeline (Phase 2)" section below for
what's actually built, and `ml/MODELS.md` for full per-model detail and honest metrics.

## Modules

```
/frontend    Next.js 14 (App Router), TypeScript strict, Tailwind, shadcn/ui, NextAuth
/backend     FastAPI, SQLAlchemy 2.0, Alembic, Pydantic v2
/ml          Training pipeline for the four self-trained models (Phase 2)
/libs        Shared Python packages used by both /ml and /backend — see "ML pipeline" below
/data        Raw and processed datasets used to train the models (git-ignored, Phase 2)
/infra       Deployment configuration for Vercel / Render / Fly.io
/docs        This directory
/.github     CI workflows
```

## Data flow

1. A user signs up (email+password via the backend, or Google via NextAuth) and gets a
   household record.
2. The user enters a one-time appliance inventory and 6–12 months of past bills through
   the onboarding wizard (Phase 4).
3. On each dashboard load, the frontend calls the backend's forecast, breakdown, anomaly,
   and recommendation endpoints. **Only the forecast endpoint exists today** — the rest
   land in Phase 3, following the same `ModelRegistry` pattern established in Phase 2 (see
   below). The backend loads pre-trained model artifacts from `backend/models/` once, at
   process startup, and predicts in-process — **no external AI API calls happen on the
   request path, ever.**
4. Recommendations acted upon are tracked in `savings_events`, which powers the
   cumulative impact scoreboard.

## ML pipeline (Phase 2)

Four models, trained offline in `/ml`, served in-process by the backend. Full design
rationale, features, metrics, and — critically — the honest limitations of validating
against synthetic data live in `ml/MODELS.md`; this section covers only the serving
architecture.

**The four models**, each with its own acceptance threshold gating CI (see
`ml/evaluation/validate.py`):

| Model | Approach | Depends on |
|---|---|---|
| Bill Forecaster | XGBoost, point + 80% prediction interval | — |
| Anomaly Detector | Seasonal residual + robust z-score | Forecaster |
| Appliance Disaggregator | Per-category XGBoost regression | — |
| Recommendation Ranker | Rule base + learned XGBoost prioritizer | All three above |

**Serialization contract**: every artifact is a single plain-JSON file
(`backend/models/<name>_v1.json`) — each model's own native export (XGBoost's
`Booster.save_raw(raw_format="json")`) as a string value, plus metadata as plain
dicts/lists. **No pickled Python object, ever.** The backend never imports from `ml`
to load a model — `backend/app/core/model_registry.py`'s loader functions are a
from-scratch reimplementation of the same plain-JSON format, and
`backend/tests/test_model_loading.py` is the test that would fail first if that
contract were ever broken.

**`ModelRegistry`** (`backend/app/core/model_registry.py`) loads all four artifacts
once, at process/module import time (not deferred into the ASGI lifespan event, which
isn't reliably triggered by every ASGI client), and cross-checks each artifact's
declared `model_version` against `backend/models/models_manifest.json`
(`ml/evaluation/generate_manifest.py` generates this after training). A load or
version-mismatch failure fails `/readyz` (503) — **never** `/healthz`: the process is
still up and shouldn't be restarted, it just shouldn't be routed traffic. See
`docs/RUNBOOK.md`'s "Retraining and rolling out a new model version" section for the
operational procedure.

**Shared libraries** (`/libs`): logic and reference data that both the training
pipeline and the serving endpoint must use identically, so they can never silently
drift apart:

- `libs/wattwise_tariffs` — the Indian electricity tariff calculator (slab/ToD billing
  math). Used to compute training-data ground-truth bills, to price every Model 4
  recommendation candidate, and to compute the forecast endpoint's
  `predicted_amount_paise` from `predicted_units_wh`.
- `libs/wattwise_climate` — city→climate-zone and zone→monthly-temperature lookups.
  Used to build training households and, at serving time, to derive a real household's
  `zone`/`target_month_temp_c` features from its stored `city` (there is no dedicated
  `zone` column — see `docs/RUNBOOK.md`'s "known operational quirks").

Both are installed editable (`pip install -e ../libs/<name>`) into both `/ml` and
`/backend`'s virtualenvs — one implementation, two consumers.

## Auth

- **Credentials (email + password):** the frontend calls `POST /auth/signup` or
  `POST /auth/login` directly from a Next.js server context, receives a backend-issued
  JWT access/refresh pair, and NextAuth stores it in an encrypted session cookie (JWT
  session strategy). Every subsequent API call attaches the backend access token as a
  Bearer token.
- **Google OAuth:** NextAuth handles the OAuth handshake. Once Google confirms the
  user's identity, NextAuth's server-side `jwt` callback calls
  `POST /auth/oauth/exchange` on the backend with a shared `INTERNAL_API_SECRET` header.
  The backend trusts this call (it never talks to Google directly) and mints its own
  JWT pair for a first-party or newly created user record. This keeps a single source of
  truth for authorization (the backend's JWT) regardless of how the user authenticated.
- Access tokens are short-lived (30 min default); the NextAuth `jwt` callback refreshes
  them transparently via `POST /auth/refresh` using the longer-lived refresh token.
- Every backend endpoint that touches household data resolves `get_current_user` from
  the bearer token and scopes all queries to that user's own households — there is no
  endpoint that accepts a household ID without checking ownership (enforced starting
  Phase 3 as household-scoped endpoints are added).

## Data model

Six core tables, all with `id (uuid)`, `created_at`, `updated_at`, `deleted_at`
(soft-delete) columns:

| Table | Purpose |
|---|---|
| `users` | Account identity; either password-based or OAuth-linked |
| `households` | One user can own multiple households; carries DISCOM, location, dwelling profile |
| `bills` | One row per billing period; `amount_paise` (integer) and `units_consumed_wh` (integer) — no floats for money or energy |
| `appliances` | The one-time appliance inventory per household |
| `recommendations` | Ranked, ₹/CO₂-quantified actions generated for a household |
| `savings_events` | Realized savings attributed to a recommendation, for longitudinal tracking |

Money is always stored in **paise** (integer) and energy in **Wh** (integer) to avoid
float drift — see `backend/app/models/`. All timestamps are UTC.

## Sustainability math (CO₂ accounting)

India's grid emission factor is dominated by coal (~75% of generation mix per the
Central Electricity Authority's most recent CO2 Baseline Database). Phase 2 will encode
a per-DISCOM (or national fallback) grid emission factor in kg CO₂ per kWh, sourced from
the CEA baseline database, and every recommendation's `estimated_co2_kg_per_year` field
is computed as:

```
kWh_saved_per_year × grid_emission_factor_kg_per_kwh
```

The emission factor and its source will be documented in `/ml/DATA.md` once Phase 2
lands, and the `calculation_method` field on every `Recommendation` row stores the exact
formula and factor used so the number is always inspectable, not a black box.

## Deployment topology

- **Frontend:** Vercel. `frontend/next.config.mjs` builds a standalone output; the
  production `Dockerfile` is available for non-Vercel hosts.
- **Backend:** Render or Fly.io, running the `backend/Dockerfile` image behind
  `uvicorn`. `alembic upgrade head` runs as a release step before the app boots.
- **Database:** managed Postgres 15 from day one (Render Postgres or Fly Postgres) — no
  SQLite in any deployed environment. Local dev uses `postgres:15-alpine` via Docker
  Compose.
- **CI:** GitHub Actions (`.github/workflows/ci.yml`) runs backend lint/typecheck/test
  against a real Postgres service container, and frontend lint/typecheck/test/build, on
  every push and PR to `main`.

## Observability

- Backend: `structlog` JSON logging with a request ID bound to every log line via
  `RequestContextMiddleware` (`backend/app/core/middleware.py`), PII redaction for
  email/password/full_name/phone/address fields, Sentry SDK initialized whenever
  `SENTRY_DSN` is set, and a Prometheus-format `/metrics` endpoint via
  `prometheus-fastapi-instrumentator`.
- Health checks: `/healthz` (liveness, no dependency checks) and `/readyz` (readiness,
  confirms the database connection is live) — used by Render/Fly health checks.
