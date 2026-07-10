# Architecture

## Status

This document describes the system as built through **Phase 1 (production scaffolding)**.
Sections describing the ML pipeline, full API surface, and dashboard UI describe the
target design for Phases 2–4 and are marked accordingly.

## Modules

```
/frontend    Next.js 14 (App Router), TypeScript strict, Tailwind, shadcn/ui, NextAuth
/backend     FastAPI, SQLAlchemy 2.0, Alembic, Pydantic v2
/ml          Training pipeline for the four self-trained models (Phase 2)
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
   and recommendation endpoints (Phase 3). The backend loads pre-trained model artifacts
   from `backend/models/` at process startup and predicts in-process — **no external AI
   API calls happen on the request path, ever.**
4. Recommendations acted upon are tracked in `savings_events`, which powers the
   cumulative impact scoreboard.

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
