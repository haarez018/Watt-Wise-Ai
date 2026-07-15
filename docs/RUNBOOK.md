# Runbook

## Local development

```bash
git clone <repo>
cd "Sustainability HACK"
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env.local
docker compose up
```

Frontend at `http://localhost:3000`, backend at `http://localhost:8000`
(`/docs` for the interactive API). The `backend` service runs
`alembic upgrade head` automatically before starting `uvicorn`.

To run backend or frontend outside Docker (faster iteration):

```bash
# backend
cd backend
python -m venv .venv && .venv/Scripts/activate   # Windows; source .venv/bin/activate on Unix
pip install -r requirements-dev.txt
alembic upgrade head
uvicorn app.main:app --reload

# frontend
cd frontend
npm install
npm run dev
```

## Deploying

### Backend (Render or Fly.io)

1. **Build context is the repo root, not `backend/`** — set the platform's root/build
   directory to `.` (repo root) with Dockerfile path `backend/Dockerfile`, not root
   directory `backend`. This changed in Phase 2: `backend/requirements.txt` now
   installs `libs/wattwise_tariffs` and `libs/wattwise_climate` as editable packages
   (`-e ../libs/...`), which only resolves if the build context includes `libs/`
   alongside `backend/` — see `backend/Dockerfile`'s top comment and
   `docker-compose.yml`'s `context: .` for the working local-dev equivalent.
   **Verified without Docker itself**, since Docker Desktop's daemon was unusable in
   the environment this was developed in (recurring corrupted socket files under
   `%LOCALAPPDATA%\Docker` — see below): recreated the exact build-context layout
   (`/app` = `backend/`, `/libs` = `libs/`, as siblings) on disk, installed
   `requirements.txt` into a fresh Python 3.11 venv from that layout, and confirmed
   `pip` resolves both `-e ../libs/...` lines correctly and `import app.main` +
   `ModelRegistry` load all four real model artifacts successfully. This proves the
   dependency-resolution logic the Phase 2 change touched is correct, but it is
   **not a substitute for an actual `docker build`** — container-specific concerns
   (the `apt-get install libpq5` step, actually running inside a Linux container)
   remain unverified. Confirm with a real `docker build -f backend/Dockerfile .`
   once Docker Desktop is usable, or treat the first Render/Fly deploy as that
   verification (Render/Fly build the image on their own infrastructure, not this
   machine, so a working deploy is real confirmation).

   **Docker Desktop itself was unusable while this was written**: its backend
   process crashed on startup with `The filename, directory name, or volume label
   syntax is incorrect` for two different socket files under `%LOCALAPPDATA%`
   (`Docker\run\dockerInference`, then `docker-secrets-engine\engine.sock`), both
   with an identical corruption timestamp suggesting a single bad shutdown event.
   Deleting the affected directories (`Remove-Item -Recurse -Force`, requires an
   **elevated** PowerShell — a non-elevated session got "the file cannot be
   accessed by the system" on the same path) fixed it twice, but it's worth
   watching for recurrence — if it comes back a third time, treat it as
   environment/security-software interference (e.g. an EDR product on a managed
   machine) rather than another one-off stale file, and escalate to whoever
   manages the machine rather than continuing to delete-and-retry.
2. Set environment variables from `backend/.env.example` — at minimum
   `DATABASE_URL`, `SYNC_DATABASE_URL`, `JWT_SECRET`, `INTERNAL_API_SECRET`
   (**must match the frontend's value exactly**), `CORS_ORIGINS` (the deployed
   frontend origin), and `SENTRY_DSN` once error tracking is wired to a real project.
3. Run `alembic upgrade head` as a release/pre-deploy step before the new instance
   takes traffic.
4. Point the platform's health check at `/healthz` and readiness/startup check at
   `/readyz`.

### Frontend (Vercel)

1. Root directory: `frontend`.
2. Set `NEXT_PUBLIC_API_URL` (the deployed backend URL), `NEXTAUTH_URL` (the deployed
   frontend URL), `NEXTAUTH_SECRET` (a fresh random value, not the local dev one),
   `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET`, and `INTERNAL_API_SECRET` (must match
   the backend's value exactly).
3. Vercel auto-detects Next.js; no custom build command needed.

### Database

Managed Postgres 15 from day one (Render Postgres, Fly Postgres, or equivalent) — never
SQLite in a deployed environment. After provisioning, run migrations once from a
machine that can reach the database:

```bash
cd backend
SYNC_DATABASE_URL=<production-sync-url> alembic upgrade head
```

## Deploy verification checklist

Run through every item below against the live URL after the **first** deploy (and any
deploy that touches auth, the database schema, or `backend/models/`). This is the
actual verification of the deploy — "the URL loads" is not enough, especially since
local Docker verification wasn't possible for the Phase 2 changes (see above); this
checklist is what confirms the real build worked, not a substitute assumption.

- [ ] **`GET /healthz` returns `200`** with `{"status": "ok"}`. This only proves the
  process is up — expect it to pass even if the database or models are broken.
- [ ] **`GET /readyz` returns `200`** with `{"status": "ready"}`. A `503` here means
  either the database is unreachable or `ModelRegistry` failed to load — check the
  response `detail` field, which names the specific failure
  (`"Models not loaded: ..."` vs a raw DB connection error), then check application
  logs for `model_registry_load_failed` or the DB driver's own error.
- [ ] **Signup + login works against the real managed Postgres**, not a cached/local
  session: `POST /auth/signup` with a fresh email, then `POST /auth/login` with the
  same credentials, confirm both return a `200` with an access/refresh token pair.
- [ ] **Alembic migrations actually ran**: connect to the managed Postgres directly
  (or via the platform's DB console) and confirm the expected tables exist — `users`,
  `households`, `bills`, `appliances`, `recommendations`, `savings_events`,
  `refresh_tokens`, plus Alembic's own `alembic_version` table with a version ID
  matching the latest migration in `backend/alembic/versions/`. A missing table here
  means the release-step migration didn't run, even if `/readyz` looks fine (readiness
  doesn't check schema completeness, only connectivity).
- [ ] **`GET /households/{id}/forecast` returns a valid forecast** for a seeded
  household with 3+ bills: `predicted_units_wh > 0`,
  `prediction_interval_80.low <= predicted_units_wh <= prediction_interval_80.high`,
  `model_version == "forecaster_v1"`. This is the one endpoint that exercises the full
  Phase 2 stack (auth → DB → `ModelRegistry` → `wattwise_tariffs`/`wattwise_climate` →
  response) in production, not just locally.
- [ ] **No PII in INFO-level logs**: trigger a few requests (signup, login, a forecast
  call) and check the platform's log viewer for the household UUID, email address, or
  bill amounts appearing in structured `INFO` log lines outside the request path itself
  (`path=/households/{uuid}/forecast` is expected and already-known — see Phase 2 audit
  Check 10 — anything beyond that, e.g. the email or bill data logged as a separate
  field, is a real regression).
- [ ] **Sentry receives a test error correctly**: trigger a deliberate 500 (e.g. a
  malformed request the validation layer doesn't catch, or temporarily break
  `DATABASE_URL` and hit any endpoint) and confirm the error shows up in the Sentry
  project tied to `SENTRY_DSN`, with the request ID from `RequestContextMiddleware`
  attached so it's correlatable with the platform's own logs.

Only once every box above is checked does the deploy count as verified — not when the
homepage loads.

## Rolling back

- **Backend:** redeploy the previous container image/tag. If the rollback crosses a
  migration boundary, run `alembic downgrade -1` against production **before**
  rolling back the code, so the schema matches what the older code expects. Every
  migration in `backend/alembic/versions/` has a working `downgrade()`.
- **Frontend:** use Vercel's "Instant Rollback" to the previous deployment.

## Retraining and rolling out a new model version

There is no separate model-registry service or out-of-band model storage — the four
`backend/models/*.json` artifacts and `models_manifest.json` are committed to git like
any other file, so **a model version is just a git commit**, and rolling one out or
back is the same procedure as any other backend deploy.

**Retraining:**

```bash
cd ml
source .venv/Scripts/activate   # Windows; source .venv/bin/activate on Unix
python train_all.py             # ~3 minutes — see ml/MODELS.md's "Full pipeline retraining"
python -m evaluation.validate   # sanity check locally before committing — same check CI runs
```

`train_all.py` overwrites all four artifacts and regenerates the manifest together, so
they're always consistent with each other (never hand-edit `models_manifest.json` or
copy in a single artifact from a different training run — `ModelRegistry`'s
version-check at backend startup exists specifically to catch that class of mistake,
but it's better caught locally first). Commit all five files
(`forecaster_v1.json`, `anomaly_v1.json`, `disaggregator_v1.json`, `recommender_v1.json`,
`models_manifest.json`) together in one commit — a partial commit (e.g. new artifacts
without the new manifest) is exactly the inconsistent state the version check is
designed to reject, so it would fail `/readyz` on the next deploy rather than serve
silently-mismatched models.

**Rolling out:** push the commit, deploy the backend normally (per "Deploying" above).
The new process reads the new artifacts from `backend/models/` at import time — no
separate model-deploy step, no feature-flag gating a model version. `/readyz` failing
after deploy (503, `"Models not loaded: ..."`) means either an artifact is missing/
corrupt or the manifest doesn't match — check the deploy actually included the new
`backend/models/*.json` files (a `.dockerignore`/`.gitignore` mistake excluding them
would produce exactly this symptom) before assuming the training run itself was bad.

**Rolling back a bad model version:** revert or redeploy the previous commit/image —
the previous `backend/models/*.json` files come back with it, and `ModelRegistry`
loads whatever's on disk at startup with no awareness of "versions" beyond the
manifest's own version-consistency check. There is deliberately no model-specific
rollback mechanism separate from normal code rollback, because the model artifacts
aren't tracked anywhere the code isn't.

## Common incidents

| Symptom | Likely cause | First checks |
|---|---|---|
| `/readyz` returning errors | Database unreachable | Check Postgres is up; check `DATABASE_URL` / `SYNC_DATABASE_URL` credentials and network access |
| `/readyz` returning 503 with `"Models not loaded: ..."` | A model artifact is missing/corrupt, or `models_manifest.json` doesn't match the artifacts' declared `model_version`s | Confirm all 5 files exist in the deployed `backend/models/`; re-run `python -m evaluation.validate` from `ml/` against the same artifacts to see the specific mismatch |
| Forecast endpoint returns a plausible-looking but wrong `predicted_amount_paise` | `wattwise_tariffs`/`wattwise_climate` not installed as editable, so the backend fell back to a stale copy — check they were never vendored/copied instead of `pip install -e`'d | `pip show wattwise-tariffs` should point at `libs/wattwise_tariffs`, not a site-packages copy |
| Frontend shows "RefreshFailed" session error | Backend JWT refresh failing | Check backend logs for `/auth/refresh` 401s; confirm `JWT_SECRET` hasn't changed (invalidates all outstanding tokens) |
| Google sign-in creates no backend session | `INTERNAL_API_SECRET` mismatch between frontend and backend | Confirm both services have the identical value; check backend logs for `403` on `/auth/oauth/exchange` |
| 429s on `/auth/login` or `/auth/signup` | Rate limiting (10/minute/IP) | Expected under abuse; if a legitimate user is blocked, check `RATE_LIMIT_DEFAULT` and the `slowapi` limits in `app/api/routes/auth.py` |
| CI failing on `mypy` only in the pre-commit/CI environment, not locally | `additional_dependencies` in `.pre-commit-config.yaml` or CI's `requirements-dev.txt` install missing a package mypy needs to resolve types | Compare the failing environment's installed packages against `backend/requirements.txt` |

## Secrets rotation

`JWT_SECRET` rotation invalidates every outstanding access/refresh token (all users are
logged out). `INTERNAL_API_SECRET` rotation must be deployed to both frontend and
backend simultaneously, or Google sign-in will fail with `403` in between.

## Known operational quirks

- **Pre-commit hooks that run tools with `pyproject.toml`-driven config
  (black, ruff, mypy) must be scoped per project root, with an explicit
  `--config` flag.** A single hook instance spanning multiple project roots
  (e.g. `files: ^(backend|ml)/` in one `black` entry) can't unambiguously
  resolve "the" config file, and will silently fall back to the tool's own
  defaults — for black, that's an 88-character line length instead of either
  project's configured 100, which mass-reformats correctly-styled files the
  moment `pre-commit run` executes. Fix: one hook instance per root, each
  with its own `args: ["--config=<root>/pyproject.toml"]` and a `files:`
  regex scoped to just that root — see `.pre-commit-config.yaml`'s
  `ruff (backend)`/`ruff (ml)` and `black (backend)`/`black (ml)` pairs. If
  you add a new project root (e.g. a `/libs` package), add matching per-root
  hook instances rather than extending an existing one's `files:` regex.

- **`mypy` can be blocked locally on Windows by an Application Control
  policy on its own compiled `partially_defined.pyd` file** (mypy 1.14.1's
  mypyc-compiled internals) — fails with `ImportError: DLL load failed
  while importing partially_defined: An Application Control policy has
  blocked this file.` This reproduces even after a clean `pip install
  --force-reinstall` of mypy, so it isn't a corrupted download or anything
  a dependency change fixes — it's the host machine's security policy
  flagging that specific compiled file, observed to affect both a project
  venv and pre-commit's own cache venv simultaneously. Not a code issue and
  not something to chase by rebuilding the local environment or disabling
  the policy. **CI (Linux) is the authoritative mypy check** — this policy
  is Windows-specific and doesn't apply there. If mypy is failing locally on
  Windows, verify correctness via `ruff` (still runs fine — it isn't a
  compiled-extension tool the policy flags) plus the test suite, and trust
  CI's mypy result over the local one.

- **`Household` has no `zone` or `tariff_name` column — the forecast
  endpoint derives both at request time, not at household-creation time.**
  `Household.city` (free text) is resolved to a climate zone via
  `wattwise_climate.city_to_zone`, falling back to `DEFAULT_ZONE`
  ("composite") for an unrecognized or missing city. `Household.discom`
  (6-value `DiscomCode` literal: TNEB/BESCOM/ADANI/TATA_POWER/MSEDCL/OTHER)
  is mapped to Model 1's 3 trained tariff structures via
  `app/services/forecast.py`'s `DISCOM_TO_TARIFF_NAME` dict — only TNEB and
  BESCOM have a dedicated modeled tariff; everything else (including OTHER)
  falls back to `tod_generic`, the same generic-ToD fallback the training
  generator itself uses for "other DISCOMs." Both fallbacks mean a
  forecast is always possible even for a household with incomplete profile
  data, but the further a real household is from "known city, TNEB or
  BESCOM," the less accurate its forecast likely is — worth surfacing in
  the product UI once there's a place to show forecast confidence
  qualitatively, not just as a numeric interval. If Phase 3/4's onboarding
  wizard ever collects zone/tariff directly from the user instead of
  inferring them, prefer the direct value and treat this inference as the
  fallback, not the other way around.

## Open items (deliberately deferred, not forgotten)

- **Phase 1 has not been deployed.** Frontend to Vercel, backend + managed
  Postgres to Render or Fly.io, per the "Deploying" section above — none of
  this has actually been done yet. It was deliberately deferred to focus on
  Phase 2 (the ML pipeline). Do this before demo day, not the day of — the
  standard gotchas (first deploy failing on the migration step if
  `alembic upgrade head` isn't wired as an actual release step, free-tier
  cold-start delay after idle) are exactly the kind of thing that's fine to
  hit a week early and not fine to hit live. **This is next** — see "Deploy
  verification checklist" above for the pass criteria; this is also the
  real verification of the `backend/Dockerfile` build-context change from
  Phase 2 (see the `docker-secrets-engine`/`Docker\run` note above), since
  local Docker verification wasn't possible in this environment.

- **Phase 3 backlog: wire Model 4's confidence gate at serve time.** Model 4
  (`ml/models/recommender.py`) has a complete, tested flip-under-perturbation
  confidence gate (`_apply_confidence_gate`, and `recommend_for_household`
  as the real serving-time entrypoint that gates against the actually-shown
  ranking) — but nothing in `backend/` calls it, because Step 6 only wired
  the forecast endpoint, not a recommendations endpoint (confirmed by-design
  in `docs/PHASE_2_AUDIT.md`'s Check 6, not an oversight). When Phase 3
  builds `GET /households/{id}/recommendations`, it must (1) call
  `recommend_for_household` or a backend-side reimplementation of it
  following the same plain-JSON/no-`ml`-import pattern
  `app/services/forecast.py` already established, so the confidence gate
  actually runs on real requests, and (2) log a
  "recommendations suppressed per household" metric (mirroring
  `low_confidence_recommendations_per_1000_households` in
  `ml/evaluation/reports/recommender_v1_metrics.json`, currently 505.6 —
  meaningfully nonzero on the offline eval set) so the gate's real-world
  behavior is observable in production, not just in that one-time
  evaluation. At that point, also move `GRID_EMISSION_FACTOR_KG_PER_KWH`
  (currently only in `ml/models/recommender.py:75`) to a shared `libs/`
  location following the tariffs/climate pattern, rather than re-declaring
  it in `backend/` (Phase 2 audit, Check 2).
