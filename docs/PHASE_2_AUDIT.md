# Phase 2 Audit Report
Date: 2026-07-15 (all 4 decisions below ruled on and closed same day — see each
Check's "Resolution"/"Ruling" note)

Scope: read-only verification of the working tree as it stands after Phase 2 (four
models, `libs/wattwise_tariffs`, `libs/wattwise_climate`, `ModelRegistry`, the
`GET /households/{id}/forecast` endpoint, CI validation, and the Step 7 wrap-up). No
code was modified as part of producing the *original* report (Checks 1-10's "Evidence"
sections below); the 4 "NEEDS DECISION" items were ruled on afterward and those fixes
are described inline in each Check's resolution note, not treated as a second silent
audit. Docker Desktop's daemon was not usable in this environment during Phase 2 (see
`docs/RUNBOOK.md`'s "Deploying" section for why) — this affects Check 8 specifically
and is called out there, not glossed over.

## Summary
- Pass count: 6 (Checks 1, 3, 4, 7, 9, 10)
- Needs decision count: 4 — **all 4 ruled on and closed**: Check 2 (dedupe the
  lag/min-bills constant) and Check 5 (manifest content-hash + git provenance) were
  fixed in code; Check 6 (Model 4's confidence gate has no caller) was confirmed
  by-design with a Phase 3 backlog item recorded; Check 8 (Docker build
  verifiability) was confirmed as "defer to the real Render/Fly deploy," not fixed
  locally (see that Check's ruling for why).
- Fail count: 0
- Overall verdict: **NO BLOCKING ISSUES**, now with zero open decisions. Phase 2's core
  claims (no pickling, single shared tariff implementation, ModelRegistry gating
  readiness, forecast endpoint correctness, no zombie CI, real integration test
  coverage) held up under direct evidence from the start; the four genuine gaps found
  have each been closed or explicitly deferred with a recorded reason, not left
  ambiguous for Phase 3 to rediscover.

## Check 1: Serialization contract (Option 2) — no pickled custom classes anywhere
Status: **PASS**

Evidence:
- `grep -rn "pickle\.dump|pickle\.load|joblib\.dump|joblib\.load|dill\." **/*.py` across
  the whole repo: zero matches. Same for `import pickle|import joblib|import dill`:
  zero matches.
- All four model artifacts plus the manifest parse as plain JSON with no custom class
  markers — confirmed by loading each with stdlib `json.load` and printing top-level
  keys: `forecaster_v1.json` → `model_version, feature_columns, categories,
  quantile_low, quantile_high, point_model_json, ...`; `anomaly_v1.json` → plain
  scalars; `disaggregator_v1.json` → `..., fridge_model_json, ac_model_json, ...`;
  `recommender_v1.json` → `..., zone_categories, tariff_categories, ...`;
  `models_manifest.json` → `generated_at, training_run_id, models`.
- `grep -rn "from ml\.|import ml\b|from ml import" backend/**/*.py`: zero matches —
  the backend genuinely never imports from `ml`.
- Load paths: [`backend/app/core/model_registry.py:38-84`](../backend/app/core/model_registry.py)
  (`load_forecaster`, `load_anomaly`, `load_disaggregator`, `load_recommender`) each
  reconstruct from `json.loads()` + `xgb.Booster().load_model(bytearray(...))` —
  primitives only, no custom deserialization.

## Check 2: Shared tariff/calc code — no duplication between train and serve
Status: **NEEDS DECISION** → **RESOLVED** (post-audit fix, same day)

**Resolution:** `backend/app/services/forecast.py` no longer declares an independent
`MIN_BILLS_REQUIRED` literal. `_lag_columns()` extracts and orders the artifact's own
`lag_N_units_wh` feature names from `forecaster.metadata["feature_columns"]`, and
`_min_bills_required()` returns their count — the single source of truth is now the
loaded artifact, not a hand-maintained number that has to be kept in sync with
`ml/features/engineering.py`'s `LAG_MONTHS` by convention. `_build_feature_vector` was
generalized to match (loops over however many lag columns exist, rather than
unpacking exactly 3), so a future retrain with a different lag window doesn't silently
zero-fill missing lag features either — fixing only the threshold check would have left
that half of the bug in place. Covered by `backend/tests/test_forecast_service.py`,
including a regression test that constructs a synthetic 6-lag model to prove the
derivation is genuinely dynamic, not coincidentally correct against today's 3-lag
artifact. The `GRID_EMISSION_FACTOR_KG_PER_KWH` flag below is unchanged — still
correctly deferred to whenever Phase 3 builds a recommendations endpoint.

Evidence (as originally audited, before the fix above):
- Training-time import: [`ml/data/generate_synthetic.py:23-27`](../ml/data/generate_synthetic.py)
  — `from wattwise_tariffs import (build_tariff_lookup, compute_bill_amount_paise,
  load_tariff_reference_tables)`.
- Serving-time import: [`backend/app/services/forecast.py:23-24`](../backend/app/services/forecast.py)
  — `from wattwise_tariffs import TariffModel, build_tariff_lookup,
  compute_bill_amount_paise` / `load_tariff_reference_tables as load_tariffs`.
- Both resolve to the single definition in
  [`libs/wattwise_tariffs/wattwise_tariffs/__init__.py:33-113`](../libs/wattwise_tariffs/wattwise_tariffs/__init__.py)
  — confirmed no second `class TariffModel` / `def build_tariff_lookup` / `def
  compute_bill_amount_paise` exists anywhere else in the repo (`grep -rn` for each
  returns exactly one definition site). **PASS** on the tariff calculator specifically.

**The actual gap:** one real duplicated constant, not imported from a shared source.
[`ml/features/engineering.py:21`](../ml/features/engineering.py) — `LAG_MONTHS = 3`
(Model 1's lag-window size, used to build training examples). Independently,
[`backend/app/services/forecast.py:30`](../backend/app/services/forecast.py) —
`MIN_BILLS_REQUIRED = 3` (the same underlying constraint: Model 1 needs exactly 3
months of history). These are two separate literals that happen to agree today. If
Model 1 is ever retrained with a different lag window, nothing enforces that
`MIN_BILLS_REQUIRED` gets updated to match — the forecast endpoint would either reject
households that actually have enough history, or (worse) build a feature vector with
the wrong number of lag columns silently. Note that `forecaster.metadata["feature_columns"]`
(already loaded by the backend) contains exactly `LAG_MONTHS` many `lag_N_units_wh`
columns, so this could be derived at load time instead of hardcoded — the information
is already present in the artifact the backend loads.

Also worth a one-line flag, not a real risk today: `GRID_EMISSION_FACTOR_KG_PER_KWH =
0.716` is defined once, in
[`ml/models/recommender.py:75`](../ml/models/recommender.py), and used only there —
it's not duplicated in `backend/` because there's no recommendations endpoint yet to
need it. When Phase 3 builds one, this constant should move to a shared location
(`libs/`) at that point, following the same pattern as tariffs/climate, rather than
being re-declared in `backend/`.

**Question for you:** derive the lag-window length from `feature_columns` at request
time (a small, low-risk change to `backend/app/services/forecast.py`), or leave
`MIN_BILLS_REQUIRED` as an independent constant with a comment cross-referencing
`LAG_MONTHS`? I'd lean toward deriving it — it's a few lines and removes a real
(if currently dormant) drift risk — but flagging as a decision since you said audit
only, no fixes.

## Check 3: ModelRegistry — loads at startup, fails readiness on failure
Status: **PASS** (with one nuance worth naming precisely)

Evidence:
- Singleton: [`backend/app/core/model_registry.py:149-150`](../backend/app/core/model_registry.py)
  — `model_registry = ModelRegistry()` followed immediately by
  `model_registry.load(MODELS_DIR)`, both at **module import time**, not inside
  FastAPI's `lifespan` async context manager. `grep -n "model_registry\.load\("
  backend/**/*.py` confirms this is the only call site anywhere.
- `/healthz` ([`app/api/routes/system.py:11-14`](../backend/app/api/routes/system.py)):
  returns `{"status": "ok"}` unconditionally — zero reference to `model_registry`.
- `/readyz` ([`app/api/routes/system.py:17-29`](../backend/app/api/routes/system.py)):
  checks `model_registry.is_ready` after the DB check, raises `503` with
  `f"Models not loaded: {model_registry.load_error}"` if not ready.
- No lazy loading: the only `.load(` call site is the module-level one above; the
  forecast route ([`app/api/routes/households.py:26-30`](../backend/app/api/routes/households.py))
  only *reads* `model_registry.is_ready`/`model_registry.forecaster`, never calls
  `.load()`.

**The nuance:** the audit prompt asks to confirm the registry "loads at FastAPI
startup." Precisely, it loads at **Python module import time**, which for a real
`uvicorn` deployment is functionally the same moment (import happens once per worker
process at boot, before the first request) — but it is not literally inside the
`lifespan` `startup` event you'll see in `app/main.py:29-41`. This was a deliberate
choice, documented in `model_registry.py`'s own module docstring: ASGI test clients
don't reliably trigger the lifespan protocol, so loading at import time keeps
`/readyz` tests deterministic without an explicit trigger. Confirmed correct for this
project's actual deployment model (single import per worker process), but worth
knowing if a future multi-app-instance-per-process pattern is ever introduced.

## Check 4: `/households/{id}/forecast` endpoint correctness
Status: **PASS**

Evidence:
- Ownership: [`backend/app/api/routes/households.py:22-23`](../backend/app/api/routes/households.py)
  — `household: Household = Depends(get_owned_household)`, the exact Phase 1
  dependency, no duplicate check.
- Response shape: [`backend/app/schemas/forecast.py`](../backend/app/schemas/forecast.py)
  — `HouseholdForecast{predicted_units_wh: int, predicted_amount_paise: int,
  prediction_interval_80: PredictionInterval80{low, high}, model_version: str,
  generated_at: datetime}` — matches the spec exactly, field for field.
- `predicted_amount_paise` derivation:
  [`backend/app/services/forecast.py:148-154`](../backend/app/services/forecast.py) —
  computed via `compute_bill_amount_paise(predicted_units_wh / 1000.0,
  _tariff_lookup()[...], sanctioned_load_kw)`, the shared `wattwise_tariffs` function,
  called from `predicted_units_wh` — never stored on the model, never computed inline
  with ad-hoc arithmetic.
- Insufficient history: [`InsufficientHistoryError`](../backend/app/services/forecast.py)
  raised at `forecast.py:137-138` when `len(sorted_bills) < MIN_BILLS_REQUIRED`,
  caught and converted to `HTTPException(400, ...)` at
  [`households.py:39-40`](../backend/app/api/routes/households.py), which the global
  handler in `app/main.py:70-72` converts to RFC 7807 `problem+json`.
- Test coverage — all 4 required paths present in
  [`backend/tests/test_forecast_endpoint.py`](../backend/tests/test_forecast_endpoint.py):
  `test_forecast_happy_path` (line 44), `test_forecast_insufficient_history_is_400`
  (line 71), `test_forecast_cross_user_is_404` (line 91),
  `test_forecast_model_missing_is_503` (line 129) — plus two extra
  (`test_forecast_nonexistent_household_is_404`, `test_forecast_requires_auth`) beyond
  what was asked.

## Check 5: Manifest consistency
Status: **NEEDS DECISION** → **RESOLVED** (post-audit fix, same day)

**Resolution:** both gaps identified below are fixed. `ml/evaluation/generate_manifest.py`
now writes a SHA-256 of each artifact file's actual bytes (`sha256_of`) and
`trained_from_commit` (`git rev-parse HEAD` at generation time, falling back to
`"unknown"` if git isn't available — not a fresh UUID per run, which is what
`training_run_id` below used to do). `backend/app/core/model_registry.py`'s
`_check_manifest_entry` (renamed from `_check_manifest_version`) and
`ml/evaluation/validate.py`'s `_check_manifest_consistency` both check the hash in
addition to the version string, sharing the same `sha256_of` implementation (imported
by `validate.py` from `generate_manifest`, not reimplemented a third time). Signing,
HMAC, and external attestation were explicitly scoped out as Phase 4+ — the audit's own
framing ("a smaller gap than it might sound... but real if the manifest is ever treated
as the source of truth independent of git history") is exactly right, and a content
hash plus git provenance is proportionate to that actual gap. Covered by
`ml/tests/test_validate.py::test_validate_catches_sha256_mismatch` and
`backend/tests/test_model_registry.py::test_load_fails_readiness_on_manifest_sha256_mismatch`.

Evidence (as originally audited, before the fix above):
- `backend/models/models_manifest.json` matches every artifact present: verified by
  loading all 5 files and cross-checking `models.<key>.version` against each
  artifact's own `model_version` field — all 4 match.
- Manifest fields present, per
  [`ml/evaluation/generate_manifest.py:33-54`](../ml/evaluation/generate_manifest.py):
  `generated_at` (timestamp), `training_run_id` (a UUID minted fresh each time the
  manifest is regenerated — see the caveat below), and `models.<key>.metrics` (the
  full metrics report per model). **Version and metric snapshot: PASS.**
- **Missing: artifact SHA / content-integrity check.** The manifest stores no hash of
  any artifact file's actual bytes. `ModelRegistry._check_manifest_version`
  ([`model_registry.py:97-108`](../backend/app/core/model_registry.py)) only compares
  the `model_version` *string* declared in the manifest against the *string* declared
  inside the artifact — it does not detect a corrupted, truncated, or hand-edited
  artifact that still happens to carry the correct `model_version` value. Same gap in
  `ml/evaluation/validate.py`'s `_manifest_version` check (line 34-38) — string
  comparison only, no hash.
- `training_run_id` caveat: it's generated fresh every time
  `generate_manifest.generate_manifest()` runs (`ml/evaluation/generate_manifest.py:52`
  — `str(uuid.uuid4())`), not derived from anything about the actual training run
  itself (no git commit SHA, no training script invocation ID persisted anywhere). Two
  manifest-generation runs against byte-identical artifacts get two different
  `training_run_id`s. This satisfies "has a training-run ID field" literally but not
  the spirit of traceability back to *which* run produced the artifacts.
- `validate.py` confirmed to check manifest-artifact consistency, not just thresholds:
  [`ml/evaluation/validate.py:50-51, 59-60, 67-68, 74-75`](../ml/evaluation/validate.py)
  — one explicit version-mismatch check per model, each appending a distinct failure
  message.

**Question for you:** add a content hash (e.g. SHA-256 of each artifact file) to the
manifest and have both `ModelRegistry` and `validate.py` check it? This would catch
the "artifact edited by hand without touching model_version" class of bug the
audit prompt specifically asks about, which the current version-string check cannot.
Given artifacts are already committed to git (so `git log` gives real provenance), the
missing SHA is a smaller gap than it might sound — but it's a real one if the manifest
is ever treated as the source of truth independent of git history.

## Check 6: Confidence gate on Model 4
Status: **NEEDS DECISION**

Evidence:
- Implementation: [`ml/models/recommender.py:480-522`](../ml/models/recommender.py)
  (`_apply_confidence_gate`) and
  [`ml/models/recommender.py:673-706`](../ml/models/recommender.py)
  (`recommend_for_household`, the real serving-time entrypoint that gates against the
  *actually shown* top-N ranking, not the rule base's raw estimate).
- Tested: [`ml/tests/test_recommender.py:211, 244-300`](../ml/tests/test_recommender.py)
  — both the standalone gate function and `recommend_for_household` have dedicated
  tests.
- Fires meaningfully, not dead code: `ml/evaluation/reports/recommender_v1_metrics.json`
  → `low_confidence_recommendations_per_1000_households: 505.6` (≈17% of shown
  recommendation slots downgraded) — confirmed non-trivial, not near-zero.
- **`grep -rn "recommend_for_household|_apply_confidence_gate" **/*.py` outside
  `ml/`: zero matches.** The confidence gate does not run at real serve time today,
  because **there is no recommendations-serving endpoint in `backend/` at all** — Step
  6 built only the forecast endpoint (`GET /households/{id}/forecast`), which is
  Model 1's output, not Model 4's. `recommend_for_household` is a complete, tested
  reference implementation with no caller.

This is not a defect relative to what Phase 2 actually committed to shipping — the
Phase 2 prompt's Step 3 ("Model serving — one endpoint wired end-to-end") named the
*forecast* endpoint specifically, not a recommendations endpoint, and that's exactly
what got built. But the audit prompt's phrasing ("confirm it actually runs on
disaggregator inputs at serve time") deserves a precise answer: **it doesn't, yet,
because nothing in the running system calls Model 4 at all.** The suppression-rate
evidence above proves the gate *works* against the evaluation set; it does not prove
it's wired into a live request path, because no live request path exercises Model 4.

**Question for you:** is this the expected state entering Phase 3 (a recommendations
endpoint is presumably next, and `recommend_for_household` is ready to be called from
one), or did you expect Step 6's "one endpoint" to also cover a minimal recommendations
path? I read Step 6's scope as forecast-only per the original Phase 2 prompt, but
flagging since the audit prompt's phrasing implies an expectation of serve-time
execution that isn't met today.

**Ruling:** confirmed as by-design, not an oversight — Step 6's original scope
(Phase 2 prompt, "Model serving — one endpoint wired end-to-end") named the forecast
endpoint specifically, so this gap is exactly what Step 6 was supposed to leave behind
for Phase 3, not something Step 6 forgot. **Phase 3 backlog item (recorded so it isn't
lost)**: the future `GET /households/{id}/recommendations` endpoint must call
`recommend_for_household` (`ml/models/recommender.py`) — or a backend-side
reimplementation of it, matching the existing plain-JSON/no-`ml`-import pattern used by
`app/services/forecast.py` — so the flip-under-perturbation confidence gate actually
runs at serve time, and must log the "recommendations suppressed per household" metric
(mirroring `low_confidence_recommendations_per_1000_households` from
`ml/evaluation/reports/recommender_v1_metrics.json`) so the gate's real-world
suppression rate is observable in production, not just in the offline evaluation set.

## Check 7: Zombie CI jobs and workflow hygiene
Status: **PASS**

Evidence:
- `grep -n "if: false|continue-on-error|run: python [a-zA-Z_]+\.py"
  .github/workflows/ci.yml`: zero matches. The old zombie (`ml-validation` job,
  `if: false`, pointing at a nonexistent `validate_models.py`) no longer exists — it
  was replaced with the `ml` job.
- [`.github/workflows/ci.yml:105-110`](../.github/workflows/ci.yml) — the `ml` job has
  no `if:` condition at all (runs unconditionally on every push/PR per the workflow's
  top-level triggers) and no `continue-on-error`. Its final step,
  [`ci.yml`](../.github/workflows/ci.yml) `python -m evaluation.validate`, is a real,
  existing, tested module (`ml/evaluation/validate.py`, covered by
  `ml/tests/test_validate.py`, 4 tests).

## Check 8: Docker build verifiability
Status: **NEEDS DECISION** → **CONFIRMED, plan retained**

**Ruling:** stick with the plan — treat the first real Render/Fly deploy as the actual
build verification, per `docs/RUNBOOK.md`'s "Deploying" section (now extended with a
concrete post-deploy checklist — see that doc). The local Docker corruption turned out
to be recurring (the same class of stale-socket error resurfaced in a second,
unrelated directory after the first fix — `%LOCALAPPDATA%\docker-secrets-engine`, not
just `%LOCALAPPDATA%\Docker\run`), consistent with active interference (this machine
shows signs of managed endpoint security software) rather than a one-time leftover —
so further local attempts were deliberately abandoned rather than chased indefinitely.
The build-context simulation described below already verified the actual risk the
Phase 2 change introduced (relative-path resolution across the `libs/` boundary);
everything else a local `docker build` would additionally catch (the Linux-container
runtime itself, `apt-get install libpq5`) is equally well verified by Render's own
build logs on first deploy.

Evidence (as originally audited):
- **A literal `docker build -f backend/Dockerfile .` has not been executed.** Docker
  Desktop's daemon was unusable throughout this environment's Phase 2 work (recurring
  corrupted socket files under `%LOCALAPPDATA%\Docker` and
  `%LOCALAPPDATA%\docker-secrets-engine`, documented in
  [`docs/RUNBOOK.md`](../docs/RUNBOOK.md)'s "Deploying" section with the exact error
  text and fix). This is honestly marked as unverified there, not silently assumed
  working.
- **Mitigating evidence collected instead:** the exact build-context layout the
  Dockerfile produces (`backend/` copied to `/app`, `libs/` copied to `/libs`, as
  siblings — [`backend/Dockerfile:7-18`](../backend/Dockerfile)) was recreated on disk
  and `pip install -r requirements.txt` run against it in a clean Python 3.11 venv.
  Both `-e ../libs/wattwise_tariffs` and `-e ../libs/wattwise_climate` resolved and
  built correctly, the full dependency tree installed cleanly, and `import app.main`
  plus `ModelRegistry` loading all four real committed artifacts succeeded
  (`is_ready == True`). This verifies the Python dependency-resolution logic the
  Phase 2 change touched, but **not** container-specific concerns (the `apt-get
  install libpq5` step, actually running inside a Linux container namespace).
- `.dockerignore` coverage against the audit's explicit list: `.git/` ✓ (line 5),
  `__pycache__` ✓ (lines 16-17, 28 — covers `backend/**/__pycache__/` and
  `libs/**/__pycache__/`), `.pytest_cache` ✓ (lines 19, 30), `.venv` ✓ (line 14),
  `node_modules` ✓ (line 7), frontend build directories ✓ (lines 8-12). **`*.pyc` as a
  standalone pattern is not present** — only `__pycache__/` directories are excluded.
  In practice this repo has no `.pyc` files outside `__pycache__/` (confirmed by the
  `.gitignore` using the same pattern), so this is a low-risk gap, but the audit asked
  for it explicitly.
- Image size: **N/A — no image was built**, so there's nothing to report against the
  500MB threshold.

**Question for you:** this is the item most worth closing before Phase 3, per your own
prior instruction that Docker verification should happen before Phase 3's first
endpoint. The `RUNBOOK.md` entry already recommends treating the first real Render/Fly
deploy as the actual verification if local Docker remains unusable — confirm that's
still the plan, or should another local Docker attempt be made first (a machine
restart was one of the options discussed and not yet tried)?

## Check 9: Test coverage of the integration path
Status: **PASS**

Evidence:
- [`backend/tests/test_forecast_endpoint.py:44-64`](../backend/tests/test_forecast_endpoint.py)
  (`test_forecast_happy_path`) exercises the full stack in one test: real `httpx`
  `AsyncClient` → real JWT bearer auth (`create_access_token`) → real
  `get_owned_household` DB lookup (seeded `User`/`Household`/`Bill` rows) → the real,
  process-wide `model_registry` (loaded from the actual committed artifacts, not
  mocked) → `generate_forecast` → the real `wattwise_tariffs` calculator → assertions
  on the actual JSON response shape.
- `grep -n "mock|Mock|patch\b|monkeypatch" backend/tests/test_forecast_endpoint.py`:
  zero matches — nothing in this test is mocked.
- 107 tests total across the monorepo (40 backend + 54 ml + 6 `wattwise_tariffs` + 7
  `wattwise_climate`), the large majority of which are legitimately per-model or
  per-function unit tests (appropriate for the rule base, feature engineering, and
  individual model training) — but the integration path specifically (auth → DB →
  ModelRegistry → prediction → tariff → response) has real, non-mocked coverage, which
  is what the audit asks to confirm.
- Caveat: this test runs against SQLite (`backend/tests/conftest.py`), not Postgres —
  the same pattern established since Phase 1, not a new gap.

## Check 10: Bonus items
Status: **PASS**, with two notes worth naming

- `grep -rn "TODO|FIXME|XXX|HACK|WORKAROUND" **/*.{py,ts,tsx}`: zero matches across
  the whole repo.
- PII in logs: `grep -n "logger\.(info|error|warning|debug)"` across
  `model_registry.py`, `main.py`, `households.py`, `forecast.py` shows only
  `trained_from_commit` (renamed from `training_run_id` by the Check 5 fix above —
  same non-PII nature, a git SHA rather than a per-run UUID), `env`, `models_ready`,
  `path`, and exception info logged — no household ID, user ID, or bill data logged
  directly by any Phase 2 addition.
  **One pre-existing note, not a Phase 2 regression:**
  [`backend/app/core/middleware.py:27-33`](../backend/app/core/middleware.py) logs
  `path=request.url.path` at INFO for every request, and
  `GET /households/{household_id}/forecast`'s path literally contains the household
  UUID. This pattern already existed for `GET /households/{id}` since Phase 1; Step 6
  extends the same existing pattern rather than introducing a new one. Worth a
  decision at some point (household UUIDs aren't directly identifying on their own,
  but are still an internal identifier logged in plaintext), but out of Phase 2's
  scope to fix unilaterally here.
- Endpoints not in OpenAPI schema: `grep -n "include_in_schema"` shows only `/metrics`
  ([`main.py:104`](../backend/app/main.py)) and `/auth/oauth/exchange`
  ([`auth.py:205`](../backend/app/api/routes/auth.py)) are deliberately excluded — the
  new forecast endpoint has no such exclusion, so it **is** included in the OpenAPI
  schema automatically.
- `model_version` sourcing: the forecast response's `model_version`
  ([`forecast.py:159`](../backend/app/services/forecast.py)) is read from the
  **artifact's own** `model_version` field, not re-read from the manifest at request
  time. This is safe in practice — `ModelRegistry`'s load-time check
  (`_check_manifest_version`) already guarantees the artifact and manifest agree
  before `is_ready` can be `True` — but it means the response value is *indirectly*
  guaranteed consistent with the manifest (via the load-time gate), not *directly*
  sourced from it per-request. Worth knowing, not worth changing.
