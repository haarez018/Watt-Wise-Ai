# Phase 1 Audit Report
Date: 2026-07-11

Scope: read-only verification of the working tree as committed at `c1cc2f8` (already
pushed to `origin/main` on GitHub prior to this audit). No code was modified as part of
producing this report.

## Summary
- Pass count: 3
- Fail count: 6
- Needs decision count: 2
- Overall verdict: **FIX REQUIRED** — and note that commit `c1cc2f8` is already pushed
  to GitHub with these issues present, so "before commit" is retroactive; treat this as
  "fix required before the next push / before Phase 2 starts."

## Check 1: Auth end-to-end wiring
Status: **NEEDS DECISION**

Evidence:
- Signing key from env, not hardcoded: [`backend/app/core/config.py:24-26`](../backend/app/core/config.py) — `jwt_secret` is a `pydantic-settings` field, resolved from the `JWT_SECRET` env var. Confirmed PASS, but see caveat below.
- Token flow: [`backend/app/api/routes/auth.py:25-99`](../backend/app/api/routes/auth.py) (signup/login/refresh/me) and [`backend/app/core/security.py`](../backend/app/core/security.py) (`create_access_token`, `create_refresh_token`, `decode_token`).
- Frontend propagation: [`frontend/src/lib/auth.ts:83-129`](../frontend/src/lib/auth.ts) — the `jwt` callback stores `accessToken`, `refreshToken`, and `accessTokenExpiresAt` (decoded via `decodeJwtExpiryMs`, line 20-27), and proactively calls `refreshBackendToken` when `Date.now() < expiresAt - 60_000` is false (line 114-117) — this **is** a pre-expiry refresh, not a post-401 reactive one. Confirmed PASS for this specific sub-item.
- Expired/stale token rejection: `decode_token` in `security.py` calls `jwt.decode(...)` with no `leeway` argument, so PyJWT's default `exp` validation applies — an expired token always raises `PyJWTError` → `InvalidTokenError` → `401` in [`backend/app/api/deps.py:29-31`](../backend/app/api/deps.py). No code path treats an expired token as valid. PASS.

**The actual gap:** `POST /auth/refresh` ([`auth.py:74-94`](../backend/app/api/routes/auth.py)) validates the incoming refresh token and issues a **new** access+refresh pair, but never invalidates the old refresh token. There is no `jti` denylist, no refresh-token version column on `users`, and no token table in the schema at all (`backend/app/models/` has no session/token model). This means a previously-issued refresh token stays valid for its full 30-day lifetime even after being "rotated" — if one leaks, rotating via legitimate use does not revoke it. Also: there is no logout/revocation endpoint anywhere, so an access or refresh token cannot be invalidated early under any circumstance short of changing `JWT_SECRET` (which invalidates everyone).

Additionally: `jwt_secret` has a dev fallback default (`"dev-secret-change-me"`) and there is no startup check asserting that `env == "production"` implies a non-default secret — nothing would stop a misconfigured deploy from silently running with the dev secret in prod.

**Question for you:** do you want refresh-token rotation/revocation (a `jti` denylist table, or a `token_version` column on `users` bumped on logout/password-change) built now, or is this acceptable to defer to Phase 3 alongside the rest of the household endpoints? Given Phase 1's stated scope was "auth from day one" but not full session management, deferring is defensible — but it should be a conscious call, not an oversight.

## Check 2: Alembic migration correctness
Status: **FAIL**

Evidence: [`backend/alembic/versions/9e7afc6ea408_initial_schema.py`](../backend/alembic/versions/9e7afc6ea408_initial_schema.py)

| Requirement | Result |
|---|---|
| `created_at`/`updated_at` server-side defaults | PASS — `_audit_columns()` (lines 21-36) sets `server_default=sa.func.now()` on both. |
| `updated_at` has `onupdate` | PASS at the ORM level — [`backend/app/models/base.py`](../backend/app/models/base.py) `TimestampMixin.updated_at` has `onupdate=func.now()`. Note: `onupdate` is a client-side SQLAlchemy behavior, not a DDL construct, so it correctly does **not** appear in the migration itself — this is not drift, just how the feature works. |
| `deleted_at` nullable | PASS — present on every table via `_audit_columns()`. |
| `deleted_at` **indexed** | **FAIL** — no `op.create_index` targets `deleted_at` on any of the six tables. Every soft-delete-aware query filters `WHERE deleted_at IS NULL`; today's dataset size makes this invisible, but it's a real gap against the explicit requirement. |
| FKs have explicit `ondelete` | **FAIL** — every `sa.ForeignKey(...)` call in the migration (lines 58, 76, 93, 111, 132, 137) omits `ondelete`, defaulting to `RESTRICT`-like implicit behavior at the DB level (actually: no `ON DELETE` clause at all, so Postgres defaults to `NO ACTION`). Confirmed the **models** match — `backend/app/models/*.py` also never pass `ondelete=` to any `ForeignKey(...)` — so there's no model/migration drift, but both fail the stated standard. Given `Household` → `User` and `Bill`/`Appliance`/`Recommendation` → `Household` are all `cascade="all, delete-orphan"` at the ORM relationship level (e.g. [`user.py:23-25`](../backend/app/models/user.py), [`household.py:34-39`](../backend/app/models/household.py)), the intent is clearly cascade-on-delete — but ORM-level cascade only fires when a delete goes through SQLAlchemy's session; a raw `DELETE FROM users` (or any tool bypassing the ORM) would hit a real FK violation instead of cascading. This should be `ondelete="CASCADE"` at the DB level to match the actual intent, especially since these are soft-delete tables and real hard-deletes should be rare, but not impossible (e.g. GDPR export/delete flow in Phase 3). |
| Money/energy columns are `BigInteger` | **FAIL** — `bill.py:24-25` (`units_consumed_wh`, `amount_paise`), `recommendation.py:26` (`estimated_savings_paise_per_month`), `savings_event.py:27` (`savings_paise`) all use plain `Integer` (max ~2.147 billion), not `BigInteger`. Model and migration agree with each other (no drift), but neither matches the stated standard. Practical severity is low for a single bill/recommendation row (₹21 million cap is far beyond any real household bill), but worth fixing for consistency and to avoid ever having to think about it again. |
| Timestamps `DateTime(timezone=True)` | PASS — used consistently in both models and migration. |

## Check 3: Household ↔ user cardinality
Status: **PASS**

Evidence: [`backend/app/models/household.py:21-22`](../backend/app/models/household.py) — `owner_id` is a plain (non-unique) indexed FK to `users.id`. [`backend/app/models/user.py:23-25`](../backend/app/models/user.py) — `households: Mapped[list["Household"]]`, a one-to-many relationship. No `UniqueConstraint` on `owner_id` anywhere in the model or migration, so the schema already correctly supports one user owning multiple households — this is **not** 1:1.

Caveat: no household-CRUD API endpoints or frontend flows exist yet to actually create/switch households — `frontend/src/app/onboarding/page.tsx` is a static placeholder ("Onboarding wizard coming in Phase 4"). This is expected per the documented Phase 1 scope (`docs/API.md` lists household endpoints under "planned Phase 3"), not a defect — flagging so it isn't mistaken for a working feature today.

## Check 4: Row-level authorization
Status: **NEEDS DECISION**

Evidence: `grep -rn "household_id" backend/app/api/` returns zero matches. There are currently **no endpoints that accept a household ID at all** — the only routers mounted in [`backend/app/main.py:85-86`](../backend/app/main.py) are `system` (`/healthz`, `/readyz`) and `auth`. There is no `get_household_or_404`-style dependency anywhere in `backend/app/api/deps.py` to audit, because nothing yet needs one.

This matches the documented Phase 1 scope (household/bill/appliance endpoints are explicitly "planned Phase 3" in `docs/API.md`), so this isn't a Phase 1 regression — but it's the single most important thing to get right before any household-scoped endpoint ships, and the audit prompt is correct to ask for it now rather than after.

**Question for you:** should I scaffold the ownership-check dependency (load household by ID, verify `household.owner_id == current_user.id`, return `404` — not `403` — on mismatch to avoid leaking existence) now as part of closing out this audit, or leave it entirely for when the first Phase 3 endpoint is written? I'd lean toward building it now since it's small, self-contained, and then every Phase 3 endpoint can just depend on it from day one instead of each author re-deriving the pattern — but flagging as a decision since you said not to fix anything yet.

## Check 5: Rate limiting scope
Status: **FAIL**

Evidence:
- [`backend/app/core/limiter.py:1-4`](../backend/app/core/limiter.py) — `Limiter(key_func=get_remote_address)`. Key is **IP-only**, confirmed. No `default_limits` argument is passed either, which matters for the next finding.
- [`backend/app/api/routes/auth.py`](../backend/app/api/routes/auth.py): `signup` (line 26) `@limiter.limit("10/minute")`, `login` (line 49) `@limiter.limit("10/minute")`, `refresh` (line 75) `@limiter.limit("30/minute")` — all present and documented in `docs/API.md`.
- `oauth_exchange` (lines 102-135): **no `@limiter.limit(...)` decorator at all.** Because `Limiter()` was constructed without `default_limits`, slowapi applies **zero rate limiting** to this route — it is not merely "not stricter than user-facing endpoints" as the audit prompt speculated, it is currently completely unlimited.

Combined with Check 7's finding that this endpoint is also unauthenticated-by-schema-visibility, this is the highest-severity concrete finding in this audit: an internal, secret-guarded, user-creating endpoint with no rate limit at all.

Login being keyed on IP alone (not IP+email) is a secondary, lower-severity finding — legitimate under shared-NAT/office-network conditions, but not urgent.

## Check 6: `.env.example` completeness
Status: **FAIL**

Evidence:
- Env vars read in backend code all funnel through `pydantic-settings` in [`backend/app/core/config.py`](../backend/app/core/config.py) (no stray `os.getenv`/`os.environ` calls found anywhere in `backend/` — confirmed via grep, zero matches). Cross-referencing all 12 `Settings` fields against [`backend/.env.example`](../backend/.env.example): all 12 are present, none missing, none extra. No drift.
- Frontend env reads (`grep -rn "process\.env\.[A-Z_]+" frontend/src`): `NEXT_PUBLIC_API_URL` ([`lib/auth.ts:5`](../frontend/src/lib/auth.ts), [`lib/api-client.ts:1`](../frontend/src/lib/api-client.ts)), `GOOGLE_CLIENT_ID`/`GOOGLE_CLIENT_SECRET` (`lib/auth.ts:78-79`), `INTERNAL_API_SECRET` (`lib/auth.ts:96`). `NEXTAUTH_URL`/`NEXTAUTH_SECRET` aren't read explicitly in our code but are read implicitly by the `next-auth` package itself — both are present in [`frontend/.env.example`](../frontend/.env.example). No missing or unused vars either side.
- **The actual failure:** `backend/.env.example` has **zero** comments on any of its 12 lines. `frontend/.env.example` has exactly **one** comment (line 9, on `INTERNAL_API_SECRET`) out of 6 lines. The explicit requirement ("each entry should have a short comment describing what it is") is unmet almost entirely.

## Check 7: Google OAuth server-to-server endpoint
Status: **FAIL**

Evidence: [`backend/app/api/routes/auth.py:102-135`](../backend/app/api/routes/auth.py)
- Secret from env only, not logged: PASS — `settings.internal_api_secret` (config.py) is compared directly (line 111); nothing in [`backend/app/core/middleware.py`](../backend/app/core/middleware.py)'s `RequestContextMiddleware` logs request bodies or headers (it only logs `method`, `path`, `status_code`, `duration_ms`), so the secret is never written to logs — but this is true because **nothing logs bodies/headers at all** right now, not because a redaction mechanism was proven against a real log line. The `_redact_pii` processor in [`backend/app/core/logging.py:8-15`](../backend/app/core/logging.py) redacts keys named `email`/`password`/`full_name`/`phone`/`address` — this list doesn't include `x_internal_secret` or `provider_subject`, but again, moot today since nothing logs them.
- Excluded from OpenAPI schema or tagged "internal": **FAIL** — the endpoint is declared on `router = APIRouter(prefix="/auth", tags=["auth"])` (line 22) with no per-route `include_in_schema=False` or distinct tag. It appears in `/docs` identically to `signup`/`login`/`me`, which could mislead an external integrator into thinking it's a public endpoint.
- Stricter rate limiting than user-facing endpoints: **FAIL** — see Check 5; it has no rate limiting at all, which is the opposite of "stricter."

## Check 8: CI reality check
Status: **FAIL**

Evidence: [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)
- Backend job (lines 14-65): starts a real `postgres:15-alpine` service container (lines 17-29), runs `alembic upgrade head` against it (lines 54-57), then runs `pytest` against the same database (lines 59-65) — **not** SQLite. Structurally correct. PASS for this sub-item.
- Frontend job (lines 67-103): runs ESLint, `tsc --noEmit`, `prettier --check`, `vitest run`, and `next build` in that order. Structurally correct. PASS for this sub-item.
- **Actual GitHub run status** (checked via the GitHub API against `haarez018/Watt-Wise-Ai`, run id `29072807260`, commit `c1cc2f8`): `status: completed`, `conclusion: failure`. Job-level breakdown: **Backend job → success**; **Frontend job → failure at the "Build" step**; ML-validation job → skipped (expected, `if: false`).
- I could not retrieve the raw build log text via unauthenticated fetch (GitHub's log viewer is JS-rendered and the log-download endpoint needs auth even for public repos), so I can't cite the exact error line. Best-effort hypothesis based on the code: the `Build` step's env block ([`ci.yml:98-102`](../.github/workflows/ci.yml)) sets `NEXT_PUBLIC_API_URL`, `NEXTAUTH_URL`, `NEXTAUTH_SECRET`, `INTERNAL_API_SECRET` — but **not** `GOOGLE_CLIENT_ID`/`GOOGLE_CLIENT_SECRET`, which `frontend/src/lib/auth.ts:77-80` passes into the `Google({...})` provider unconditionally. If NextAuth v5 validates provider config at import/build time when Next.js statically analyzes routes, missing Google credentials could throw during `next build`. This is a hypothesis, not a confirmed root cause — needs the actual log to verify.
- **This is the most important finding in the whole audit**: CI has not passed on the actual GitHub repository despite passing locally in every check I ran during Phase 1 development. The "8 backend tests + 7 frontend tests + 3 e2e tests all green" status reported earlier in this project was accurate for local runs only; it was never confirmed against the real CI environment until this audit. Do not treat local-green as CI-green going forward.

## Check 9: Frontend token storage & cookie flags
Status: **PASS**

Evidence:
- `grep -rn "localStorage|sessionStorage" frontend/src` → zero matches. No token, refresh token, or PII touches web storage anywhere.
- `grep -rn "cookies|httpOnly|secure|sameSite" frontend/src` → zero matches — there is no custom cookie configuration anywhere in the codebase, meaning NextAuth v5's defaults apply as-is: `httpOnly: true` always, `secure: true` when it detects an `https://` deployment URL, `sameSite: 'lax'`. Session strategy is `"jwt"` ([`frontend/src/lib/auth.ts:48`](../frontend/src/lib/auth.ts)), so the backend access/refresh tokens live inside NextAuth's own encrypted, httpOnly session cookie, not in a separate readable cookie.
- Caveat (not a code defect, but worth stating): this "secure by default" behavior is contingent on `NEXTAUTH_URL` (or `AUTH_URL`) actually being set to an `https://` URL in the production environment. If that's misconfigured to `http://` in prod, the secure-cookie default would silently regress. Worth a deploy-checklist item, not a code fix.

## Check 10: Next.js App Router hygiene
Status: **PASS**

Every file starting with `"use client"` (13 total, via grep):

| File | Justification |
|---|---|
| `src/app/(auth)/login/page.tsx` | `useState`, `useForm`, `useRouter`, `useSearchParams`, `signIn` — genuinely interactive. |
| `src/app/(auth)/signup/page.tsx` | Same as above, plus `apiFetch` client call. |
| `src/app/providers.tsx` | Wraps `SessionProvider` (React context) — must be client. |
| `src/components/providers/query-provider.tsx` | `useState` to hold a stable `QueryClient` instance. |
| `src/components/ui/form.tsx` | `useFormContext`, `useId`, custom React context (`react-hook-form` is client-only). |
| `src/components/ui/select.tsx`, `tabs.tsx`, `dialog.tsx`, `dropdown-menu.tsx`, `avatar.tsx` | Radix UI primitives — these use `forwardRef` and internal `useEffect`/measurement logic that require a client boundary; this is a structural requirement of the underlying library, not a wasteful choice. |
| `src/components/ui/label.tsx`, `separator.tsx` | Same Radix-primitive constraint, even though these two are visually simple — Radix ships them from the same client-only primitive package, so "use client" is still required despite the component itself having no obvious interactivity. |
| `src/components/ui/sonner.tsx` | Uses `next-themes`' `useTheme` hook — client-only. |

`card.tsx`, `button.tsx`, `badge.tsx`, `input.tsx`, and `skeleton.tsx` are **not** in this list and correctly remain server components. `src/app/dashboard/page.tsx` and `src/app/onboarding/page.tsx` are also server components (they call `auth()` server-side and only pass a server action to a client `<Button>`). No unnecessary client-boundary bloat found.

## Check 11: Bonus production sanity items
Status: **FAIL**

- **CORS:** [`backend/app/main.py:47-53`](../backend/app/main.py) — `allow_origins=settings.cors_origins`, an explicit env-driven allowlist (default `["http://localhost:3000"]`), not `"*"`. PASS. (`allow_methods=["*"]`/`allow_headers=["*"]` are wildcarded, which is standard/acceptable when origins themselves are locked down and `allow_credentials=True` — Starlette wouldn't even permit `"*"` origins combined with credentials.)
- **RFC 7807 shape on every error path: FAIL.** Only two handlers are registered — `StarletteHTTPException` and `RequestValidationError` (`main.py:71-80`), both correctly routed through `_problem_detail`. But there is **no generic `Exception` handler**, so an unhandled 500 falls through to FastAPI's default handler, which returns a plain `{"detail": "Internal Server Error"}` — not 7807-shaped. Separately, the rate-limit-exceeded handler is slowapi's own `_rate_limit_exceeded_handler` (`main.py:44`), which returns its own plain-text/JSON shape, also not 7807-shaped. So the "every error path" bar is not met.
- **`/metrics` guard: FAIL.** [`backend/app/main.py:83`](../backend/app/main.py) — `Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)`. `include_in_schema=False` only hides it from the OpenAPI docs; it does **not** add authentication or an internal-network guard. The endpoint is fully publicly reachable today with zero protection.
- **TODOs/FIXMEs/commented-out code:** clean. `grep -rn "TODO|FIXME|XXX"` across the tracked tree returns only one incidental match inside `frontend/package-lock.json` (a base64 integrity hash that happens to contain the substring "XXX" — not an actual comment). No real TODO/FIXME/commented-out code in any tracked source file.
- **Dependency CVE quick pass** (via web search, not exhaustive):
  - `next==14.2.35` ([`frontend/package.json:22`](../frontend/package.json)) — **confirmed this is the current officially-patched version** for the 14.x line as of the most recent (May 2026) Next.js security release. PASS.
  - `python-multipart==0.0.20` ([`backend/requirements.txt:17`](../backend/requirements.txt)) — outdated. CVE-2026-24486 (path traversal, fixed in 0.0.22) and CVE-2026-40347 (DoS via multipart preamble/epilogue, fixed in 0.0.26) both postdate our pin. No route currently accepts `multipart/form-data` (no `File(...)`/`Form(...)` params found anywhere in `backend/app/`), so today's exploitability is effectively nil, but the package will matter as soon as Phase 4's bill-upload feature lands, and it costs nothing to bump now. **FAIL.**
  - `pyjwt[crypto]==2.10.1` ([`backend/requirements.txt:10`](../backend/requirements.txt)) — outdated. CVE-2026-32597 (unvalidated `crit` header, fixed in 2.12.0) and CVE-2026-48526 (HMAC/asymmetric key-confusion, fixed in 2.13.0). Our usage pins a single algorithm explicitly (`algorithms=[settings.jwt_algorithm]` in `security.py`, always HS256), which mitigates the classic alg-confusion exploit path regardless of the library bug — but the dependency is still outdated and should be bumped. **FAIL** (low current exploitability, still worth fixing).
  - `next-auth==5.0.0-beta.31` — not a CVE, but worth flagging: this is still a **beta** major version. Reasonable and common for App Router projects today, but it's a conscious risk (breaking changes between betas, less battle-tested) rather than a stable dependency — a NEEDS DECISION-flavored note, not a hard fail.
  - This was a manual spot-check of the highest-risk packages, not a full `pip-audit`/`npm audit` run — recommend running both tools for full coverage before the next deploy.

## Recommended fix order
1. **Get CI green on GitHub** (Check 8) — nothing else in this list can be trusted as "verified" until the frontend build actually passes in the real CI environment, not just locally.
2. **Add a rate limit to `/auth/oauth/exchange`** (Check 5/7) — currently the single most exploitable gap: an internal, secret-guarded, user-creating endpoint with zero request throttling.
3. **Guard `/metrics`** (Check 11) — trivial fix (network-level restriction or a lightweight auth check), currently fully public.
4. **Fix the Alembic migration** (Check 2) — add `ondelete="CASCADE"` (or the appropriate behavior per relationship) to every FK, index `deleted_at` on all six tables, switch money/energy columns to `BigInteger`. Do this now, before any real data exists in a deployed database, since schema changes get more expensive later.
5. **Bump `python-multipart` and `pyjwt`** (Check 11) to their patched versions.
6. **Add comments to every `.env.example` line** (Check 6) — quick, low-risk.
7. **Add a catch-all `Exception` handler and a 7807-shaped rate-limit handler** (Check 11) so every error path is consistently shaped.
8. **Decide on refresh-token rotation/revocation scope** (Check 1) — either build a minimal `jti`/token-version mechanism now, or explicitly defer to Phase 3 and document the decision.
9. **Decide whether to scaffold the household-ownership-check dependency now** (Check 4) — small and self-contained; building it before the first Phase 3 endpoint means every future endpoint inherits the correct pattern instead of each one re-deriving it.
10. **Revisit the `next-auth` beta pin** (Check 11) before the actual production launch — not urgent for continued Phase 2 work, but shouldn't be forgotten.
