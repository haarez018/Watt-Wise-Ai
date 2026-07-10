# API

## OpenAPI spec

FastAPI auto-generates the OpenAPI schema and interactive docs from the code — they are
always in sync with what's deployed:

- Interactive Swagger UI: `GET /docs`
- ReDoc: `GET /redoc`
- Raw schema: `GET /openapi.json`

This file is the hand-written usage guide; treat the running `/docs` endpoint as the
source of truth for exact request/response shapes.

## Auth model

All endpoints except `/healthz`, `/readyz`, `/metrics`, and `/auth/*` require a bearer
token:

```
Authorization: Bearer <access_token>
```

Access tokens are short-lived JWTs (30 minutes by default) signed with `JWT_SECRET`
(HS256). Get one via `/auth/signup` or `/auth/login`, or transparently through the
frontend's NextAuth session.

## Endpoints implemented (Phase 1)

### `POST /auth/signup`

Creates a user with a bcrypt/argon2-hashed password. Rate-limited to 10 requests/minute
per IP.

```json
// Request
{ "email": "user@example.com", "password": "at-least-10-characters", "full_name": "Optional Name" }

// 201 Response
{ "access_token": "...", "refresh_token": "...", "token_type": "bearer" }
```

Returns `409 Conflict` if the email is already registered.

### `POST /auth/login`

```json
// Request
{ "email": "user@example.com", "password": "..." }

// 200 Response
{ "access_token": "...", "refresh_token": "...", "token_type": "bearer" }
```

Returns `401 Unauthorized` for a wrong password or unknown email (same error either way,
to avoid leaking which emails are registered).

### `POST /auth/refresh`

```json
// Request
{ "refresh_token": "..." }

// 200 Response
{ "access_token": "...", "refresh_token": "...", "token_type": "bearer" }
```

### `GET /auth/me`

Returns the authenticated user's profile. Requires a valid access token.

### `POST /auth/oauth/exchange`

**Server-to-server only** — called by the frontend's NextAuth backend after Google has
already verified the user, never called from the browser. Requires an
`X-Internal-Secret` header matching the backend's `INTERNAL_API_SECRET`. Finds or
creates a user by email and mints a backend JWT pair.

### `GET /healthz`

Liveness probe. Always returns `{"status": "ok"}` if the process is running; touches no
dependencies.

### `GET /readyz`

Readiness probe. Runs `SELECT 1` against Postgres; returns `503`-equivalent failure
(via exception) if the database is unreachable.

### `GET /metrics`

Prometheus-format metrics (request counts, latencies) via
`prometheus-fastapi-instrumentator`. Not included in the OpenAPI schema.

## Endpoints planned (Phase 3)

These are designed in `PROBLEM_STATEMENT.md` and the household/bill/appliance models
already exist in the database; the routes themselves land in Phase 3:

| Method & path | Purpose |
|---|---|
| `POST /households` | Create a household profile |
| `GET /households/{id}` | Fetch a household (owner-scoped) |
| `PATCH /households/{id}` | Update household profile / appliance inventory |
| `POST /households/{id}/bills` | Add a bill |
| `GET /households/{id}/bills` | List bill history |
| `GET /households/{id}/forecast` | Next-month prediction with confidence interval |
| `GET /households/{id}/breakdown` | Appliance-level disaggregation for a given month |
| `GET /households/{id}/anomalies` | Flagged anomalies with plain-language explanations |
| `GET /households/{id}/recommendations` | Ranked, ₹/CO₂-quantified actions |
| `GET /households/{id}/impact` | Cumulative ₹ saved and kg CO₂ avoided since baseline |
| `POST /households/{id}/export` | GDPR-style data export |
| `DELETE /households/{id}` | Account/household deletion (soft-delete) |

Every one of these will resolve the authenticated user from the bearer token and
verify `household.owner_id == current_user.id` before returning or mutating anything —
there is no household-scoped endpoint that skips this check.

## Error format

Errors follow [RFC 7807](https://www.rfc-editor.org/rfc/rfc7807) `application/problem+json`:

```json
{
  "type": "about:blank",
  "title": "HTTPException",
  "status": 401,
  "detail": "Incorrect email or password",
  "instance": "/auth/login"
}
```

## Rate limiting

`slowapi` enforces per-IP limits on public endpoints. `/auth/signup` and `/auth/login`
are limited to 10/minute; `/auth/refresh` to 30/minute. Limits are configurable via
`RATE_LIMIT_DEFAULT` in `.env` for endpoints that don't declare their own.
