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

1. Build from `backend/Dockerfile`.
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

## Rolling back

- **Backend:** redeploy the previous container image/tag. If the rollback crosses a
  migration boundary, run `alembic downgrade -1` against production **before**
  rolling back the code, so the schema matches what the older code expects. Every
  migration in `backend/alembic/versions/` has a working `downgrade()`.
- **Frontend:** use Vercel's "Instant Rollback" to the previous deployment.

## Common incidents

| Symptom | Likely cause | First checks |
|---|---|---|
| `/readyz` returning errors | Database unreachable | Check Postgres is up; check `DATABASE_URL` / `SYNC_DATABASE_URL` credentials and network access |
| Frontend shows "RefreshFailed" session error | Backend JWT refresh failing | Check backend logs for `/auth/refresh` 401s; confirm `JWT_SECRET` hasn't changed (invalidates all outstanding tokens) |
| Google sign-in creates no backend session | `INTERNAL_API_SECRET` mismatch between frontend and backend | Confirm both services have the identical value; check backend logs for `403` on `/auth/oauth/exchange` |
| 429s on `/auth/login` or `/auth/signup` | Rate limiting (10/minute/IP) | Expected under abuse; if a legitimate user is blocked, check `RATE_LIMIT_DEFAULT` and the `slowapi` limits in `app/api/routes/auth.py` |
| CI failing on `mypy` only in the pre-commit/CI environment, not locally | `additional_dependencies` in `.pre-commit-config.yaml` or CI's `requirements-dev.txt` install missing a package mypy needs to resolve types | Compare the failing environment's installed packages against `backend/requirements.txt` |

## Secrets rotation

`JWT_SECRET` rotation invalidates every outstanding access/refresh token (all users are
logged out). `INTERNAL_API_SECRET` rotation must be deployed to both frontend and
backend simultaneously, or Google sign-in will fail with `403` in between.
