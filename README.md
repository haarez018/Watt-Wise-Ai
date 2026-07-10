# WattWise AI

Household electricity intelligence for Indian households: bill forecasting, anomaly
detection, software-only appliance disaggregation, and ₹/CO₂-quantified recommendations
— all served by models we trained ourselves, with no external AI calls at inference
time.

See [`docs/PROBLEM_STATEMENT.md`](docs/PROBLEM_STATEMENT.md) for the product thesis,
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for how it fits together, and
[`docs/RUNBOOK.md`](docs/RUNBOOK.md) for deployment and incident response.

## Stack

- **Frontend:** Next.js 14 (App Router), TypeScript strict, Tailwind, shadcn/ui,
  TanStack Query, NextAuth
- **Backend:** FastAPI, SQLAlchemy 2.0, Alembic, Pydantic v2
- **Database:** Postgres 15
- **ML:** scikit-learn / XGBoost / statsmodels (Phase 2 — see `docs/ML.md`)

## Quickstart (local dev)

Requires Docker Desktop, Node 20+, and Python 3.11+ if you want to run services outside
Docker.

```bash
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env.local
docker compose up
```

- Frontend: http://localhost:3000
- Backend: http://localhost:8000 (interactive docs at `/docs`)

The backend container runs `alembic upgrade head` automatically before starting, so the
database schema is ready on first boot.

### Running without Docker

```bash
# Backend
cd backend
python -m venv .venv
.venv\Scripts\activate   # Windows — use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements-dev.txt
alembic upgrade head
uvicorn app.main:app --reload

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

You'll need a local Postgres instance reachable at the `DATABASE_URL` /
`SYNC_DATABASE_URL` in `backend/.env` if you skip Docker.

## Testing

```bash
# Backend
cd backend && pytest

# Frontend
cd frontend && npm run test        # Vitest unit/component tests
cd frontend && npm run test:e2e    # Playwright, starts its own dev server
```

## Linting & formatting

```bash
# Backend
cd backend && ruff check app tests && black --check app tests && mypy app

# Frontend
cd frontend && npm run lint && npm run typecheck && npm run format:check
```

Both are enforced via `.pre-commit-config.yaml` — install it once with:

```bash
pip install pre-commit
pre-commit install
```

## Repo layout

```
/frontend   Next.js app
/backend    FastAPI app, Alembic migrations, model artifacts (backend/models/)
/ml         Training pipeline for the self-trained models (Phase 2)
/data       Datasets used to train models (git-ignored)
/infra      Deployment configuration
/docs       Architecture, API, ML, runbook, and problem-statement docs
/.github    CI workflows
```

## Contributing

- Every schema change is an Alembic migration — never hand-edit the database.
- Never commit `.env` files; only `.env.example` is tracked.
- Run the linters and tests above before opening a PR; CI enforces the same checks.
- Monetary values are always integers in paise; energy values are always integers in
  Wh. Don't introduce floats for either.
