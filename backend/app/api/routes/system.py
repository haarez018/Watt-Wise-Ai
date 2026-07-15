from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db_session
from app.core.model_registry import model_registry

router = APIRouter(tags=["system"])


@router.get("/healthz", status_code=status.HTTP_200_OK)
async def healthz() -> dict[str, str]:
    """Liveness probe: process is up. Does not touch dependencies."""
    return {"status": "ok"}


@router.get("/readyz", status_code=status.HTTP_200_OK)
async def readyz(db: AsyncSession = Depends(get_db_session)) -> dict[str, str]:
    """Readiness probe: process can serve traffic (DB reachable, models loaded).
    A model-loading failure fails readiness, not liveness — the process is
    still up and Kubernetes/Render shouldn't restart it, but shouldn't route
    traffic to it either."""
    await db.execute(text("SELECT 1"))
    if not model_registry.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Models not loaded: {model_registry.load_error}",
        )
    return {"status": "ready"}
