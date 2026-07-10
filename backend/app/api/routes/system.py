from fastapi import APIRouter, Depends, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db_session

router = APIRouter(tags=["system"])


@router.get("/healthz", status_code=status.HTTP_200_OK)
async def healthz() -> dict[str, str]:
    """Liveness probe: process is up. Does not touch dependencies."""
    return {"status": "ok"}


@router.get("/readyz", status_code=status.HTTP_200_OK)
async def readyz(db: AsyncSession = Depends(get_db_session)) -> dict[str, str]:
    """Readiness probe: process can serve traffic (DB reachable)."""
    await db.execute(text("SELECT 1"))
    return {"status": "ready"}
