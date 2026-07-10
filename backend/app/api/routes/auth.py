import uuid

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db_session
from app.core.config import get_settings
from app.core.limiter import limiter
from app.core.security import (
    InvalidTokenError,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.models.user import User
from app.schemas.auth import RefreshRequest, TokenPair, UserLogin, UserRead, UserSignup
from app.schemas.oauth import OAuthExchangeRequest

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/signup", response_model=TokenPair, status_code=status.HTTP_201_CREATED)
@limiter.limit("10/minute")
async def signup(
    request: Request, body: UserSignup, db: AsyncSession = Depends(get_db_session)
) -> TokenPair:
    existing = await db.execute(select(User).where(User.email == body.email))
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="An account with this email already exists"
        )

    user = User(
        email=body.email, password_hash=hash_password(body.password), full_name=body.full_name
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    return TokenPair(
        access_token=create_access_token(user.id), refresh_token=create_refresh_token(user.id)
    )


@router.post("/login", response_model=TokenPair)
@limiter.limit("10/minute")
async def login(
    request: Request, body: UserLogin, db: AsyncSession = Depends(get_db_session)
) -> TokenPair:
    result = await db.execute(
        select(User).where(User.email == body.email, User.deleted_at.is_(None))
    )
    user = result.scalar_one_or_none()

    if (
        user is None
        or user.password_hash is None
        or not verify_password(body.password, user.password_hash)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password"
        )
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is deactivated")

    return TokenPair(
        access_token=create_access_token(user.id), refresh_token=create_refresh_token(user.id)
    )


@router.post("/refresh", response_model=TokenPair)
@limiter.limit("30/minute")
async def refresh(
    request: Request, body: RefreshRequest, db: AsyncSession = Depends(get_db_session)
) -> TokenPair:
    try:
        payload = decode_token(body.refresh_token, expected_type="refresh")
    except InvalidTokenError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc

    user_id = uuid.UUID(payload.sub)
    result = await db.execute(select(User).where(User.id == user_id, User.deleted_at.is_(None)))
    user = result.scalar_one_or_none()
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or inactive"
        )

    return TokenPair(
        access_token=create_access_token(user.id), refresh_token=create_refresh_token(user.id)
    )


@router.get("/me", response_model=UserRead)
async def me(current_user: User = Depends(get_current_user)) -> User:
    return current_user


@router.post("/oauth/exchange", response_model=TokenPair)
async def oauth_exchange(
    body: OAuthExchangeRequest,
    db: AsyncSession = Depends(get_db_session),
    x_internal_secret: str | None = Header(default=None),
) -> TokenPair:
    """Server-to-server endpoint: mints a backend JWT for a user already verified by an OAuth
    provider (e.g. Google via NextAuth). Only callable with the shared internal secret."""
    settings = get_settings()
    if x_internal_secret != settings.internal_api_secret:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid internal secret")

    result = await db.execute(
        select(User).where(User.email == body.email, User.deleted_at.is_(None))
    )
    user = result.scalar_one_or_none()

    if user is None:
        user = User(
            email=body.email,
            full_name=body.full_name,
            oauth_provider=body.provider,
            oauth_subject=body.provider_subject,
            is_verified=True,
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
    elif not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is deactivated")

    return TokenPair(
        access_token=create_access_token(user.id), refresh_token=create_refresh_token(user.id)
    )
