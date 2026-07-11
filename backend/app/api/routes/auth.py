import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from sqlalchemy import select, update
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
from app.models.refresh_token import RefreshToken
from app.models.user import User
from app.schemas.auth import RefreshRequest, TokenPair, UserLogin, UserRead, UserSignup
from app.schemas.oauth import OAuthExchangeRequest

router = APIRouter(prefix="/auth", tags=["auth"])


def _client_ip(request: Request) -> str | None:
    return request.client.host if request.client else None


async def _issue_token_pair(db: AsyncSession, request: Request, user_id: uuid.UUID) -> TokenPair:
    """Mints a fresh access+refresh pair and persists the refresh token's row.

    Used for signup/login/oauth-exchange, where there's no prior token to rotate.
    """
    access_token = create_access_token(user_id)
    refresh_token, jti, issued_at = create_refresh_token(user_id)

    db.add(
        RefreshToken(
            user_id=user_id,
            jti=jti,
            issued_at=issued_at,
            user_agent=request.headers.get("user-agent"),
            ip=_client_ip(request),
        )
    )
    await db.commit()

    return TokenPair(access_token=access_token, refresh_token=refresh_token)


async def _rotate_refresh_token(
    db: AsyncSession, request: Request, old_row: RefreshToken, user_id: uuid.UUID
) -> TokenPair:
    """Mints a fresh pair and marks `old_row` revoked, pointing it at the new row."""
    access_token = create_access_token(user_id)
    refresh_token, jti, issued_at = create_refresh_token(user_id)

    new_row = RefreshToken(
        user_id=user_id,
        jti=jti,
        issued_at=issued_at,
        user_agent=request.headers.get("user-agent"),
        ip=_client_ip(request),
    )
    db.add(new_row)
    await db.flush()

    old_row.revoked_at = datetime.now(UTC)
    old_row.replaced_by = new_row.id
    await db.commit()

    return TokenPair(access_token=access_token, refresh_token=refresh_token)


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

    return await _issue_token_pair(db, request, user.id)


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

    return await _issue_token_pair(db, request, user.id)


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

    token_result = await db.execute(select(RefreshToken).where(RefreshToken.jti == payload.jti))
    token_row = token_result.scalar_one_or_none()
    if token_row is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token not recognized"
        )

    if token_row.revoked_at is not None:
        # This exact token was already rotated once before. Presenting it again means
        # it leaked (or a rotation response was lost and replayed) — revoke every
        # still-active token for this user so the legitimate session must re-authenticate.
        await db.execute(
            update(RefreshToken)
            .where(RefreshToken.user_id == user_id, RefreshToken.revoked_at.is_(None))
            .values(revoked_at=datetime.now(UTC))
        )
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token reuse detected; all sessions have been revoked",
        )

    return await _rotate_refresh_token(db, request, token_row, user_id)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
@limiter.limit("30/minute")
async def logout(
    request: Request, body: RefreshRequest, db: AsyncSession = Depends(get_db_session)
) -> None:
    """Revokes the given refresh token. Idempotent: an already-invalid token is a no-op."""
    try:
        payload = decode_token(body.refresh_token, expected_type="refresh")
    except InvalidTokenError:
        return

    await db.execute(
        update(RefreshToken)
        .where(RefreshToken.jti == payload.jti, RefreshToken.revoked_at.is_(None))
        .values(revoked_at=datetime.now(UTC))
    )
    await db.commit()


@router.post("/logout-all", status_code=status.HTTP_204_NO_CONTENT)
@limiter.limit("30/minute")
async def logout_all(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> None:
    """Revokes every active refresh token for the authenticated user (all devices)."""
    await db.execute(
        update(RefreshToken)
        .where(RefreshToken.user_id == current_user.id, RefreshToken.revoked_at.is_(None))
        .values(revoked_at=datetime.now(UTC))
    )
    await db.commit()


@router.get("/me", response_model=UserRead)
async def me(current_user: User = Depends(get_current_user)) -> User:
    return current_user


@router.post("/oauth/exchange", response_model=TokenPair, include_in_schema=False)
@limiter.limit("10/minute")
async def oauth_exchange(
    request: Request,
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

    return await _issue_token_pair(db, request, user.id)
