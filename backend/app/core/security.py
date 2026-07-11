import uuid
from datetime import UTC, datetime, timedelta
from typing import Literal

import jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from app.core.config import get_settings

_pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")


def hash_password(plain_password: str) -> str:
    return _pwd_context.hash(plain_password)


def verify_password(plain_password: str, password_hash: str) -> bool:
    return _pwd_context.verify(plain_password, password_hash)


class TokenPayload(BaseModel):
    sub: str
    type: Literal["access", "refresh"]
    exp: datetime
    iat: datetime
    jti: str


def _create_token(
    subject: uuid.UUID, token_type: Literal["access", "refresh"], expires_delta: timedelta
) -> tuple[str, str, datetime]:
    settings = get_settings()
    now = datetime.now(UTC)
    jti = str(uuid.uuid4())
    payload = {
        "sub": str(subject),
        "type": token_type,
        "iat": now,
        "exp": now + expires_delta,
        "jti": jti,
    }
    token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    return token, jti, now


def create_access_token(subject: uuid.UUID) -> str:
    settings = get_settings()
    token, _jti, _issued_at = _create_token(
        subject, "access", timedelta(minutes=settings.access_token_expire_minutes)
    )
    return token


def create_refresh_token(subject: uuid.UUID) -> tuple[str, str, datetime]:
    """Returns (token, jti, issued_at) so the caller can persist a RefreshToken row."""
    settings = get_settings()
    return _create_token(subject, "refresh", timedelta(days=settings.refresh_token_expire_days))


class InvalidTokenError(Exception):
    pass


def decode_token(token: str, expected_type: Literal["access", "refresh"]) -> TokenPayload:
    settings = get_settings()
    try:
        raw = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except jwt.PyJWTError as exc:
        raise InvalidTokenError("Token is invalid or expired") from exc

    parsed = TokenPayload.model_validate(raw)
    if parsed.type != expected_type:
        raise InvalidTokenError(f"Expected a {expected_type} token, got {parsed.type}")
    return parsed
