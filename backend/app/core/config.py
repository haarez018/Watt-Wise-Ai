from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, loaded from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    env: Literal["development", "test", "production"] = Field(default="development")

    database_url: str = Field(
        default="postgresql+asyncpg://wattwise:wattwise@localhost:5432/wattwise",
        description="Async SQLAlchemy connection string used by the running app.",
    )
    sync_database_url: str = Field(
        default="postgresql+psycopg://wattwise:wattwise@localhost:5432/wattwise",
        description="Sync connection string used by Alembic migrations.",
    )

    jwt_secret: str = Field(
        default="dev-secret-change-me", description="HMAC signing key for access/refresh JWTs."
    )
    internal_api_secret: str = Field(
        default="dev-internal-secret-change-me",
        description="Shared secret for server-to-server calls from NextAuth (e.g. OAuth exchange).",
    )
    metrics_token: str = Field(
        default="dev-metrics-token-change-me",
        description="Required X-Metrics-Token header value to read /metrics.",
    )
    jwt_algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=30)
    refresh_token_expire_days: int = Field(default=30)

    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    sentry_dsn: str | None = Field(default=None)

    rate_limit_default: str = Field(default="60/minute")

    log_level: str = Field(default="INFO")

    @property
    def is_production(self) -> bool:
        return self.env == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
