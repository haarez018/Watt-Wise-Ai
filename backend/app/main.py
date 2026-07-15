from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import sentry_sdk
import structlog
from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi.errors import RateLimitExceeded
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes import auth, households, system
from app.core.config import get_settings
from app.core.limiter import limiter
from app.core.logging import configure_logging
from app.core.middleware import RequestContextMiddleware
from app.core.model_registry import model_registry

settings = get_settings()
configure_logging(settings.log_level)
logger = structlog.get_logger("app")

if settings.sentry_dsn:
    sentry_sdk.init(dsn=settings.sentry_dsn, environment=settings.env, traces_sample_rate=0.1)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    # Models are loaded eagerly at `app.core.model_registry` import time, not
    # here — see that module's docstring for why. This just logs the result.
    if model_registry.is_ready:
        logger.info("app_startup", env=settings.env, models_ready=True)
    else:
        logger.error(
            "app_startup", env=settings.env, models_ready=False, error=model_registry.load_error
        )
    yield
    logger.info("app_shutdown")


app = FastAPI(
    title="WattWise AI API",
    description="Household electricity intelligence: forecasting, anomalies, disaggregation.",
    version="0.1.0",
    lifespan=lifespan,
)

app.state.limiter = limiter

app.add_middleware(RequestContextMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _problem_detail(status_code: int, title: str, detail: str, request: Request) -> JSONResponse:
    """RFC 7807 problem+json error body."""
    return JSONResponse(
        status_code=status_code,
        content={
            "type": "about:blank",
            "title": title,
            "status": status_code,
            "detail": detail,
            "instance": str(request.url.path),
        },
        media_type="application/problem+json",
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    return _problem_detail(exc.status_code, exc.__class__.__name__, str(exc.detail), request)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return _problem_detail(422, "Validation Error", str(exc.errors()), request)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return _problem_detail(429, "Rate Limit Exceeded", str(exc.detail), request)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("unhandled_exception", exc_info=exc, path=request.url.path)
    return _problem_detail(500, "Internal Server Error", "An unexpected error occurred.", request)


Instrumentator().instrument(app)


@app.get("/metrics", include_in_schema=False)
async def metrics(x_metrics_token: str | None = Header(default=None)) -> Response:
    """Prometheus scrape endpoint. Gated by a shared token, not publicly scrapable."""
    if x_metrics_token != settings.metrics_token:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


app.include_router(system.router)
app.include_router(auth.router)
app.include_router(households.router)
