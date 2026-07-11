from collections.abc import AsyncGenerator

from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db_session


async def test_rate_limit_response_is_rfc7807_shaped(client: AsyncClient) -> None:
    payload = {"email": "ratelimit@example.com", "password": "wrong-password"}

    responses = [await client.post("/auth/login", json=payload) for _ in range(11)]
    limited = [r for r in responses if r.status_code == 429]

    assert limited, "expected at least one 429 after exceeding the 10/minute login limit"
    body = limited[0].json()
    assert body["status"] == 429
    assert body["type"] == "about:blank"
    assert "title" in body and "detail" in body and "instance" in body


async def test_unhandled_exception_is_rfc7807_shaped(app: object) -> None:
    async def _broken_db_session() -> AsyncGenerator[AsyncSession, None]:
        raise RuntimeError("simulated unexpected failure")
        yield  # pragma: no cover - unreachable, satisfies generator typing

    fastapi_app = app
    fastapi_app.dependency_overrides[get_db_session] = _broken_db_session  # type: ignore[attr-defined]

    # raise_app_exceptions=False: let Starlette's own ServerErrorMiddleware convert the
    # exception into the response a real deployment (uvicorn) would send, instead of
    # httpx re-raising it into the test.
    transport = ASGITransport(app=fastapi_app, raise_app_exceptions=False)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as isolated_client:
            response = await isolated_client.get("/readyz")
    finally:
        del fastapi_app.dependency_overrides[get_db_session]  # type: ignore[attr-defined]

    assert response.status_code == 500
    body = response.json()
    assert body["status"] == 500
    assert body["type"] == "about:blank"
    assert body["detail"] == "An unexpected error occurred."
