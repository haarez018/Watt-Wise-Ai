from httpx import AsyncClient


async def test_healthz(client: AsyncClient) -> None:
    response = await client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_readyz(client: AsyncClient) -> None:
    response = await client.get("/readyz")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


async def test_metrics_without_token_rejected(client: AsyncClient) -> None:
    response = await client.get("/metrics")
    assert response.status_code == 404


async def test_metrics_with_wrong_token_rejected(client: AsyncClient) -> None:
    response = await client.get("/metrics", headers={"X-Metrics-Token": "wrong"})
    assert response.status_code == 404


async def test_metrics_with_correct_token_allowed(client: AsyncClient) -> None:
    from app.core.config import get_settings

    settings = get_settings()
    response = await client.get("/metrics", headers={"X-Metrics-Token": settings.metrics_token})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
