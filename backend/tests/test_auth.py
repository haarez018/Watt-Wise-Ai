from httpx import AsyncClient

VALID_PASSWORD = "correct-horse-battery-staple"


async def test_signup_then_me(client: AsyncClient) -> None:
    signup_response = await client.post(
        "/auth/signup",
        json={"email": "new@example.com", "password": VALID_PASSWORD, "full_name": "New User"},
    )
    assert signup_response.status_code == 201
    access_token = signup_response.json()["access_token"]

    me_response = await client.get("/auth/me", headers={"Authorization": f"Bearer {access_token}"})
    assert me_response.status_code == 200
    assert me_response.json()["email"] == "new@example.com"


async def test_signup_duplicate_email_rejected(client: AsyncClient) -> None:
    payload = {"email": "dup@example.com", "password": VALID_PASSWORD}
    first = await client.post("/auth/signup", json=payload)
    assert first.status_code == 201

    second = await client.post("/auth/signup", json=payload)
    assert second.status_code == 409


async def test_login_wrong_password_rejected(client: AsyncClient) -> None:
    await client.post(
        "/auth/signup", json={"email": "login@example.com", "password": VALID_PASSWORD}
    )

    response = await client.post(
        "/auth/login", json={"email": "login@example.com", "password": "wrong-password"}
    )
    assert response.status_code == 401


async def test_me_without_token_rejected(client: AsyncClient) -> None:
    response = await client.get("/auth/me")
    assert response.status_code == 401


async def test_oauth_exchange_requires_internal_secret(client: AsyncClient) -> None:
    response = await client.post(
        "/auth/oauth/exchange",
        json={"email": "g@example.com", "provider": "google", "provider_subject": "sub123"},
    )
    assert response.status_code == 403


async def test_oauth_exchange_creates_user(client: AsyncClient) -> None:
    from app.core.config import get_settings

    settings = get_settings()
    response = await client.post(
        "/auth/oauth/exchange",
        json={"email": "g2@example.com", "provider": "google", "provider_subject": "sub456"},
        headers={"X-Internal-Secret": settings.internal_api_secret},
    )
    assert response.status_code == 200
    access_token = response.json()["access_token"]

    me_response = await client.get("/auth/me", headers={"Authorization": f"Bearer {access_token}"})
    assert me_response.status_code == 200
    assert me_response.json()["email"] == "g2@example.com"
