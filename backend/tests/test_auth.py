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


async def test_refresh_rotates_token(client: AsyncClient) -> None:
    signup_response = await client.post(
        "/auth/signup", json={"email": "rotate@example.com", "password": VALID_PASSWORD}
    )
    old_refresh_token = signup_response.json()["refresh_token"]

    first_refresh = await client.post("/auth/refresh", json={"refresh_token": old_refresh_token})
    assert first_refresh.status_code == 200
    new_refresh_token = first_refresh.json()["refresh_token"]
    assert new_refresh_token != old_refresh_token

    second_refresh = await client.post("/auth/refresh", json={"refresh_token": new_refresh_token})
    assert second_refresh.status_code == 200


async def test_refresh_reuse_detected_revokes_entire_chain(client: AsyncClient) -> None:
    signup_response = await client.post(
        "/auth/signup", json={"email": "reuse@example.com", "password": VALID_PASSWORD}
    )
    old_refresh_token = signup_response.json()["refresh_token"]

    rotated = await client.post("/auth/refresh", json={"refresh_token": old_refresh_token})
    new_refresh_token = rotated.json()["refresh_token"]

    # Replaying the already-rotated (old) token is a reuse signal -> rejected.
    replay = await client.post("/auth/refresh", json={"refresh_token": old_refresh_token})
    assert replay.status_code == 401

    # Reuse detection revokes the whole chain, so the token issued by the rotation
    # above is now dead too, even though it was valid moments earlier.
    now_also_dead = await client.post("/auth/refresh", json={"refresh_token": new_refresh_token})
    assert now_also_dead.status_code == 401


async def test_logout_revokes_refresh_token(client: AsyncClient) -> None:
    signup_response = await client.post(
        "/auth/signup", json={"email": "logout@example.com", "password": VALID_PASSWORD}
    )
    refresh_token = signup_response.json()["refresh_token"]

    logout_response = await client.post("/auth/logout", json={"refresh_token": refresh_token})
    assert logout_response.status_code == 204

    refresh_after_logout = await client.post("/auth/refresh", json={"refresh_token": refresh_token})
    assert refresh_after_logout.status_code == 401


async def test_logout_is_idempotent_for_an_already_invalid_token(client: AsyncClient) -> None:
    response = await client.post("/auth/logout", json={"refresh_token": "not-a-real-token"})
    assert response.status_code == 204


async def test_logout_all_revokes_every_session(client: AsyncClient) -> None:
    await client.post(
        "/auth/signup", json={"email": "multisession@example.com", "password": VALID_PASSWORD}
    )
    session_a = await client.post(
        "/auth/login", json={"email": "multisession@example.com", "password": VALID_PASSWORD}
    )
    session_b = await client.post(
        "/auth/login", json={"email": "multisession@example.com", "password": VALID_PASSWORD}
    )
    access_token_a = session_a.json()["access_token"]
    refresh_token_a = session_a.json()["refresh_token"]
    refresh_token_b = session_b.json()["refresh_token"]

    logout_all_response = await client.post(
        "/auth/logout-all", headers={"Authorization": f"Bearer {access_token_a}"}
    )
    assert logout_all_response.status_code == 204

    for token in (refresh_token_a, refresh_token_b):
        response = await client.post("/auth/refresh", json={"refresh_token": token})
        assert response.status_code == 401
