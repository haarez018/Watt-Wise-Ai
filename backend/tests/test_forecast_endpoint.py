import uuid
from datetime import date

from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.model_registry import model_registry
from app.core.security import create_access_token
from app.models.bill import Bill
from app.models.household import Household
from app.models.user import User


async def _create_user_and_household(
    db_session: AsyncSession, email: str, **household_kwargs: object
) -> tuple[User, Household]:
    user = User(email=email, password_hash=None, is_verified=True)
    db_session.add(user)
    await db_session.flush()

    household = Household(owner_id=user.id, name="Test Home", **household_kwargs)
    db_session.add(household)
    await db_session.commit()
    await db_session.refresh(household)
    return user, household


async def _add_bills(
    db_session: AsyncSession, household: Household, n_months: int, units_wh: int = 150_000
) -> None:
    for i in range(n_months):
        month = 12 - n_months + i + 1  # e.g. 3 months -> Oct, Nov, Dec
        bill = Bill(
            household_id=household.id,
            billing_period_start=date(2025, month, 1),
            billing_period_end=date(2025, month, 28),
            units_consumed_wh=units_wh + i * 1_000,
            amount_paise=20_000,
        )
        db_session.add(bill)
    await db_session.commit()


async def test_forecast_happy_path(client: AsyncClient, db_session: AsyncSession) -> None:
    user, household = await _create_user_and_household(
        db_session,
        "forecast-owner@example.com",
        city="Chennai",
        discom="TNEB",
        occupants=4,
        sanctioned_load_kw=4.0,
    )
    await _add_bills(db_session, household, n_months=3)
    access_token = create_access_token(user.id)

    response = await client.get(
        f"/households/{household.id}/forecast",
        headers={"Authorization": f"Bearer {access_token}"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_units_wh"] > 0
    assert body["predicted_amount_paise"] >= 0
    assert body["prediction_interval_80"]["low"] <= body["predicted_units_wh"]
    assert body["prediction_interval_80"]["high"] >= body["predicted_units_wh"]
    assert body["model_version"] == "forecaster_v1"
    assert "generated_at" in body


async def test_forecast_insufficient_history_is_400(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    user, household = await _create_user_and_household(
        db_session, "insufficient-history@example.com"
    )
    await _add_bills(db_session, household, n_months=2)  # fewer than the required 3
    access_token = create_access_token(user.id)

    response = await client.get(
        f"/households/{household.id}/forecast",
        headers={"Authorization": f"Bearer {access_token}"},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["status"] == 400
    assert "3" in body["detail"]


async def test_forecast_cross_user_is_404(client: AsyncClient, db_session: AsyncSession) -> None:
    _owner, household = await _create_user_and_household(db_session, "forecast-owner2@example.com")
    await _add_bills(db_session, household, n_months=3)

    intruder = User(email="forecast-intruder@example.com", password_hash=None, is_verified=True)
    db_session.add(intruder)
    await db_session.commit()
    await db_session.refresh(intruder)
    access_token = create_access_token(intruder.id)

    response = await client.get(
        f"/households/{household.id}/forecast",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert response.status_code == 404


async def test_forecast_nonexistent_household_is_404(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    user = User(email="forecast-noone@example.com", password_hash=None, is_verified=True)
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    access_token = create_access_token(user.id)

    response = await client.get(
        f"/households/{uuid.uuid4()}/forecast",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert response.status_code == 404


async def test_forecast_requires_auth(client: AsyncClient) -> None:
    response = await client.get(f"/households/{uuid.uuid4()}/forecast")
    assert response.status_code == 401


async def test_forecast_model_missing_is_503(client: AsyncClient, db_session: AsyncSession) -> None:
    user, household = await _create_user_and_household(db_session, "model-missing@example.com")
    await _add_bills(db_session, household, n_months=3)
    access_token = create_access_token(user.id)

    original_forecaster = model_registry.forecaster
    model_registry.forecaster = None
    try:
        response = await client.get(
            f"/households/{household.id}/forecast",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        assert response.status_code == 503
        body = response.json()
        assert body["status"] == 503
    finally:
        model_registry.forecaster = original_forecaster
