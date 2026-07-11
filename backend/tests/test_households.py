import uuid

from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import create_access_token
from app.models.household import Household
from app.models.user import User


async def _create_user_and_household(
    db_session: AsyncSession, email: str
) -> tuple[User, Household]:
    user = User(email=email, password_hash=None, is_verified=True)
    db_session.add(user)
    await db_session.flush()

    household = Household(owner_id=user.id, name="Test Home")
    db_session.add(household)
    await db_session.commit()
    await db_session.refresh(household)
    return user, household


async def test_owner_can_fetch_own_household(client: AsyncClient, db_session: AsyncSession) -> None:
    user, household = await _create_user_and_household(db_session, "owner@example.com")
    access_token = create_access_token(user.id)

    response = await client.get(
        f"/households/{household.id}", headers={"Authorization": f"Bearer {access_token}"}
    )
    assert response.status_code == 200
    assert response.json()["id"] == str(household.id)


async def test_other_user_gets_404_not_403(client: AsyncClient, db_session: AsyncSession) -> None:
    _owner, household = await _create_user_and_household(db_session, "owner2@example.com")

    intruder = User(email="intruder@example.com", password_hash=None, is_verified=True)
    db_session.add(intruder)
    await db_session.commit()
    await db_session.refresh(intruder)

    access_token = create_access_token(intruder.id)
    response = await client.get(
        f"/households/{household.id}", headers={"Authorization": f"Bearer {access_token}"}
    )
    assert response.status_code == 404


async def test_nonexistent_household_is_404(client: AsyncClient, db_session: AsyncSession) -> None:
    user = User(email="noone@example.com", password_hash=None, is_verified=True)
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    access_token = create_access_token(user.id)

    response = await client.get(
        f"/households/{uuid.uuid4()}", headers={"Authorization": f"Bearer {access_token}"}
    )
    assert response.status_code == 404


async def test_household_requires_auth(client: AsyncClient) -> None:
    response = await client.get(f"/households/{uuid.uuid4()}")
    assert response.status_code == 401
