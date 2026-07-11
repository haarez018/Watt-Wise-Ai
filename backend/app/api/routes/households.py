from fastapi import APIRouter, Depends

from app.api.deps import get_owned_household
from app.models.household import Household
from app.schemas.household import HouseholdRead

router = APIRouter(prefix="/households", tags=["households"])


@router.get("/{household_id}", response_model=HouseholdRead)
async def get_household(household: Household = Depends(get_owned_household)) -> Household:
    return household
