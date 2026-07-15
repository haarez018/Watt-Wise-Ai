from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db_session, get_owned_household
from app.core.model_registry import model_registry
from app.models.bill import Bill
from app.models.household import Household
from app.schemas.forecast import HouseholdForecast, PredictionInterval80
from app.schemas.household import HouseholdRead
from app.services.forecast import InsufficientHistoryError, generate_forecast

router = APIRouter(prefix="/households", tags=["households"])


@router.get("/{household_id}", response_model=HouseholdRead)
async def get_household(household: Household = Depends(get_owned_household)) -> Household:
    return household


@router.get("/{household_id}/forecast", response_model=HouseholdForecast)
async def get_household_forecast(
    household: Household = Depends(get_owned_household),
    db: AsyncSession = Depends(get_db_session),
) -> HouseholdForecast:
    if not model_registry.is_ready or model_registry.forecaster is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Forecasting model is not currently available.",
        )

    result = await db.execute(
        select(Bill).where(Bill.household_id == household.id, Bill.deleted_at.is_(None))
    )
    bills = list(result.scalars().all())

    try:
        forecast = generate_forecast(household, bills, model_registry.forecaster)
    except InsufficientHistoryError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return HouseholdForecast(
        predicted_units_wh=forecast.predicted_units_wh,
        predicted_amount_paise=forecast.predicted_amount_paise,
        prediction_interval_80=PredictionInterval80(
            low=forecast.prediction_interval_low_wh,
            high=forecast.prediction_interval_high_wh,
        ),
        model_version=forecast.model_version,
        generated_at=forecast.generated_at,
    )
