from datetime import datetime

from pydantic import BaseModel


class PredictionInterval80(BaseModel):
    low: int
    high: int


class HouseholdForecast(BaseModel):
    predicted_units_wh: int
    predicted_amount_paise: int
    prediction_interval_80: PredictionInterval80
    model_version: str
    generated_at: datetime
