import uuid

from pydantic import BaseModel, ConfigDict


class HouseholdRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    owner_id: uuid.UUID
    name: str
    state: str | None
    city: str | None
    postal_code: str | None
    discom: str
    dwelling_type: str | None
    occupants: int | None
    sanctioned_load_kw: float | None
