import uuid
from datetime import date
from typing import TYPE_CHECKING

from sqlalchemy import Date, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, SoftDeleteMixin, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.household import Household


class SavingsEvent(UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin, Base):
    """Records realized savings attributed to a recommendation, for longitudinal tracking."""

    __tablename__ = "savings_events"

    household_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("households.id"), nullable=False, index=True
    )
    recommendation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("recommendations.id"), nullable=True
    )
    observed_month: Mapped[date] = mapped_column(Date, nullable=False)
    savings_paise: Mapped[int] = mapped_column(Integer, nullable=False)
    co2_avoided_kg: Mapped[float] = mapped_column(nullable=False)

    household: Mapped["Household"] = relationship()
