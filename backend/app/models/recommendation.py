import uuid
from datetime import date
from typing import TYPE_CHECKING

from sqlalchemy import Date, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, SoftDeleteMixin, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.household import Household


class Recommendation(UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin, Base):
    """A ranked, quantified action surfaced to a household."""

    __tablename__ = "recommendations"

    household_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("households.id"), nullable=False, index=True
    )
    category: Mapped[str] = mapped_column(String(50), nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    explanation: Mapped[str] = mapped_column(Text, nullable=False)
    estimated_savings_paise_per_month: Mapped[int] = mapped_column(Integer, nullable=False)
    estimated_co2_kg_per_year: Mapped[float] = mapped_column(nullable=False)
    calculation_method: Mapped[str] = mapped_column(Text, nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="active")
    generated_for_date: Mapped[date] = mapped_column(Date, nullable=False)

    household: Mapped["Household"] = relationship()
