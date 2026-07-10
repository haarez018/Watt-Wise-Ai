import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Date, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, SoftDeleteMixin, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.household import Household


class Bill(UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin, Base):
    """A single monthly bill. Monetary values are stored in paise; energy in Wh."""

    __tablename__ = "bills"

    household_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("households.id"), nullable=False, index=True
    )
    billing_period_start: Mapped[Date] = mapped_column(Date, nullable=False)
    billing_period_end: Mapped[Date] = mapped_column(Date, nullable=False)
    units_consumed_wh: Mapped[int] = mapped_column(Integer, nullable=False)
    amount_paise: Mapped[int] = mapped_column(Integer, nullable=False)
    source: Mapped[str] = mapped_column(String(20), nullable=False, default="manual")

    household: Mapped["Household"] = relationship(back_populates="bills")
