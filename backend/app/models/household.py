import uuid
from typing import TYPE_CHECKING, Literal

from sqlalchemy import ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, SoftDeleteMixin, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.appliance import Appliance
    from app.models.bill import Bill
    from app.models.user import User

DiscomCode = Literal["TNEB", "BESCOM", "ADANI", "TATA_POWER", "MSEDCL", "OTHER"]


class Household(UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin, Base):
    __tablename__ = "households"

    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False, default="My Home")
    state: Mapped[str | None] = mapped_column(String(100), nullable=True)
    city: Mapped[str | None] = mapped_column(String(100), nullable=True)
    postal_code: Mapped[str | None] = mapped_column(String(10), nullable=True)
    discom: Mapped[str] = mapped_column(String(50), nullable=False, default="OTHER")
    dwelling_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    occupants: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sanctioned_load_kw: Mapped[float | None] = mapped_column(nullable=True)

    owner: Mapped["User"] = relationship(back_populates="households")
    bills: Mapped[list["Bill"]] = relationship(
        back_populates="household", cascade="all, delete-orphan"
    )
    appliances: Mapped[list["Appliance"]] = relationship(
        back_populates="household", cascade="all, delete-orphan"
    )
