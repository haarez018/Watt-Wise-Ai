"""initial schema

Revision ID: 9e7afc6ea408
Revises:
Create Date: 2026-07-09 14:37:28.323144

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "9e7afc6ea408"
down_revision: str | None = None
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def _audit_columns() -> list[sa.Column]:
    return [
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    ]


def upgrade() -> None:
    op.create_table(
        "users",
        *_audit_columns(),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=True),
        sa.Column("full_name", sa.String(length=200), nullable=True),
        sa.Column("oauth_provider", sa.String(length=50), nullable=True),
        sa.Column("oauth_subject", sa.String(length=255), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("is_verified", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "households",
        *_audit_columns(),
        sa.Column(
            "owner_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False
        ),
        sa.Column("name", sa.String(length=200), nullable=False, server_default="My Home"),
        sa.Column("state", sa.String(length=100), nullable=True),
        sa.Column("city", sa.String(length=100), nullable=True),
        sa.Column("postal_code", sa.String(length=10), nullable=True),
        sa.Column("discom", sa.String(length=50), nullable=False, server_default="OTHER"),
        sa.Column("dwelling_type", sa.String(length=50), nullable=True),
        sa.Column("occupants", sa.Integer(), nullable=True),
        sa.Column("sanctioned_load_kw", sa.Float(), nullable=True),
    )
    op.create_index("ix_households_owner_id", "households", ["owner_id"])

    op.create_table(
        "bills",
        *_audit_columns(),
        sa.Column(
            "household_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("households.id"),
            nullable=False,
        ),
        sa.Column("billing_period_start", sa.Date(), nullable=False),
        sa.Column("billing_period_end", sa.Date(), nullable=False),
        sa.Column("units_consumed_wh", sa.Integer(), nullable=False),
        sa.Column("amount_paise", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(length=20), nullable=False, server_default="manual"),
    )
    op.create_index("ix_bills_household_id", "bills", ["household_id"])

    op.create_table(
        "appliances",
        *_audit_columns(),
        sa.Column(
            "household_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("households.id"),
            nullable=False,
        ),
        sa.Column("category", sa.String(length=50), nullable=False),
        sa.Column("label", sa.String(length=100), nullable=True),
        sa.Column("quantity", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("star_rating", sa.Integer(), nullable=True),
        sa.Column("age_years", sa.Integer(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
    )
    op.create_index("ix_appliances_household_id", "appliances", ["household_id"])

    op.create_table(
        "recommendations",
        *_audit_columns(),
        sa.Column(
            "household_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("households.id"),
            nullable=False,
        ),
        sa.Column("category", sa.String(length=50), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("explanation", sa.Text(), nullable=False),
        sa.Column("estimated_savings_paise_per_month", sa.Integer(), nullable=False),
        sa.Column("estimated_co2_kg_per_year", sa.Float(), nullable=False),
        sa.Column("calculation_method", sa.Text(), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="active"),
        sa.Column("generated_for_date", sa.Date(), nullable=False),
    )
    op.create_index("ix_recommendations_household_id", "recommendations", ["household_id"])

    op.create_table(
        "savings_events",
        *_audit_columns(),
        sa.Column(
            "household_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("households.id"),
            nullable=False,
        ),
        sa.Column(
            "recommendation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("recommendations.id"),
            nullable=True,
        ),
        sa.Column("observed_month", sa.Date(), nullable=False),
        sa.Column("savings_paise", sa.Integer(), nullable=False),
        sa.Column("co2_avoided_kg", sa.Float(), nullable=False),
    )
    op.create_index("ix_savings_events_household_id", "savings_events", ["household_id"])


def downgrade() -> None:
    op.drop_table("savings_events")
    op.drop_table("recommendations")
    op.drop_table("appliances")
    op.drop_table("bills")
    op.drop_table("households")
    op.drop_table("users")
