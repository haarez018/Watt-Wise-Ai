"""harden schema: ondelete, deleted_at index, bigint

Revision ID: 84388f07cf30
Revises: 9e7afc6ea408
Create Date: 2026-07-11 18:00:49.190952

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "84388f07cf30"
down_revision: str | None = "9e7afc6ea408"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None

# (old default-named constraint, new explicit name, table, column, ref table, ref column, ondelete)
_FK_CHANGES = [
    (
        "households_owner_id_fkey",
        "fk_households_owner_id_users",
        "households",
        "owner_id",
        "users",
        "id",
        "CASCADE",
    ),
    (
        "bills_household_id_fkey",
        "fk_bills_household_id_households",
        "bills",
        "household_id",
        "households",
        "id",
        "CASCADE",
    ),
    (
        "appliances_household_id_fkey",
        "fk_appliances_household_id_households",
        "appliances",
        "household_id",
        "households",
        "id",
        "CASCADE",
    ),
    (
        "recommendations_household_id_fkey",
        "fk_recommendations_household_id_households",
        "recommendations",
        "household_id",
        "households",
        "id",
        "CASCADE",
    ),
    (
        "savings_events_household_id_fkey",
        "fk_savings_events_household_id_households",
        "savings_events",
        "household_id",
        "households",
        "id",
        "CASCADE",
    ),
    (
        "savings_events_recommendation_id_fkey",
        "fk_savings_events_recommendation_id_recommendations",
        "savings_events",
        "recommendation_id",
        "recommendations",
        "id",
        "SET NULL",
    ),
]

_SOFT_DELETE_TABLES = [
    "users",
    "households",
    "bills",
    "appliances",
    "recommendations",
    "savings_events",
]

_BIGINT_COLUMNS = [
    ("bills", "units_consumed_wh"),
    ("bills", "amount_paise"),
    ("recommendations", "estimated_savings_paise_per_month"),
    ("savings_events", "savings_paise"),
]


def upgrade() -> None:
    for old_name, new_name, table, column, ref_table, ref_column, ondelete in _FK_CHANGES:
        op.drop_constraint(old_name, table, type_="foreignkey")
        op.create_foreign_key(new_name, table, ref_table, [column], [ref_column], ondelete=ondelete)

    for table in _SOFT_DELETE_TABLES:
        op.create_index(f"ix_{table}_deleted_at", table, ["deleted_at"])

    for table, column in _BIGINT_COLUMNS:
        op.alter_column(table, column, type_=sa.BigInteger(), existing_nullable=False)


def downgrade() -> None:
    for table, column in _BIGINT_COLUMNS:
        op.alter_column(table, column, type_=sa.Integer(), existing_nullable=False)

    for table in _SOFT_DELETE_TABLES:
        op.drop_index(f"ix_{table}_deleted_at", table_name=table)

    for old_name, new_name, table, column, ref_table, ref_column, _ondelete in _FK_CHANGES:
        op.drop_constraint(new_name, table, type_="foreignkey")
        op.create_foreign_key(old_name, table, ref_table, [column], [ref_column])
