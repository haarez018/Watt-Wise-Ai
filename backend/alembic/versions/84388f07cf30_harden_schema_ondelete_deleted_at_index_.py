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

# (new explicit name, table, column, ref table, ref column, ondelete)
_FK_CHANGES = [
    ("fk_households_owner_id_users", "households", "owner_id", "users", "id", "CASCADE"),
    ("fk_bills_household_id_households", "bills", "household_id", "households", "id", "CASCADE"),
    (
        "fk_appliances_household_id_households",
        "appliances",
        "household_id",
        "households",
        "id",
        "CASCADE",
    ),
    (
        "fk_recommendations_household_id_households",
        "recommendations",
        "household_id",
        "households",
        "id",
        "CASCADE",
    ),
    (
        "fk_savings_events_household_id_households",
        "savings_events",
        "household_id",
        "households",
        "id",
        "CASCADE",
    ),
    (
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


def _existing_fk_name(inspector: sa.Inspector, table: str, column: str, ref_table: str) -> str:
    """Finds the (likely DB-default-assigned) name of an existing FK constraint.

    The original migration created these FKs inline without an explicit name, so
    Postgres assigned its own default — this looks it up rather than guessing it,
    since the exact default naming isn't something to hard-code and trust blindly.
    """
    for fk in inspector.get_foreign_keys(table):
        if fk["constrained_columns"] == [column] and fk["referred_table"] == ref_table:
            name = fk["name"]
            if name is None:
                raise RuntimeError(f"FK on {table}.{column} -> {ref_table} has no name to drop")
            return name
    raise RuntimeError(f"No existing FK found on {table}.{column} -> {ref_table}")


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    # Resolve every old constraint name up front, before any DDL in this migration
    # runs, so later drops/creates can't affect an inspector still in use.
    resolved = [
        (
            _existing_fk_name(inspector, table, column, ref_table),
            new_name,
            table,
            column,
            ref_table,
            ref_column,
            ondelete,
        )
        for new_name, table, column, ref_table, ref_column, ondelete in _FK_CHANGES
    ]

    for old_name, new_name, table, column, ref_table, ref_column, ondelete in resolved:
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

    for new_name, table, column, ref_table, ref_column, _ondelete in _FK_CHANGES:
        op.drop_constraint(new_name, table, type_="foreignkey")
        op.create_foreign_key(None, table, ref_table, [column], [ref_column])
