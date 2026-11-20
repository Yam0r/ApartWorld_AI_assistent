"""Initial schema: requests and calendar events.

Revision ID: 0001
Revises:
Create Date: 2026-11-19
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "requests",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("received_at", sa.DateTime, nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("resolved_at", sa.DateTime, nullable=True),
        sa.Column("payload", JSONB, nullable=False),
    )
    op.create_index("ix_requests_status", "requests", ["status"])

    op.create_table(
        "calendar_events",
        sa.Column("seq", sa.BigInteger, sa.Identity(), primary_key=True),
        sa.Column("event_id", sa.String(128), nullable=False),
        sa.Column("received_at", sa.DateTime, nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("unit_id", sa.String(64), nullable=True),
        sa.Column("event", JSONB, nullable=False),
        sa.Column("result", JSONB, nullable=False),
        sa.Column("duplicates", sa.Integer, nullable=False, server_default="0"),
        sa.Column("last_duplicate_at", sa.DateTime, nullable=True),
        sa.UniqueConstraint("event_id", name="uq_calendar_events_event_id"),
    )


def downgrade() -> None:
    op.drop_table("calendar_events")
    op.drop_index("ix_requests_status", table_name="requests")
    op.drop_table("requests")
