"""PostgreSQL storage (SQLAlchemy Core + psycopg 3).

Schema is created by Alembic migrations (migrations/), not by this module.
Each document is kept as JSONB next to the columns used for lookups.
"""

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Engine,
    Identity,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    create_engine,
    func,
    insert,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import IntegrityError

from apartworld_ai_assistent.calendar_sync import CalendarEvent, EventResult, StoredEvent
from apartworld_ai_assistent.pipeline import ProcessedRequest, RequestStatus, Resolution
from apartworld_ai_assistent.stores import check_resolvable

metadata = MetaData()

requests_table = Table(
    "requests", metadata,
    Column("id", String(32), primary_key=True),
    Column("received_at", DateTime, nullable=False),
    Column("status", String(32), nullable=False, index=True),
    Column("resolved_at", DateTime, nullable=True),
    Column("payload", JSONB, nullable=False),
)

calendar_events_table = Table(
    "calendar_events", metadata,
    Column("seq", BigInteger, Identity(), primary_key=True),          # arrival order for replay
    Column("event_id", String(128), nullable=False),
    Column("received_at", DateTime, nullable=False),
    Column("status", String(32), nullable=False),
    Column("unit_id", String(64), nullable=True),
    Column("event", JSONB, nullable=False),
    Column("result", JSONB, nullable=False),
    Column("duplicates", Integer, nullable=False, server_default="0"),
    Column("last_duplicate_at", DateTime, nullable=True),
    UniqueConstraint("event_id", name="uq_calendar_events_event_id"),  # idempotency guard
)


def make_engine(url: str) -> Engine:
    return create_engine(url, pool_pre_ping=True)


class SqlRequestStore:
    def __init__(self, engine: Engine):
        self._engine = engine

    def add(self, item: ProcessedRequest) -> None:
        with self._engine.begin() as conn:
            conn.execute(insert(requests_table).values(
                id=item.id, received_at=item.received_at, status=item.status.value,
                payload=item.model_dump(mode="json"),
            ))

    def get(self, item_id: str) -> ProcessedRequest | None:
        with self._engine.connect() as conn:
            payload = conn.scalar(select(requests_table.c.payload).where(requests_table.c.id == item_id))
        return ProcessedRequest.model_validate(payload) if payload else None

    def list(self, status: RequestStatus | None = None) -> list[ProcessedRequest]:
        query = select(requests_table.c.payload).order_by(requests_table.c.received_at, requests_table.c.id)
        if status:
            query = query.where(requests_table.c.status == status.value)
        with self._engine.connect() as conn:
            return [ProcessedRequest.model_validate(p) for p in conn.scalars(query)]

    def resolve(self, item_id: str, resolution: Resolution) -> ProcessedRequest:
        with self._engine.begin() as conn:
            # Row lock: two staff members cannot close the same item at once.
            payload = conn.scalar(
                select(requests_table.c.payload).where(requests_table.c.id == item_id).with_for_update()
            )
            item = check_resolvable(ProcessedRequest.model_validate(payload) if payload else None, item_id)
            updated = item.model_copy(update={"resolution": resolution})
            conn.execute(
                update(requests_table).where(requests_table.c.id == item_id)
                .values(resolved_at=resolution.resolved_at, payload=updated.model_dump(mode="json"))
            )
        return updated


class SqlEventStore:
    def __init__(self, engine: Engine):
        self._engine = engine

    def save(self, event: CalendarEvent, result: EventResult) -> bool:
        try:
            with self._engine.begin() as conn:
                conn.execute(insert(calendar_events_table).values(
                    event_id=event.event_id, received_at=result.received_at,
                    status=result.status.value, unit_id=result.unit_id,
                    event=event.model_dump(mode="json"), result=result.model_dump(mode="json"),
                ))
            return True
        except IntegrityError:  # unique event_id: someone stored it first
            return False

    def get(self, event_id: str) -> StoredEvent | None:
        with self._engine.connect() as conn:
            row = conn.execute(
                select(calendar_events_table).where(calendar_events_table.c.event_id == event_id)
            ).mappings().first()
        return _stored(row) if row else None

    def mark_duplicate(self, event_id: str) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                update(calendar_events_table).where(calendar_events_table.c.event_id == event_id)
                .values(duplicates=calendar_events_table.c.duplicates + 1, last_duplicate_at=func.now())
            )

    def all(self) -> list[StoredEvent]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                select(calendar_events_table).order_by(calendar_events_table.c.seq)
            ).mappings().all()
        return [_stored(r) for r in rows]


def _stored(row) -> StoredEvent:
    return StoredEvent(
        event=CalendarEvent.model_validate(row["event"]),
        result=EventResult.model_validate(row["result"]),
        duplicates=row["duplicates"],
    )
