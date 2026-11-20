"""Calendar webhooks: keep unit calendars in sync with channels (Guesty, OTA).

Spec cases covered:
- the same webhook delivered twice must not change anything twice;
- several listings of one apartment block the same physical unit;
- a booking that overlaps an existing one (parallel OTA sale) is recorded
  but flagged for a person.
"""

import threading
from collections.abc import Callable
from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from apartworld_ai_assistent.availability import periods_overlap
from apartworld_ai_assistent.inventory import Inventory
from apartworld_ai_assistent.models import BookedPeriod


class EventType(StrEnum):
    BOOKING_CREATED = "booking_created"
    BOOKING_CANCELLED = "booking_cancelled"


class CalendarEvent(BaseModel):
    event_id: str = Field(min_length=1, examples=["evt-001"])
    type: EventType
    listing_id: str = Field(examples=["booking-river-deluxe-2"])
    check_in: date = Field(examples=["2027-03-10"])
    check_out: date = Field(examples=["2027-03-13"])

    @model_validator(mode="after")
    def dates_in_order(self) -> CalendarEvent:
        if self.check_out <= self.check_in:
            raise ValueError("check_out must be after check_in")
        return self


class EventStatus(StrEnum):
    APPLIED = "applied"                  # calendar updated
    DUPLICATE = "duplicate"              # same event_id seen before: nothing changed
    CONFLICT = "conflict"                # applied, but overlaps another booking: staff alert
    NOT_FOUND = "not_found"              # cancellation for a booking we do not have
    UNKNOWN_LISTING = "unknown_listing"  # listing not in the registry: staff alert


class EventResult(BaseModel):
    event_id: str
    status: EventStatus
    unit_id: str | None = None
    detail: str | None = None
    received_at: datetime
    duplicates: int = 0  # in the log: how many times the same event came again


class StoredEvent(BaseModel):
    event: CalendarEvent
    result: EventResult
    duplicates: int = 0


class CalendarSync:
    """Applies webhook events to unit calendars and keeps the event journal.

    The journal lives in an EventStore (memory or PostgreSQL). On start the
    stored events are replayed over units.json, so bookings survive restarts.
    """

    def __init__(self, inventory: Inventory, now: Callable[[], datetime], store=None):
        if store is None:
            from apartworld_ai_assistent.stores import InMemoryEventStore

            store = InMemoryEventStore()
        self._store = store
        self._now = now
        self._units = {u.id: u for u in inventory.units}
        self._listings = {li.id: li for li in inventory.listings}
        self._lock = threading.Lock()
        for stored in self._store.all():
            _, change = self._evaluate(stored.event, stored.result.received_at)
            if change:
                change()

    def apply(self, event: CalendarEvent) -> EventResult:
        with self._lock:
            moment = self._now()
            existing = self._store.get(event.event_id)
            if existing is None:
                result, change = self._evaluate(event, moment)
                # Saved first: the unique event_id in the store is the real guard.
                if self._store.save(event, result):
                    if change:
                        change()
                    return result
                existing = self._store.get(event.event_id)

            self._store.mark_duplicate(event.event_id)
            return EventResult(
                event_id=event.event_id, status=EventStatus.DUPLICATE,
                unit_id=existing.result.unit_id if existing else None,
                detail="already processed", received_at=moment,
            )

    def _evaluate(self, event: CalendarEvent, moment: datetime):
        """Decide the outcome without touching calendars; return (result, change)."""

        def done(status: EventStatus, unit_id: str | None = None, detail: str | None = None):
            return EventResult(event_id=event.event_id, status=status, unit_id=unit_id,
                               detail=detail, received_at=moment)

        listing = self._listings.get(event.listing_id)
        if listing is None:
            detail = f"listing {event.listing_id} not in registry"
            return done(EventStatus.UNKNOWN_LISTING, detail=detail), None
        unit = self._units[listing.unit_id]

        if event.type == EventType.BOOKING_CANCELLED:
            for period in unit.booked:
                if (period.check_in, period.check_out) == (event.check_in, event.check_out):
                    def remove(period=period):
                        unit.booked.remove(period)
                        unit.calendar_updated_at = moment
                    return done(EventStatus.APPLIED, unit.id, "booking removed"), remove
            return done(EventStatus.NOT_FOUND, unit.id, "no booking with these dates"), None

        overlapping = [
            p for p in unit.booked
            if periods_overlap(event.check_in, event.check_out, p.check_in, p.check_out)
        ]

        def add():
            unit.booked.append(BookedPeriod(check_in=event.check_in, check_out=event.check_out))
            unit.calendar_updated_at = moment

        if overlapping:
            dates = ", ".join(f"{p.check_in}–{p.check_out}" for p in overlapping)
            return done(EventStatus.CONFLICT, unit.id, f"overlaps existing booking {dates}"), add
        return done(EventStatus.APPLIED, unit.id, "booking added"), add

    def log(self, only_alerts: bool = False) -> list[EventResult]:
        """First delivery of every event, with how many times it was repeated."""
        alerts = {EventStatus.CONFLICT, EventStatus.UNKNOWN_LISTING}
        results = [s.result.model_copy(update={"duplicates": s.duplicates}) for s in self._store.all()]
        return [r for r in results if r.status in alerts] if only_alerts else results
