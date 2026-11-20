"""One contract, two implementations: every test runs on memory and on PostgreSQL."""

from datetime import datetime, timedelta

import pytest

from apartworld_ai_assistent.calendar_sync import CalendarEvent, EventResult, EventStatus
from apartworld_ai_assistent.pipeline import (
    ProcessedRequest,
    RequestStatus,
    Resolution,
    ResolutionAction,
)
from apartworld_ai_assistent.stores import InMemoryEventStore, InMemoryRequestStore, NotInQueueError

NOW = datetime(2026, 11, 19, 12, 0)


@pytest.fixture(params=["memory", "postgres"])
def request_store(request):
    if request.param == "memory":
        return InMemoryRequestStore()
    from apartworld_ai_assistent.db import SqlRequestStore

    return SqlRequestStore(request.getfixturevalue("pg_engine"))


@pytest.fixture(params=["memory", "postgres"])
def event_store(request):
    if request.param == "memory":
        return InMemoryEventStore()
    from apartworld_ai_assistent.db import SqlEventStore

    return SqlEventStore(request.getfixturevalue("pg_engine"))


def item(item_id="r1", status=RequestStatus.MANUAL_REVIEW, minutes=0) -> ProcessedRequest:
    return ProcessedRequest(
        id=item_id, received_at=NOW + timedelta(minutes=minutes),
        email="Dobrý den, 40 studentů, 10.–13. března", status=status, reason="large_group",
    )


def resolution() -> Resolution:
    return Resolution(action=ResolutionAction.DISMISSED, resolved_by="staff", resolved_at=NOW, note="ok")


# --- requests ---

def test_request_round_trip_keeps_every_field(request_store):
    original = item()
    request_store.add(original)
    assert request_store.get("r1") == original


def test_unknown_request_is_none(request_store):
    assert request_store.get("nope") is None


def test_list_is_ordered_and_filtered(request_store):
    request_store.add(item("late", minutes=5))
    request_store.add(item("early", minutes=1))
    request_store.add(item("ready", status=RequestStatus.READY, minutes=3))
    assert [i.id for i in request_store.list()] == ["early", "ready", "late"]
    assert [i.id for i in request_store.list(RequestStatus.MANUAL_REVIEW)] == ["early", "late"]


def test_resolve_once(request_store):
    request_store.add(item())
    resolved = request_store.resolve("r1", resolution())
    assert resolved.resolution.action == ResolutionAction.DISMISSED
    assert request_store.get("r1").resolution == resolved.resolution
    with pytest.raises(NotInQueueError, match="already resolved"):
        request_store.resolve("r1", resolution())


def test_resolve_only_review_items(request_store):
    request_store.add(item(status=RequestStatus.READY))
    with pytest.raises(NotInQueueError, match="not in the review queue"):
        request_store.resolve("r1", resolution())


def test_resolve_unknown_raises_key_error(request_store):
    with pytest.raises(KeyError):
        request_store.resolve("nope", resolution())


# --- calendar events ---

def event(event_id="evt-1") -> CalendarEvent:
    return CalendarEvent(event_id=event_id, type="booking_created", listing_id="booking-river-deluxe",
                         check_in="2027-03-10", check_out="2027-03-13")


def result(event_id="evt-1") -> EventResult:
    return EventResult(event_id=event_id, status=EventStatus.APPLIED, unit_id="river-deluxe-suite",
                       detail="booking added", received_at=NOW)


def test_event_is_stored_once(event_store):
    assert event_store.save(event(), result()) is True
    assert event_store.save(event(), result()) is False  # unique event_id
    assert len(event_store.all()) == 1


def test_event_round_trip(event_store):
    event_store.save(event(), result())
    stored = event_store.get("evt-1")
    assert stored.event == event()
    assert stored.result == result()
    assert event_store.get("nope") is None


def test_duplicates_are_counted(event_store):
    event_store.save(event(), result())
    event_store.mark_duplicate("evt-1")
    event_store.mark_duplicate("evt-1")
    assert event_store.get("evt-1").duplicates == 2


def test_events_come_back_in_arrival_order(event_store):
    for event_id in ["c", "a", "b"]:
        event_store.save(event(event_id), result(event_id))
    assert [s.event.event_id for s in event_store.all()] == ["c", "a", "b"]
