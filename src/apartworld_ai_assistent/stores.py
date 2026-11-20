"""Storage interfaces and in-memory implementations.

The in-memory versions keep unit tests fast; db.py has the PostgreSQL ones.
Both pass the same contract tests (tests/test_stores.py).
"""

import threading
from typing import Protocol

from apartworld_ai_assistent.calendar_sync import CalendarEvent, EventResult, StoredEvent
from apartworld_ai_assistent.pipeline import ProcessedRequest, RequestStatus, Resolution


class NotInQueueError(ValueError):
    """Resolving something that is not an open review item."""


class RequestStore(Protocol):
    def add(self, item: ProcessedRequest) -> None: ...
    def get(self, item_id: str) -> ProcessedRequest | None: ...
    def list(self, status: RequestStatus | None = None) -> list[ProcessedRequest]: ...
    def resolve(self, item_id: str, resolution: Resolution) -> ProcessedRequest:
        """Close a review item once. KeyError if unknown, NotInQueueError otherwise."""
        ...


class EventStore(Protocol):
    def save(self, event: CalendarEvent, result: EventResult) -> bool:
        """Store the first delivery of an event. False if event_id is already stored."""
        ...
    def get(self, event_id: str) -> StoredEvent | None: ...
    def mark_duplicate(self, event_id: str) -> None: ...
    def all(self) -> list[StoredEvent]:
        """Every stored event in arrival order (used to rebuild calendars)."""
        ...


def check_resolvable(item: ProcessedRequest | None, item_id: str) -> ProcessedRequest:
    if item is None:
        raise KeyError(item_id)
    if item.status != RequestStatus.MANUAL_REVIEW:
        raise NotInQueueError("request is not in the review queue")
    if item.resolution is not None:
        raise NotInQueueError("already resolved")
    return item


class InMemoryRequestStore:
    def __init__(self):
        self._items: dict[str, ProcessedRequest] = {}
        self._lock = threading.Lock()

    def add(self, item: ProcessedRequest) -> None:
        with self._lock:
            self._items[item.id] = item

    def get(self, item_id: str) -> ProcessedRequest | None:
        return self._items.get(item_id)

    def list(self, status: RequestStatus | None = None) -> list[ProcessedRequest]:
        with self._lock:
            items = list(self._items.values())
        if status:
            items = [i for i in items if i.status == status]
        return sorted(items, key=lambda i: (i.received_at, i.id))

    def resolve(self, item_id: str, resolution: Resolution) -> ProcessedRequest:
        with self._lock:
            item = check_resolvable(self._items.get(item_id), item_id)
            updated = item.model_copy(update={"resolution": resolution})
            self._items[item_id] = updated
            return updated


class InMemoryEventStore:
    def __init__(self):
        self._events: dict[str, StoredEvent] = {}  # insertion order = arrival order
        self._lock = threading.Lock()

    def save(self, event: CalendarEvent, result: EventResult) -> bool:
        with self._lock:
            if event.event_id in self._events:
                return False
            self._events[event.event_id] = StoredEvent(event=event, result=result)
            return True

    def get(self, event_id: str) -> StoredEvent | None:
        return self._events.get(event_id)

    def mark_duplicate(self, event_id: str) -> None:
        with self._lock:
            stored = self._events[event_id]
            self._events[event_id] = stored.model_copy(update={"duplicates": stored.duplicates + 1})

    def all(self) -> list[StoredEvent]:
        with self._lock:
            return list(self._events.values())
