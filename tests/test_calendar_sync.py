from datetime import date, datetime

import pytest
from fastapi.testclient import TestClient

from apartworld_ai_assistent.api import create_app
from apartworld_ai_assistent.availability import Verdict, check_unit
from apartworld_ai_assistent.calendar_sync import CalendarEvent, CalendarSync, EventStatus
from apartworld_ai_assistent.inventory import load_inventory

NOW = datetime(2026, 11, 19, 12, 0)


@pytest.fixture
def inventory():
    return load_inventory()  # fresh copy: webhooks change it


@pytest.fixture
def sync(inventory):
    return CalendarSync(inventory, now=lambda: NOW)


def event(event_id="evt-1", type="booking_created", listing_id="direct-hidden-garden-residence",
          check_in="2027-03-10", check_out="2027-03-13"):
    return CalendarEvent(event_id=event_id, type=type, listing_id=listing_id,
                         check_in=check_in, check_out=check_out)


def unit(inventory, unit_id):
    return next(u for u in inventory.units if u.id == unit_id)


def is_free(inventory, unit_id, check_in=date(2027, 3, 10), check_out=date(2027, 3, 13)):
    result = check_unit(unit(inventory, unit_id), check_in, check_out, guests=2, now=NOW)
    return result.verdict == Verdict.AVAILABLE


def test_booking_blocks_the_unit(sync, inventory):
    assert is_free(inventory, "hidden-garden-residence")
    result = sync.apply(event())
    assert result.status == EventStatus.APPLIED
    assert result.unit_id == "hidden-garden-residence"
    assert not is_free(inventory, "hidden-garden-residence")


def test_repeated_webhook_changes_nothing(sync, inventory):
    sync.apply(event())
    second = sync.apply(event())
    assert second.status == EventStatus.DUPLICATE
    assert len(unit(inventory, "hidden-garden-residence").booked) == 1


def test_booking_on_one_listing_blocks_the_physical_unit(sync, inventory):
    # booked on Booking.com as "River Deluxe 2" -> River Deluxe Suite is taken for direct too
    sync.apply(event(listing_id="booking-river-deluxe-2"))
    assert not is_free(inventory, "river-deluxe-suite")


def test_same_address_different_unit_is_not_blocked(sync, inventory):
    sync.apply(event(listing_id="direct-milady-horakove-5"))
    assert not is_free(inventory, "milady-horakove-5")
    assert is_free(inventory, "milady-horakove-6")


def test_parallel_ota_sale_is_flagged_as_conflict(sync, inventory):
    sync.apply(event(event_id="evt-direct", listing_id="direct-river-deluxe-suite"))
    second = sync.apply(event(event_id="evt-ota", listing_id="booking-river-deluxe"))
    assert second.status == EventStatus.CONFLICT
    assert "2027-03-10" in second.detail
    assert [r.event_id for r in sync.log(only_alerts=True)] == ["evt-ota"]


def test_checkout_day_booking_is_not_a_conflict(sync):
    sync.apply(event(event_id="a", check_in="2027-03-10", check_out="2027-03-13"))
    assert sync.apply(event(event_id="b", check_in="2027-03-13", check_out="2027-03-15")).status \
        == EventStatus.APPLIED


def test_cancellation_frees_the_unit(sync, inventory):
    sync.apply(event(event_id="create"))
    result = sync.apply(event(event_id="cancel", type="booking_cancelled"))
    assert result.status == EventStatus.APPLIED
    assert is_free(inventory, "hidden-garden-residence")


def test_cancellation_of_unknown_booking(sync):
    assert sync.apply(event(type="booking_cancelled")).status == EventStatus.NOT_FOUND


def test_unknown_listing_is_an_alert(sync):
    result = sync.apply(event(listing_id="airbnb-ghost"))
    assert result.status == EventStatus.UNKNOWN_LISTING
    assert sync.log(only_alerts=True)[0].event_id == "evt-1"


def test_webhook_refreshes_calendar_timestamp(sync, inventory):
    later = datetime(2026, 11, 25, 9, 0)
    CalendarSync(inventory, now=lambda: later).apply(event())
    assert unit(inventory, "hidden-garden-residence").calendar_updated_at == later


def test_bad_dates_rejected():
    with pytest.raises(ValueError):
        event(check_in="2027-03-13", check_out="2027-03-10")


# --- through the HTTP API ---

class NoLlm:
    def extract(self, *args):
        raise AssertionError("LLM not needed for webhooks")


def test_webhook_endpoint_and_alert_log(inventory):
    client = TestClient(create_app(NoLlm(), inventory, now=lambda: NOW))
    body = {"event_id": "evt-9", "type": "booking_created", "listing_id": "booking-river-deluxe",
            "check_in": "2027-03-10", "check_out": "2027-03-13"}
    assert client.post("/webhooks/calendar", json=body).json()["status"] == "applied"
    assert client.post("/webhooks/calendar", json=body).json()["status"] == "duplicate"
    assert len(client.get("/calendar-events").json()) == 2
    assert client.get("/calendar-events", params={"alerts_only": True}).json() == []
    bad = {**body, "event_id": "evt-10", "check_out": "2027-03-01"}
    assert client.post("/webhooks/calendar", json=bad).status_code == 422
