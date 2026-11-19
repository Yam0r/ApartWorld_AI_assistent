from datetime import date, datetime

import pytest

from apartworld_ai_assistent.availability import Verdict, check_unit
from apartworld_ai_assistent.models import BookedPeriod, Unit

NOW = datetime(2026, 11, 19, 12, 0)

# Default request: 3 nights, 10-13 March 2027, 2 guests.
CHECK_IN = date(2027, 3, 10)
CHECK_OUT = date(2027, 3, 13)


def make_unit(**overrides) -> Unit:
    data = dict(
        id="u1", name="Test", district="Karlín",
        max_guests=4, min_nights=3, price_per_night_czk=2000,
        calendar_updated_at=datetime(2026, 11, 19, 10, 0),
    )
    data.update(overrides)
    return Unit(**data)


def booked(check_in: date, check_out: date) -> BookedPeriod:
    return BookedPeriod(check_in=check_in, check_out=check_out)


def check(unit: Unit, check_in=CHECK_IN, check_out=CHECK_OUT, guests=2):
    return check_unit(unit, check_in, check_out, guests=guests, now=NOW)


def test_unit_is_created():
    unit = make_unit()
    assert unit.max_guests == 4


# --- happy path ---

def test_fitting_unit_is_available():
    result = check(make_unit())
    assert result.verdict == Verdict.AVAILABLE
    assert result.unit_id == "u1"


# --- 1. capacity ---

def test_too_many_guests_rejected():
    result = check(make_unit(max_guests=2), guests=4)
    assert result.verdict == Verdict.REJECTED
    assert result.reason == "capacity"


def test_guests_equal_to_capacity_available():
    result = check(make_unit(max_guests=4), guests=4)
    assert result.verdict == Verdict.AVAILABLE


# --- 2. minimum stay ---

def test_stay_shorter_than_minimum_rejected():
    # 3 nights requested, unit requires 5
    result = check(make_unit(min_nights=5))
    assert result.verdict == Verdict.REJECTED
    assert result.reason == "min_nights=5"


def test_stay_equal_to_minimum_available():
    result = check(make_unit(min_nights=3))
    assert result.verdict == Verdict.AVAILABLE


# --- 3. overlap with existing bookings ---

def test_overlapping_booking_rejected():
    unit = make_unit(booked=[booked(date(2027, 3, 11), date(2027, 3, 14))])
    result = check(unit)
    assert result.verdict == Verdict.REJECTED
    assert result.reason == "booked"


def test_booking_inside_request_rejected():
    unit = make_unit(booked=[booked(date(2027, 3, 11), date(2027, 3, 12))])
    assert check(unit).verdict == Verdict.REJECTED


def test_request_inside_booking_rejected():
    unit = make_unit(booked=[booked(date(2027, 3, 1), date(2027, 3, 31))])
    assert check(unit).verdict == Verdict.REJECTED


def test_new_guest_arrives_on_previous_checkout_day_available():
    # previous guest leaves 10 March, new guest arrives 10 March
    unit = make_unit(booked=[booked(date(2027, 3, 7), date(2027, 3, 10))])
    assert check(unit).verdict == Verdict.AVAILABLE


def test_guest_leaves_on_next_checkin_day_available():
    # our guest leaves 13 March, next guest arrives 13 March
    unit = make_unit(booked=[booked(date(2027, 3, 13), date(2027, 3, 16))])
    assert check(unit).verdict == Verdict.AVAILABLE


def test_one_of_several_bookings_overlaps_rejected():
    unit = make_unit(booked=[
        booked(date(2027, 2, 1), date(2027, 2, 5)),
        booked(date(2027, 3, 12), date(2027, 3, 15)),
    ])
    assert check(unit).verdict == Verdict.REJECTED


# --- 4. stale calendar ---

def test_stale_calendar_goes_to_manual_review():
    unit = make_unit(calendar_updated_at=datetime(2026, 11, 17, 12, 0))  # 2 days old
    result = check(unit)
    assert result.verdict == Verdict.MANUAL_REVIEW
    assert result.reason == "stale_calendar"


def test_calendar_exactly_24h_old_still_trusted():
    unit = make_unit(calendar_updated_at=datetime(2026, 11, 18, 12, 0))
    assert check(unit).verdict == Verdict.AVAILABLE


# --- 5. unknown price ---

def test_unknown_price_goes_to_manual_review():
    result = check(make_unit(price_per_night_czk=None))
    assert result.verdict == Verdict.MANUAL_REVIEW
    assert result.reason == "unknown_price"


# --- order of checks ---

def test_hard_rejection_wins_over_manual_review():
    # does not fit anyway, so no point asking a human about the stale calendar
    unit = make_unit(
        max_guests=2,
        calendar_updated_at=datetime(2026, 11, 1, 12, 0),
        price_per_night_czk=None,
    )
    result = check(unit, guests=4)
    assert result.verdict == Verdict.REJECTED
    assert result.reason == "capacity"


# --- invalid input ---

def test_checkout_before_checkin_raises():
    with pytest.raises(ValueError):
        check(make_unit(), check_in=CHECK_OUT, check_out=CHECK_IN)


def test_zero_nights_raises():
    with pytest.raises(ValueError):
        check(make_unit(), check_in=CHECK_IN, check_out=CHECK_IN)


def test_zero_guests_raises():
    with pytest.raises(ValueError):
        check(make_unit(), guests=0)
