from datetime import date, datetime, timedelta
from enum import StrEnum

from pydantic import BaseModel

from apartworld_ai_assistent.models import Unit

MAX_CALENDAR_AGE = timedelta(hours=24)


class Verdict(StrEnum):
    AVAILABLE = "available"
    REJECTED = "rejected"
    MANUAL_REVIEW = "manual_review"


class CheckResult(BaseModel):
    unit_id: str
    verdict: Verdict
    reason: str


def periods_overlap(a_in: date, a_out: date, b_in: date, b_out: date) -> bool:
    """Half-open periods [in, out): check-out day is free for a new check-in."""
    return a_in < b_out and b_in < a_out


def check_unit(
    unit: Unit,
    check_in: date,
    check_out: date,
    guests: int,
    now: datetime,
) -> CheckResult:
    """Decide whether a unit can be offered for the requested stay.

    Hard rejections are checked first: if the unit does not fit anyway,
    there is no point sending it to manual review.
    """
    if check_out <= check_in:
        raise ValueError("check_out must be after check_in")
    if guests < 1:
        raise ValueError("guests must be at least 1")

    def result(verdict: Verdict, reason: str) -> CheckResult:
        return CheckResult(unit_id=unit.id, verdict=verdict, reason=reason)

    nights = (check_out - check_in).days

    if guests > unit.max_guests:
        return result(Verdict.REJECTED, "capacity")

    if nights < unit.min_nights:
        return result(Verdict.REJECTED, f"min_nights={unit.min_nights}")

    for period in unit.booked:
        if periods_overlap(check_in, check_out, period.check_in, period.check_out):
            return result(Verdict.REJECTED, "booked")

    if now - unit.calendar_updated_at > MAX_CALENDAR_AGE:
        return result(Verdict.MANUAL_REVIEW, "stale_calendar")

    if unit.price_per_night_czk is None:
        return result(Verdict.MANUAL_REVIEW, "unknown_price")

    return result(Verdict.AVAILABLE, "ok")
