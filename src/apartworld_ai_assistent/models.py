from datetime import date, datetime

from pydantic import BaseModel


class BookedPeriod(BaseModel):
    check_in: date    # inclusive
    check_out: date   # exclusive


class Unit(BaseModel):
    """A physical apartment. Several listings can point to one unit."""

    id: str
    name: str
    district: str
    max_guests: int
    min_nights: int
    price_per_night_czk: int | None   # None = price unknown -> manual review
    calendar_updated_at: datetime
    booked: list[BookedPeriod] = []

    # Descriptive fields from the public catalog (optional).
    address: str | None = None
    bedrooms: int | None = None
    bathrooms: int | None = None
    area_m2: int | None = None
    price_30d_czk: int | None = None
    check_in_time: str | None = None
    check_out_time: str | None = None
    guesty_id: str | None = None
    synthetic: bool = False           # invented for a test case, not on the website
    note: str | None = None


class Listing(BaseModel):
    """A listing on one channel (website, Booking.com, Airbnb...)."""

    id: str
    unit_id: str
    channel: str
    name: str | None = None
    url: str | None = None
