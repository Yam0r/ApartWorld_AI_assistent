from datetime import date, datetime
from apartworld_ai_assistent.models import Unit, BookedPeriod

def make_unit(**overrides) -> Unit:
    data = dict(
        id="u1", name="Test", district="Karlín",
        max_guests=4, min_nights=3, price_per_night_czk=2000,
        calendar_updated_at=datetime(2026, 11, 19, 10, 0),
    )
    data.update(overrides)
    return Unit(**data)

def test_unit_is_created():
    unit = make_unit()
    assert unit.max_guests == 4