from datetime import date, datetime

import pytest

from apartworld_ai_assistent.availability import Verdict, check_unit
from apartworld_ai_assistent.inventory import Inventory, load_inventory

# Scenario from data/units.json -> _meta.scenario
NOW = datetime(2026, 11, 19, 12, 0)
CHECK_IN = date(2027, 3, 10)
CHECK_OUT = date(2027, 3, 13)
GUESTS = 4


@pytest.fixture(scope="module")
def inventory() -> Inventory:
    return load_inventory()


def test_test_data_loads(inventory):
    assert len(inventory.units) >= 10
    assert len(inventory.listings) >= len(inventory.units)


def test_every_unit_has_a_listing(inventory):
    for unit in inventory.units:
        assert inventory.listings_for(unit.id), unit.id


def test_river_deluxe_listings_are_one_physical_unit(inventory):
    names = {l.name for l in inventory.listings_for("river-deluxe-suite")}
    assert {"River Deluxe", "River Deluxe 2", "River Deluxe Suite"} <= names


def test_milady_horakove_5_and_6_are_separate_units(inventory):
    units = {u.id: u for u in inventory.units}
    no5, no6 = units["milady-horakove-5"], units["milady-horakove-6"]
    assert no5.address == no6.address
    assert no5.id != no6.id


def test_listing_pointing_to_unknown_unit_is_rejected():
    with pytest.raises(ValueError):
        Inventory(units=[], listings=[{"id": "l1", "unit_id": "ghost", "channel": "direct"}])


EXPECTED = {
    "river-deluxe-suite": (Verdict.AVAILABLE, "ok"),             # checkout on our check-in day
    "old-town-residence": (Verdict.AVAILABLE, "ok"),             # check-in on our check-out day
    "golden-cat": (Verdict.REJECTED, "min_nights=5"),
    "dicaprio-penthouse": (Verdict.REJECTED, "booked"),
    "white-rabbit-residence": (Verdict.MANUAL_REVIEW, "stale_calendar"),
    "london-royal-residence": (Verdict.MANUAL_REVIEW, "unknown_price"),
    "black-dragon-haven": (Verdict.REJECTED, "booked"),
    "luxury-lion-suite": (Verdict.AVAILABLE, "ok"),
    "hidden-garden-residence": (Verdict.AVAILABLE, "ok"),
    "victory-castle-suite": (Verdict.AVAILABLE, "ok"),
    "milady-horakove-5": (Verdict.AVAILABLE, "ok"),
    "milady-horakove-6": (Verdict.AVAILABLE, "ok"),
}


def test_scenario_2_three_night_request(inventory):
    actual = {
        u.id: (r.verdict, r.reason)
        for u in inventory.units
        for r in [check_unit(u, CHECK_IN, CHECK_OUT, guests=GUESTS, now=NOW)]
    }
    assert actual == EXPECTED
