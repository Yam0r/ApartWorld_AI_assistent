from datetime import date, datetime

import pytest

from apartworld_ai_assistent.inventory import Inventory, load_inventory
from apartworld_ai_assistent.search import rank_by_preferences, search

NOW = datetime(2026, 11, 19, 12, 0)
CHECK_IN = date(2027, 3, 10)
CHECK_OUT = date(2027, 3, 13)


def fresh_unit(unit_id: str, price: int | None = 1000, **overrides) -> dict:
    data = dict(
        id=unit_id, name=unit_id, district="Prague 1",
        max_guests=4, min_nights=3, price_per_night_czk=price,
        calendar_updated_at=datetime(2026, 11, 19, 10, 0),
    )
    data.update(overrides)
    return data


def listing(listing_id: str, unit_id: str, channel: str = "direct") -> dict:
    return dict(id=listing_id, unit_id=unit_id, channel=channel)


def run(inventory: Inventory, guests: int = 2):
    return search(inventory, CHECK_IN, CHECK_OUT, guests=guests, now=NOW)


# --- unit tests on small hand-made inventories ---

def test_several_listings_of_one_unit_give_one_offer():
    inv = Inventory(
        units=[fresh_unit("river")],
        listings=[
            listing("booking-river", "river", "booking"),
            listing("booking-river-2", "river", "booking"),
            listing("direct-river", "river", "direct"),
        ],
    )
    result = run(inv)
    assert len(result.available) == 1
    assert len(result.available[0].listings) == 3


def test_direct_listing_comes_first():
    inv = Inventory(
        units=[fresh_unit("river")],
        listings=[
            listing("booking-river", "river", "booking"),
            listing("living-river", "river", "living"),
            listing("direct-river", "river", "direct"),
        ],
    )
    channels = [li.channel for li in run(inv).available[0].listings]
    assert channels == ["direct", "living", "booking"]


def test_total_price_is_nightly_price_times_nights():
    inv = Inventory(units=[fresh_unit("a", price=1500)], listings=[listing("la", "a")])
    offer = run(inv).available[0]
    assert offer.nights == 3
    assert offer.total_price_czk == 4500


def test_offers_sorted_by_total_price():
    inv = Inventory(
        units=[
            fresh_unit("pricey", price=3000),
            fresh_unit("cheap", price=1000),
            fresh_unit("mid", price=2000),
        ],
        listings=[listing("l1", "pricey"), listing("l2", "cheap"), listing("l3", "mid")],
    )
    assert [o.unit.id for o in run(inv).available] == ["cheap", "mid", "pricey"]


def test_results_are_split_into_three_lists():
    inv = Inventory(
        units=[
            fresh_unit("ok"),
            fresh_unit("small", max_guests=1),
            fresh_unit("no-price", price=None),
        ],
        listings=[listing("l1", "ok"), listing("l2", "small"), listing("l3", "no-price")],
    )
    result = run(inv, guests=2)
    assert [o.unit.id for o in result.available] == ["ok"]
    assert [r.unit_id for r in result.rejected] == ["small"]
    assert [r.unit_id for r in result.manual_review] == ["no-price"]


def test_empty_inventory_gives_empty_result():
    result = run(Inventory(units=[], listings=[]))
    assert result.available == [] and result.manual_review == [] and result.rejected == []


# --- scenario 2 on the real test data ---

@pytest.fixture(scope="module")
def scenario_result():
    return search(load_inventory(), CHECK_IN, CHECK_OUT, guests=4, now=NOW)


def test_scenario_river_deluxe_offered_once_with_all_links(scenario_result):
    river = [o for o in scenario_result.available if o.unit.id == "river-deluxe-suite"]
    assert len(river) == 1
    assert len(river[0].listings) == 4
    assert river[0].listings[0].channel == "direct"


def test_scenario_milady_5_and_6_are_two_offers(scenario_result):
    ids = {o.unit.id for o in scenario_result.available}
    assert {"milady-horakove-5", "milady-horakove-6"} <= ids


def test_scenario_cheapest_offer_first(scenario_result):
    prices = [o.total_price_czk for o in scenario_result.available]
    assert prices == sorted(prices)


def test_scenario_unverified_units_never_offered(scenario_result):
    offered = {o.unit.id for o in scenario_result.available}
    held_back = {r.unit_id for r in scenario_result.manual_review + scenario_result.rejected}
    assert offered.isdisjoint(held_back)
    assert {r.unit_id for r in scenario_result.manual_review} == {
        "white-rabbit-residence", "london-royal-residence",
    }


# --- client preferences ---

def district_inventory(*units: tuple[str, int, str]) -> Inventory:
    return Inventory(
        units=[fresh_unit(uid, price=price, district=district) for uid, price, district in units],
        listings=[listing(f"l-{uid}", uid) for uid, _, _ in units],
    )


def test_requested_then_nearby_and_far_districts_dropped():
    inv = district_inventory(
        ("cheap-holesovice", 1000, "Prague 7 · Holešovice"),   # other end of Prague
        ("pricey-smichov", 3000, "Prague 5 · Smíchov"),
        ("mid-smichov", 2000, "Prague 5 · Smíchov"),
        ("cheap-novemesto", 1500, "Prague 2 · Nové Město"),    # across the river
    )
    ranked = rank_by_preferences(run(inv).available, ["near Smíchov"])
    assert [o.unit.id for o in ranked] == ["mid-smichov", "pricey-smichov", "cheap-novemesto"]
    assert [o.proximity for o in ranked] == ["requested", "requested", "nearby"]


def test_only_far_districts_left_are_offered_and_marked_other():
    inv = district_inventory(
        ("b-holesovice", 2000, "Prague 7 · Holešovice"),
        ("a-vrsovice", 1000, "Prague 10 · Vršovice"),
    )
    ranked = rank_by_preferences(run(inv).available, ["Smíchov"])
    assert [o.unit.id for o in ranked] == ["a-vrsovice", "b-holesovice"]
    assert {o.proximity for o in ranked} == {"other"}


def test_unknown_preference_keeps_everything():
    inv = district_inventory(("a", 1000, "Prague 7 · Holešovice"), ("b", 2000, "Prague 5 · Smíchov"))
    ranked = rank_by_preferences(run(inv).available, ["quiet", "balcony"])
    assert [o.unit.id for o in ranked] == ["a", "b"]
    assert all(o.proximity is None for o in ranked)


def test_district_match_ignores_accents_and_case():
    inv = Inventory(
        units=[fresh_unit("a", price=1000, district="Prague 7 · Holešovice"),
               fresh_unit("b", price=2000, district="Prague 5 · Smíchov")],
        listings=[listing("l1", "a"), listing("l2", "b")],
    )
    ranked = rank_by_preferences(run(inv).available, ["SMICHOV please"])
    assert ranked[0].unit.id == "b"


def test_no_preferences_keeps_price_order():
    inv = Inventory(
        units=[fresh_unit("a", price=2000), fresh_unit("b", price=1000)],
        listings=[listing("l1", "a"), listing("l2", "b")],
    )
    assert [o.unit.id for o in rank_by_preferences(run(inv).available, [])] == ["b", "a"]
