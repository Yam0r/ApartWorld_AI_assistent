from datetime import date, datetime

from pydantic import BaseModel

from apartworld_ai_assistent.availability import CheckResult, Verdict, check_unit
from apartworld_ai_assistent.geography import proximity, requested_districts
from apartworld_ai_assistent.inventory import Inventory
from apartworld_ai_assistent.models import Listing, Unit

# Direct booking is the business goal, so direct links go first.
CHANNEL_ORDER = {"direct": 0, "living": 1}


class Offer(BaseModel):
    """One physical unit that passed every check, with all its listings."""

    unit: Unit
    listings: list[Listing]
    nights: int
    total_price_czk: int
    # set by rank_by_preferences: "requested" | "nearby" | "other"
    proximity: str | None = None


class SearchResult(BaseModel):
    available: list[Offer]             # can be offered to the client
    manual_review: list[CheckResult]   # a human has to check before offering
    rejected: list[CheckResult]        # does not fit, never shown to the client


PROXIMITY_ORDER = {"requested": 0, "nearby": 1, "other": 2}


def rank_by_preferences(offers: list[Offer], preferences: list[str]) -> list[Offer]:
    """Apply the client's district wish.

    - requested district first, then neighbouring districts, price order kept;
    - other districts are dropped, so the far end of Prague is not offered;
    - only if nothing is in or near the district, other districts are returned
      (marked "other") so the client still gets an answer.
    No known district in the preferences -> offers unchanged.
    """
    wanted = requested_districts(preferences)
    if not wanted:
        return offers
    tagged = [o.model_copy(update={"proximity": proximity(o.unit.district, wanted)}) for o in offers]
    close = [o for o in tagged if o.proximity != "other"]
    return sorted(close or tagged, key=lambda o: PROXIMITY_ORDER[o.proximity])


def _sorted_listings(listings: list[Listing]) -> list[Listing]:
    return sorted(listings, key=lambda l: (CHANNEL_ORDER.get(l.channel, 99), l.id))


def search(
    inventory: Inventory,
    check_in: date,
    check_out: date,
    guests: int,
    now: datetime,
) -> SearchResult:
    """Check every physical unit and split the results into three lists.

    Units, not listings, are checked: several listings of one apartment
    produce one offer, never several.
    """
    nights = (check_out - check_in).days
    available: list[Offer] = []
    manual_review: list[CheckResult] = []
    rejected: list[CheckResult] = []

    for unit in inventory.units:
        result = check_unit(unit, check_in, check_out, guests=guests, now=now)

        if result.verdict == Verdict.REJECTED:
            rejected.append(result)
        elif result.verdict == Verdict.MANUAL_REVIEW:
            manual_review.append(result)
        else:
            # check_unit guarantees a known price for AVAILABLE units
            assert unit.price_per_night_czk is not None
            available.append(Offer(
                unit=unit,
                listings=_sorted_listings(inventory.listings_for(unit.id)),
                nights=nights,
                total_price_czk=unit.price_per_night_czk * nights,
            ))

    available.sort(key=lambda o: (o.total_price_czk, o.unit.id))
    return SearchResult(available=available, manual_review=manual_review, rejected=rejected)
