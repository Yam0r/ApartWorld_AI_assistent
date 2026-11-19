from datetime import date, datetime

from pydantic import BaseModel

from apartworld_ai_assistent.availability import CheckResult, Verdict, check_unit
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


class SearchResult(BaseModel):
    available: list[Offer]             # can be offered to the client
    manual_review: list[CheckResult]   # a human has to check before offering
    rejected: list[CheckResult]        # does not fit, never shown to the client


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
