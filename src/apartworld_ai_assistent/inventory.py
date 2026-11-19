import json
from pathlib import Path

from pydantic import BaseModel, model_validator

from apartworld_ai_assistent.models import Listing, Unit

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "units.json"


class Inventory(BaseModel):
    units: list[Unit]
    listings: list[Listing]

    @model_validator(mode="after")
    def check_references(self) -> "Inventory":
        unit_ids = [u.id for u in self.units]
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("duplicate unit id")
        listing_ids = [l.id for l in self.listings]
        if len(listing_ids) != len(set(listing_ids)):
            raise ValueError("duplicate listing id")
        known = set(unit_ids)
        for listing in self.listings:
            if listing.unit_id not in known:
                raise ValueError(f"listing {listing.id} points to unknown unit {listing.unit_id}")
        return self

    def listings_for(self, unit_id: str) -> list[Listing]:
        return [l for l in self.listings if l.unit_id == unit_id]


def load_inventory(path: Path = DEFAULT_PATH) -> Inventory:
    data = json.loads(path.read_text(encoding="utf-8"))
    return Inventory(units=data["units"], listings=data["listings"])
