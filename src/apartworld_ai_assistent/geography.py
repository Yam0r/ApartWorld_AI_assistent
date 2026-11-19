"""Prague districts and which ones are next to each other.

Used to answer "Smíchov or something nearby" without offering the other
end of the city. Neighbours are districts with a shared border or one bridge
/ a few tram stops apart. Kept small on purpose: extend when new units appear.
"""

import re
import unicodedata

# Display name -> neighbours (display names). Made symmetric below.
_NEIGHBOURS: dict[str, set[str]] = {
    "Smíchov": {"Malá Strana", "Košíře", "Radlice", "Nové Město", "Vyšehrad"},
    "Malá Strana": {"Hradčany", "Staré Město", "Nové Město"},
    "Staré Město": {"Nové Město", "Josefov", "Holešovice"},
    "Nové Město": {"Vinohrady", "Vyšehrad", "Karlín", "Žižkov", "Nusle"},
    "Vinohrady": {"Žižkov", "Vršovice", "Nusle", "Strašnice"},
    "Žižkov": {"Karlín", "Libeň", "Strašnice"},
    "Vršovice": {"Nusle", "Strašnice", "Michle"},
    "Vyšehrad": {"Nusle", "Podolí"},
    "Holešovice": {"Letná", "Bubeneč", "Libeň", "Karlín", "Troja"},
    "Karlín": {"Libeň"},
    "Libeň": {"Vysočany"},
    "Horní Počernice": {"Černý Most", "Kyje"},
}


def plain(text: str) -> str:
    """Lowercase without accents: 'Smíchov' -> 'smichov'."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _build() -> tuple[dict[str, str], dict[str, set[str]]]:
    display: dict[str, str] = {}
    neighbours: dict[str, set[str]] = {}
    for district, near in _NEIGHBOURS.items():
        for name in {district, *near}:
            display[plain(name)] = name
            neighbours.setdefault(plain(name), set())
        for name in near:
            neighbours[plain(district)].add(plain(name))
            neighbours[plain(name)].add(plain(district))
    return display, neighbours


DISPLAY, NEIGHBOURS = _build()


def district_key(unit_district: str) -> str:
    """'Prague 5 · Smíchov' -> 'smichov' (the part after the dot)."""
    return plain(unit_district.split("·")[-1].strip())


def requested_districts(preferences: list[str]) -> set[str]:
    """Known district names mentioned in the client's preferences."""
    text = " " + " ".join(re.findall(r"\w+", plain(" ".join(preferences)))) + " "
    return {key for key in DISPLAY if f" {key} " in text}


def proximity(unit_district: str, wanted: set[str]) -> str:
    """'requested', 'nearby' or 'other'."""
    key = district_key(unit_district)
    if key in wanted:
        return "requested"
    if any(key in NEIGHBOURS.get(w, set()) for w in wanted):
        return "nearby"
    return "other"
