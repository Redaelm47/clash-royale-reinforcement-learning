"""Card names: IL_Replay (RoyaleAPI) keys to the row names of the game tables.

The il_card_names.json table comes from RoyaleImitate (MIT license,
https://github.com/RoyaleGym/RoyaleImitate). All 119 of its names exist in the crforge
16.635.8 tables (checked).
"""

import json
import re
from pathlib import Path

NAMES: dict[str, str] = json.loads(Path(__file__).with_name("il_card_names.json").read_text())

# The 2.6 deck with the table names, in the most played variant in IL_Replay
# (8,552 decks out of 14,305): evolved Cannon, hero Musketeer, evolved Skeletons.
DECK_26 = ["HogRider", "Musketeer", "Cannon", "IceGolemite", "IceSpirits", "Skeletons", "Fireball", "Log"]
SLOT_EVOLUTION, SLOT_HERO = 1, 2
SLOTS_26 = [0, SLOT_HERO, SLOT_EVOLUTION, 0, 0, SLOT_EVOLUTION, 0, 0]

TOWERS = {
    "tower-princess": "King_PrincessTowers",
    "cannoneer": "King_CannonTowers",
    "dagger-duchess": "King_KnifeTowers",
    "royal-chef": "King_ChefTowers",
}

_SUFFIX = re.compile(r"-(ev\d+|hero)$")


def base_key(key: str) -> str:
    """The key without its variant suffix (-ev1, -hero)."""
    return _SUFFIX.sub("", key)


def slot_flag(key: str) -> int:
    """The slot flag of a deck card: 1 evolution, 2 hero, 0 otherwise."""
    m = _SUFFIX.search(key)
    if m is None:
        return 0
    return SLOT_HERO if m.group(1) == "hero" else SLOT_EVOLUTION


def card_name(key: str | None) -> str | None:
    """The card's name in the tables, or None if we don't know it."""
    if key is None:
        return None
    return NAMES.get(base_key(key))
