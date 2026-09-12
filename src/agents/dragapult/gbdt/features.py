"""Public candidate features and sequential legal-selection contract."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from agents.dragapult.gbdt.rich_features import append as append_rich

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from typing import Any


SCHEMA_PATH = Path(__file__).resolve().with_name("schema.json")
SCHEMA = json.loads(SCHEMA_PATH.read_text())
PROFILE = SCHEMA["profile"]
SCHEMA_ID = hashlib.sha256(SCHEMA_PATH.read_bytes()).hexdigest()
NAMES = SCHEMA["feature_names"]
IDS = sorted(map(int, SCHEMA["deck_counts"]))
DECK_COUNTS = {int(k): v for k, v in SCHEMA["deck_counts"].items()}


def check_deck(deck: Sequence[int]) -> None:
    """Reject card counts that differ from the model deck."""
    if Counter(deck) != DECK_COUNTS:
        raise ValueError("Dragapult ex deck does not match the feature/model contract")


def card_at(
    obs: dict[str, Any], area: int, index: int | None, player: int
) -> dict[str, Any] | None:
    """Resolve an observable card; hidden zones never reveal card identities."""
    if area not in (1, 2, 3, 4, 5, 6, 7, 12):
        return None
    if player not in (0, 1) or index is None or index < 0:
        raise ValueError("Invalid card reference")
    current, selection = obs["current"], obs["select"]
    if area == 2 and player != current["yourIndex"]:
        return None
    ps = current["players"][player]
    zones = {
        1: selection.get("deck"),
        2: ps.get("hand"),
        3: ps["discard"],
        4: ps["active"],
        5: ps["bench"],
        6: ps["prize"],
        7: current["stadium"],
        12: current.get("looking"),
    }
    cards = zones[area]
    if cards is None:
        return None
    return cards[index]


def candidate(obs: dict[str, Any], option: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve the public card associated with a legal action option."""
    kind = option["type"]
    actor = obs["current"]["yourIndex"]
    if kind == 7:
        return card_at(obs, 2, option["index"], actor)
    if kind in (8, 9):
        return card_at(obs, option["area"], option["index"], actor)
    if kind in (10, 11):
        return card_at(obs, option["area"], option["index"], actor)
    if kind in (3, 4, 5, 6):
        return card_at(obs, option["area"], option["index"], option["playerIndex"])
    if kind == 13:
        return next(iter(obs["current"]["players"][actor]["active"]), None)
    return None


def options_for(obs: dict[str, Any], prefix: Sequence[int]) -> list[int]:
    """Return remaining legal option indices, followed by STOP when allowed."""
    sel = obs["select"]
    if len(prefix) > sel["maxCount"] or len(set(prefix)) != len(prefix):
        raise ValueError("Invalid selection prefix")
    if any(i < 0 or i >= len(sel["option"]) for i in prefix):
        raise ValueError("Prefix is outside legal options")
    if len(prefix) == sel["maxCount"]:
        return []
    values = [i for i in range(len(sel["option"])) if i not in prefix]
    if len(prefix) >= sel["minCount"]:
        values.append(-1)  # STOP follows real candidates, including on ties.
    return values


def card_values(card: dict[str, Any] | None, table: dict[int, dict[str, Any]]) -> list[float]:
    """Encode public card properties in schema order."""
    if not card:
        return [0.0] * len(SCHEMA["card_fields"])
    master = table[card["id"]]
    hp, maximum = card.get("hp", 0), card.get("maxHp", 0)
    prizes = 3 if master["megaEx"] else 2 if master["ex"] else 1
    return [
        1,
        int("hp" in card),
        hp,
        maximum,
        maximum - hp,
        len(card.get("energies", [])),
        len(card.get("tools", [])),
        int(master["stage1"]),
        int(master["stage2"]),
        int(master["ex"]),
        int(master["megaEx"]),
        int(master.get("weakness") == PROFILE["attack_type"]),
        int(master.get("resistance") == PROFILE["attack_type"]),
        int(card.get("appearThisTurn", False)),
        prizes,
    ]


def feature_row(
    obs: dict[str, Any],
    option_index: int,
    prefix: Sequence[int],
    table: dict[int, dict[str, Any]],
    *,
    rich: bool = True,
) -> list[float]:
    """Build one candidate feature row using only the acting player's observation."""
    cur, sel = obs["current"], obs["select"]
    actor = cur["yourIndex"]
    me, opponent = cur["players"][actor], cur["players"][1 - actor]
    stop = option_index == -1
    option = {} if stop else sel["option"][option_index]
    kind = option.get("type", -1)
    card = None if stop else candidate(obs, option)
    target = None
    if kind in (8, 9):
        target = card_at(obs, option["inPlayArea"], option["inPlayIndex"], actor)
    elif kind in (3, 4, 5, 6):
        target = card
    my_active = next(iter(me["active"]), None)
    op_active = next(iter(opponent["active"]), None)
    if kind == 13:
        target = op_active
    area = option.get("area", -1)
    target_area = option.get("inPlayArea", area)
    owner = option.get("playerIndex", actor)
    row = [
        cur["turn"],
        int(cur["firstPlayer"] == actor),
        int(cur["supporterPlayed"]),
        int(cur["energyAttached"]),
        me["handCount"],
        me["deckCount"],
        len(me["prize"]),
        opponent["handCount"],
        opponent["deckCount"],
        len(opponent["prize"]),
        len(me["bench"]),
        len(opponent["bench"]),
        int(me["asleep"]),
        int(me["paralyzed"]),
        sel["minCount"],
        sel["maxCount"],
        len(sel["option"]),
        sel["remainDamageCounter"],
        int(sel.get("deck") is not None),
        int(cur.get("looking") is not None),
        len(prefix),
        sel["maxCount"] - len(prefix),
        int(stop),
        option.get("number", 0),
        int(card is not None and owner == actor),
        int(area == 4),
        int(area == 5),
        int(target_area == 4),
        int(target_area == 5),
        int(target is not None and (owner != actor or kind == 13)),
    ]
    for value in (card, target, my_active, op_active):
        row.extend(card_values(value, table))
    row.extend(int(sel["context"] == i) for i in range(48))
    row.extend(int(kind == i) for i in range(18))
    row.extend(int(area == i) for i in range(1, 13))
    row.extend(
        int(option.get("attackId", -1) >= 0 and option["attackId"] % 8 == i) for i in range(8)
    )
    zones = [
        me["hand"],
        me["discard"],
        me["active"],
        me["bench"],
        sel.get("deck") or [],
        [card],
        [target],
        [candidate(obs, sel["option"][i]) for i in prefix],
        [sel.get("effect")],
        [sel.get("contextCard")],
    ]
    for cards in zones:
        counts = Counter(c["id"] for c in cards if c)
        row.extend(counts[id] for id in IDS)
    board = [c for c in me["active"] + me["bench"] if c]
    counts = Counter(c["id"] for c in board)
    hand = Counter(c["id"] for c in me["hand"] if c)
    energy_ids = PROFILE["basic_energy_ids"]
    row.extend(
        [
            sum(counts[i] for i in PROFILE["pokemon_ids"]),
            sum(min(counts[a], hand[b]) for a, b in PROFILE["evolutions"]),
            sum(counts[i] for i in PROFILE["attackers"]),
            sum(len(c.get("energies", [])) >= 1 for c in me["bench"] if c),
            sum(len(c.get("energies", [])) >= 2 for c in me["bench"] if c),
            sum(c["id"] in energy_ids for c in me["discard"] if c),
            sum(len(c.get("energies", [])) for c in board),
            sum(hand[i] for i in energy_ids),
            0,
            0,
            0,
            0,
        ]
    )  # Attack outcomes are unknown; no private-state simulation.
    if not rich:
        return row
    append_rich(
        row, obs, None if stop else option, prefix, table, IDS, card_values, candidate, card_at
    )
    if len(row) != len(NAMES):
        raise ValueError(f"Feature width mismatch: {len(row)} != {len(NAMES)}")
    return row


def action_queries(
    obs: dict[str, Any], action: Sequence[int], table: dict[int, dict[str, Any]]
) -> Iterator[dict[str, Any]]:
    """Expand an action into ordered prefix queries, including an optional STOP target."""
    sel = obs["select"]
    if not sel["minCount"] <= len(action) <= sel["maxCount"]:
        raise ValueError("Invalid teacher action length")
    prefix = []
    for target in [*action, *([-1] if len(action) < sel["maxCount"] else [])]:
        options = options_for(obs, prefix)
        if target not in options:
            raise ValueError("Invalid teacher action/STOP")
        yield dict(
            prefix=prefix.copy(),
            options=options,
            target=target,
            features=[feature_row(obs, i, prefix, table) for i in options],
        )
        if target != -1:
            prefix.append(target)
