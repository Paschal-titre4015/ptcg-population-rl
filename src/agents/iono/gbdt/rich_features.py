"""Public card and board features.

Canonical slots are sorted by their entire public feature vector, never position.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any


HASH_MODULI = (67, 71)
SLOTS = ["candidate", "target", "own_active", "opponent_active"] + [
    f"{side}_canonical_bench_{i}" for side in ("own", "opponent") for i in range(5)
]
ZONES = (
    "own_hand",
    "own_discard",
    "own_board",
    "opponent_board",
    "opponent_discard",
    "stadium",
    "public_search",
    "candidates",
    "targets",
    "selected",
)


def names(ids: Sequence[int], card_fields: Sequence[str]) -> list[str]:
    """Return canonical board and identity feature names in export order."""
    fields = list(card_fields)
    fields += [f"identity_hash_{m}_{i}" for m in HASH_MODULI for i in range(m + 1)]
    fields += [f"deck_id_{i}" for i in ids] + [f"energy_type_{i}" for i in range(12)]
    fields += [f"{zone}_hash_17_{i}" for zone in ("tools", "evolution") for i in range(18)]
    result = [f"rich_{slot}_{f}" for slot in SLOTS for f in fields]
    result += [f"rich_{z}_hash_{m}_{i}" for z in ZONES for m in HASH_MODULI for i in range(m + 1)]
    result += [f"deck_{z}_{i}" for z in ("board", "bench", "hand", "discard") for i in ids]
    result += [f"{z}_energy_type_{i}" for z in ("active", "target") for i in range(12)]
    result += [f"legal_attack_hash_{i}" for i in range(8)]
    return result


def hashes(
    cards: Sequence[dict[str, Any] | None], moduli: Sequence[int] = HASH_MODULI
) -> list[float]:
    """Count public card identities in deterministic hash bins."""
    ids = [c["id"] for c in cards if c]
    out = []
    for m in moduli:
        counts = Counter(1 + i % m if i > 0 else 0 for i in ids)
        out.extend(counts[i] for i in range(m + 1))
    return out


def target_card(
    obs: dict[str, Any],
    option: dict[str, Any] | None,
    candidate_fn: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any] | None],
    card_at_fn: Callable[[dict[str, Any], int, int, int], dict[str, Any] | None],
) -> dict[str, Any] | None:
    """Resolve an action's public target with the supplied observation helpers."""
    actor = obs["current"]["yourIndex"]
    kind = option["type"]
    if kind in (8, 9):
        return card_at_fn(obs, option["inPlayArea"], option["inPlayIndex"], actor)
    if kind in (3, 4, 5, 6):
        return candidate_fn(obs, option)
    if kind == 13:
        return next(iter(obs["current"]["players"][1 - actor]["active"]), None)
    return None


def append(
    row: list[float],
    obs: dict[str, Any],
    option: dict[str, Any] | None,
    prefix: Sequence[int],
    table: dict[int, dict[str, Any]],
    ids: Sequence[int],
    card_values: Callable[[dict[str, Any] | None, dict[int, dict[str, Any]]], list[float]],
    candidate: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any] | None],
    card_at: Callable[[dict[str, Any], int, int, int], dict[str, Any] | None],
) -> None:
    """Append canonical board, public-zone, and deck-specific features in place."""
    cur, sel = obs["current"], obs["select"]
    actor = cur["yourIndex"]
    me, op = cur["players"][actor], cur["players"][1 - actor]
    board = [c for c in me["active"] + me["bench"] if c]
    enemy = [c for c in op["active"] + op["bench"] if c]
    active = next(iter(me["active"]), None)

    def fields(c: dict[str, Any] | None) -> list[float]:
        """Encode one public card for canonical board sorting."""
        energies = Counter((c or {}).get("energies", []))
        return (
            card_values(c, table)
            + hashes([c])
            + [int(bool(c) and c["id"] == i) for i in ids]
            + [energies[i] for i in range(12)]
            + hashes((c or {}).get("tools", []), (17,))
            + hashes((c or {}).get("preEvolution", []), (17,))
        )

    width = len(fields(None))
    c = candidate(obs, option) if option else None
    target = target_card(obs, option, candidate, card_at) if option else None
    for value in (c, target, active, next(iter(op["active"]), None)):
        row.extend(fields(value))
    for ps in (me, op):
        if len(ps["bench"]) > 5:
            raise ValueError("GBDT public board exceeds five bench slots")
        canonical = sorted((fields(c) for c in ps["bench"] if c), reverse=True)
        for values in canonical:
            row.extend(values)
        row.extend([0] * (width * (5 - len(canonical))))
    options = sel["option"]
    for zone in (
        me["hand"],
        me["discard"],
        board,
        enemy,
        op["discard"],
        cur["stadium"],
        sel["deck"] if sel.get("deck") is not None else cur.get("looking") or [],
        [candidate(obs, o) for o in options],
        [target_card(obs, o, candidate, card_at) for o in options],
        [candidate(obs, options[i]) for i in prefix],
    ):
        row.extend(hashes(zone))
    for zone in (board, me["bench"], me["hand"], me["discard"]):
        counts = Counter(c["id"] for c in zone if c)
        row.extend(counts[i] for i in ids)
    for card in (active, target):
        energies = Counter((card or {}).get("energies", []))
        row.extend(energies[i] for i in range(12))
    legal = {o["attackId"] % 8 for o in options if o["type"] == 13}
    row.extend(int(i in legal) for i in range(8))
