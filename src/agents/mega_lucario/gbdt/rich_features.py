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
POKEMON_IDS = (673, 674, 675, 676, 677, 678)
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
DOMAIN_FIELDS = (
    "fighting_hand",
    "fighting_discard",
    "attachment_available",
    "empty_bench",
    "opponent_multi_prize",
    "opponent_mega",
    "damaged_lucario",
    "gravity_mountain",
    "opponent_stage2",
    "opponent_total_hp",
    "own_total_hp",
    "attack_ignores_weakness",
    "aura_jab_recovery_capacity",
    "lunar_cycle_resource_ready",
    "riolu_evolution_pairs",
    "makuhita_evolution_pairs",
)
INTERACTIONS = tuple(f"legal_attack_{i}" for i in range(976, 984)) + (
    "attach_completes_one",
    "attach_completes_two",
    "attach_completes_three",
    "mega_ready_brave",
    "bench_fighting",
    "lunar_discard_aura_fuel",
    "lillie_draw_capacity",
    "carmine_draw_capacity",
    "carmine_first_turn",
    "dusk_search_prompt",
    "gravity_candidate_stage2_targets",
    "evolve_lucario",
    "evolve_hariyama",
    "wild_press_self_damage",
    "wild_press_self_ko",
    "aura_bench_targets",
)


def names(ids: Sequence[int], card_fields: Sequence[str]) -> list[str]:
    """Return canonical board and identity feature names in export order."""
    fields = list(card_fields)
    fields += [f"identity_hash_{m}_{i}" for m in HASH_MODULI for i in range(m + 1)]
    fields += [f"deck_id_{i}" for i in ids] + [f"energy_type_{i}" for i in range(12)]
    fields += [f"{zone}_hash_17_{i}" for zone in ("tools", "evolution") for i in range(18)]
    result = [f"rich_{slot}_{f}" for slot in SLOTS for f in fields]
    result += [f"rich_{z}_hash_{m}_{i}" for z in ZONES for m in HASH_MODULI for i in range(m + 1)]
    result += [
        f"lucario_{z}_{i}" for z in ("board", "bench", "hand", "discard") for i in POKEMON_IDS
    ]
    result += [f"lucario_ready_plus_{extra}_{i}" for extra in (0, 1) for i in POKEMON_IDS]
    result += [f"lucario_active_energy_type_{i}" for i in range(12)]
    result += ["lucario_" + f for f in DOMAIN_FIELDS]
    result += ["lucario_" + f for f in INTERACTIONS]
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
    counts = []
    for zone in (board, me["bench"], me["hand"], me["discard"]):
        counts.append(Counter(c["id"] for c in zone if c))
        row.extend(counts[-1][i] for i in POKEMON_IDS)
    needed = (1, 3, 2, 1, 1, 1)
    lunar_bench = any(c and c["id"] == 675 for c in me["bench"])
    for extra in (0, 1):
        row.extend(
            sum(
                c["id"] == i
                and (i != 676 or lunar_bench)
                and c.get("energies", []).count(6) + extra >= n
                for c in board
            )
            for i, n in zip(POKEMON_IDS, needed)
        )
    energies = Counter((active or {}).get("energies", []))
    row.extend(energies[i] for i in range(12))
    b, _, hand, discard = counts

    def prizes(c: dict[str, Any]) -> int:
        """Return the public prize value of knocking out this Pokemon."""
        return 3 if table[c["id"]]["megaEx"] else 2 if table[c["id"]]["ex"] else 1

    attack = (option or {}).get("attackId", -1)
    row.extend(
        [
            hand[6],
            discard[6],
            int(not cur["energyAttached"]),
            max(0, 5 - len(me["bench"])),
            sum(prizes(c) > 1 for c in enemy),
            sum(table[c["id"]]["megaEx"] for c in enemy),
            sum(c["id"] == 678 and c.get("hp", 0) < c.get("maxHp", 0) for c in board),
            int(any(c and c["id"] == 1252 for c in cur["stadium"])),
            sum(table[c["id"]]["stage2"] for c in enemy),
            sum(c.get("hp", 0) for c in enemy),
            sum(c.get("hp", 0) for c in board),
            int(attack == 980),
            min(3, discard[6]) if attack == 982 and me["bench"] else 0,
            int(b[675] > 0 and b[676] > 0 and hand[6] > 0),
            min(b[677], hand[678]),
            min(b[673], hand[674]),
        ]
    )
    legal_attacks = {o.get("attackId") for o in options if o["type"] == 13}
    row.extend(int(i in legal_attacks) for i in range(976, 984))
    attaching = bool(option and option["type"] == 8 and c and c["id"] == 6 and target)
    target_energy = (target or {}).get("energies", []).count(6)
    row.extend(int(attaching and target_energy == n - 1) for n in (1, 2, 3))
    effect = (sel.get("effect") or {}).get("id", 0)
    cid = (c or {}).get("id", 0)
    row.extend(
        [
            sum(c["id"] == 678 and c.get("energies", []).count(6) >= 2 for c in board),
            sum(c.get("energies", []).count(6) for c in me["bench"] if c),
            int(effect == 675 and cid == 6),
            min(
                me["deckCount"],
                max(0, (8 if len(me["prize"]) == 6 else 6) - max(0, me["handCount"] - 1)),
            ),
            min(5, me["deckCount"]),
            int(cid == 1192 and cur["turn"] == 1),
            int(effect == 1102 and (sel.get("deck") is not None or cur.get("looking") is not None)),
            sum(table[c["id"]]["stage2"] for c in enemy) if cid == 1252 else 0,
            int(bool(option) and option["type"] == 9 and cid == 678),
            int(bool(option) and option["type"] == 9 and cid == 674),
            70 if attack == 978 else 0,
            int(attack == 978 and bool(active) and 0 < active.get("hp", 0) <= 70),
            len(me["bench"]) if attack == 982 else 0,
        ]
    )
