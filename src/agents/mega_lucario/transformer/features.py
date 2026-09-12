"""Public entity tokens. Fixed state slots, variable legal-option suffix, STOP last."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from agents.mega_lucario.gbdt.features import (
    DECK_COUNTS,
    IDS,
    SCHEMA_ID,
    candidate,
    card_values,
    feature_row,
    options_for,
)
from agents.mega_lucario.transformer.contract import SCORE_TIE_TOLERANCE

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    import numpy as np
    from numpy.typing import ArrayLike, NDArray


WIDTH = 198
NAMES = (
    ["valid", "type", "legal", "selected", "card_slot", "context_slot"]
    + [f"generic_{i}" for i in range(96)]
    + [f"counts_{i}" for i in range(32)]
    + [f"evolution_{i}" for i in range(32)]
    + [f"result_{i}" for i in range(32)]
)
CONTRACT = dict(
    version="lucario-entity-transformer-v2",
    feature_schema=SCHEMA_ID,
    width=WIDTH,
    max_tokens=120,
    max_options=100,
    board_slots=12,
    zone_slots=6,
    deck_slots=32,
    normalization="generic sign-log1p float32; card counts / 60",
    score_tie_tolerance=SCORE_TIE_TOLERANCE,
    tokens="Global, 12 Board, 6 Zone, all original Options, STOP; no position embedding",
    selection="selected hidden sum and step embedding; selected mask; STOP at min; end at max",
    features="public-lucario-entity-v2; conservative deck/prize belief; approximate attack results flagged uncertain",
)
CONTRACT_ID = hashlib.sha256(
    json.dumps(CONTRACT, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
SLOTS = {card: i for i, card in enumerate(IDS)}


def encode(matrix: ArrayLike) -> NDArray[np.float32]:
    """Validate token count, feature width, and finite values; return float32 tokens."""
    import numpy as np

    x = np.asarray(matrix, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != WIDTH or not 20 <= len(x) <= 120 or not np.isfinite(x).all():
        raise ValueError("Invalid structured Transformer input")
    return x


def token_indices(obs: dict[str, Any], options: Sequence[int]) -> list[int]:
    """Map original option indices and STOP to positions in the token sequence."""
    return [19 + i if i >= 0 else 19 + len(obs["select"]["option"]) for i in options]


def tokenize(
    obs: dict[str, Any],
    prefix: Sequence[int],
    table: dict[int, dict[str, Any]],
    option_rows: dict[int, Sequence[float]] | None = None,
) -> NDArray[np.float32]:
    """Build public state and option tokens, rejecting rather than truncating overflow."""
    import numpy as np

    cur, sel = obs["current"], obs["select"]
    actor = cur["yourIndex"]
    count = len(sel["option"])
    if count > 100:
        raise ValueError("Transformer option overflow: maximum 100; no truncation")
    legal = options_for(obs, prefix)
    x = np.zeros((20 + count, WIDTH), np.float32)
    x[:, 4:6] = -1

    def slot(c: dict[str, Any] | None) -> int:
        """Map a public card to its deck vocabulary slot, or -1 when unknown."""
        return SLOTS.get((c or {}).get("id"), -1)

    def counts(cards: Sequence[dict[str, Any] | None]) -> NDArray[np.float32]:
        """Count known card identities in the fixed deck vocabulary."""
        a = np.zeros(32, np.float32)
        for c in cards:
            s = slot(c)
            if s >= 0:
                a[s] += 1
        return a

    def generic(row: int, values: Sequence[float]) -> None:
        """Write sign-log-scaled scalar features into a token row."""
        v = np.asarray(values, np.float32)
        x[row, 6 : 6 + len(v)] = np.sign(v) * np.log1p(np.abs(v))

    base = feature_row(obs, -1, [], table, rich=False)
    x[0, 0] = 1
    x[0, 4] = slot(sel.get("effect"))
    x[0, 5] = slot(sel.get("contextCard"))
    generic(0, base[:20] + base[90:138] + base[346:354])
    me = cur["players"][actor]
    op = cur["players"][1 - actor]
    known = list(me["hand"]) + list(me["discard"])
    for side, ps in enumerate((me, op)):
        board = list(ps["active"])[:1] + [None] * (not ps["active"]) + list(ps["bench"])
        if len(board) > 6:
            raise ValueError("Board exceeds 6 slots per side")
        for j in range(6):
            r = 1 + 6 * side + j
            x[r, 1] = 1
            c = board[j] if j < len(board) else None
            if not c:
                continue
            x[r, 0] = 1
            x[r, 4] = slot(c)
            generic(r, [side, int(j == 0)] + card_values(c, table))
            attached = c.get("energyCards", []) + c.get("tools", [])
            evolution = c.get("preEvolution", [])
            x[r, 102:134] = counts(attached) / 60
            x[r, 134:166] = counts(evolution) / 60
            if side == 0:
                known += [c] + attached + evolution
    # Only publicly identified own cards are subtracted. Unknown pool is split
    # proportionally between deck and prizes; never read hidden card identities.
    known += [c for c in cur.get("stadium", []) if c and c.get("playerIndex") == actor]
    remaining = np.zeros(32, np.float32)
    for card, n in DECK_COUNTS.items():
        remaining[SLOTS[card]] = n
    remaining = np.maximum(0, remaining - counts(known))
    total = me["deckCount"] + len(me["prize"])
    exact = sel.get("deck")
    exact_ok = (
        exact is not None
        and len(exact) == me["deckCount"]
        and all(c for c in exact)
        and np.all(counts(exact) <= remaining)
    )
    deck = counts(exact) if exact_ok else remaining * (me["deckCount"] / max(1, total))
    prize = remaining - deck if exact_ok else remaining * (len(me["prize"]) / max(1, total))
    zones = [
        counts(me["hand"]),
        counts(me["discard"]),
        deck,
        prize,
        counts(exact or []),
        counts(op["discard"]),
    ]
    for z, values in enumerate(zones):
        r = 13 + z
        x[r, 0] = int(z != 4 or exact is not None)
        x[r, 1] = 2
        generic(
            r,
            [int(z == i) for i in range(6)]
            + [float(values.sum()), int(z in (2, 3)), int(exact_ok and z in (2, 3))],
        )
        x[r, 102:134] = values / 60
    for i, o in enumerate(sel["option"]):
        r = 19 + i
        x[r, 0] = 1
        x[r, 1] = 3
        x[r, 2] = int(i in legal)
        x[r, 3] = int(i in prefix)
        x[r, 4] = slot(candidate(obs, o))
        f = (
            option_rows[i]
            if option_rows is not None and i in option_rows
            else feature_row(obs, i, [], table, rich=False)
        )
        generic(r, f[22:60] + f[138:176] + f[346:358])
        # Approximate public damage is explicitly uncertain: no exact transition
        # claim for protection, damage modifiers or random card effects.
        result = np.zeros(32, np.float32)
        if o["type"] == 13:
            result[0] = 1
            result[3] = 1
            result[29] = f[354] / 600
            result[30] = f[356]
            result[31] = f[357]
        x[r, 166:198] = result
    x[-1, 0] = 1
    x[-1, 1] = 4
    x[-1, 2] = int(-1 in legal)
    return x


def with_prefix(
    tokens: NDArray[np.float32], obs: dict[str, Any], prefix: Sequence[int]
) -> NDArray[np.float32]:
    """Encoder inputs are prefix-independent; only pointer selection flags change."""
    result = tokens.copy()
    result[:, 2:4] = 0
    result[token_indices(obs, options_for(obs, prefix)), 2] = 1
    if prefix:
        result[[19 + i for i in prefix], 3] = 1
    return result
