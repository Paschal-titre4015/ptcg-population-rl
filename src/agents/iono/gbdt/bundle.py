"""Read native trees, LightGBM boosters and training metadata from one bundle."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import lightgbm as lgb

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any


class Boosters(dict):
    """Reference trees plus exported selection policy."""


def load_bundle(path: Path) -> tuple[dict[str, Any], Boosters]:
    """Validate a native bundle and restore its LightGBM banks and selection thresholds."""
    raw = path.read_bytes()
    magic, schema, identity, body = raw.split(b"\n", 3)
    if (
        magic not in (b"PTCG_GBDT_V2", b"PTCG_GBDT_V3")
        or identity != b"MODEL_ID " + hashlib.sha256(schema + b"\n" + body).hexdigest().encode()
    ):
        raise ValueError("Invalid model format/checksum")
    _, archive = body.split(b"\nARCHIVE_BYTES ", 1)
    length, archive = archive.split(b"\n", 1)
    size = int(length)
    if not 0 < size <= 67108864 or archive[size:] != b"\nEND\n":
        raise ValueError("Invalid model archive length")
    payload = json.loads(archive[:size])
    training = payload["training"]
    if training.get("schema_id") != schema.decode():
        raise ValueError("Archive schema mismatch")
    training.update(
        model_id=identity.decode().split()[1], model_sha256=hashlib.sha256(raw).hexdigest()
    )
    boosters = Boosters(
        {name: lgb.Booster(model_str=payload["boosters"][name]) for name in ("general", "main")}
    )
    boosters.thresholds = training.get("selection_thresholds", {})
    thresholds = boosters.thresholds
    if bool(thresholds) != (magic == b"PTCG_GBDT_V3"):
        raise ValueError("Threshold format mismatch")
    if thresholds:
        expected = (
            f"STOP_FEATURE {training['feature_names'].index('is_stop')}\nTHRESHOLDS {len(thresholds)}\n"
            + "".join(
                f"{training['feature_names'].index('context_' + str(k))} {v:.17g}\n"
                for k, v in sorted(thresholds.items(), key=lambda item: int(item[0]))
            )
        ).encode()
        if expected not in body:
            raise ValueError("Native/archive selection threshold mismatch")
    boosters.names = training.get("feature_names", [])
    return training, boosters
