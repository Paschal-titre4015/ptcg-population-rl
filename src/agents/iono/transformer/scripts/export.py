"""Export a .pt checkpoint to a standalone little-endian .bin C++ asset."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "src"))
import argparse
import struct

from agents.iono.replay import sha256
from agents.iono.transformer.checkpoint import load_checkpoint
from agents.iono.transformer.features import CONTRACT_ID, DECK_COUNTS, NAMES, SCHEMA_ID
from agents.iono.transformer.model import ARCHITECTURE

MAGIC = b"PTCG_TFM_V3\0\0\0\0\0"


def export(checkpoint: Path, output: Path) -> None:
    """Write a validated checkpoint to the native little-endian binary format."""
    if checkpoint.resolve() == output.resolve():
        raise ValueError("Checkpoint and binary output must differ")
    model, _ = load_checkpoint(checkpoint, CONTRACT_ID, DECK_COUNTS, NAMES)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as f:
        f.write(MAGIC)
        f.write(CONTRACT_ID.encode())
        f.write(SCHEMA_ID.encode())
        f.write(sha256(checkpoint).encode())
        f.write(
            struct.pack(
                "<6I",
                len(NAMES),
                *[ARCHITECTURE[k] for k in ("dim", "heads", "layers", "ff")],
                len(DECK_COUNTS),
            )
        )
        for card, count in sorted(DECK_COUNTS.items()):
            f.write(struct.pack("<ii", card, count))
        state = model.state_dict()
        f.write(struct.pack("<I", len(state)))
        for name, tensor in state.items():
            array = tensor.detach().numpy().astype("<f4")
            encoded = name.encode()
            f.write(struct.pack("<II", len(encoded), array.ndim))
            f.write(encoded)
            f.write(struct.pack("<" + "I" * array.ndim, *array.shape))
            f.write(array.tobytes())
    temporary.replace(output)
    print(f"Exported {output}: {output.stat().st_size} bytes", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    export(a.checkpoint, a.output)
