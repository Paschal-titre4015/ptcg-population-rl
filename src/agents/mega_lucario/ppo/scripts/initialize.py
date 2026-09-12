"""Freeze a structured Policy/Value checkpoint for the first PPO rollout."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "src"))
import argparse
import shutil

from agents.mega_lucario.transformer.checkpoint import load_checkpoint
from agents.mega_lucario.transformer.features import CONTRACT_ID, DECK_COUNTS, NAMES


def initialize(checkpoint: Path, output: Path) -> Path:
    """Validate and copy the starting Policy/Value checkpoint without changing weights."""
    if output.exists():
        raise ValueError("Initial PPO checkpoint already exists")
    load_checkpoint(checkpoint, CONTRACT_ID, DECK_COUNTS, NAMES)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, output)
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(initialize(a.checkpoint, a.output))
