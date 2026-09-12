"""Train the official-deck Dragapult structured Transformer on GBDT replays."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "src"))
import argparse

from agents.dragapult.transformer.features import CONTRACT_ID
from agents.dragapult.transformer.train import train


def main() -> None:
    """Parse command-line arguments and run this tool."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--hard-weight", type=float, default=0.1)
    p.add_argument("--value-weight", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=204)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--max-updates", type=int, help="Maximum optimizer updates")
    a = p.parse_args()
    train(
        a.feature_dir,
        a.output_dir,
        CONTRACT_ID,
        epochs=a.epochs,
        batch_size=a.batch_size,
        learning_rate=a.learning_rate,
        temperature=a.temperature,
        hard_weight=a.hard_weight,
        value_weight=a.value_weight,
        seed=a.seed,
        device=a.device,
        max_updates=a.max_updates,
    )


if __name__ == "__main__":
    main()
