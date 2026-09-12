"""Train and export the Mega Abomasnow ex GBDT from extracted game-split features."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "src"))
import argparse

from agents.mega_abomasnow.gbdt.features import DECK_COUNTS, NAMES, SCHEMA_ID
from agents.mega_abomasnow.gbdt.train import train


def main() -> None:
    """Parse command-line arguments and run this tool."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--feature-dir", type=Path, default=ROOT / "outputs/mega_abomasnow/gbdt_features"
    )
    p.add_argument("--output-dir", type=Path, default=ROOT / "models/mega_abomasnow")
    p.add_argument(
        "--profile",
        choices=["standard", "search", "quick"],
        default="standard",
        help="standard: fixed bank sizes and refit rounds; search: tune bank sizes; quick: small validation run",
    )
    p.add_argument("--rounds", type=int)
    p.add_argument("--threads", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--refit-general-rounds", type=int, default=650)
    p.add_argument("--refit-main-rounds", type=int, default=1100)
    a = p.parse_args()
    full = a.profile != "quick"
    rounds = (
        a.rounds
        if a.rounds is not None
        else (2400 if a.profile == "standard" else 1600 if full else 80)
    )
    threads = a.threads if a.threads is not None else (32 if full else 2)
    seed = a.seed if a.seed is not None else (204 if full else 42)
    train(
        a.feature_dir,
        a.output_dir,
        rounds,
        threads,
        seed,
        profile=a.profile,
        schema_id=SCHEMA_ID,
        names=NAMES,
        deck_counts=DECK_COUNTS,
        refit_rounds=dict(general=a.refit_general_rounds, main=a.refit_main_rounds),
    )


if __name__ == "__main__":
    main()
