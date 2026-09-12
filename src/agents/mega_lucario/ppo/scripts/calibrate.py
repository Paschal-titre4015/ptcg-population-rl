"""Calibrate PPO learning rate on a fixed rollout and excluded global-KL holdout."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "src"))
import argparse
import json

from agents.mega_lucario.ppo.scripts.collect import (
    CONTRACT_ID,
    DECK_COUNTS,
    DEFAULTS,
    NAMES,
    load_checkpoint,
)
from agents.mega_lucario.ppo.scripts.train import add_training_args, train, training_config
from agents.mega_lucario.replay import sha256

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any


def calibrate(
    checkpoint: Path,
    rollout: Path,
    output: Path,
    rates: Sequence[float],
    config: dict[str, Any],
    target: float = 0.004,
    tolerance: float = 0.001,
) -> dict[str, Any]:
    """Select a learning rate whose held-out policy KL is closest to the target."""
    import math

    if (
        not rates
        or len(set(rates)) != len(rates)
        or any(not math.isfinite(v) or v <= 0 for v in rates)
        or not 0 < tolerance < target
    ):
        raise ValueError("Invalid calibration rates/target")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Calibration output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for index, rate in enumerate(rates):
        # Every candidate starts from exactly the same weights AND optimizer.
        model, payload = load_checkpoint(checkpoint, CONTRACT_ID, DECK_COUNTS, NAMES)
        report = train(
            model,
            payload,
            checkpoint,
            rollout,
            output / f"trial-{index:02d}",
            dict(config, learning_rate=rate),
        )
        results.append(
            dict(
                learning_rate=rate,
                global_kl=report["heldout_global_approximate_kl"],
                holdout_indices_sha256=report["holdout_indices_sha256"],
                updates=report["updates"],
                decisions_used=report["decisions_used"],
            )
        )
    if len({r["holdout_indices_sha256"] for r in results}) != 1:
        raise ValueError("Calibration holdout changed")
    selected = min(results, key=lambda r: abs(r["global_kl"] - target))
    report = dict(
        checkpoint_sha256=sha256(checkpoint),
        rollout_manifest_sha256=sha256(rollout / "manifest.json"),
        config=config,
        target=target,
        tolerance=tolerance,
        results=results,
        selected=selected,
        selected_within_tolerance=abs(selected["global_kl"] - target) <= tolerance,
        use="Apply selected learning_rate to subsequent fresh rollouts; do not deploy trial weights",
    )
    (output / "calibration.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "rollout-dir", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--rates", nargs="+", type=float, default=[2e-5, 3e-5, 4e-5, 5e-5, 6e-5])
    p.add_argument("--target", type=float, default=DEFAULTS["kl_target"])
    p.add_argument("--tolerance", type=float, default=DEFAULTS["kl_tolerance"])
    p.add_argument("--seed", type=int, default=204)
    add_training_args(p)
    a = p.parse_args()
    report = calibrate(
        a.checkpoint,
        a.rollout_dir,
        a.output_dir,
        a.rates,
        training_config(a, a.seed),
        a.target,
        a.tolerance,
    )
    if not report["selected_within_tolerance"]:
        raise SystemExit(
            "No rate reached the requested KL range; inspect calibration.json and extend the sweep"
        )
