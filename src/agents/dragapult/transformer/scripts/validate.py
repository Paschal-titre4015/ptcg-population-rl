"""Exercise standalone .bin inference, self-play, deterministic replays and parity."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "src"))
import argparse
import json
import shutil
import subprocess
import tempfile

from agents.dragapult.replay import sha256
from agents.dragapult.transformer.scripts.audit import ROOT, audit

if TYPE_CHECKING:
    from typing import Any


def validate(
    checkpoint: Path,
    model: Path,
    teacher_model: Path,
    output: Path,
    binary: Path,
    features: Path | None = None,
    probe_queries: int = 1000,
) -> dict[str, Any]:
    """Check standalone inference, self-play, replay determinism, and Python/C++ parity."""
    if output.exists() and any(output.iterdir()):
        raise ValueError("Validation output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    matchups = [
        ("dragapult_transformer", "dragapult_gbdt"),
        ("dragapult_gbdt", "dragapult_transformer"),
        ("dragapult_transformer", "dragapult_transformer"),
    ]
    results = []
    with tempfile.TemporaryDirectory() as temporary:
        standalone = Path(temporary) / "model.bin"
        shutil.copy2(model, standalone)
        for i, (a, b) in enumerate(matchups):
            command = [
                str(binary.resolve()),
                "--agent-a",
                a,
                "--agent-b",
                b,
                "--deck-a",
                str(ROOT / "src/agents/dragapult/deck.csv"),
                "--deck-b",
                str(ROOT / "src/agents/dragapult/deck.csv"),
                "--games",
                "2",
                "--seed",
                str(202620000 + i * 10),
                "--replay-dir",
                str(output / "replays" / f"pair-{i}"),
            ]
            for seat, name in (("a", a), ("b", b)):
                command += [
                    f"--model-{seat}",
                    str(standalone if name.endswith("_transformer") else teacher_model.resolve()),
                ]
            run = subprocess.run(command, capture_output=True, text=True, check=True, timeout=600)
            games = [json.loads(line) for line in run.stdout.splitlines()]
            if len(games) != 2 or any(g["agent_errors"] or g["engine_errors"] for g in games):
                raise ValueError("Native battle errors")
            results += games
            print(f"Validated {a} vs {b}", flush=True)
        command[command.index("--replay-dir") + 1] = str(output / "determinism")
        subprocess.run(command, capture_output=True, text=True, check=True, timeout=600)
        for i in range(2):
            if sha256(output / "determinism" / f"game-{i}.jsonl") != sha256(
                output / "replays/pair-2" / f"game-{i}.jsonl"
            ):
                raise ValueError("Non-deterministic replay")
        parity = audit(
            output / "replays",
            checkpoint,
            standalone,
            output / "parity.json",
            features,
            binary.parent / "dragapult_transformer_probe",
            probe_queries,
        )
    report = dict(games=results, deterministic_replays=True, standalone_bin=True, parity=parity)
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("checkpoint", "model", "teacher-model", "output-dir"):
        p.add_argument("--" + flag, type=Path, required=True)
    p.add_argument("--binary", type=Path, default=ROOT / "build/arena_cpp")
    p.add_argument("--feature-dir", type=Path)
    p.add_argument("--probe-queries", type=int, default=1000)
    a = p.parse_args()
    validate(
        a.checkpoint,
        a.model,
        a.teacher_model,
        a.output_dir,
        a.binary,
        a.feature_dir,
        a.probe_queries,
    )
