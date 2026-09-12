"""Collect seeded rule/GBDT Dragapult ex games against the four baselines."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor

from agents.dragapult.gbdt.features import SCHEMA_ID, check_deck
from agents.dragapult.replay import inspect_replay, replay_schema, sha256

if TYPE_CHECKING:
    from typing import Any


def collect(
    binary: Path,
    output: Path,
    games: int,
    seed: int,
    games_per_opponent: int | None = None,
    teacher: str = "dragapult",
    model: Path | None = None,
    workers: int = 1,
    resume: bool = False,
) -> dict[str, Any]:
    """Collect or resume seeded teacher matches against each official rule policy."""
    if teacher not in ("dragapult", "dragapult_gbdt"):
        raise ValueError("Unsupported teacher")
    if (teacher == "dragapult_gbdt") != (model is not None):
        raise ValueError("The GBDT teacher requires --model; rule teacher does not use one")
    per_seat = (
        [games, games]
        if games_per_opponent is None
        else [(games_per_opponent + 1) // 2, games_per_opponent // 2]
    )
    total = sum(per_seat) * 4
    if min(per_seat) < 1 or workers < 1 or seed < 0 or seed + total - 1 > 2**32 - 1:
        raise ValueError("Invalid game count / seed range / workers")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()) and not resume:
        raise ValueError("Output directory must be empty (or use --resume)")
    check_deck(list(map(int, (ROOT / "src/agents/dragapult/deck.csv").read_text().split())))
    model_id = model.read_text().splitlines()[2].split()[1] if model is not None else None
    jobs = []
    next_seed = seed
    for opponent in ("mega_lucario", "dragapult", "iono", "mega_abomasnow"):
        for seat in (0, 1):
            jobs.append((opponent, seat, per_seat[seat], next_seed))
            next_seed += per_seat[seat]

    def pairing(job: tuple[str, int, int, int]) -> list[dict[str, Any]]:
        """Collect one opponent/seat pairing, reusing only a verified seed prefix."""
        opponent, seat, count, first_seed = job
        a, b = (teacher, opponent) if seat == 0 else (opponent, teacher)
        destination = output / f"{opponent}-seat{seat}"
        entries = []

        def inspect(path: Path) -> dict[str, Any]:
            """Validate one replay against the requested teacher, deck, and model."""
            header, terminal = inspect_replay(path)
            if (
                header["agents"] != [a, b]
                or replay_schema(header) != SCHEMA_ID
                or header["models"][seat] != model_id
            ):
                raise ValueError("Resume replay teacher/schema/model mismatch")
            check_deck(header["decks"][seat])
            return dict(
                path=str(path.relative_to(output)),
                sha256=sha256(path),
                seed=header["seed"],
                **terminal,
            )

        if resume:
            entries = [inspect(p) for p in sorted(destination.rglob("game-*.jsonl"))]
            if (
                sorted(e["seed"] for e in entries)
                != list(range(first_seed, first_seed + len(entries)))
                or len(entries) > count
            ):
                raise ValueError("Resume requires a complete contiguous prefix of game seeds")
        remaining = count - len(entries)
        if remaining:
            target = (
                destination
                if not destination.exists()
                else destination / f"continuation-{len(entries)}"
            )
            deck_a = "dragapult" if a == "dragapult_gbdt" else a
            deck_b = "dragapult" if b == "dragapult_gbdt" else b
            command = [
                str(binary.resolve()),
                "--agent-a",
                a,
                "--agent-b",
                b,
                "--deck-a",
                str(ROOT / f"src/agents/{deck_a}/deck.csv"),
                "--deck-b",
                str(ROOT / f"src/agents/{deck_b}/deck.csv"),
                "--games",
                str(remaining),
                "--seed",
                str(first_seed + len(entries)),
                "--replay-dir",
                str(target),
            ]
            if model is not None:
                command += ["--model-a" if seat == 0 else "--model-b", str(model.resolve())]
            run = subprocess.run(
                command, capture_output=True, text=True, timeout=max(300, remaining * 10)
            )
            if run.returncode:
                raise RuntimeError(f"Arena failed: {run.stderr}")
            results = [json.loads(line) for line in run.stdout.splitlines()]
            if len(results) != remaining or any(
                r["agent_errors"] or r["engine_errors"] for r in results
            ):
                raise RuntimeError("Invalid arena results")
            entries.extend(inspect(path) for path in sorted(target.glob("game-*.jsonl")))
        if len(entries) != count:
            raise ValueError("Wrong replay count")
        print(f"{a} vs {b}: {count} games recorded", flush=True)
        return sorted(entries, key=lambda e: e["seed"])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        entries = [
            entry for pairing_entries in pool.map(pairing, jobs) for entry in pairing_entries
        ]
    manifest = dict(
        format="ptcg-rule-corpus-v1",
        schema_id=SCHEMA_ID,
        games=entries,
        teacher=teacher,
        model_sha256=sha256(model) if model is not None else None,
        seed=seed,
        games_per_pairing=per_seat[0] if per_seat[0] == per_seat[1] else None,
        games_by_seat=per_seat,
        games_per_opponent=sum(per_seat),
        total_games=total,
        workers=workers,
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    """Parse command-line arguments and run this tool."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, default=ROOT / "build/arena_cpp")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/dragapult/rule_replays")
    p.add_argument("--games", type=int, default=2, help="Games per opponent/seat (8 pairings)")
    p.add_argument(
        "--games-per-opponent",
        type=int,
        help="Total games per opponent, split across seats (e.g. 100 gives 400 total)",
    )
    p.add_argument("--seed", type=int, default=20260910)
    p.add_argument("--teacher", choices=["dragapult", "dragapult_gbdt"], default="dragapult")
    p.add_argument("--model", type=Path)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument(
        "--resume", action="store_true", help="Validate and continue complete seed prefixes"
    )
    a = p.parse_args()
    collect(
        a.binary,
        a.output_dir,
        a.games,
        a.seed,
        a.games_per_opponent,
        a.teacher,
        a.model,
        a.workers,
        a.resume,
    )


if __name__ == "__main__":
    main()
