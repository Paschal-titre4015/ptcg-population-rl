"""Seat-balanced C++ matches and reproducible Bayesian rating estimates."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import itertools
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

from rating import estimate

if TYPE_CHECKING:
    from typing import Any


ROOT = Path(__file__).resolve().parents[2]
AGENTS = ("mega_lucario", "dragapult", "iono", "mega_abomasnow")


def sha(path: Path | str) -> str:
    """Hash an executable, model, deck, or completed match corpus."""
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Replace a JSON output atomically after writing its temporary file."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def roster(config: Path | None) -> dict[str, dict[str, str]]:
    """Resolve roster paths and hash each model and bundled deck."""
    entries = json.loads(config.read_text()) if config else {n: dict(agent=n) for n in AGENTS}
    if not isinstance(entries, dict) or len(entries) < 2:
        raise ValueError("Roster must map at least two labels to agent/model objects")
    result = {}
    for label, item in entries.items():
        if (
            not isinstance(label, str)
            or not label
            or not label.replace("_", "").replace("-", "").isalnum()
        ):
            raise ValueError("Roster labels must be identifiers")
        agent = item["agent"]
        deck = next((n for n in AGENTS if agent in (n, n + "_gbdt", n + "_transformer")), None)
        if deck is None:
            raise ValueError("Unknown agent " + agent)
        path = ROOT / f"src/agents/{deck}/deck.csv"
        entry = dict(agent=agent, deck=str(path.resolve()), deck_sha256=sha(path))
        if agent != deck:
            model = Path(item["model"])
            model = model if model.is_absolute() else config.parent / model
            entry.update(model=str(model.resolve()), model_sha256=sha(model))
        elif "model" in item:
            raise ValueError("Rule agents do not accept models")
        result[label] = entry
    return result


def collect(
    binary: Path,
    output: Path,
    entries: dict[str, dict[str, str]],
    games_per_pair: int = 200,
    seed: int = 202700000,
    workers: int = 2,
    resume: bool = False,
    prior_sigma: float = 600.0,
    pair_cap: int = 500,
    raw_bt: bool = False,
    max_sigma: float | None = None,
) -> dict[str, Any]:
    """Collect or resume seat-balanced matches and save Bayesian rating estimates."""
    if games_per_pair < 2 or games_per_pair % 2 or workers < 1:
        raise ValueError("Use positive workers and a positive even game count per pair")
    names = sorted(entries)
    pairs = list(itertools.combinations(names, 2))
    count = games_per_pair // 2
    if seed < 0 or seed + len(pairs) * 1000000 + count > 2**32:
        raise ValueError("Seed range overflow")
    if count >= 1000000:
        raise ValueError("At most 1,999,998 games per pair in one seeded corpus")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "collection.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = dict(
            format="ptcg-rating-corpus-v1",
            binary_sha256=sha(binary),
            roster=entries,
            seed=seed,
            schedule="sorted unordered pairs; same game seeds for both seats; pair stride 1000000",
        )
        manifest = output / "manifest.json"
        records_path = output / "games.jsonl"
        if manifest.exists():
            if not resume:
                raise ValueError("Use a new output directory or --resume")
            if json.loads(manifest.read_text()) != identity:
                raise ValueError("Binary, model, deck, roster or seed changed; use a new corpus")
        else:
            if records_path.exists():
                raise ValueError("Unidentified existing game log")
            atomic_json(manifest, identity)
        records = (
            [json.loads(line) for line in records_path.read_text().splitlines()]
            if records_path.exists()
            else []
        )
        seen = {(r["a"], r["b"], r["seed"]) for r in records}
        if len(seen) != len(records):
            raise ValueError("Duplicate game identity in saved corpus")
        if any(r["a"] not in names or r["b"] not in names or r["a"] == r["b"] for r in records):
            raise ValueError("Saved games are outside the comparison schedule")
        jobs = []
        for pair_index, (left, right) in enumerate(pairs):
            base = seed + pair_index * 1000000
            for a, b in ((left, right), (right, left)):
                completed = sorted(r["seed"] for r in records if r["a"] == a and r["b"] == b)
                if completed != list(range(base, base + len(completed))) or len(completed) > count:
                    raise ValueError(
                        "Resume requires a contiguous prefix and a nondecreasing game target"
                    )
                if len(completed) < count:
                    jobs.append((a, b, base + len(completed), count - len(completed)))

        def run(job: tuple[str, str, int, int]) -> list[dict[str, Any]]:
            """Run one seat assignment and retain results with explicit error attribution."""
            a, b, start, n = job
            command = [str(binary.resolve()), "--games", str(n), "--seed", str(start)]
            for seat, label in (("a", a), ("b", b)):
                e = entries[label]
                command += [f"--agent-{seat}", e["agent"], f"--deck-{seat}", e["deck"]]
                if "model" in e:
                    command += [f"--model-{seat}", e["model"]]
            env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
            run = subprocess.run(
                command, capture_output=True, text=True, timeout=max(300, n * 30), env=env
            )
            rows = [json.loads(line) for line in run.stdout.splitlines()]
            if len(rows) != n or [r.get("game") for r in rows] != list(range(n)):
                raise ValueError(f"Incomplete arena process {a}/{b}: {run.stderr[-1000:]}")
            if run.returncode not in (0, 1) or (
                run.returncode and not any(r["agent_errors"] or r["engine_errors"] for r in rows)
            ):
                raise ValueError(f"Unexpected arena exit {run.returncode}: {run.stderr[-1000:]}")
            return [
                dict(
                    a=a,
                    b=b,
                    seed=start + i,
                    result=r["result"],
                    steps=r["steps"],
                    agent_errors=r["agent_errors"],
                    engine_errors=r["engine_errors"],
                    error_seat=r.get("error_seat", -1),
                )
                for i, r in enumerate(rows)
            ]

        with ThreadPoolExecutor(max_workers=workers) as pool, records_path.open("a") as out:
            for rows in pool.map(run, jobs):
                for row in rows:
                    out.write(json.dumps(row, separators=(",", ":")) + "\n")
                out.flush()
                os.fsync(out.fileno())
                records.extend(rows)
                print(f"Recorded {len(records)} games", flush=True)
        report = estimate(names, records, prior_sigma, pair_cap, raw_bt, max_sigma)
        report.update(
            corpus_sha256=sha(records_path),
            games_per_pair=games_per_pair,
            workers=workers,
            blas_threads=1,
        )
        atomic_json(output / "ratings.json", report)
        with (output / "ratings.csv").open("w", newline="") as out:
            writer = csv.DictWriter(
                out,
                fieldnames=("agent", "mu", "sigma", "lcb", "games", "weight"),
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(report["ratings"])
        print(json.dumps(report["ratings"], ensure_ascii=False, indent=2))
        return report


def main() -> None:
    """Parse command-line arguments and run this tool."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--roster",
        type=Path,
        help="JSON mapping labels to {agent, model?}; model paths relative to this JSON",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--binary", type=Path, default=ROOT / "build/arena_cpp")
    p.add_argument("--games-per-pair", type=int, default=200)
    p.add_argument("--seed", type=int, default=202700000)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--prior-sigma", type=float, default=600.0)
    p.add_argument("--pair-cap", type=int, default=500)
    p.add_argument("--raw-bt", action="store_true", help="Disable profile weights and pair cap")
    p.add_argument(
        "--max-sigma",
        type=float,
        help="Fail after saving results if any posterior uncertainty exceeds this target",
    )
    a = p.parse_args()
    r = collect(
        a.binary,
        a.output_dir,
        roster(a.roster),
        a.games_per_pair,
        a.seed,
        a.workers,
        a.resume,
        a.prior_sigma,
        a.pair_cap,
        a.raw_bt,
        a.max_sigma,
    )
    if r["precision_met"] is False:
        p.exit(
            2,
            "Uncertainty target not met; collect more games or broaden the connected comparison pool.\n",
        )


if __name__ == "__main__":
    main()
