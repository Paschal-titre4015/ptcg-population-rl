"""Paired end-to-end self-play benchmark with identical seeds, models and threads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import statistics
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


ROOT = Path(__file__).resolve().parents[5]


def digest(path: Path) -> str:
    """Hash an executable or model used in the benchmark."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(
    binary: Path, model: Path, games: int, seed: int, output: Path, record: bool, sampled: bool
) -> dict[str, Any]:
    """Measure one arena process, including model loading and requested replay output."""
    deck = ROOT / "src/agents/mega_lucario/deck.csv"
    command = [
        str(binary.resolve()),
        "--agent-a",
        "mega_lucario_transformer",
        "--agent-b",
        "mega_lucario_transformer",
        "--model-a",
        str(model.resolve()),
        "--model-b",
        str(model.resolve()),
        "--deck-a",
        str(deck),
        "--deck-b",
        str(deck),
        "--games",
        str(games),
        "--seed",
        str(seed),
    ]
    if record:
        command += ["--replay-dir", str(output)]
    if sampled:
        command += ["--sample-seed-a", str(seed + 1000000), "--sample-seed-b", str(seed + 2000000)]
    env = dict(
        os.environ, PTCG_TRANSFORMER_BACKEND="blas", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"
    )
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_start = usage.ru_utime + usage.ru_stime
    start = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True, check=True, env=env)
    seconds = time.perf_counter() - start
    results = [json.loads(line) for line in result.stdout.splitlines()]
    if len(results) != games or any(r["agent_errors"] or r["engine_errors"] for r in results):
        raise ValueError("Self-play failed")
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_seconds = usage.ru_utime + usage.ru_stime - cpu_start
    steps = sum(r["steps"] for r in results)
    return dict(
        seconds=seconds,
        cpu_seconds=cpu_seconds,
        games_per_second=games / seconds,
        steps_per_second=steps / seconds,
        steps=steps,
        results=results,
    )


def benchmark(
    before: Path,
    after: Path | None,
    model: Path,
    output: Path,
    games: int = 16,
    repetitions: int = 3,
    seed: int = 202690000,
    record: bool = False,
    sampled: bool = False,
) -> dict[str, Any]:
    """Alternate seeded self-play runs and compare outcomes, actions, and throughput."""
    if games < 1 or repetitions < 1:
        raise ValueError("Positive games and repetitions required")
    if sampled and not record:
        raise ValueError("Sampling requires --replay to retain on-policy statistics")
    if output.exists():
        raise ValueError("Use a new output directory")
    output.mkdir(parents=True)
    rows = []
    for repetition in range(repetitions):
        pair = {}
        # Alternate order to reduce systematic cache/frequency drift.
        for name, binary in (
            [("before", before), ("after", after)]
            if repetition % 2 == 0
            else [("after", after), ("before", before)]
        ):
            if binary is None:
                continue
            pair[name] = run(
                binary,
                model,
                games,
                seed + repetition * games,
                output / f"{name}-{repetition}",
                record,
                sampled,
            )
            print(
                json.dumps(
                    dict(
                        repetition=repetition,
                        side=name,
                        **{k: v for k, v in pair[name].items() if k != "results"},
                    )
                ),
                flush=True,
            )
        if "after" in pair:
            if [
                {k: r[k] for k in ("game", "result", "steps", "agent_errors", "engine_errors")}
                for r in pair["before"]["results"]
            ] != [
                {k: r[k] for k in ("game", "result", "steps", "agent_errors", "engine_errors")}
                for r in pair["after"]["results"]
            ]:
                raise ValueError("Seeded outcomes or step counts differ")
            if record:
                for i in range(games):

                    def actions(path: Path) -> list[tuple[int, int, list[int]]]:
                        """Read the ordered decision, seat, and action sequence from a replay."""
                        return [
                            (r["step"], r["seat"], r["action"])
                            for line in path.open()
                            if (r := json.loads(line))["type"] == "decision"
                        ]

                    if actions(output / f"before-{repetition}/game-{i}.jsonl") != actions(
                        output / f"after-{repetition}/game-{i}.jsonl"
                    ):
                        raise ValueError("Seeded action sequences differ")
            pair["speedup"] = pair["before"]["seconds"] / pair["after"]["seconds"]
            pair["cpu_speedup"] = pair["before"]["cpu_seconds"] / pair["after"]["cpu_seconds"]
        rows.append(pair)
        (output / "progress.json").write_text(json.dumps(rows, indent=2) + "\n")
    cpu = next(
        line.split(":", 1)[1].strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines()
        if line.startswith("model name")
    )
    report = dict(
        cpu=cpu,
        workers=1,
        blas_threads=1,
        backend="blas",
        games_per_repetition=games,
        repetitions=repetitions,
        seed=seed,
        replay=record,
        sampled=sampled,
        scope="process wall time including model loading, observation/features, inference, engine and requested replay writing",
        before_binary_sha256=digest(before),
        after_binary_sha256=digest(after) if after else None,
        model_sha256=digest(model),
        rows=rows,
    )
    if after:
        report.update(
            median_speedup=statistics.median(r["speedup"] for r in rows),
            minimum_speedup=min(r["speedup"] for r in rows),
            median_cpu_speedup=statistics.median(r["cpu_speedup"] for r in rows),
        )
    (output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--games", type=int, default=16)
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--seed", type=int, default=202690000)
    p.add_argument("--replay", action="store_true")
    p.add_argument("--sampled", action="store_true")
    a = p.parse_args()
    benchmark(
        a.before,
        a.after,
        a.model,
        a.output_dir,
        a.games,
        a.repetitions,
        a.seed,
        a.replay,
        a.sampled,
    )
