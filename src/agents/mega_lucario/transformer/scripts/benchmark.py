"""Measure forward-only CPU inference and compare float32 backends with double."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "src"))
import argparse
import io
import json
import os
import platform
import shlex
import subprocess

import numpy as np
import scipy

from agents.mega_lucario.gbdt.generate_schema import generate
from agents.mega_lucario.replay import sha256

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any


def benchmark(
    model: Path, output: Path, repetitions: int = 20, tokens: Sequence[int] = (32, 50, 120)
) -> dict[str, Any]:
    """Measure CPU forward latency and compare available backends with double precision."""
    if repetitions < 1 or not tokens or any(n < 20 or n > 120 for n in tokens):
        raise ValueError("Positive repetitions and 20..120 tokens required")
    output.mkdir(parents=True, exist_ok=True)
    generate(output / "mega_lucario_schema.h")
    binary = output / "benchmark"
    libraries = sorted(
        Path(scipy.__file__).resolve().parent.parent.glob("scipy.libs/*openblas*.so")
    )
    command = shlex.split(os.environ.get("CXX", "g++")) + [
        "-std=c++20",
        "-O3",
        "-I",
        str(output),
        str(ROOT / "src/agents/mega_lucario/transformer/cpp/benchmark.cpp"),
        "-ldl",
        "-o",
        str(binary),
    ]
    subprocess.run(command, check=True)
    backends = ["reference", "scalar"]
    cpuinfo = Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else ""
    if "avx2" in cpuinfo and "fma" in cpuinfo:
        backends.append("avx2")
    if libraries:
        backends.append("blas")
    records = []
    for n in tokens:
        rng = np.random.default_rng(n)
        x = np.zeros((n, 198))
        x[:, 0] = 1
        x[:, 4:6] = -1
        x[1:13, 1] = 1
        x[13:19, 1] = 2
        x[19:-1, 1] = 3
        x[-1, 1] = 4
        x[19:, 2] = 1
        x[:, 6:102] = rng.normal(0, 0.1, (n, 96))
        stream = io.StringIO()
        stream.write(f"{n} 198\n")
        np.savetxt(stream, x, fmt="%.9g")
        reference = None
        for backend in backends:
            env = dict(
                os.environ,
                PTCG_TRANSFORMER_BACKEND=backend,
                OPENBLAS_NUM_THREADS="1",
                OMP_NUM_THREADS="1",
            )
            if libraries:
                env["PTCG_OPENBLAS_LIBRARY"] = str(libraries[0])
            result = subprocess.run(
                [str(binary.resolve()), str(model.resolve()), str(repetitions)],
                input=stream.getvalue(),
                text=True,
                capture_output=True,
                check=True,
                env=env,
            )
            row = json.loads(result.stdout)
            scores = np.array([v for v in row.pop("scores") if v is not None])
            values = np.r_[scores, row["value"]]
            if reference is None:
                reference = values
            np.testing.assert_allclose(values, reference, atol=2e-5, rtol=2e-5)
            if np.argmax(scores) != np.argmax(reference[:-1]):
                raise ValueError("Backend action mismatch")
            row.update(
                requested_backend=backend,
                tokens=n,
                max_abs_error=float(abs(values - reference).max()),
            )
            records.append(row)
            print(json.dumps(row), flush=True)
    hardware = next(
        (
            line.split(":", 1)[1].strip()
            for line in cpuinfo.splitlines()
            if line.startswith("model name")
        ),
        platform.processor(),
    )
    report = dict(
        hardware=hardware,
        workers=1,
        blas_threads=1,
        repetitions=repetitions,
        warmup=3,
        scope="forward only, including raw conversion/temporary allocation; excludes model loading, features and arena",
        inputs="seeded synthetic valid tokens; no match-throughput claim",
        token_counts=list(tokens),
        compiler=command,
        model_sha256=sha256(model),
        atol=2e-5,
        rtol=2e-5,
        results=records,
    )
    (output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--repetitions", type=int, default=20)
    p.add_argument("--tokens", nargs="+", type=int, default=[32, 50, 120])
    a = p.parse_args()
    benchmark(a.model, a.output_dir, a.repetitions, a.tokens)
