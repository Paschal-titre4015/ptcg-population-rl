"""Verify the unmodified upstream engine and build the C++ arena with official policies."""

from __future__ import annotations

import argparse
import importlib
import os
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

AGENTS = ("mega_lucario", "dragapult", "iono", "mega_abomasnow")

from fetch_cg_engine import DISTRIBUTION, ROOT, verify_tree


def main() -> None:
    """Parse command-line arguments and run this tool."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-dir", type=Path, default=ROOT / "src/engine/cg")
    parser.add_argument("--output", type=Path, default=ROOT / "build/arena_cpp")
    args = parser.parse_args()
    verify_tree(args.engine_dir, DISTRIBUTION)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generated = args.output.parent / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    for agent in AGENTS:
        importlib.import_module(f"agents.{agent}.gbdt.generate_schema").generate(
            generated / f"{agent}_schema.h"
        )
    # SciPy is already a LightGBM dependency; record its local BLAS path only in
    # the ignored binary, overridable at runtime with PTCG_OPENBLAS_LIBRARY.
    import scipy

    libraries = sorted(
        Path(scipy.__file__).resolve().parent.parent.glob("scipy.libs/*openblas*.so")
    )
    numeric_flags = ["-ldl"]
    if libraries:
        numeric_flags.append('-DPTCG_OPENBLAS_DEFAULT="' + str(libraries[0]) + '"')
    command = shlex.split(os.environ.get("CXX", "g++")) + [
        "-std=c++20",
        "-O3",
        "-DNDEBUG",
        "-pthread",
        "-I",
        str(args.engine_dir),
        "-I",
        str(generated),
        str(ROOT / "src/engine/cpp/arena_main.cpp"),
        "-o",
        str(args.output),
    ]
    command += numeric_flags
    print(shlex.join(command), flush=True)
    subprocess.run(command, check=True)
    for agent in AGENTS:
        for method in ("gbdt", "transformer"):
            name = f"{method}_probe" if agent == "mega_lucario" else f"{agent}_{method}_probe"
            command = shlex.split(os.environ.get("CXX", "g++")) + [
                "-std=c++20",
                "-O3",
                "-I",
                str(generated),
                str(ROOT / f"src/agents/{agent}/{method}/cpp/probe.cpp"),
                "-o",
                str(args.output.parent / name),
            ]
            if method == "transformer":
                command += numeric_flags
            subprocess.run(command, check=True)
    verify_tree(args.engine_dir, DISTRIBUTION)


if __name__ == "__main__":
    main()
