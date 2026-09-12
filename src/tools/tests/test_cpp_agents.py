"""Compile native policy contract tests without any external engine/assets."""

from __future__ import annotations

import importlib
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agents.mega_lucario.gbdt.generate_schema import generate


class CppAgentContracts(unittest.TestCase):
    """Validate cpp agent contracts."""

    def test_native_contracts(self) -> None:
        """Check native contracts."""
        source = Path(__file__).with_name("official_agents_test.cpp")
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "official_agents_test"
            header = generate(Path(directory) / "mega_lucario_schema.h")
            for agent in ("dragapult", "iono", "mega_abomasnow"):
                importlib.import_module(f"agents.{agent}.gbdt.generate_schema").generate(
                    Path(directory) / f"{agent}_schema.h"
                )
            command = shlex.split(os.environ.get("CXX", "g++"))
            subprocess.run(
                command
                + ["-std=c++20", "-O0", "-I", str(header.parent), str(source), "-o", str(binary)],
                check=True,
            )
            subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    unittest.main()
