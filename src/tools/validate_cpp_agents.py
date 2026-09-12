"""Run all official C++ matchups and compare every input/action with Python."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import itertools
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

from fetch_official_assets import CG_SHA256, OFFICIAL_AGENTS

if TYPE_CHECKING:
    from types import ModuleType
    from typing import Any


ROOT = Path(__file__).resolve().parents[2]
AGENTS = ROOT / "src/agents"


def verify(path: Path, expected: str) -> None:
    """Verify a notebook, API file, or deck before loading it."""
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")


def load(
    agent: dict[str, str], seat: str | int, notebook_root: Path | None
) -> tuple[ModuleType, ModuleType | None]:
    """Load isolated local and optional original-notebook policies for one seat."""
    path = AGENTS / agent["name"] / "main.py"
    verify(path.with_name("deck.csv"), agent["deck_sha256"])
    spec = importlib.util.spec_from_file_location(f"{agent['name']}_{seat}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference = None
    if notebook_root is not None:
        slug = agent["notebook"].split("/")[1]
        notebook = notebook_root / slug / f"{slug}.ipynb"
        verify(notebook, agent["notebook_sha256"])
        cells = json.loads(notebook.read_text())["cells"]
        sources = [
            "".join(c["source"]).split("\n", 1)[1]
            for c in cells
            if c["cell_type"] == "code" and "".join(c["source"]).startswith("%%writefile main.py\n")
        ]
        if len(sources) != 1:
            raise ValueError(f"Expected one main.py cell: {notebook}")
        source = sources[0]
        reference = types.ModuleType(f"reference_{agent['name']}_{seat}")
        # The unmodified official code reads deck.csv from cwd during import.
        previous = Path.cwd()
        try:
            os.chdir(path.parent)
            exec(compile(source, str(notebook), "exec"), reference.__dict__)
        finally:
            os.chdir(previous)
    return module, reference


NAMES = tuple(agent["name"] for agent in OFFICIAL_AGENTS)


def project(observation: dict[str, Any]) -> dict[str, Any]:
    """Independent projection of the upstream JSON onto native policy fields."""
    source = observation["current"]
    current = {
        key: source[key]
        for key in (
            "turn",
            "yourIndex",
            "firstPlayer",
            "supporterPlayed",
            "energyAttached",
            "stadium",
            "looking",
        )
    }
    current["players"] = []
    for player in source["players"]:
        view = {
            key: player[key]
            for key in (
                "handCount",
                "deckCount",
                "asleep",
                "paralyzed",
                "active",
                "bench",
                "discard",
                "prize",
            )
        }
        view["hand"] = player["hand"] or []
        current["players"].append(view)
    source = observation["select"]
    selection = {
        key: source[key]
        for key in (
            "context",
            "minCount",
            "maxCount",
            "remainDamageCounter",
            "deck",
            "contextCard",
            "effect",
        )
    }
    selection["option"] = [
        {
            key: option.get(key, -1)
            for key in (
                "type",
                "number",
                "area",
                "index",
                "playerIndex",
                "inPlayArea",
                "inPlayIndex",
                "attackId",
            )
        }
        for option in source["option"]
    ]
    logs = []
    for log in observation["logs"]:
        item = dict(type=log["type"], playerIndex=-1, attackId=-1, fromArea=-1, toArea=-1)
        if log["type"] in (3, 6, 7, 15):
            item["playerIndex"] = log["playerIndex"]
        if log["type"] == 15:
            item["attackId"] = log["attackId"]
        if log["type"] in (6, 7):
            item["fromArea"], item["toArea"] = log["fromArea"], log["toArea"]
        logs.append(item)
    return dict(current=current, select=selection, logs=logs)


def compare(
    trace_path: Path,
    agents: dict[str, dict[str, str]],
    metadata: list[dict[str, Any]],
    notebook_root: Path | None = None,
) -> dict[str, Any]:
    """Compare every native decision with Python and the optional original notebook."""
    modules = {}
    count = 0
    contexts = set()
    multiple = optional_stop = 0
    with trace_path.open() as trace:
        for line in trace:
            record = json.loads(line)
            game, seat = record["game"], record["seat"]
            location = f"{trace_path.name} game={game} step={record['step']} seat={seat}"
            obs = record["observation"]
            if project(obs) != record["view"]:
                raise AssertionError(f"Observation projection mismatch: {location}")
            key = (game, seat)
            if key not in modules:
                policy, notebook = load(
                    agents[record["agent"]], f"cpp_{game}_{seat}", notebook_root
                )
                # The reference uses the exact card database of this C++ engine.
                # cg-lib supplies only the Python observation classes here.
                for module in (policy, notebook):
                    if module is not None:
                        module.card_table = {
                            card["cardId"]: SimpleNamespace(**card) for card in metadata
                        }
                modules[key] = policy, notebook
            for module in modules[key]:
                if module is None:
                    continue
                expected = module.agent(copy.deepcopy(obs))
                if expected != record["action"]:
                    raise AssertionError(
                        f"Action mismatch: {location} agent={record['agent']} "
                        f"context={obs['select']['context']} Python={expected} C++={record['action']}"
                    )
            selection = obs["select"]
            action = record["action"]
            if not (
                selection["minCount"] <= len(action) <= selection["maxCount"]
                and len(action) == len(set(action))
                and all(0 <= index < len(selection["option"]) for index in action)
            ):
                raise AssertionError(f"Invalid action: {location}")
            contexts.add(selection["context"])
            multiple += len(action) > 1
            optional_stop += len(action) < selection["maxCount"]
            count += 1
    return dict(
        decisions=count,
        contexts=sorted(contexts),
        multiple_selections=multiple,
        optional_stops=optional_stop,
    )


def main() -> None:
    """Parse command-line arguments and run this tool."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/arena_cpp")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/cpp_agents")
    parser.add_argument("--cg-root", type=Path, default=ROOT / "external/official_assets/cg-lib")
    parser.add_argument("--notebook-root", type=Path)
    parser.add_argument(
        "--games", type=int, default=1, help="Games per ordered pairing (16 pairings)"
    )
    args = parser.parse_args()
    if args.games < 1:
        parser.error("--games must be positive")
    for filename, expected in CG_SHA256.items():
        verify(args.cg_root / "cg" / filename, expected)
    sys.path.insert(0, str(args.cg_root.resolve()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    agents = {agent["name"]: agent for agent in OFFICIAL_AGENTS}
    reports = []
    failed = False
    for a, b in itertools.product(NAMES, repeat=2):
        trace = args.output_dir / f"{a}__{b}.jsonl"
        metadata_path = args.output_dir / "card_metadata.json"
        command = [
            str(args.binary.resolve()),
            "--agent-a",
            a,
            "--agent-b",
            b,
            "--deck-a",
            str(ROOT / f"src/agents/{a}/deck.csv"),
            "--deck-b",
            str(ROOT / f"src/agents/{b}/deck.csv"),
            "--games",
            str(args.games),
            "--trace",
            str(trace),
            "--metadata",
            str(metadata_path),
        ]
        report = dict(a=a, b=b, agent_errors=0, engine_errors=0, parity_errors=0)
        try:
            run = subprocess.run(command, capture_output=True, text=True, timeout=120)
            report["games"] = [json.loads(line) for line in run.stdout.splitlines()]
            for game in report["games"]:
                report["agent_errors"] += game["agent_errors"]
                report["engine_errors"] += game["engine_errors"]
            if run.returncode or len(report["games"]) != args.games:
                raise RuntimeError(f"Arena failed ({run.returncode}): {run.stderr.strip()}")
            report.update(
                compare(trace, agents, json.loads(metadata_path.read_text()), args.notebook_root)
            )
            if report["decisions"] != sum(game["steps"] for game in report["games"]):
                raise AssertionError("Trace does not cover every decision")
        except Exception as error:
            if isinstance(error, AssertionError):
                report["parity_errors"] += 1
            report["error"] = str(error)
            failed = True
        reports.append(report)
        print(json.dumps(report), flush=True)
        (args.output_dir / "validation.json").write_text(json.dumps(reports, indent=2) + "\n")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
