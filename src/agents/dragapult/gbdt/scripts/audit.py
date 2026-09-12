"""Compare Python features/LightGBM scores/actions against native C++ evidence."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "src"))
import argparse
import io
import json
import subprocess
import tempfile

import numpy as np

from agents.dragapult.gbdt.bundle import load_bundle
from agents.dragapult.gbdt.features import NAMES, SCHEMA_ID, action_queries, check_deck
from agents.dragapult.gbdt.train import predict_banks
from agents.dragapult.replay import decisions, inspect_replay, replay_schema, sha256

if TYPE_CHECKING:
    from typing import Any


ATOL = RTOL = 1e-10


def audit(
    replay_dir: Path,
    model_dir: Path,
    output: Path,
    feature_dir: Path | None = None,
    probe: Path | None = None,
) -> dict[str, Any]:
    """Compare replay features, scores, and actions against Python and a native probe."""
    model_path = model_dir / "model.gbdt" if model_dir.is_dir() else model_dir
    training, boosters = load_bundle(model_path)
    if training["schema_id"] != SCHEMA_ID:
        raise ValueError("Model schema mismatch")
    for booster in boosters.values():
        if booster.feature_name() != NAMES:
            raise ValueError("LightGBM feature order mismatch")
    report = dict(
        schema_id=SCHEMA_ID,
        model_id=training["model_id"],
        atol=ATOL,
        rtol=RTOL,
        games=0,
        decisions=0,
        queries=0,
        feature_rows=0,
        feature_max_abs_error=0.0,
        prediction_max_abs_error=0.0,
        action_mismatches=0,
        agent_errors=0,
        engine_errors=0,
        probe_rows=0,
        replays=[],
    )
    for path in sorted(replay_dir.rglob("game-*.jsonl")):
        header, terminal = inspect_replay(path)
        if replay_schema(header) != SCHEMA_ID:
            raise ValueError("Replay schema mismatch")
        for seat, agent in enumerate(header["agents"]):
            if agent == "dragapult_gbdt":
                check_deck(header["decks"][seat])
                if header["models"][seat] != training["model_id"]:
                    raise ValueError("Replay was collected with another model")
        table = {
            c["cardId"]: c for c in json.loads((path.parent / "card_metadata.json").read_text())
        }
        matrices, native_scores, choices = [], [], []
        for record in decisions(path):
            if record["agent"] != "dragapult_gbdt":
                continue
            report["decisions"] += 1
            queries = list(action_queries(record["observation"], record["action"], table))
            cpp_queries = record.get("gbdt_queries", [])
            if len(queries) != len(cpp_queries):
                raise ValueError("Missing C++ queries")
            for query, cpp in zip(queries, cpp_queries, strict=True):
                if any(query[key] != cpp[key] for key in ("prefix", "options", "target")):
                    raise AssertionError("C++ candidate/prefix/STOP mismatch")
                x = np.asarray(query["features"], dtype=np.float64)
                native = np.asarray(cpp["features"], dtype=np.float64)
                if not np.array_equal(x, native):
                    raise AssertionError(f"Feature mismatch: {path} step={record['step']}")
                if cpp["scores"] is None or len(cpp["scores"]) != len(x):
                    raise ValueError("Missing C++ scores")
                matrices.append(x)
                native_scores.extend(cpp["scores"])
                choices.append((query["options"], query["target"]))
        if matrices:
            matrix = np.concatenate(matrices)
            scores = predict_banks(boosters, matrix, NAMES.index("context_0"))
            native = np.asarray(native_scores)
            np.testing.assert_allclose(scores, native, atol=ATOL, rtol=RTOL)
            report["prediction_max_abs_error"] = max(
                report["prediction_max_abs_error"], float(np.max(np.abs(scores - native)))
            )
            offset = 0
            for options, target in choices:
                if options[int(np.argmax(scores[offset : offset + len(options)]))] != target:
                    raise AssertionError(f"Python/C++ action mismatch: {path}")
                offset += len(options)
            report["feature_rows"] += len(matrix)
            report["queries"] += len(choices)
            report["games"] += 1
            report["replays"].append(
                dict(
                    path=str(path.relative_to(replay_dir)),
                    sha256=sha256(path),
                    steps=terminal["steps"],
                )
            )
    if not report["decisions"]:
        raise ValueError("No GBDT decisions were audited")
    if feature_dir is not None:
        if probe is None:
            raise ValueError("A native probe is required for dataset parity")
        for split in json.loads((feature_dir / "manifest.json").read_text())["splits"]:
            with np.load(feature_dir / f"{split}.npz", allow_pickle=False) as data:
                matrix = data["x"]
            # Use a temporary stream instead of command-line arguments for data.
            with tempfile.TemporaryFile(mode="w+") as stream:
                stream.write(f"{len(matrix)} {matrix.shape[1]}\n")
                np.savetxt(stream, matrix, fmt="%.17g")
                stream.seek(0)
                result = subprocess.run(
                    [str(probe.resolve()), str(model_path.resolve())],
                    stdin=stream,
                    capture_output=True,
                    text=True,
                    check=True,
                )
            native = np.loadtxt(io.StringIO(result.stdout), ndmin=1)
            scores = predict_banks(boosters, matrix, NAMES.index("context_0"))
            np.testing.assert_allclose(scores, native, atol=ATOL, rtol=RTOL)
            report["prediction_max_abs_error"] = max(
                report["prediction_max_abs_error"], float(np.max(np.abs(scores - native)))
            )
            report["probe_rows"] += len(matrix)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "replays"}), flush=True)
    return report


def main() -> None:
    """Parse command-line arguments and run this tool."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay-dir", type=Path, required=True)
    p.add_argument(
        "--model", "--model-dir", dest="model_dir", type=Path, default=ROOT / "models/dragapult"
    )
    p.add_argument("--output", type=Path, default=ROOT / "outputs/dragapult/gbdt_parity.json")
    p.add_argument("--feature-dir", type=Path)
    p.add_argument("--probe", type=Path, default=ROOT / "build/dragapult_gbdt_probe")
    a = p.parse_args()
    audit(a.replay_dir, a.model_dir, a.output, a.feature_dir, a.probe)


if __name__ == "__main__":
    main()
