"""Fit context-specific STOP thresholds on games excluded from tree fitting/refit."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "src"))
import argparse
import json

import numpy as np

from agents.mega_abomasnow.gbdt.bundle import load_bundle
from agents.mega_abomasnow.gbdt.features import DECK_COUNTS, NAMES, SCHEMA_ID
from agents.mega_abomasnow.gbdt.train import export_model, predict_banks
from agents.mega_abomasnow.replay import sha256

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any


def fit_threshold(examples: Sequence[tuple[float, bool]]) -> tuple[float, float]:
    """Fit the STOP threshold and return its stop/continue accuracy."""
    scores = np.array([r[0] for r in examples])
    stop = np.array([r[1] for r in examples])
    values = np.unique(scores)
    if len(values) > 401:
        values = values[np.linspace(0, len(values) - 1, 401, dtype=int)]
    candidates = np.r_[
        np.nextafter(values[0], -np.inf),
        values,
        np.nextafter(values[-1], np.inf),
        (values[1:] + values[:-1]) / 2,
    ]
    # STOP appears last: tied scores choose a regular candidate in both runtimes.
    threshold = max(candidates, key=lambda t: (np.mean((scores < t) == stop), -abs(t)))
    return float(threshold), float(np.mean((scores < threshold) == stop))


def calibrate(model: Path, features: Path, output: Path, min_queries: int = 10) -> dict[str, Any]:
    """Fit STOP thresholds on calibration games and export the calibrated bundle."""
    if output.exists():
        raise ValueError("Calibration output already exists")
    training, boosters = load_bundle(model)
    manifest = json.loads((features / "manifest.json").read_text())
    split = manifest["splits"]["calibration"]
    if training["schema_id"] != SCHEMA_ID or manifest["schema_id"] != SCHEMA_ID:
        raise ValueError("Schema mismatch")
    used = {g for games in training["split_games"].values() for g in games}
    if used.intersection(split["games"]):
        raise ValueError("Calibration games overlap model selection/training/test")
    if (
        sha256(features / "calibration.npz") != split["arrays_sha256"]
        or sha256(features / "calibration.groups.json") != split["groups_sha256"]
    ):
        raise ValueError("Calibration data checksum mismatch")
    meta = json.loads((features / "calibration.groups.json").read_text())
    with np.load(features / "calibration.npz", allow_pickle=False) as d:
        x = d["x"]
        groups = d["groups"]
    if (
        len(meta) != len(groups)
        or sum(groups) != len(x)
        or any(m["game"] not in split["games"] for m in meta)
    ):
        raise ValueError("Calibration alignment mismatch")
    scores = predict_banks(boosters, x, NAMES.index("context_0"))
    contexts = {}
    banks = {}
    offset = 0
    for size, m in zip(groups, meta, strict=True):
        row = scores[offset : offset + size]
        offset += size
        options = m["options"]
        if -1 not in options or len(options) < 2:
            continue
        maximum = max(v for o, v in zip(options, row, strict=True) if o != -1)
        example = (maximum, m["target"] == -1)
        context = m["context"]
        contexts.setdefault(context, []).append(example)
        banks.setdefault(context == 0, []).append(example)
    if not contexts:
        raise ValueError("No optional-selection decisions to calibrate")
    thresholds = {}
    report = {}
    for context, examples in contexts.items():
        threshold, accuracy = fit_threshold(
            examples if len(examples) >= min_queries else banks[context == 0]
        )
        thresholds[str(context)] = threshold
        report[str(context)] = dict(
            threshold=threshold,
            queries=len(examples),
            stop_accuracy=accuracy,
            fallback=len(examples) < min_queries,
        )
    training = dict(
        training,
        selection_thresholds=thresholds,
        calibration=dict(
            source_model_sha256=sha256(model),
            games=split["games"],
            contexts=report,
            objective="prefix stop/continue accuracy; bank fallback for rare contexts",
            unknown_context="learned STOP score",
            feature_manifest_sha256=sha256(features / "manifest.json"),
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    identity = export_model(boosters, output, SCHEMA_ID, NAMES, DECK_COUNTS, training)
    print(json.dumps(dict(model_id=identity, calibration=training["calibration"])), flush=True)
    return training


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--feature-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    calibrate(a.model, a.feature_dir, a.output)
