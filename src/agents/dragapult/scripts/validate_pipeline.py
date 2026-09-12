"""Validate rule play, GBDT, Transformer distillation and PPO end to end.

The resulting weights are test fixtures, not trained playing agents.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
import argparse
import json
import os
import subprocess

import torch

from agents.dragapult.gbdt.features import DECK_COUNTS, SCHEMA_ID
from agents.dragapult.gbdt.features import NAMES as GBDT_NAMES
from agents.dragapult.gbdt.scripts.audit import audit as audit_gbdt
from agents.dragapult.gbdt.scripts.extract import extract as extract_gbdt
from agents.dragapult.gbdt.train import train as train_gbdt
from agents.dragapult.ppo.scripts.collect import DEFAULTS, collect
from agents.dragapult.ppo.train import train as train_ppo
from agents.dragapult.transformer.checkpoint import load_checkpoint
from agents.dragapult.transformer.features import CONTRACT_ID, NAMES
from agents.dragapult.transformer.scripts.audit import audit as audit_transformer
from agents.dragapult.transformer.scripts.export import export
from agents.dragapult.transformer.scripts.extract import extract as extract_transformer
from agents.dragapult.transformer.train import train as train_transformer

if TYPE_CHECKING:
    from typing import Any


AGENT = "dragapult"
AGENTS = ("mega_lucario", "dragapult", "iono", "mega_abomasnow")


def validate(output: Path, binary: Path, device: str = "cuda") -> dict[str, Any]:
    """Run a bounded rule-to-GBDT-to-Transformer-to-PPO validation cycle."""
    if output.exists() and any(output.iterdir()):
        raise ValueError("Validation output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    binary = binary.resolve()
    output = output.resolve()
    games = []

    def battle(
        left: str,
        right: str,
        directory: Path,
        seed: int,
        count: int = 1,
        models: dict[str, Path] | None = None,
    ) -> None:
        """Record one matchup and reject incomplete or failed arena results."""
        command = [
            str(binary),
            "--games",
            str(count),
            "--seed",
            str(seed),
            "--replay-dir",
            str(directory),
        ]
        for seat, name in (("a", left), ("b", right)):
            deck = next(n for n in AGENTS if name in (n, n + "_gbdt", n + "_transformer"))
            command += [
                f"--agent-{seat}",
                name,
                f"--deck-{seat}",
                str(ROOT / f"src/agents/{deck}/deck.csv"),
            ]
            if name != deck:
                command += [f"--model-{seat}", str(models[name])]
        run = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
            env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"),
        )
        records = [json.loads(line) for line in run.stdout.splitlines()]
        if len(records) != count or any(
            r["agent_errors"] or r["engine_errors"] or r["result"] not in (0, 1, 2) for r in records
        ):
            raise ValueError("Validation battle failed")
        games.extend(records)

    other = AGENTS[(AGENTS.index(AGENT) + 1) % len(AGENTS)]
    battle(AGENT, AGENT, output / "rule/self", 205000000, 2)
    battle(AGENT, other, output / "rule/left", 205000010)
    battle(other, AGENT, output / "rule/right", 205000020)
    extract_gbdt(output / "rule", output / "gbdt_features")
    train_gbdt(
        output / "gbdt_features",
        output / "gbdt",
        rounds=2,
        threads=1,
        schema_id=SCHEMA_ID,
        names=GBDT_NAMES,
        deck_counts=DECK_COUNTS,
    )
    teacher = output / "gbdt/model.gbdt"
    models = {AGENT + "_gbdt": teacher}
    for i, (left, right) in enumerate(
        ((AGENT + "_gbdt", AGENT + "_gbdt"), (AGENT + "_gbdt", other), (other, AGENT + "_gbdt"))
    ):
        battle(left, right, output / f"teacher/pair-{i}", 205000100 + i, models=models)
    gp = audit_gbdt(
        output / "teacher",
        teacher,
        output / "gbdt_parity.json",
        output / "gbdt_features",
        binary.parent / "dragapult_gbdt_probe",
    )
    extract_transformer(output / "teacher", teacher, output / "distillation")
    td = train_transformer(
        output / "distillation",
        output / "transformer",
        CONTRACT_ID,
        epochs=1,
        batch_size=2,
        device=device,
        max_updates=1,
    )
    checkpoint = output / "transformer/model.pt"
    model_bin = output / "transformer/model.bin"
    export(checkpoint, model_bin)
    models[AGENT + "_transformer"] = model_bin
    battle(
        AGENT + "_transformer",
        AGENT + "_transformer",
        output / "transformer_replays/self",
        205000200,
        models=models,
    )
    battle(
        other,
        AGENT + "_transformer",
        output / "transformer_replays/cross",
        205000201,
        models=models,
    )
    tp = audit_transformer(
        output / "transformer_replays",
        checkpoint,
        model_bin,
        output / "transformer_parity.json",
        output / "distillation",
        binary.parent / "dragapult_transformer_probe",
        8,
    )
    rollout = collect(
        checkpoint,
        output / "rollout",
        binary,
        opponents=["self"],
        games=2,
        workers=1,
        seed=205000300,
        sample_seed=205000400,
    )
    model, payload = load_checkpoint(checkpoint, CONTRACT_ID, DECK_COUNTS, NAMES)
    config = dict(
        DEFAULTS,
        device=device,
        seed=204,
        updates=1,
        batch_size=2,
        micro_batch_size=2,
        holdout_decisions=2,
    )
    pp = train_ppo(model, payload, checkpoint, output / "rollout", output / "ppo", config)
    export(output / "ppo/model.pt", output / "ppo/model.bin")
    models[AGENT + "_transformer"] = output / "ppo/model.bin"
    battle(
        AGENT + "_transformer",
        AGENT + "_transformer",
        output / "updated/self",
        205000500,
        models=models,
    )
    up = audit_transformer(
        output / "updated",
        output / "ppo/model.pt",
        output / "ppo/model.bin",
        output / "ppo_parity.json",
    )
    report = dict(
        agent=AGENT,
        purpose="validation fixtures only; not a strength evaluation",
        gbdt_rounds=2,
        distillation_updates=td["updates"],
        ppo_updates=pp["updates"],
        games=games,
        rollout_games=rollout["arena_results"],
        gbdt_parity=gp,
        transformer_parity=tp,
        ppo_parity=up,
        hardware=torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
    )
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Validation complete: {output}/validation.json", flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--binary", type=Path, default=ROOT / "build/arena_cpp")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    a = p.parse_args()
    validate(a.output_dir, a.binary, a.device)
