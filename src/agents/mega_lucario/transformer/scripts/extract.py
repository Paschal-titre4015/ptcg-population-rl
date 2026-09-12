"""Distillation shards: whole-game splits, observable tokens and GBDT score targets."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "src"))
import argparse
import json

import numpy as np

from agents.mega_lucario.gbdt.bundle import load_bundle
from agents.mega_lucario.gbdt.features import (
    DECK_COUNTS,
    NAMES,
    SCHEMA_ID,
    action_queries,
    check_deck,
)
from agents.mega_lucario.gbdt.train import predict_banks
from agents.mega_lucario.replay import decisions, inspect_replay, replay_schema, sha256
from agents.mega_lucario.transformer.features import (
    CONTRACT,
    CONTRACT_ID,
    token_indices,
    tokenize,
    with_prefix,
)
from agents.mega_lucario.transformer.features import (
    NAMES as TOKEN_NAMES,
)

if TYPE_CHECKING:
    from typing import Any


def extract_shard(job: tuple[Path, Path, Path, str, int, list[Path]]) -> dict[str, Any]:
    """Write one distillation shard after verifying teacher actions and native scores."""
    replay_dir, model, output, split, index, selected = job
    training, boosters = load_bundle(model)
    raw_features = []
    raw_scores = []
    features = []
    scores = []
    sizes = []
    targets = []
    metadata = []
    games = []
    forced = 0
    outcomes = []
    decision_sizes = []
    for path in selected:
        header, terminal = inspect_replay(path)
        digest = sha256(path)
        if replay_schema(header) != SCHEMA_ID or "mega_lucario_gbdt" not in header["agents"]:
            raise ValueError("Wrong teacher/schema")
        for seat, name in enumerate(header["agents"]):
            if name == "mega_lucario_gbdt":
                check_deck(header["decks"][seat])
                if header["models"][seat] != training["model_id"]:
                    raise ValueError("Mixed teacher weights")
        table = {
            c["cardId"]: c for c in json.loads((path.parent / "card_metadata.json").read_text())
        }
        games.append(
            dict(
                path=str(path.relative_to(replay_dir)),
                sha256=digest,
                seed=header["seed"],
                agents=header["agents"],
                metadata_sha256=sha256(path.parent / "card_metadata.json"),
                **terminal,
            )
        )
        for record in decisions(path):
            if record["agent"] != "mega_lucario_gbdt":
                continue
            queries = list(action_queries(record["observation"], record["action"], table))
            native = record.get("gbdt_queries", [])
            if len(queries) != len(native):
                raise ValueError("Missing teacher queries")
            initial_tokens = None
            begin_queries = len(sizes)
            for query, cpp in zip(queries, native, strict=True):
                if any(query[k] != cpp[k] for k in ("options", "prefix", "target")):
                    raise ValueError("Selection mismatch")
                x = np.asarray(query["features"], dtype=np.float64)
                if not np.array_equal(x, np.asarray(cpp["features"])):
                    raise ValueError("Python/C++ feature mismatch")
                score = np.asarray(cpp["scores"], dtype=np.float64)
                if score.shape != (len(x),) or not np.isfinite(score).all():
                    raise ValueError("Missing teacher scores")
                target = query["options"].index(query["target"])
                if int(np.argmax(score)) != target:
                    raise ValueError("Teacher action/score mismatch")
                if len(x) == 1:
                    forced += 1
                raw_features.append(x)
                raw_scores.append(score)
                if initial_tokens is None:
                    rows_by_option = dict(
                        zip(queries[0]["options"], queries[0]["features"], strict=True)
                    )
                    initial_tokens = tokenize(record["observation"], [], table, rows_by_option)
                tokens = with_prefix(initial_tokens, record["observation"], query["prefix"])
                indices = token_indices(record["observation"], query["options"])
                labels = np.full(len(tokens), -1e9, np.float32)
                labels[indices] = score
                features.append(tokens)
                scores.append(labels)
                sizes.append(len(tokens))
                targets.append(indices[target])
                metadata.append(
                    dict(
                        game=digest,
                        step=record["step"],
                        seat=record["seat"],
                        context=record["observation"]["select"]["context"],
                        prefix=query["prefix"],
                        options=query["options"],
                        target=query["target"],
                    )
                )
            if len(sizes) > begin_queries:
                decision_sizes.append(len(sizes) - begin_queries)
                outcomes.append(
                    0.0
                    if terminal["result"] == 2
                    else (1.0 if terminal["result"] == record["seat"] else -1.0)
                )
    if not features:
        raise ValueError("No trainable queries in shard")
    raw = np.concatenate(raw_features)
    teacher = np.concatenate(raw_scores)
    reference = predict_banks(boosters, raw, NAMES.index("context_0"), threads=1)
    np.testing.assert_allclose(reference, teacher, atol=1e-10, rtol=1e-10)
    name = f"{split}-{index:04d}"
    np.savez_compressed(
        output / f"{name}.npz",
        x=np.concatenate(features),
        scores=np.concatenate(scores),
        sizes=np.asarray(sizes, dtype=np.int32),
        targets=np.asarray(targets, dtype=np.int32),
        decision_sizes=np.asarray(decision_sizes, dtype=np.int32),
        outcomes=np.asarray(outcomes, dtype=np.float32),
    )
    (output / f"{name}.json").write_text(json.dumps(metadata, separators=(",", ":")) + "\n")
    shard = dict(
        path=f"{name}.npz",
        sha256=sha256(output / f"{name}.npz"),
        metadata_sha256=sha256(output / f"{name}.json"),
        decisions=len(outcomes),
        queries=len(sizes),
        rows=sum(sizes),
        max_tokens=max(sizes),
    )
    result = dict(
        split=split,
        games=games,
        forced=forced,
        shard=shard,
        score_error=float(np.max(np.abs(reference - teacher))),
    )
    print(
        json.dumps(dict(split=split, shard=index, games=len(games), queries=len(sizes))), flush=True
    )
    return result


def extract(
    replay_dir: Path,
    model: Path,
    output: Path,
    seed: int = 204,
    shard_games: int = 50,
    workers: int = 1,
    validation_fraction: float = 0.05,
    test_fraction: float = 0.0,
) -> dict[str, Any]:
    """Partition whole games and extract teacher-scored token shards."""
    if (
        not 0 < validation_fraction < 1
        or not 0 <= test_fraction < 1
        or validation_fraction + test_fraction >= 1
    ):
        raise ValueError("Invalid game split fractions")
    if shard_games < 1 or workers < 1:
        raise ValueError("shard-games/workers must be positive")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty")
    training, _ = load_bundle(model)
    if training["schema_id"] != SCHEMA_ID:
        raise ValueError("Teacher schema mismatch")

    def seed_key(path: Path) -> tuple[int, str]:
        """Order replays deterministically by seed and path."""
        with path.open() as stream:
            header = json.loads(next(stream))
        if type(header.get("seed")) is not int:
            raise ValueError("Seeded replays required")
        return header["seed"], str(path)

    paths = sorted(replay_dir.rglob("game-*.jsonl"), key=seed_key)
    if len(paths) < 3:
        raise ValueError("At least three games required")
    permutation = np.random.default_rng(seed).permutation(len(paths))
    valid = max(1, round(len(paths) * validation_fraction))
    test = max(1, round(len(paths) * test_fraction)) if test_fraction else 0
    if valid + test >= len(paths):
        raise ValueError("No training games remain")
    assignment = {
        int(i): ("validation" if j < valid else "test" if j < valid + test else "train")
        for j, i in enumerate(permutation)
    }
    output.mkdir(parents=True, exist_ok=True)
    report = dict(
        format="ptcg-distillation-v2",
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        contract=CONTRACT,
        contract_id=CONTRACT_ID,
        feature_names=TOKEN_NAMES,
        deck_counts={str(k): v for k, v in DECK_COUNTS.items()},
        teacher_model_id=training["model_id"],
        teacher_model_sha256=sha256(model),
        split_seed=seed,
        splits={},
        feature_max_abs_error=0.0,
        teacher_score_max_abs_error=0.0,
    )
    jobs = []
    for split in ("train", "validation", *(["test"] if test else [])):
        report["splits"][split] = dict(games=[], shards=[], forced_queries=0)
        selected = [p for i, p in enumerate(paths) if assignment[i] == split]
        for index, start in enumerate(range(0, len(selected), shard_games)):
            jobs.append(
                (replay_dir, model, output, split, index, selected[start : start + shard_games])
            )
    seen = set()
    episodes = set()

    def add(result: dict[str, Any]) -> None:
        """Add a completed shard to the split manifest."""
        for g in result["games"]:
            episode = (g["seed"], tuple(g["agents"]))
            if g["sha256"] in seen or episode in episodes:
                raise ValueError("Duplicate replay/seeded episode")
            seen.add(g["sha256"])
            episodes.add(episode)
        split = report["splits"][result["split"]]
        split["games"] += result["games"]
        split["shards"].append(result["shard"])
        split["forced_queries"] += result["forced"]
        report["teacher_score_max_abs_error"] = max(
            report["teacher_score_max_abs_error"], result["score_error"]
        )

    if workers == 1:
        for job in jobs:
            add(extract_shard(job))
    else:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            for result in pool.map(extract_shard, jobs):
                add(result)
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    """Parse command-line arguments and run this tool."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay-dir", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=204)
    p.add_argument("--shard-games", type=int, default=50)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--validation-fraction", type=float, default=0.05)
    p.add_argument("--test-fraction", type=float, default=0.0)
    a = p.parse_args()
    extract(
        a.replay_dir,
        a.model,
        a.output_dir,
        a.seed,
        a.shard_games,
        a.workers,
        a.validation_fraction,
        a.test_fraction,
    )


if __name__ == "__main__":
    main()
