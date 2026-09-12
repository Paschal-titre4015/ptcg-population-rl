"""LightGBM candidate ranking, numerical tree export, and validation reporting."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import lightgbm as lgb
import numpy as np
from scipy import sparse

from agents.iono.replay import sha256

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
    from typing import Any

    from numpy.typing import NDArray


def export_model(
    boosters: dict[str, lgb.Booster],
    path: Path,
    schema_id: str,
    names: Sequence[str],
    deck_counts: dict[int, int],
    training: dict[str, Any] | None = None,
) -> str:
    """Write both numeric tree banks and their metadata to one bundle; return its ID."""
    body = [f"FEATURES {len(names)}", *names, f"DECK {len(deck_counts)}"]
    body.extend(f"{id} {count}" for id, count in sorted(deck_counts.items()))
    body += [f"ROUTE_FEATURE {names.index('context_0')}", "BANKS 2"]
    for bank, name in enumerate(("general", "main")):
        model = boosters[name].dump_model()
        if model["objective"] != "lambdarank":
            raise ValueError("Only LambdaRank numeric trees are supported")
        body += [f"BANK {bank}", f"TREES {len(model['tree_info'])}"]
        for tree in model["tree_info"]:
            nodes = []

            def visit(node: dict[str, Any]) -> int:
                """Flatten a numeric tree recursively and return the node index."""
                index = len(nodes)
                nodes.append(None)
                if "leaf_value" in node:
                    nodes[index] = (-1, 0.0, -1, -1, float(node["leaf_value"]))
                else:
                    if node["decision_type"] != "<=":
                        raise ValueError("Categorical/non-numeric splits are not supported")
                    left, right = visit(node["left_child"]), visit(node["right_child"])
                    nodes[index] = (
                        node["split_feature"],
                        float(node["threshold"]),
                        left,
                        right,
                        0.0,
                    )
                return index

            visit(tree["tree_structure"])
            body.append(f"NODES {len(nodes)}")
            for feature, threshold, left, right, value in nodes:
                if not np.isfinite([threshold, value]).all():
                    raise ValueError("Non-finite model parameter")
                body.append(f"{feature} {threshold:.17g} {left} {right} {value:.17g}")
    thresholds = (training or {}).get("selection_thresholds", {})
    if thresholds:
        body += [f"STOP_FEATURE {names.index('is_stop')}", f"THRESHOLDS {len(thresholds)}"]
        body += [
            f"{names.index('context_' + str(context))} {value:.17g}"
            for context, value in sorted(thresholds.items(), key=lambda item: int(item[0]))
        ]
    archive = json.dumps(
        dict(
            training=training if training is not None else dict(schema_id=schema_id),
            boosters={name: b.model_to_string() for name, b in boosters.items()},
        ),
        ensure_ascii=True,
        separators=(",", ":"),
    )
    if len(archive.encode()) > 67108864:
        raise ValueError("Model archive exceeds the C++ runtime limit")
    body += [f"ARCHIVE_BYTES {len(archive.encode())}", archive, "END"]
    payload = "\n".join(body) + "\n"
    model_id = hashlib.sha256((schema_id + "\n" + payload).encode()).hexdigest()
    version = "PTCG_GBDT_V3" if thresholds else "PTCG_GBDT_V2"
    path.write_text(f"{version}\n{schema_id}\nMODEL_ID {model_id}\n" + payload)
    return model_id


def bank_rows(data: dict[str, NDArray[Any]], main: bool, route: int) -> dict[str, NDArray[Any]]:
    """Select complete ranking queries belonging to one context bank."""
    parts, labels, groups, weights = [], [], [], []
    offset = 0
    for size in data["groups"]:
        x, y = data["x"][offset : offset + size], data["y"][offset : offset + size]
        if not np.all(x[:, route] == x[0, route]) or x[0, route] not in (0, 1):
            raise ValueError("Inconsistent MAIN context within a query")
        if (x[0, route] == 1) == main:
            parts.append(x)
            labels.append(y)
            groups.append(size)
            weights.append(data["weights"][offset : offset + size])
        offset += int(size)
    if not groups:
        raise ValueError("Both MAIN and non-MAIN ranking queries are required in each split")
    return dict(
        x=np.concatenate(parts),
        y=np.concatenate(labels),
        groups=np.asarray(groups, dtype=np.int32),
        weights=np.concatenate(weights),
    )


def predict_banks(
    boosters: dict[str, lgb.Booster], matrix: NDArray[Any], route: int, threads: int = 2
) -> NDArray[np.float64]:
    """Score candidates with their context bank and apply calibrated STOP thresholds."""
    main = matrix[:, route] == 1
    scores = np.empty(len(matrix), dtype=np.float64)
    for name, mask in (("main", main), ("general", ~main)):
        if mask.any():
            scores[mask] = boosters[name].predict(matrix[mask], num_threads=threads)
    for context, value in getattr(boosters, "thresholds", {}).items():
        mask = (matrix[:, boosters.names.index("is_stop")] == 1) & (
            matrix[:, boosters.names.index("context_" + str(context))] == 1
        )
        scores[mask] = value
    return scores


def query_accuracy(scores: NDArray[Any], labels: NDArray[Any], groups: NDArray[Any]) -> float:
    """Measure whether the first maximum-scoring candidate is the teacher target."""
    correct = offset = 0
    for size in groups:
        correct += labels[offset + int(np.argmax(scores[offset : offset + size]))] == 1
        offset += int(size)
    return float(correct / len(groups))


def context_weights(
    metadata: Sequence[dict[str, Any]], groups: NDArray[Any]
) -> NDArray[np.float64]:
    # Balance contexts, counting each observable decision once.
    """Balance context frequencies while counting each observed decision once."""
    counts, seen = {}, set()
    for m in metadata:
        key = (m["game"], m["step"], m["seat"])
        if key not in seen:
            seen.add(key)
            context = m["context"]
            if context != 0:
                counts[context] = counts.get(context, 0.0) + (1 / 6 if context == 14 else 1.0)
    positive = sorted(counts.values())
    median = positive[len(positive) // 2] if positive else 1.0
    weights = []
    for m, size in zip(metadata, groups, strict=True):
        context = m["context"]
        weight = (
            1.0
            if context == 0
            else (1 / 6 if context == 14 else 1.0)
            * min(4.0, max(0.25, (median / counts[context]) ** 0.5))
        )
        weights.extend([weight] * int(size))
    return np.asarray(weights)


def train(
    feature_dir: Path,
    output: Path,
    rounds: int = 80,
    threads: int = 2,
    seed: int = 42,
    *,
    schema_id: str,
    names: Sequence[str],
    deck_counts: dict[int, int],
    profile: str = "quick",
    refit_rounds: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Train and validate two LambdaRank banks, optionally refit, and export one model."""
    if profile not in ("search", "standard", "quick"):
        raise ValueError("Unknown training profile")
    full = profile != "quick"
    refit_rounds = dict(general=650, main=1100) if refit_rounds is None else dict(refit_rounds)
    if profile == "standard" and (
        set(refit_rounds) != {"general", "main"} or any(v < 1 for v in refit_rounds.values())
    ):
        raise ValueError("Both bank refit rounds must be positive")
    manifest_deck = {str(k): v for k, v in sorted(deck_counts.items())}
    if rounds < 1 or threads < 1:
        raise ValueError("Rounds and threads must be positive")
    manifest = json.loads((feature_dir / "manifest.json").read_text())
    if (
        manifest["schema_id"] != schema_id
        or manifest["feature_names"] != names
        or manifest["deck_counts"] != manifest_deck
    ):
        raise ValueError("Feature/deck schema mismatch")
    splits = ("train", "validation", "test") if full else ("train", "validation")
    if any(split not in manifest["splits"] for split in splits):
        raise ValueError(
            f"{profile} requires game splits: {splits}; extract with --test-fraction .1 for standard/search profiles"
        )
    all_games = [g for split in splits for g in manifest["splits"][split]["games"]]
    if len(set(all_games)) != len(all_games):
        raise ValueError("Game split overlap")
    arrays = {}
    for split in splits:
        path = feature_dir / f"{split}.npz"
        if sha256(path) != manifest["splits"][split]["arrays_sha256"]:
            raise ValueError("Feature array checksum mismatch")
        with np.load(path, allow_pickle=False) as data:
            arrays[split] = {key: data[key] for key in ("x", "y", "groups")}
        meta_path = feature_dir / f"{split}.groups.json"
        if sha256(meta_path) != manifest["splits"][split]["groups_sha256"]:
            raise ValueError("Group metadata checksum mismatch")
        metadata = json.loads(meta_path.read_text())
        d = arrays[split]
        d["metadata"] = metadata
        d["weights"] = context_weights(metadata, d["groups"]) if full else np.ones(len(d["y"]))
        if len(metadata) != len(d["groups"]) or any(
            m["game"] not in manifest["splits"][split]["games"] for m in metadata
        ):
            raise ValueError("Group belongs to another game split")
        if np.any(d["groups"] < 2) or not np.isin(d["y"], [0, 1]).all():
            raise ValueError("Invalid ranking groups/labels")
        offset = 0
        for size in d["groups"]:
            if d["y"][offset : offset + size].sum() != 1:
                raise ValueError("Each ranking query must have exactly one teacher target")
            offset += int(size)
        if (
            d["x"].shape != (len(d["y"]), len(names))
            or d["groups"].sum() != len(d["y"])
            or not np.isfinite(d["x"]).all()
        ):
            raise ValueError("Invalid training arrays")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Model output directory must be empty")
    params = dict(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[1],
        learning_rate=0.08,
        num_leaves=15,
        max_depth=5,
        min_data_in_leaf=10,
        feature_pre_filter=False,
        seed=seed,
        num_threads=threads,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
        use_missing=False,
        zero_as_missing=False,
        label_gain=[0, 1],
    )
    if full:
        params = dict(
            objective="lambdarank",
            metric="ndcg",
            ndcg_eval_at=[1, 3, 5],
            learning_rate=0.05,
            feature_fraction=0.85,
            bagging_fraction=0.9,
            bagging_freq=1,
            min_child_samples=10,
            max_depth=-1,
            seed=seed,
            feature_fraction_seed=seed + 1,
            bagging_seed=seed + 2,
            data_random_seed=seed + 3,
            deterministic=True,
            force_col_wise=True,
            num_threads=threads,
            verbosity=-1,
            lambdarank_truncation_level=30,
        )
    if profile == "standard":
        params["min_child_samples"] = 20
    boosters, metrics, iterations, trials, selected = {}, {}, {}, {}, {}
    fitted_iterations = {}
    for name in ("general", "main"):
        data = {
            split: bank_rows(d, name == "main", names.index("context_0"))
            for split, d in arrays.items()
        }
        datasets = {}
        for split, d in data.items():
            datasets[split] = lgb.Dataset(
                sparse.csr_matrix(d["x"]),
                label=d["y"],
                group=d["groups"],
                weight=d["weights"],
                feature_name=names,
                reference=datasets.get("train"),
                free_raw_data=False,
            )
        leaves = ([63, 127] if name == "main" else [31, 63]) if full else [15]
        if profile == "standard":
            leaves = [127 if name == "main" else 63]
        trials[name] = []
        best_score, best = -1.0, None
        for leaf_count in leaves:
            config = dict(params, num_leaves=leaf_count)
            print(f"Training {name}: leaves={leaf_count}", flush=True)
            booster = lgb.train(
                config,
                datasets["train"],
                num_boost_round=rounds,
                valid_sets=[datasets["validation"]],
                valid_names=["validation"],
                callbacks=[
                    lgb.early_stopping(100 if full else 10, verbose=False),
                    lgb.log_evaluation(100),
                ],
            )
            d = data["validation"]
            accuracy = query_accuracy(
                booster.predict(d["x"], num_threads=threads), d["y"], d["groups"]
            )
            trials[name].append(
                dict(
                    params=config,
                    best_iteration=booster.best_iteration,
                    validation_top1=accuracy,
                    selection_score=1.25 * accuracy,
                )
            )
            if accuracy > best_score:
                best_score, best = accuracy, booster
                selected[name] = config
        iterations[name] = best.best_iteration
        metrics[name] = {
            split: dict(
                query_top1=query_accuracy(
                    best.predict(d["x"], num_threads=threads), d["y"], d["groups"]
                ),
                queries=len(d["groups"]),
                rows=len(d["y"]),
            )
            for split, d in data.items()
        }
        if full:
            combined = {
                key: np.concatenate([arrays[s][key] for s in ("train", "validation")])
                for key in ("x", "y", "groups")
            }
            metadata = arrays["train"]["metadata"] + arrays["validation"]["metadata"]
            combined["weights"] = context_weights(metadata, combined["groups"])
            refit = bank_rows(combined, name == "main", names.index("context_0"))
            dataset = lgb.Dataset(
                sparse.csr_matrix(refit["x"]),
                label=refit["y"],
                group=refit["groups"],
                weight=refit["weights"],
                feature_name=names,
            )
            best = lgb.train(
                selected[name],
                dataset,
                num_boost_round=refit_rounds[name] if profile == "standard" else iterations[name],
            )
        fitted_iterations[name] = best.current_iteration()
        boosters[name] = best
    report = dict(
        schema_id=schema_id,
        deck_counts=manifest_deck,
        feature_names=names,
        profile=profile,
        params=params,
        selected_params=selected,
        trials=trials,
        requested_rounds=rounds,
        early_stopping_rounds=100 if full else 10,
        best_iteration=iterations,
        fitted_iteration=fitted_iterations,
        requested_refit_rounds=refit_rounds if profile == "standard" else None,
        metrics=metrics,
        metrics_stage="validation-selected, before refit",
        refit_splits=["train", "validation"] if full else ["train"],
        split_games={split: manifest["splits"][split]["games"] for split in splits},
        feature_manifest_sha256=sha256(feature_dir / "manifest.json"),
        lightgbm_version=lgb.__version__,
    )
    model_id = export_model(boosters, output / "model.gbdt", schema_id, names, deck_counts, report)
    report.update(model_id=model_id, model_sha256=sha256(output / "model.gbdt"))
    print(
        json.dumps(dict(model_id=model_id, best_iteration=iterations, metrics=metrics)), flush=True
    )
    return report
