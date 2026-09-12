"""Masked soft-score distillation with held-out game validation and test."""

from __future__ import annotations

import hashlib
import json
import random
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.nn import functional as F

from agents.iono.replay import sha256
from agents.iono.transformer.model import ARCHITECTURE, ObservationTransformer, select_logits

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from typing import Any

    from numpy.typing import NDArray


def load_split(directory: Path, manifest: dict[str, Any], split: str) -> dict[str, NDArray[Any]]:
    """Load and validate shards while preserving whole-game and decision boundaries."""
    features = []
    scores = []
    sizes = []
    targets = []
    decision_sizes = []
    outcomes = []
    for shard in manifest["splits"][split]["shards"]:
        path = directory / shard["path"]
        if (
            sha256(path) != shard["sha256"]
            or sha256(path.with_suffix(".json")) != shard["metadata_sha256"]
        ):
            raise ValueError("Shard checksum mismatch")
        with np.load(path, allow_pickle=False) as d:
            if "outcomes" not in d or "decision_sizes" not in d:
                raise ValueError("Re-extract distillation data with terminal Value targets (v2)")
            ds = d["decision_sizes"]
            reward = d["outcomes"]
            if (
                ds.ndim != 1
                or not np.issubdtype(ds.dtype, np.integer)
                or np.any(ds < 1)
                or ds.sum() != len(d["sizes"])
                or reward.shape != ds.shape
                or not np.isin(reward, [-1, 0, 1]).all()
            ):
                raise ValueError("Invalid decision/outcome alignment")
            decision_sizes.append(ds)
            outcomes.append(reward)
            x, s, n, t = (d[k] for k in ("x", "scores", "sizes", "targets"))
            if (
                x.shape != (len(s), manifest["contract"]["width"])
                or n.sum() != len(x)
                or len(n) != len(t)
            ):
                raise ValueError("Invalid shard shapes")
            if (
                not np.isfinite(x).all()
                or not np.isfinite(s).all()
                or np.any(n < 2)
                or np.any(t < 0)
                or np.any(t >= n)
            ):
                raise ValueError("Invalid distillation targets")
            meta = json.loads(path.with_suffix(".json").read_text())
            games = {g["sha256"] for g in manifest["splits"][split]["games"]}
            if len(meta) != len(n) or any(m["game"] not in games for m in meta):
                raise ValueError("Query belongs to another split")
            game_results = {g["sha256"]: g["result"] for g in manifest["splits"][split]["games"]}
            begin = 0
            for count, value in zip(ds, reward, strict=True):
                records = meta[begin : begin + count]
                begin += int(count)
                if len({(r["game"], r["seat"], r["step"]) for r in records}) != 1:
                    raise ValueError("Decision crosses game/seat/step boundary")
                record = records[0]
                result = game_results[record["game"]]
                expected = 0.0 if result == 2 else (1.0 if result == record["seat"] else -1.0)
                if value != expected:
                    raise ValueError("Value target is not acting-seat terminal outcome")
            features.append(x)
            scores.append(s)
            sizes.append(n)
            targets.append(t)
    if not features:
        raise ValueError("Empty dataset split")
    sizes = np.concatenate(sizes)
    return dict(
        x=np.concatenate(features),
        scores=np.concatenate(scores),
        sizes=sizes,
        targets=np.concatenate(targets),
        offsets=np.r_[0, np.cumsum(sizes)],
        decision_offsets=np.r_[0, np.cumsum(np.concatenate(decision_sizes))],
        outcomes=np.concatenate(outcomes),
    )


def batches(
    data: dict[str, NDArray[Any]],
    batch_size: int,
    device: str,
    rng: np.random.Generator | None = None,
) -> Iterator[
    tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
]:
    """Yield padded tensor batches with prefix ownership and terminal outcomes."""
    order = np.arange(len(data["outcomes"]))
    if rng is not None:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        decisions = order[start : start + batch_size]
        indices = []
        owners = []
        first = []
        for owner, d in enumerate(decisions):
            begin, end = data["decision_offsets"][d : d + 2]
            first.append(len(indices))
            indices.extend(range(begin, end))
            owners.extend([owner] * (end - begin))
        width = int(data["sizes"][indices].max())
        x = np.zeros((len(indices), width, data["x"].shape[1]), np.float32)
        score = np.full((len(indices), width), -np.inf, np.float32)
        mask = np.zeros((len(indices), width), bool)
        for row, i in enumerate(indices):
            begin, end = data["offsets"][i : i + 2]
            n = end - begin
            x[row, :n] = data["x"][begin:end]
            score[row, :n] = data["scores"][begin:end]
            mask[row, :n] = True
        yield tuple(
            torch.as_tensor(a, device=device)
            for a in (
                x,
                mask,
                score,
                data["targets"][indices].astype(np.int64),
                np.asarray(owners),
                np.asarray(first),
                data["outcomes"][decisions],
            )
        )


def value_targets(outcomes: torch.Tensor) -> torch.Tensor:
    """Convert terminal outcomes into targets for the 51-bin value distribution."""
    position = (outcomes.clamp(-1, 1) + 1) * 25
    lower = position.floor().long()
    upper = position.ceil().long()
    fraction = position - lower
    target = outcomes.new_zeros((len(outcomes), 51))
    target.scatter_add_(1, lower[:, None], (1 - fraction)[:, None])
    target.scatter_add_(1, upper[:, None], fraction[:, None])
    return target


def decision_loss(
    model: ObservationTransformer,
    b: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    temperature: float,
    hard_weight: float,
    value_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """Return the combined decision loss, component metrics, and policy logits."""
    x, mask, teacher, target, owners, first, outcome = b
    logits, values, value_logits = model.forward_details(x, mask)
    # The teacher distribution is applied once per decision; NLL covers its
    # complete ordered selection including forced choices and STOP.
    _, kl = losses(
        logits[first], teacher[first], mask[first], target[first], temperature, hard_weight
    )
    nll = F.cross_entropy(logits, target, reduction="none")
    joint = logits.new_zeros(len(first)).index_add(0, owners, nll).mean()
    value_loss = -(value_targets(outcome) * value_logits[first].log_softmax(-1)).sum(-1).mean()
    loss = kl + hard_weight * joint + value_weight * value_loss
    return (
        loss,
        dict(
            kl=kl,
            action_nll=joint,
            value_loss=value_loss,
            value_mse=(values[first] - outcome).square().mean(),
        ),
        logits,
    )


def losses(
    logits: torch.Tensor,
    teacher: torch.Tensor,
    mask: torch.Tensor,
    target: torch.Tensor,
    temperature: float,
    hard_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine teacher KL and hard-action loss while masking illegal candidates."""
    mask = mask & torch.isfinite(logits)
    probability = (teacher.masked_fill(~mask, -torch.inf) / temperature).softmax(-1)
    log_probability = logits.log_softmax(-1)
    # Never multiply zero by -inf in padded positions.
    hard = F.cross_entropy(logits, target)
    kl = (
        (
            probability
            * (probability.clamp_min(1e-30).log() - log_probability.masked_fill(~mask, 0.0))
        )
        .sum(-1)
        .mean()
    )
    return hard_weight * hard + kl, kl


@torch.no_grad()
def evaluate(
    model: ObservationTransformer,
    data: dict[str, NDArray[Any]],
    batch_size: int,
    device: str,
    temperature: float,
    hard_weight: float,
    value_weight: float = 0.25,
) -> dict[str, Any]:
    """Aggregate held-out policy and value metrics without gradient updates."""
    model.eval()
    total = correct = queries = 0
    sums = {k: 0.0 for k in ("loss", "kl", "action_nll", "value_loss", "value_mse")}
    for b in batches(data, batch_size, device):
        loss, metrics, logits = decision_loss(model, b, temperature, hard_weight, value_weight)
        n = len(b[-1])
        total += n
        queries += len(logits)
        correct += int((select_logits(logits) == b[3]).sum())
        for k, v in dict(loss=loss, **metrics).items():
            sums[k] += float(v) * n
    return dict(
        decisions=total,
        queries=queries,
        teacher_top1=correct / queries,
        **{k: v / total for k, v in sums.items()},
    )


def train(
    directory: Path,
    output: Path,
    contract_id: str,
    *,
    epochs: int = 5,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    temperature: float = 1.0,
    hard_weight: float = 0.1,
    value_weight: float = 0.25,
    seed: int = 204,
    device: str = "cuda",
    max_updates: int | None = None,
) -> dict[str, Any]:
    """Distill policy and value from game-split shards and save the best checkpoint."""
    if max_updates is not None and max_updates < 1:
        raise ValueError("max_updates must be positive")
    if (
        epochs < 1
        or batch_size < 1
        or not np.isfinite([learning_rate, temperature, hard_weight, value_weight]).all()
        or learning_rate <= 0
        or temperature <= 0
        or not 0 <= hard_weight <= 1
        or value_weight <= 0
    ):
        raise ValueError("Invalid training settings")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty")
    manifest = json.loads((directory / "manifest.json").read_text())
    digest = hashlib.sha256(
        json.dumps(manifest["contract"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if manifest["contract_id"] != contract_id or digest != contract_id:
        raise ValueError("Transformer contract mismatch")
    games = [g["sha256"] for split in manifest["splits"].values() for g in split["games"]]
    if len(games) != len(set(games)):
        raise ValueError("Game leakage across splits")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rng = np.random.default_rng(seed)
    model = ObservationTransformer(manifest["contract"]["width"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    train_data = load_split(directory, manifest, "train")
    validation = load_split(directory, manifest, "validation")
    output.mkdir(parents=True, exist_ok=True)
    config = dict(
        epochs=epochs,
        max_updates=max_updates,
        batch_size=batch_size,
        learning_rate=learning_rate,
        temperature=temperature,
        hard_weight=hard_weight,
        value_weight=value_weight,
        value_target="terminal outcome per acting seat",
        loss="KL + action_weight * joint NLL + value_weight * distribution CE",
        seed=seed,
        device=device,
        weight_decay=1e-4,
        gradient_clip=0.5,
        hardware=torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
        torch_version=str(torch.__version__),
    )
    updates = 0
    history = []
    best = float("inf")
    for epoch in range(epochs):
        model.train()
        total = 0
        accumulated = 0.0
        for b in batches(train_data, batch_size, device, rng):
            if max_updates is not None and updates >= max_updates:
                break
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = decision_loss(model, b, temperature, hard_weight, value_weight)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5, error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            total += len(b[-1])
            accumulated += float(loss.detach()) * len(b[-1])
        metrics = evaluate(
            model, validation, batch_size, device, temperature, hard_weight, value_weight
        )
        history.append(dict(epoch=epoch + 1, train_loss=accumulated / total, validation=metrics))
        print(json.dumps(history[-1]), flush=True)
        if metrics["loss"] < best:
            best = metrics["loss"]
            payload = dict(
                format="ptcg-transformer-checkpoint-v1",
                contract_id=contract_id,
                contract=manifest["contract"],
                architecture=ARCHITECTURE,
                deck_counts=manifest["deck_counts"],
                feature_names=manifest["feature_names"],
                teacher_model_id=manifest["teacher_model_id"],
                dataset_sha256=sha256(directory / "manifest.json"),
                split_games={
                    s: [g["sha256"] for g in d["games"]] for s, d in manifest["splits"].items()
                },
                config=config,
                epoch=epoch + 1,
                validation=metrics,
                state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            )
            torch.save(payload, output / "model.pt")
        if max_updates is not None and updates >= max_updates:
            break
    payload = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(payload["state_dict"])
    test = (
        evaluate(
            model,
            load_split(directory, manifest, "test"),
            batch_size,
            device,
            temperature,
            hard_weight,
            value_weight,
        )
        if "test" in manifest["splits"]
        else None
    )
    report = dict(
        config=config,
        updates=updates,
        max_updates=max_updates,
        history=history,
        selected_epoch=payload["epoch"],
        test=test,
        checkpoint_sha256=sha256(output / "model.pt"),
        teacher_model_id=manifest["teacher_model_id"],
    )
    (output / "training.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(selected_epoch=payload["epoch"], test=test)), flush=True)
    return report
