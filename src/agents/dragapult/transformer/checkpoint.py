"""Load only tensor/basic-type checkpoints with an independently supplied contract."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from agents.dragapult.transformer.model import ARCHITECTURE, ObservationTransformer

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
    from typing import Any


def load_checkpoint(
    path: Path, contract_id: str, deck_counts: dict[int, int], names: Sequence[str]
) -> tuple[ObservationTransformer, dict[str, Any]]:
    """Validate checkpoint contracts and finite weights, then load the model in eval mode."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("format") != "ptcg-transformer-checkpoint-v1"
        or payload["contract_id"] != contract_id
        or payload["architecture"] != ARCHITECTURE
    ):
        raise ValueError("Incompatible Transformer checkpoint")
    if (
        payload["deck_counts"] != {str(k): v for k, v in deck_counts.items()}
        or payload["feature_names"] != names
    ):
        raise ValueError("Checkpoint deck/features mismatch")
    model = ObservationTransformer(len(names))
    model.load_state_dict(payload["state_dict"], strict=True)
    if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError("Non-finite checkpoint weights")
    return model.eval(), payload
