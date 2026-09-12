"""120-token shared encoder, pointer policy and 51-bin distributional critic."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from agents.dragapult.transformer.contract import SCORE_TIE_TOLERANCE

ARCHITECTURE = dict(
    dim=128,
    heads=4,
    layers=6,
    ff=256,
    activation="gelu",
    norm_eps=1e-5,
    max_tokens=120,
    value_head="global_distribution_51",
    parameters=980916,
)


class Stem(nn.Module):
    """Project one token type into the shared hidden representation."""

    def __init__(self, width: int) -> None:
        """Create the projection and normalization layers for one token type."""
        super().__init__()
        self.linear1 = nn.Linear(width, 128)
        self.linear2 = nn.Linear(128, 128)
        self.norm = nn.LayerNorm(128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project raw token features into the shared hidden space."""
        return self.norm(self.linear2(F.gelu(self.linear1(x))))


class Block(nn.Module):
    """Pre-normalized Transformer attention and feed-forward block."""

    def __init__(self) -> None:
        """Create one pre-normalized attention and feed-forward block."""
        super().__init__()
        self.norm1 = nn.LayerNorm(128)
        self.qkv = nn.Linear(128, 384)
        self.out = nn.Linear(128, 128)
        self.norm2 = nn.LayerNorm(128)
        self.ff1 = nn.Linear(128, 256)
        self.ff2 = nn.Linear(256, 128)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply masked self-attention and the residual feed-forward network."""
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b, n, 3, 4, 32).permute(2, 0, 3, 1, 4).unbind(0)
        attention = (
            ((q @ k.transpose(-2, -1)) / (32**0.5))
            .masked_fill(~mask[:, None, None, :], -torch.inf)
            .softmax(-1)
        )
        x = x + self.out((attention @ v).transpose(1, 2).reshape(b, n, d))
        return x + self.ff2(F.gelu(self.ff1(self.norm2(x))))


class ObservationTransformer(nn.Module):
    """Public-token pointer policy with a distributional value head."""

    def __init__(self, width: int = 198) -> None:
        """Build the fixed 128-dimensional Policy/Value architecture."""
        super().__init__()
        if width != 198:
            raise ValueError("Structured Transformer requires 198-field tokens")
        self.deck_embedding = nn.Embedding(32, 16)
        self.global_stem = Stem(128)
        self.board_stem = Stem(144)
        self.zone_stem = Stem(64)
        self.option_stem = Stem(144)
        self.token_type = nn.Embedding(5, 128)
        self.stop = nn.Parameter(torch.zeros(128))
        self.blocks = nn.ModuleList(Block() for _ in range(6))
        self.norm = nn.LayerNorm(128)
        self.policy_query = nn.Linear(128, 128)
        self.policy_key = nn.Linear(128, 128)
        self.policy_bias = nn.Linear(128, 1)
        self.step_embedding = nn.Embedding(6, 128)
        self.result_delta = nn.Linear(32, 128)
        self.value1 = nn.Linear(128, 96)
        self.value2 = nn.Linear(96, 51)
        self.register_buffer("value_support", torch.linspace(-1, 1, 51), persistent=False)
        nn.init.zeros_(self.value2.weight)
        nn.init.zeros_(self.value2.bias)
        if sum(p.numel() for p in self.parameters()) != 980916:
            raise RuntimeError("Parameter contract mismatch")

    @property
    def value_head(self) -> nn.Linear:
        """Expose the final value layer for initialization and calibration."""
        return self.value2

    def forward_details(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return masked policy logits, expected Value, and the 51-bin value logits."""
        if x.shape[1] > 120 or x.shape[1] < 20:
            raise ValueError("Transformer supports 20–120 packed tokens")
        valid = mask & x[:, :, 0].bool()
        kind = x[:, :, 1].long()

        def embedding(slot: torch.Tensor) -> torch.Tensor:
            """Embed known deck slots and zero the representation of unknown cards."""
            return self.deck_embedding(slot.clamp_min(0).long()) * (slot >= 0).unsqueeze(-1)

        identity = embedding(x[:, :, 4])
        extra = embedding(x[:, :, 5])
        counts = x[:, :, 102:134] @ self.deck_embedding.weight
        evolution = x[:, :, 134:166] @ self.deck_embedding.weight
        generic = x[:, :, 6:102]
        result = x[:, :, 166:198]
        h = self.global_stem(torch.cat([generic, identity, extra], -1)) * (kind == 0).unsqueeze(-1)
        h = h + self.board_stem(torch.cat([generic, identity, counts, evolution], -1)) * (
            kind == 1
        ).unsqueeze(-1)
        h = h + self.zone_stem(torch.cat([generic[:, :, :48], counts], -1)) * (kind == 2).unsqueeze(
            -1
        )
        h = h + (
            self.option_stem(torch.cat([generic, identity, result], -1)) + self.result_delta(result)
        ) * (kind == 3).unsqueeze(-1)
        h = h + self.stop * (kind == 4).unsqueeze(-1) + self.token_type(kind)
        for block in self.blocks:
            h = block(h, valid)
        h = self.norm(h)
        selected = x[:, :, 3] * valid
        summed = (h * selected.unsqueeze(-1)).sum(1)
        step = selected.sum(1).long().clamp_max(5)
        query = self.policy_query(h[:, 0] + summed + self.step_embedding(step))
        logits = (self.policy_key(h) * query.unsqueeze(1)).sum(-1) / (128**0.5) + self.policy_bias(
            h
        ).squeeze(-1)
        logits = logits.masked_fill(~(valid & x[:, :, 2].bool()), -torch.inf)
        value_logits = self.value2(F.gelu(self.value1(h[:, 0])))
        value = (value_logits.softmax(-1) * self.value_support.to(value_logits)).sum(-1)
        return logits, value, value_logits

    def forward_with_value(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return masked policy logits and expected Value for each observation."""
        return self.forward_details(x, mask)[:2]

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return masked policy logits for the supplied token batch."""
        return self.forward_with_value(x, mask)[0]


def select_logits(logits: torch.Tensor) -> torch.Tensor:
    """Select the first candidate within the shared score tie tolerance."""
    return (
        (logits >= logits.max(dim=-1, keepdim=True).values - SCORE_TIE_TOLERANCE).long().argmax(-1)
    )
