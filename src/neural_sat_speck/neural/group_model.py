"""Pair-aware permutation-invariant rankers for the Phase 6 study."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import torch
from torch import nn

from .model import (
    GohrResNet,
    GohrResNetConfig,
    ResBlock1D,
    _BatchNorm1dWithSingleSampleFallback,
)

PAIR_GROUP_SIZE: Final[int] = 64
PAIR_EMBEDDING_DIM: Final[int] = 64
PAIR_GROUP_VARIANTS: tuple[str, str, str] = (
    "b1_mean",
    "b2_moments",
    "b3_weighted",
)
PAIR_GROUP_MODEL_IDS: Mapping[str, str] = {
    "b1_mean": "speck32_64_b1_mean_g64",
    "b2_moments": "speck32_64_b2_moments_g64",
    "b3_weighted": "speck32_64_b3_weighted_g64",
}


class SharedGohrPairEncoder(nn.Module):
    """The project-owned B0 feature extractor up to its first dense layer."""

    def __init__(self, config: GohrResNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or GohrResNetConfig()
        self.stem_conv = nn.Conv1d(
            self.config.in_channels,
            self.config.num_filters,
            kernel_size=1,
            bias=False,
        )
        self.stem_bn = nn.BatchNorm1d(self.config.num_filters)
        self.relu = nn.ReLU(inplace=True)
        self.blocks = nn.ModuleList(
            [
                ResBlock1D(
                    self.config.num_filters,
                    kernel_size=self.config.kernel_size,
                )
                for _ in range(self.config.num_blocks)
            ]
        )
        flattened_dim = self.config.num_filters * self.config.word_bits
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(flattened_dim, self.config.dense_dim, bias=False)
        self.dense_bn = _BatchNorm1dWithSingleSampleFallback(
            self.config.dense_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``(N,4,16)`` pair features into ``(N,dense_dim)`` vectors."""

        expected_shape = (
            self.config.in_channels,
            self.config.word_bits,
        )
        if x.ndim != 3 or tuple(x.shape[1:]) != expected_shape:
            raise ValueError(
                f"expected pair input shape (N, {expected_shape[0]}, "
                f"{expected_shape[1]}), got {tuple(x.shape)}"
            )
        out = self.stem_conv(x)
        out = self.stem_bn(out)
        out = self.relu(out)
        for block in self.blocks:
            out = block(out)
        out = self.flatten(out)
        out = self.dense(out)
        out = self.dense_bn(out)
        return self.relu(out)

    def initialize_from_b0(self, baseline: GohrResNet) -> None:
        """Copy the compatible B0 encoder weights into this new module."""

        if self.config.to_dict() != baseline.config.to_dict():
            raise ValueError("B1/B2/B3 encoder config must match the B0 config")
        if len(baseline.head) < 4 or not isinstance(baseline.head[1], nn.Linear):
            raise ValueError("B0 checkpoint does not expose the expected dense head")
        if not isinstance(baseline.head[2], nn.BatchNorm1d):
            raise ValueError("B0 checkpoint does not expose the expected dense BN")
        self.stem_conv.load_state_dict(baseline.stem_conv.state_dict())
        self.stem_bn.load_state_dict(baseline.stem_bn.state_dict())
        self.blocks.load_state_dict(baseline.blocks.state_dict())
        self.dense.load_state_dict(baseline.head[1].state_dict())
        self.dense_bn.load_state_dict(baseline.head[2].state_dict())


class PairGroupRanker(nn.Module):
    """One of the predeclared B1/B2/B3 permutation-invariant rankers."""

    def __init__(
        self,
        config: GohrResNetConfig | None = None,
        *,
        variant: str,
        group_size: int = PAIR_GROUP_SIZE,
    ) -> None:
        super().__init__()
        if variant not in PAIR_GROUP_VARIANTS:
            raise ValueError(
                f"variant must be one of {list(PAIR_GROUP_VARIANTS)}, got {variant!r}"
            )
        if isinstance(group_size, bool) or not isinstance(group_size, int):
            raise TypeError("group_size must be an integer")
        if group_size != PAIR_GROUP_SIZE:
            raise ValueError(f"group_size must be exactly {PAIR_GROUP_SIZE}")
        self.config = config or GohrResNetConfig()
        self.variant = variant
        self.group_size = group_size
        self.embedding_dim = self.config.dense_dim
        self.encoder = SharedGohrPairEncoder(self.config)

        if variant == "b2_moments":
            head_input_dim = 2 * self.embedding_dim
        else:
            head_input_dim = self.embedding_dim
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )
        self.reliability: nn.Linear | None
        if variant == "b3_weighted":
            self.reliability = nn.Linear(self.embedding_dim, 1)
            # Uniform initial weights make B3 start as the same mean
            # aggregation used by B1, while remaining fully trainable.
            nn.init.zeros_(self.reliability.weight)
            nn.init.zeros_(self.reliability.bias)
        else:
            self.reliability = None

    @property
    def model_id(self) -> str:
        """Return the stable study identifier for this variant."""

        return PAIR_GROUP_MODEL_IDS[self.variant]

    def initialize_from_b0(self, baseline: GohrResNet) -> None:
        """Initialize the shared encoder from a separately loaded B0 model."""

        self.encoder.initialize_from_b0(baseline)

    def pair_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Return one shared-encoder embedding for every pair in each group."""

        if x.ndim != 4 or tuple(x.shape[1:]) != (
            self.group_size,
            self.config.in_channels,
            self.config.word_bits,
        ):
            raise ValueError(
                "expected group input shape "
                f"(N,{self.group_size},{self.config.in_channels},"
                f"{self.config.word_bits}), got {tuple(x.shape)}"
            )
        if x.dtype != torch.float32:
            raise TypeError("pair-group model inputs must have dtype torch.float32")
        batch_size = x.shape[0]
        flattened = x.reshape(
            batch_size * self.group_size,
            self.config.in_channels,
            self.config.word_bits,
        )
        embeddings = self.encoder(flattened)
        return embeddings.reshape(batch_size, self.group_size, self.embedding_dim)

    def aggregate_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply the variant's symmetric group aggregation."""

        if embeddings.ndim != 3 or embeddings.shape[1:] != (
            self.group_size,
            self.embedding_dim,
        ):
            raise ValueError("embeddings have an invalid group shape")
        mean = embeddings.mean(dim=1)
        if self.variant == "b1_mean":
            return mean
        if self.variant == "b2_moments":
            variance = (embeddings - mean.unsqueeze(1)).square().mean(dim=1)
            return torch.cat((mean, variance), dim=1)
        if self.reliability is None:
            raise RuntimeError("weighted ranker is missing its reliability layer")
        weights = torch.sigmoid(self.reliability(embeddings)).squeeze(-1)
        denominator = weights.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(embeddings.dtype).eps
        )
        return (embeddings * weights.unsqueeze(-1)).sum(dim=1) / denominator

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one raw group logit for a batched or single group."""

        if x.ndim == 3:
            x = x.unsqueeze(0)
        embeddings = self.pair_embeddings(x)
        aggregate = self.aggregate_embeddings(embeddings)
        return self.head(aggregate)

    @property
    def parameter_count(self) -> int:
        """Return the count of trainable parameters."""

        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    @property
    def input_shape(self) -> tuple[int, int, int]:
        """Return the per-group tensor shape excluding the batch dimension."""

        return self.group_size, self.config.in_channels, self.config.word_bits


def build_pair_group_model(
    baseline: GohrResNet,
    *,
    variant: str,
    seed: int,
    deterministic: bool = True,
) -> PairGroupRanker:
    """Construct a deterministically initialized candidate from B0 weights."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    model = PairGroupRanker(baseline.config, variant=variant)
    model.initialize_from_b0(baseline)
    return model


__all__ = [
    "PAIR_EMBEDDING_DIM",
    "PAIR_GROUP_MODEL_IDS",
    "PAIR_GROUP_SIZE",
    "PAIR_GROUP_VARIANTS",
    "PairGroupRanker",
    "SharedGohrPairEncoder",
    "build_pair_group_model",
]
