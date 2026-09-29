"""Faithful SIMON32/64 adaptation of the SPECK B0->B1 Gohr ResNet architecture.

AUDIT AND SCIENTIFIC COMPARISON WITH TOY PROTOTYPE:
===================================================
1. Architectural Discrepancy:
   - The toy prototype (`src/neural_sat_speck/simon/simon_b1_model.py`) employed a small
     2-layer MLP (`Linear(64, 128) -> BatchNorm1d -> ReLU -> Linear(128, 64) -> BatchNorm1d -> ReLU`)
     operating on flat 64-bit vectors. It possessed only 18,849 trainable parameters,
     completely omitted the 1D bit-slice spatial structure across 16-bit words, contained
     no convolutional layers, and had no residual connections.
   - This faithful module implements the authentic Gohr (2019) residual architecture and the
     repository's SPECK B0->B1 recipe:
     * SimonGohrResNet: input shape (N, 4, 16) where 4 channels are (L0, R0, L1, R1),
       1x1 stem conv (4 -> 32 channels), 4 ResBlock1D blocks (two 1D convolutions of kernel size 3,
       padding 1, bias=False, BatchNorm1d, ReLU, with residual addition), flatten to 512,
       dense projection Linear(512, 64, bias=False), BatchNorm1d, ReLU, and classification head.
     * SharedSimonPairEncoder: extracts pair features up to the 64-dimensional dense projection.
     * SimonPairGroupRanker ('b1_mean'): deep sets permutation-invariant ranker that encodes
       each pair independently through SharedSimonPairEncoder, applies symmetric mean pooling
       across the N pairs, and scores the 64-dim group embedding using the
       frozen SPECK B1 head: Linear(64, 32) -> ReLU -> Linear(32, 1).
     * Default trainable parameters: B0 62,337; B1 60,289.

2. Weight Transfer (B0 -> B1):
   - The toy prototype could not perform B0->B1 transfer because no B0 Gohr model existed and
     its flat MLP encoder had an incompatible parameter structure.
   - The faithful architecture supports exact weight copying (`initialize_from_b0`):
     stem_conv, stem_bn, all 4 residual blocks, dense projection, and dense_bn are transferred
     directly from a trained SimonGohrResNet into SimonPairGroupRanker.

3. Speed vs Cryptanalytic Validity:
   - In preliminary throughput tests (`simon_32_vs_64_resource_pilots.json`), the toy prototype
     recorded ~0.165 s screening time for 65,536 candidates across 16 pairs. That 0.17 s timing
     was an artifact of evaluating a tiny MLP with random weights. It provides zero evidence
     regarding cryptanalytic distinguishing advantage, ranking accuracy, or true B1 transfer.
   - A genuine 4-block 1D ResNet evaluates 1,048,576 pairs through 8 convolutional layers
     (>800 GFLOPs on CPU), requiring ~55-65 seconds on CPU for 65,536 candidates x 16 pairs.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cipher import Block
from .classical_ranking import FullPredictedDdt, rank_classical_domain
from .ranking_features import bao_full_value_bits, candidate_value_words


class _BatchNorm1dWithFallback(nn.BatchNorm1d):
    """BatchNorm1d that safely falls back to stored running stats for batch_size=1 during training."""

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.training and input_tensor.ndim == 2 and input_tensor.shape[0] == 1:
            return F.batch_norm(
                input_tensor,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                momentum=0.0,
                eps=self.eps,
            )
        return super().forward(input_tensor)


class ResBlock1D(nn.Module):
    """Residual block containing two 1D convolutions with batch normalisation."""

    def __init__(self, channels: int = 32, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve residual shape")
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out + residual


@dataclass(frozen=True)
class SimonGohrResNetConfig:
    """Hyperparameters for SimonGohrResNet and SimonPairGroupRanker."""

    in_channels: int = 4
    word_bits: int = 16
    num_filters: int = 32
    num_blocks: int = 4
    dense_dim: int = 64
    kernel_size: int = 3
    dropout_rate: float = 0.0

    def __post_init__(self) -> None:
        integer_fields = (
            ("in_channels", self.in_channels),
            ("word_bits", self.word_bits),
            ("num_filters", self.num_filters),
            ("num_blocks", self.num_blocks),
            ("dense_dim", self.dense_dim),
            ("kernel_size", self.kernel_size),
        )
        for name, value in integer_fields:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve residual shape")
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in [0.0, 1.0)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimonGohrResNetConfig:
        return cls(
            in_channels=int(data.get("in_channels", 4)),
            word_bits=int(data.get("word_bits", 16)),
            num_filters=int(data.get("num_filters", 32)),
            num_blocks=int(data.get("num_blocks", 4)),
            dense_dim=int(data.get("dense_dim", 64)),
            kernel_size=int(data.get("kernel_size", 3)),
            dropout_rate=float(data.get("dropout_rate", 0.0)),
        )


class SimonGohrResNet(nn.Module):
    """Compact Gohr-style 1D convolutional residual distinguisher for SIMON32/64."""

    def __init__(
        self,
        config: SimonGohrResNetConfig | None = None,
        *,
        head_type: str = "speck",
    ) -> None:
        super().__init__()
        self.config = config or SimonGohrResNetConfig()
        self.head_type = head_type

        # Initial stem layer mapping 4 word channels to num_filters
        self.stem_conv = nn.Conv1d(
            self.config.in_channels,
            self.config.num_filters,
            kernel_size=1,
            bias=False,
        )
        self.stem_bn = nn.BatchNorm1d(self.config.num_filters)
        self.relu = nn.ReLU(inplace=True)

        # Residual tower (4 ResBlock1D blocks)
        self.blocks = nn.ModuleList(
            [
                ResBlock1D(
                    self.config.num_filters,
                    kernel_size=self.config.kernel_size,
                )
                for _ in range(self.config.num_blocks)
            ]
        )

        # First dense projection
        flattened_dim = self.config.num_filters * self.config.word_bits
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(flattened_dim, self.config.dense_dim, bias=False)
        self.dense_bn = _BatchNorm1dWithFallback(self.config.dense_dim)

        # Classifier head
        if head_type == "speck":
            # Matches SPECK GohrResNet head with secondary linear layer
            self.head: nn.Module = nn.Sequential(
                nn.Linear(self.config.dense_dim, self.config.dense_dim, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(self.config.dense_dim, 1),
            )
        elif head_type == "direct":
            self.head = nn.Linear(self.config.dense_dim, 1)
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 4, 16) or (batch_size, 64).

        Returns:
            Raw logits of shape (batch_size, 1).
        """
        input_width = self.config.in_channels * self.config.word_bits
        if x.ndim == 2 and x.shape[1] == input_width:
            x = x.view(-1, self.config.in_channels, self.config.word_bits)
        elif x.ndim != 3 or x.shape[1:] != (
            self.config.in_channels,
            self.config.word_bits,
        ):
            raise ValueError(
                f"expected input shape (N, {self.config.in_channels}, "
                f"{self.config.word_bits}) or (N, {input_width}), got {x.shape}"
            )

        out = self.stem_conv(x)
        out = self.stem_bn(out)
        out = self.relu(out)

        for block in self.blocks:
            out = block(out)

        out = self.flatten(out)
        out = self.dense(out)
        out = self.dense_bn(out)
        out = self.relu(out)

        logits = self.head(out)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute sigmoid probability scores in [0, 1]."""
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                return torch.sigmoid(self.forward(x))
        finally:
            if was_training:
                self.train()

    @property
    def parameter_count(self) -> int:
        """Return total trainable parameter count."""
        return sum(param.numel() for param in self.parameters() if param.requires_grad)


class SharedSimonPairEncoder(nn.Module):
    """B0-compatible SIMON feature extractor up to the first dense 64 layer."""

    def __init__(self, config: SimonGohrResNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or SimonGohrResNetConfig()
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
        self.dense_bn = _BatchNorm1dWithFallback(self.config.dense_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode (N, 4, 16) pair features into (N, dense_dim) embeddings."""
        expected_shape = (self.config.in_channels, self.config.word_bits)
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

    def initialize_from_b0(self, baseline: Any) -> None:
        """Copy compatible B0 weights into this encoder."""
        self.stem_conv.load_state_dict(baseline.stem_conv.state_dict())
        self.stem_bn.load_state_dict(baseline.stem_bn.state_dict())
        self.blocks.load_state_dict(baseline.blocks.state_dict())
        if hasattr(baseline, "dense") and hasattr(baseline, "dense_bn"):
            self.dense.load_state_dict(baseline.dense.state_dict())
            self.dense_bn.load_state_dict(baseline.dense_bn.state_dict())
        elif hasattr(baseline, "head") and len(baseline.head) >= 3:
            self.dense.load_state_dict(baseline.head[1].state_dict())
            self.dense_bn.load_state_dict(baseline.head[2].state_dict())
        else:
            raise ValueError("baseline model does not expose compatible dense projection")


class SimonPairGroupRanker(nn.Module):
    """Pair-aware permutation-invariant Deep Sets ranker for SIMON32/64.

    Recipe:
    - Shared pair encoder: SharedSimonPairEncoder maps each (4, 16) pair to 64-dim embedding.
    - Symmetric aggregation: permutation-invariant mean pooling over all N pairs.
    - Head: Linear(64, 64) -> BatchNorm1d(64) -> ReLU -> Linear(64, 1).
    """

    def __init__(
        self,
        config: SimonGohrResNetConfig | None = None,
        *,
        variant: str = "b1_mean",
        head_variant: str = "speck_b1",
    ) -> None:
        super().__init__()
        if variant != "b1_mean":
            raise ValueError(f"Unsupported variant: {variant!r}. Expected 'b1_mean'.")
        self.config = config or SimonGohrResNetConfig()
        self.variant = variant
        self.head_variant = head_variant
        self.embedding_dim = self.config.dense_dim
        self.encoder = SharedSimonPairEncoder(self.config)

        if head_variant == "b1_mlp_bn":
            # Linear(64, 64) -> BN -> ReLU -> Linear(64, 1)
            self.head = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                _BatchNorm1dWithFallback(self.embedding_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.embedding_dim, 1),
            )
        elif head_variant == "speck_b1":
            # Linear(64, 32) -> ReLU -> Linear(32, 1) matching group_model.py
            self.head = nn.Sequential(
                nn.Linear(self.embedding_dim, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 1),
            )
        else:
            raise ValueError(f"Unknown head_variant: {head_variant}")

    def initialize_from_b0(self, baseline: Any) -> None:
        """Initialize the shared pair encoder from a B0 model."""
        self.encoder.initialize_from_b0(baseline)

    def encode_pairs(self, x: torch.Tensor) -> torch.Tensor:
        """Encode pairs into embeddings of shape (batch_size, num_pairs, embedding_dim)."""
        if x.ndim == 4:
            # Shape: (batch_size, num_pairs, 4, 16)
            batch_size, num_pairs, in_c, word_bits = x.shape
            if (in_c, word_bits) != (self.config.in_channels, self.config.word_bits):
                raise ValueError(
                    f"Expected pair shape ({self.config.in_channels}, {self.config.word_bits}), "
                    f"got ({in_c}, {word_bits})"
                )
            flat = x.reshape(batch_size * num_pairs, in_c, word_bits)
        elif x.ndim == 3:
            # Shape: (batch_size, num_pairs, 64)
            batch_size, num_pairs, feat_dim = x.shape
            if feat_dim != self.config.in_channels * self.config.word_bits:
                raise ValueError(
                    f"Expected 64-bit flat pairs, got feature dim {feat_dim}"
                )
            flat = x.reshape(batch_size * num_pairs, self.config.in_channels, self.config.word_bits)
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {tuple(x.shape)}")

        embeddings = self.encoder(flat)
        return embeddings.reshape(batch_size, num_pairs, self.embedding_dim)

    def pool_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Permutation-invariant mean pooling across the pair dimension (dim=1)."""
        return embeddings.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass over a batch of candidate groups.

        Args:
            x: Tensor of shape (batch_size, num_pairs, 4, 16) or (batch_size, num_pairs, 64).
               Single group of shape (num_pairs, 4, 16) is unsqueezed automatically.

        Returns:
            Tensor of shape (batch_size,) with scalar candidate scores.
        """
        if x.ndim == 3 and x.shape[1] == self.config.in_channels and x.shape[2] == self.config.word_bits:
            x = x.unsqueeze(0)
        embeddings = self.encode_pairs(x)
        pooled = self.pool_embeddings(embeddings)
        scores = self.head(pooled).squeeze(-1)
        return scores

    @property
    def parameter_count(self) -> int:
        """Return total trainable parameter count."""
        return sum(param.numel() for param in self.parameters() if param.requires_grad)


@dataclass(frozen=True)
class SimonB1RankingResult:
    """Detailed metrics from a SimonB1 ranking pass."""

    order: np.ndarray
    screen_scores: np.ndarray
    full_scores: np.ndarray
    finalists: np.ndarray
    feature_seconds: float
    screen_seconds: float
    rescore_seconds: float
    sort_seconds: float
    total_seconds: float
    screen_pairs_used: int
    full_pairs_used: int
    screen_tie_count: int
    full_tie_count: int
    screen_collision_count: int
    full_collision_count: int


def extract_simon_b1_features(
    ciphertext_pairs: Sequence[tuple[Block, Block]],
    candidates: Sequence[int],
    *,
    final_key_candidate: int | None = None,
    flat_bits: bool = False,
) -> torch.Tensor:
    """Extract candidate-inverted pair features as a float32 tensor.

    Returns:
        Tensor of shape (len(candidates), len(ciphertext_pairs), 4, 16) by default,
        or (len(candidates), len(ciphertext_pairs), 64) if flat_bits=True.
    """
    states = candidate_value_words(
        ciphertext_pairs,
        candidates,
        final_key_candidate=final_key_candidate,
    )
    bits = bao_full_value_bits(states)  # shape: (key, pair, 4, 16) uint8
    if flat_bits:
        bits = bits.reshape(len(candidates), len(ciphertext_pairs), 64)
    return torch.from_numpy(bits).to(torch.float32)


def rank_simon_b1_domain(
    ciphertext_pairs: Sequence[tuple[Block, Block]],
    model: SimonPairGroupRanker,
    *,
    final_key_candidate: int | None = None,
    screen_pairs: int = 16,
    full_pairs: int | None = None,
    rescore_top: int = 512,
    batch_size: int = 2048,
    device: torch.device | str = "cpu",
) -> SimonB1RankingResult:
    """Rank all 65,536 candidate subkeys using faithful SimonPairGroupRanker.

    Screening uses screen_pairs (default 16).
    Rescoring uses full_pairs (default len(ciphertext_pairs), e.g. 32 or 64).
    Ties broken deterministically by ascending candidate word.
    """
    actual_full_pairs = len(ciphertext_pairs) if full_pairs is None else full_pairs
    if not 1 <= screen_pairs <= actual_full_pairs <= len(ciphertext_pairs):
        raise ValueError(
            f"Invalid pair counts: screen_pairs={screen_pairs}, "
            f"full_pairs={actual_full_pairs}, total_pairs={len(ciphertext_pairs)}"
        )
    if not 1 <= rescore_top <= 1 << 16:
        raise ValueError(f"Invalid rescore_top={rescore_top}")

    started = time.perf_counter()
    model.eval()
    model.to(device)

    all_candidates = list(range(1 << 16))
    screen_p = ciphertext_pairs[:screen_pairs]
    full_p = ciphertext_pairs[:actual_full_pairs]

    # Stage 1: Screening on screen_pairs across all 65,536 candidates
    t_feat = 0.0
    t_screen_start = time.perf_counter()
    screen_scores = np.empty(1 << 16, dtype=np.float32)

    with torch.no_grad():
        for start in range(0, 1 << 16, batch_size):
            cand_batch = all_candidates[start : min(start + batch_size, 1 << 16)]
            t0 = time.perf_counter()
            feat = extract_simon_b1_features(
                screen_p, cand_batch, final_key_candidate=final_key_candidate
            ).to(device)
            t_feat += time.perf_counter() - t0

            out = model(feat)
            screen_scores[start : start + len(cand_batch)] = out.cpu().numpy()

    screen_seconds = time.perf_counter() - t_screen_start
    if not np.isfinite(screen_scores).all():
        raise ValueError("B1 screening produced non-finite candidate scores")

    # Screening sort with deterministic ascending-candidate tie breaking
    t_sort_start = time.perf_counter()
    candidates_all = np.arange(1 << 16, dtype=np.uint16)
    screen_order = np.lexsort((candidates_all, -screen_scores))
    finalists = candidates_all[screen_order[:rescore_top]]
    screen_tie_count = int(np.count_nonzero(screen_scores == screen_scores[screen_order[0]]))
    screen_collision_count = int(np.sum(np.diff(np.sort(screen_scores)) == 0))
    sort_time_1 = time.perf_counter() - t_sort_start

    # Stage 2: Rescore top finalists on all available pairs (e.g. 32 or 64)
    t_rescore_start = time.perf_counter()
    t0 = time.perf_counter()
    rescore_feat = extract_simon_b1_features(
        full_p, finalists, final_key_candidate=final_key_candidate
    ).to(device)
    t_feat += time.perf_counter() - t0

    with torch.no_grad():
        full_scores = model(rescore_feat).cpu().numpy()
    rescore_seconds = time.perf_counter() - t_rescore_start
    if not np.isfinite(full_scores).all():
        raise ValueError("B1 rescoring produced non-finite candidate scores")

    # Final sort on full rescore scores
    t_sort_start = time.perf_counter()
    rescore_order = np.lexsort((finalists, -full_scores))
    order = finalists[rescore_order]
    full_tie_count = int(np.count_nonzero(full_scores == full_scores[rescore_order[0]]))
    full_collision_count = int(np.sum(np.diff(np.sort(full_scores)) == 0))
    sort_seconds = sort_time_1 + (time.perf_counter() - t_sort_start)

    total_seconds = time.perf_counter() - started

    return SimonB1RankingResult(
        order=order,
        screen_scores=screen_scores,
        full_scores=full_scores,
        finalists=finalists,
        feature_seconds=t_feat,
        screen_seconds=screen_seconds,
        rescore_seconds=rescore_seconds,
        sort_seconds=sort_seconds,
        total_seconds=total_seconds,
        screen_pairs_used=screen_pairs,
        full_pairs_used=actual_full_pairs,
        screen_tie_count=screen_tie_count,
        full_tie_count=full_tie_count,
        screen_collision_count=screen_collision_count,
        full_collision_count=full_collision_count,
    )


def generate_b1_hybrid_8parent_queue(
    pairs: Sequence[tuple[Block, Block]],
    b1_model: SimonPairGroupRanker,
    table_7r: FullPredictedDdt,
    *,
    screen_pairs: int = 16,
    full_pairs: int | None = None,
    parent_count: int = 8,
    children_per_parent: int = 128,
    return_diagnostics: bool = False,
) -> list[tuple[int, int]] | tuple[list[tuple[int, int]], dict[str, Any]]:
    """Assemble 1,024 candidate tuples (k8, k9) in chronological SAT order from pairs alone.

    Stage 1: SimonPairGroupRanker ranks final-round subkey k9 (picks top 8 parents).
    Stage 2: 7R Classical DDT ranks penultimate-round subkey k8 conditioned on each parent k9 (picks top 128 children).
    Tuples sorted by rank-sum with duplicate suppression.
    """
    actual_full_pairs = len(pairs) if full_pairs is None else full_pairs

    # Stage 1
    stage1 = rank_simon_b1_domain(
        pairs,
        b1_model,
        screen_pairs=screen_pairs,
        full_pairs=actual_full_pairs,
    )
    parents = [int(k) for k in stage1.order[:parent_count]]

    # Stage 2
    t_s2_0 = time.perf_counter()
    stage2_results = [
        rank_classical_domain(
            pairs,
            table_7r,
            final_key_candidate=parent,
            expected_rounds=7,
            screen_pairs=screen_pairs,
            full_pairs=actual_full_pairs,
        )
        for parent in parents
    ]
    stage2_seconds = time.perf_counter() - t_s2_0

    # Queue assembly
    t_q0 = time.perf_counter()
    raw_tuples = []
    for r1, parent_k9 in enumerate(parents):
        children_k8 = stage2_results[r1].order[:children_per_parent]
        for r2, child_k8 in enumerate(children_k8):
            raw_tuples.append((r1 + r2, r1, r2, int(parent_k9), int(child_k8)))

    raw_tuples.sort()
    seen = set()
    queue = []
    duplicate_count = 0
    for _, _, _, parent_k9, child_k8 in raw_tuples:
        cand_tuple = (int(child_k8), int(parent_k9))  # Chronological order: (k8, k9)
        if cand_tuple in seen:
            duplicate_count += 1
            continue
        seen.add(cand_tuple)
        queue.append(cand_tuple)
    queue_seconds = time.perf_counter() - t_q0

    if not return_diagnostics:
        return queue

    diagnostics = {
        "stage1_seconds": stage1.total_seconds,
        "stage1_result": stage1,
        "stage2_total_seconds": stage2_seconds,
        "stage2_per_parent_seconds": [d.total_seconds for d in stage2_results],
        "queue_construction_seconds": queue_seconds,
        "candidate_tuples_produced": len(queue),
        "candidate_duplicates_suppressed": duplicate_count,
        "screen_pairs_used": screen_pairs,
        "full_pairs_used": actual_full_pairs,
    }
    return queue, diagnostics


__all__ = [
    "ResBlock1D",
    "SharedSimonPairEncoder",
    "SimonB1RankingResult",
    "SimonGohrResNet",
    "SimonGohrResNetConfig",
    "SimonPairGroupRanker",
    "extract_simon_b1_features",
    "generate_b1_hybrid_8parent_queue",
    "rank_simon_b1_domain",
]
