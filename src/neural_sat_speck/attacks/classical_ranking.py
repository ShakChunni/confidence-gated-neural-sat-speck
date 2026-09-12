"""Sparse empirical difference-distribution likelihood ranker for SPECK32/64.

This module implements an empirical differential cryptanalysis baseline using a
sparse Monte Carlo sample ($10^7$ pairs) of penultimate-round XOR differences.
It is an empirical surrogate baseline, NOT a full 2^32-state Markov predicted
difference distribution table (Gohr 2019, Section 4 & 5).

Key methodological characteristics:
1. Monte Carlo generation: Empirical output differences are sampled under
   independent random round keys (Markov model assumption).
2. Sparse thresholding: Only observed differences with occurrence count >= min_count
   (default 5) are retained to filter finite-sampling noise.
3. Positive LLR clipping: Weights are computed as log2(p(diff)) - log2(2^-32) =
   log2(p(diff)) + 32.0. Differences with probability <= 2^-32 or unobserved
   differences receive weight 0.0. Negative-evidence penalties are not applied.
4. Deterministic tie-breaking: Candidates with identical total scores are ordered
   by ascending numerical candidate value. Because candidate scores frequently tie
   (e.g., k and k ^ 0x8000 share penultimate differences), rank intervals
   (best rank, worst rank, midrank) are computed alongside sorted rank.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from neural_sat_speck.attacks.ranking import (
    SubkeyRanking,
    _validate_candidate_values,
)
from neural_sat_speck.data.attack import AttackInstance

BACKGROUND_LOG2_PROB: float = -32.0  # log2(2^-32) for 32-bit differences


def compute_weights_digest(weights: Mapping[int, float]) -> str:
    """Compute deterministic SHA-256 digest of the sorted (difference, weight) mapping."""
    canonical_repr = json.dumps(
        {f"0x{int(k):x}": float(v) for k, v in sorted(weights.items(), key=lambda x: int(x[0]))},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical_repr).hexdigest()


@dataclass(frozen=True)
class RankInterval:
    """Tie-aware rank statistics for a candidate under a scoring vector."""

    best_rank: int
    worst_rank: int
    midrank: float
    tied_count: int
    sorted_rank: int
    score: float


@dataclass(frozen=True)
class ClassicalDistributionTable:
    """Lookup table of log-likelihood weights for 32-bit differential transitions.

    Maps packed 32-bit difference ``(dx << 16) | dy`` to its positive LLR weight:
    ``weight = log2(p(diff)) - log2(2^-32) = log2(p(diff)) + 32.0``.
    Differences with probability <= 2^-32 or unobserved differences have weight 0.0.
    """

    rounds: int
    weights: dict[int, float]
    sample_count: int
    min_count: int
    seed: int = 999999
    seed_role: str = "demo"
    input_difference: tuple[int, int] = (0x0040, 0x0000)
    independent_round_keys: bool = True
    numpy_version: str = ""
    generator_script_hash: str | None = None
    ranking_implementation_hash: str | None = None
    weights_digest: str = ""

    def __post_init__(self) -> None:
        if self.rounds <= 0:
            raise ValueError("rounds must be positive")
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if self.min_count < 1:
            raise ValueError("min_count must be at least 1")
        for k, v in self.weights.items():
            if not isinstance(k, int) or not (0 <= k <= 0xFFFFFFFF):
                raise ValueError(f"invalid 32-bit difference key: {k!r}")
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"weight for {k:#010x} must be a finite number, got {v!r}")
        if not self.weights_digest:
            object.__setattr__(self, "weights_digest", compute_weights_digest(self.weights))
        if not self.numpy_version:
            object.__setattr__(self, "numpy_version", np.__version__)

    def weight_for(self, diff_32: int) -> float:
        """Return the positive LLR weight for a 32-bit packed difference."""
        return self.weights.get(diff_32, 0.0)

    def to_dict(self) -> dict[str, Any]:
        """Serialize distribution table to a versioned JSON dictionary."""
        return {
            "schema": "neural_sat_speck.classical_distribution_table.v2",
            "type": "sparse_empirical_monte_carlo",
            "rounds": self.rounds,
            "sample_count": self.sample_count,
            "min_count": self.min_count,
            "seed": self.seed,
            "seed_role": self.seed_role,
            "input_difference": list(self.input_difference),
            "independent_round_keys": self.independent_round_keys,
            "numpy_version": self.numpy_version,
            "generator_script_hash": self.generator_script_hash,
            "ranking_implementation_hash": self.ranking_implementation_hash,
            "weights_count": len(self.weights),
            "weights_digest": self.weights_digest,
            # Keys converted to hex strings for JSON portability
            "weights": {hex(k): v for k, v in self.weights.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClassicalDistributionTable:
        """Construct distribution table from serialized dictionary."""
        if "rounds" not in data or "sample_count" not in data or "weights" not in data:
            raise KeyError("data dictionary missing required distribution table fields")
        weights = {
            int(k, 16) if isinstance(k, str) else int(k): float(v)
            for k, v in data["weights"].items()
        }
        raw_diff = data.get("input_difference", [0x0040, 0x0000])
        input_diff = (int(raw_diff[0]), int(raw_diff[1]))
        digest = data.get("weights_digest") or compute_weights_digest(weights)
        return cls(
            rounds=int(data["rounds"]),
            weights=weights,
            sample_count=int(data["sample_count"]),
            min_count=int(data.get("min_count", 1)),
            seed=int(data.get("seed", 999999)),
            seed_role=str(data.get("seed_role", "demo")),
            input_difference=input_diff,
            independent_round_keys=bool(data.get("independent_round_keys", True)),
            numpy_version=str(data.get("numpy_version", "")),
            generator_script_hash=data.get("generator_script_hash"),
            ranking_implementation_hash=data.get("ranking_implementation_hash"),
            weights_digest=digest,
        )

    def save(self, path: Path | str, *, allow_overwrite: bool = False) -> None:
        """Save distribution table to JSON file using atomic write plus rename."""
        target = Path(path)
        if target.exists() and not allow_overwrite:
            raise FileExistsError(
                f"Destination {target} already exists. Refusing to overwrite authoritative artifact."
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_path = target.with_suffix(f".tmp.{os.getpid()}")
        try:
            temp_path.write_text(json.dumps(self.to_dict()), encoding="utf-8")
            temp_path.replace(target)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

    @classmethod
    def load(cls, path: Path | str) -> ClassicalDistributionTable:
        """Load distribution table from JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


SparseEmpiricalDistributionTable = ClassicalDistributionTable


def compute_empirical_speck_distribution(
    rounds: int,
    sample_count: int = 5_000_000,
    *,
    input_difference: tuple[int, int] = (0x0040, 0x0000),
    min_count: int = 5,
    seed: int = 999999,
    seed_role: str = "demo",
    generator_script_hash: str | None = None,
    ranking_implementation_hash: str | None = None,
) -> ClassicalDistributionTable:
    """Generate reference difference distribution via vectorized simulation.

    Encrypts ``sample_count`` pairs under independent random round keys (Markov
    cipher model assumption) with input difference ``input_difference``.
    Computes log2 likelihood weights for all observed output differences with
    occurrence count >= ``min_count`` (to filter finite-sampling noise).

    Negative and neutral transitions (LLR <= 0.0) are omitted from the table,
    yielding a sparse positive-evidence surrogate rather than a full 2^32 DDT.
    """
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if min_count < 1:
        raise ValueError("min_count must be at least 1")

    rng = np.random.default_rng(seed)
    # Generate random plaintexts and round keys
    x0 = rng.integers(0, 65536, size=sample_count, dtype=np.uint16)
    y0 = rng.integers(0, 65536, size=sample_count, dtype=np.uint16)
    x1 = x0 ^ np.uint16(input_difference[0])
    y1 = y0 ^ np.uint16(input_difference[1])

    keys = rng.integers(0, 65536, size=(rounds, sample_count), dtype=np.uint16)

    # Vectorized SPECK round encryption
    for r in range(rounds):
        kr = keys[r]
        # x = ((x >>> 7) + y) ^ kr
        x0 = (((x0 >> 7) | (x0 << 9)) + y0) ^ kr
        # y = (y <<< 2) ^ x
        y0 = ((y0 << 2) | (y0 >> 14)) ^ x0

        x1 = (((x1 >> 7) | (x1 << 9)) + y1) ^ kr
        y1 = ((y1 << 2) | (y1 >> 14)) ^ x1

    dx = (x0 ^ x1).astype(np.uint64)
    dy = (y0 ^ y1).astype(np.uint64)
    diffs = (dx << 16) | dy

    unique_vals, counts = np.unique(diffs, return_counts=True)
    mask = counts >= min_count

    # LLR = log2(count / sample_count) - log2(2^-32) = log2(count / sample_count) + 32.0
    llr_weights = np.log2(counts[mask] / sample_count) + 32.0
    # Keep only positive weights (differences more probable than uniform random)
    positive_mask = llr_weights > 0.0

    valid_diffs = unique_vals[mask][positive_mask]
    valid_weights = llr_weights[positive_mask]

    weights_dict = {
        int(diff): float(weight)
        for diff, weight in zip(valid_diffs, valid_weights, strict=True)
    }

    digest = compute_weights_digest(weights_dict)

    return ClassicalDistributionTable(
        rounds=rounds,
        weights=weights_dict,
        sample_count=sample_count,
        min_count=min_count,
        seed=seed,
        seed_role=seed_role,
        input_difference=input_difference,
        independent_round_keys=True,
        numpy_version=np.__version__,
        generator_script_hash=generator_script_hash,
        ranking_implementation_hash=ranking_implementation_hash,
        weights_digest=digest,
    )


def rank_classical_subkeys(
    instance: AttackInstance,
    distribution: ClassicalDistributionTable | Mapping[int, float],
    *,
    candidate_values: Sequence[int] | None = None,
) -> SubkeyRanking:
    """Rank candidate 16-bit final-round subkeys using classical difference distribution.

    Decrypts the final round across all pairs for each candidate subkey, looks up
    the log-likelihood ratio for each penultimate difference, and sorts candidates
    by descending total score. Ties are broken deterministically by ascending
    candidate subkey value.
    """
    if not isinstance(instance, AttackInstance):
        raise TypeError("instance must be an AttackInstance")
    if not instance.pairs:
        raise ValueError("attack instance must contain at least one pair")

    candidates = _validate_candidate_values(candidate_values)
    num_candidates = len(candidates)
    weights = distribution.weights if isinstance(distribution, ClassicalDistributionTable) else distribution

    t_start = time.perf_counter()

    # Extract ciphertext pairs into arrays (P, 2)
    c0 = np.array(
        [(p.ciphertext0[0], p.ciphertext0[1]) for p in instance.pairs],
        dtype=np.uint16,
    )
    c1 = np.array(
        [(p.ciphertext1[0], p.ciphertext1[1]) for p in instance.pairs],
        dtype=np.uint16,
    )

    t_dec_start = time.perf_counter()
    # In SPECK decryption:
    # y = (c1 ^ c0) >>> 2 is independent of candidate key k!
    y0_dec = ((c0[:, 1] ^ c0[:, 0]) >> 2) | ((c0[:, 1] ^ c0[:, 0]) << 14)
    y1_dec = ((c1[:, 1] ^ c1[:, 0]) >> 2) | ((c1[:, 1] ^ c1[:, 0]) << 14)
    dy = y0_dec ^ y1_dec  # shape (P,)
    dy_shifted = dy.astype(np.uint64)
    decryption_time = time.perf_counter() - t_dec_start

    t_score_start = time.perf_counter()
    scores: list[float] = [0.0] * num_candidates

    # Score each candidate key
    c0_x = c0[:, 0]
    c1_x = c1[:, 0]
    for idx, k in enumerate(candidates):
        # x0_dec = ((c0_x ^ k) - y0_dec) <<< 7
        term0 = ((c0_x ^ k) - y0_dec) & 0xFFFF
        x0_dec = ((term0 << 7) | (term0 >> 9)) & 0xFFFF

        term1 = ((c1_x ^ k) - y1_dec) & 0xFFFF
        x1_dec = ((term1 << 7) | (term1 >> 9)) & 0xFFFF

        dx = (x0_dec ^ x1_dec).astype(np.uint64)
        diff_32 = (dx << 16) | dy_shifted

        # Sum positive LLR weights
        cand_score = 0.0
        for d in diff_32:
            w = weights.get(int(d), 0.0)
            if w > 0.0:
                cand_score += w
        scores[idx] = cand_score

    scoring_time = time.perf_counter() - t_score_start

    t_sort_start = time.perf_counter()
    # Deterministic sorting: descending score, ascending numerical key value
    candidate_order = tuple(
        cand
        for cand, _ in sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: (-item[1], item[0]),
        )
    )
    sorting_time = time.perf_counter() - t_sort_start
    total_time = time.perf_counter() - t_start

    return SubkeyRanking(
        candidate_values=candidates,
        candidate_scores=tuple(scores),
        candidate_order=candidate_order,
        pair_count=len(instance.pairs),
        candidate_batch_size=len(candidates),
        score_aggregation="sum_logits",
        partial_decryption_seconds=decryption_time,
        inference_seconds=scoring_time,
        sorting_seconds=sorting_time,
        total_seconds=total_time,
        peak_feature_tensor_bytes=len(candidates) * 8,
    )


def compute_rank_interval(
    ranking: SubkeyRanking,
    target_candidate: int,
) -> RankInterval:
    """Compute tie-aware rank statistics for target_candidate under ranking.

    Returns a RankInterval with:
    - best_rank: 1 + count(score > target_score)
    - worst_rank: count(score >= target_score)
    - midrank: (best_rank + worst_rank) / 2.0
    - tied_count: worst_rank - best_rank + 1
    - sorted_rank: 1-indexed position in ranking.candidate_order
    - score: candidate score
    """
    scores_by_cand = dict(
        zip(ranking.candidate_values, ranking.candidate_scores, strict=True)
    )
    if target_candidate not in scores_by_cand:
        raise KeyError(f"candidate {target_candidate} not present in ranking")
    target_score = scores_by_cand[target_candidate]
    scores_arr = np.array(ranking.candidate_scores, dtype=np.float64)

    best_rank = int(np.sum(scores_arr > target_score)) + 1
    worst_rank = int(np.sum(scores_arr >= target_score))
    midrank = (best_rank + worst_rank) / 2.0
    tied_count = worst_rank - best_rank + 1
    sorted_rank = ranking.rank(target_candidate)

    return RankInterval(
        best_rank=best_rank,
        worst_rank=worst_rank,
        midrank=midrank,
        tied_count=tied_count,
        sorted_rank=sorted_rank,
        score=float(target_score),
    )
