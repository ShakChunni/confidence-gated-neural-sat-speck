"""Deterministic neural ranking of complete SPECK32/64 final-round keys.

This module implements the Phase 4 ranking boundary only. It does not make
SAT assumptions, recover a master key, or decide whether a neural ranking is
trustworthy. A candidate is scored by inverse-decrypting one final round from
each ciphertext in an attack instance, encoding the resulting pair with the
classifier's established bit layout, and summing the model's raw logits over
all pairs.
"""

from __future__ import annotations

import time
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from neural_sat_speck.data.attack import AttackInstance
from neural_sat_speck.data.generator import encode_ciphertext_pair
from neural_sat_speck.neural.dataset import features_to_tensor
from neural_sat_speck.neural.numpy_inference import (
    NumpyInferenceModule,
    vectorized_final_round_features,
)
from neural_sat_speck.speck.partial import partial_decrypt_final_round
from neural_sat_speck.speck.primitives import WORD_MASK

ALL_FINAL_ROUND_SUBKEYS: tuple[int, ...] = tuple(range(1 << 16))
SCORE_AGGREGATION: str = "sum_logits"


def _validate_candidate_values(
    candidate_values: Sequence[int] | None,
) -> tuple[int, ...]:
    """Return a non-empty ascending candidate set with strict word checks."""

    values = ALL_FINAL_ROUND_SUBKEYS if candidate_values is None else tuple(
        candidate_values
    )
    if not values:
        raise ValueError("candidate_values cannot be empty")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("candidate_values must contain integer words")
        if not 0 <= value <= WORD_MASK:
            raise ValueError(
                f"candidate value {value} is outside the 16-bit word range"
            )
    if len(set(values)) != len(values):
        raise ValueError("candidate_values must not contain duplicates")
    return tuple(sorted(values))


def _resolve_device(model: nn.Module, device: torch.device | str | None) -> torch.device:
    """Resolve an explicit device or use the model's current parameter device."""

    if device is not None:
        return torch.device(device)
    parameter = next(model.parameters(), None)
    return parameter.device if parameter is not None else torch.device("cpu")


def _model_logits(model: nn.Module, features: torch.Tensor) -> torch.Tensor:
    """Run a model and normalize its output to one finite logit per row."""

    output = model(features)
    if not isinstance(output, torch.Tensor):
        raise TypeError("ranking model must return a torch.Tensor")
    logits = output.reshape(-1)
    if logits.numel() != features.shape[0]:
        raise ValueError(
            "ranking model must return one scalar logit per feature row; "
            f"got {tuple(output.shape)} for {features.shape[0]} rows"
        )
    if not torch.isfinite(logits).all():
        raise ValueError("ranking model returned a non-finite logit")
    return logits


def _validate_pair_subsets(
    pair_subsets: Sequence[Sequence[int]] | None,
    *,
    pair_count: int,
    candidate_count: int,
    top_k: int,
) -> tuple[tuple[int, ...], ...]:
    """Validate fixed pair subsets used for optional ranking stability."""

    if pair_subsets is None:
        return ()
    if isinstance(pair_subsets, (str, bytes)):
        raise TypeError("pair_subsets must be a sequence of index sequences")
    subsets = tuple(pair_subsets)
    if len(subsets) < 2:
        raise ValueError("pair_subsets must contain at least two subsets")
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("stability_top_k must be an integer")
    if not 1 <= top_k <= candidate_count:
        raise ValueError("stability_top_k must be between one and the candidate count")

    validated: list[tuple[int, ...]] = []
    subset_size: int | None = None
    for subset in subsets:
        if isinstance(subset, (str, bytes)):
            raise TypeError("each pair subset must be a sequence of indices")
        indices = tuple(subset)
        if not indices:
            raise ValueError("pair subsets cannot be empty")
        if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < pair_count
            for index in indices
        ):
            raise ValueError("pair subset indices must be valid pair positions")
        if len(set(indices)) != len(indices):
            raise ValueError("pair subset indices must not repeat")
        if subset_size is None:
            subset_size = len(indices)
        elif len(indices) != subset_size:
            raise ValueError("pair subsets must contain the same number of pairs")
        validated.append(tuple(sorted(indices)))
    return tuple(validated)


@dataclass(frozen=True)
class PairSubsetStability:
    """Stability summary for fixed pair-subset rankings."""

    pair_subsets: tuple[tuple[int, ...], ...]
    top_k: int
    subset_top_k_candidates: tuple[tuple[int, ...], ...]
    full_top_k_jaccard: float
    pairwise_top_k_jaccard: float
    top1_full_order_agreement: float

    def __post_init__(self) -> None:
        if len(self.pair_subsets) < 2:
            raise ValueError("stability requires at least two pair subsets")
        if len(self.pair_subsets) != len(self.subset_top_k_candidates):
            raise ValueError("stability subset metadata must have equal length")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int):
            raise TypeError("stability top_k must be an integer")
        if self.top_k <= 0:
            raise ValueError("stability top_k must be positive")
        for subset, candidates in zip(
            self.pair_subsets,
            self.subset_top_k_candidates,
            strict=True,
        ):
            if not subset or len(set(subset)) != len(subset):
                raise ValueError("stability pair subsets must be non-empty and unique")
            if len(candidates) != self.top_k or len(set(candidates)) != len(candidates):
                raise ValueError("stability top-k candidates are invalid")
        for value in (
            self.full_top_k_jaccard,
            self.pairwise_top_k_jaccard,
            self.top1_full_order_agreement,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("stability scores must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        """Serialize stability metadata and aggregate scores."""

        return {
            "pair_subsets": [list(subset) for subset in self.pair_subsets],
            "top_k": self.top_k,
            "subset_top_k_candidates": [
                list(candidates) for candidates in self.subset_top_k_candidates
            ],
            "full_top_k_jaccard": self.full_top_k_jaccard,
            "pairwise_top_k_jaccard": self.pairwise_top_k_jaccard,
            "top1_full_order_agreement": self.top1_full_order_agreement,
        }


@dataclass(frozen=True)
class SubkeyRanking:
    """Scores and deterministic order for one final-round ranking run.

    ``candidate_values`` and ``candidate_scores`` are parallel ascending-value
    tuples. ``candidate_order`` contains those values sorted by descending
    score, with ascending numeric value as the deterministic tie-breaker.
    Timings are wall-clock measurements for this invocation; the feature
    tensor value is the largest single input tensor allocated by the ranker,
    not the model's total activation memory.
    """

    candidate_values: tuple[int, ...]
    candidate_scores: tuple[float, ...]
    candidate_order: tuple[int, ...]
    pair_count: int
    candidate_batch_size: int
    score_aggregation: str
    partial_decryption_seconds: float
    inference_seconds: float
    sorting_seconds: float
    total_seconds: float
    peak_feature_tensor_bytes: int
    pair_subset_stability: PairSubsetStability | None = None

    def __post_init__(self) -> None:
        if len(self.candidate_values) != len(self.candidate_scores):
            raise ValueError("candidate values and scores must have equal length")
        if set(self.candidate_order) != set(self.candidate_values):
            raise ValueError("candidate order must contain every candidate once")
        if self.pair_count <= 0:
            raise ValueError("pair_count must be positive")
        if self.candidate_batch_size <= 0:
            raise ValueError("candidate_batch_size must be positive")
        if self.score_aggregation != SCORE_AGGREGATION:
            raise ValueError("unsupported score aggregation")
        if self.pair_subset_stability is not None:
            if self.pair_subset_stability.top_k > len(self.candidate_values):
                raise ValueError("stability top_k exceeds the candidate count")
            candidate_set = set(self.candidate_values)
            for subset_candidates in (
                self.pair_subset_stability.subset_top_k_candidates
            ):
                if not set(subset_candidates) <= candidate_set:
                    raise ValueError("stability contains an unscored candidate")

    def score_for(self, candidate: int) -> float:
        """Return the aggregate score for one candidate value."""

        if isinstance(candidate, bool) or not isinstance(candidate, int):
            raise TypeError("candidate must be an integer")
        position = bisect_left(self.candidate_values, candidate)
        if (
            position == len(self.candidate_values)
            or self.candidate_values[position] != candidate
        ):
            raise KeyError(f"candidate {candidate:#06x} was not scored")
        return self.candidate_scores[position]

    def rank(self, candidate: int) -> int:
        """Return the deterministic one-based rank of a scored candidate."""

        if isinstance(candidate, bool) or not isinstance(candidate, int):
            raise TypeError("candidate must be an integer")
        try:
            return self.candidate_order.index(candidate) + 1
        except ValueError as error:
            raise KeyError(f"candidate {candidate:#06x} was not scored") from error

    def rank_interval(self, candidate: int) -> tuple[int, int]:
        """Return the one-based rank interval implied by score ties."""

        target = self.score_for(candidate)
        greater = sum(score > target for score in self.candidate_scores)
        greater_or_equal = sum(
            score >= target for score in self.candidate_scores
        )
        return greater + 1, greater_or_equal

    def top_k(self, k: int) -> tuple[int, ...]:
        """Return the first ``k`` candidates from the deterministic order."""

        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an integer")
        if not 0 <= k <= len(self.candidate_order):
            raise ValueError("k must be between zero and the candidate count")
        return self.candidate_order[:k]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the ranking and resource measurements to JSON data."""

        result: dict[str, Any] = {
            "candidate_values": list(self.candidate_values),
            "candidate_scores": list(self.candidate_scores),
            "candidate_order": list(self.candidate_order),
            "pair_count": self.pair_count,
            "candidate_batch_size": self.candidate_batch_size,
            "score_aggregation": self.score_aggregation,
            "timing_seconds": {
                "partial_decryption": self.partial_decryption_seconds,
                "inference": self.inference_seconds,
                "sorting": self.sorting_seconds,
                "total": self.total_seconds,
            },
            "peak_feature_tensor_bytes": self.peak_feature_tensor_bytes,
        }
        if self.pair_subset_stability is not None:
            result["pair_subset_stability"] = self.pair_subset_stability.to_dict()
        return result


def rank_final_round_subkeys(
    instance: AttackInstance,
    model: nn.Module,
    *,
    candidate_values: Sequence[int] | None = None,
    candidate_batch_size: int = 1024,
    device: torch.device | str | None = None,
    stability_pair_subsets: Sequence[Sequence[int]] | None = None,
    stability_top_k: int = 16,
) -> SubkeyRanking:
    """Rank candidate 16-bit final-round keys for one attack instance.

    The candidate set defaults to all ``2**16`` 16-bit values. Candidates are
    processed in ascending numeric order, in batches, and each candidate is
    paired with every ciphertext pair in the instance. The model receives
    raw logits; no sigmoid or threshold is applied. The known subkey stored in
    ``instance`` is never used during scoring.
    """

    if not isinstance(instance, AttackInstance):
        raise TypeError("instance must be an AttackInstance")
    if not instance.pairs:
        raise ValueError("attack instance must contain at least one pair")
    if isinstance(candidate_batch_size, bool) or not isinstance(
        candidate_batch_size, int
    ):
        raise TypeError("candidate_batch_size must be an integer")
    if candidate_batch_size <= 0:
        raise ValueError("candidate_batch_size must be positive")
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")

    candidates = _validate_candidate_values(candidate_values)
    validated_pair_subsets = _validate_pair_subsets(
        stability_pair_subsets,
        pair_count=len(instance.pairs),
        candidate_count=len(candidates),
        top_k=stability_top_k,
    )
    resolved_device = _resolve_device(model, device)
    original_parameter = next(model.parameters(), None)
    original_device = (
        original_parameter.device if original_parameter is not None else None
    )
    was_training = model.training
    total_started = time.perf_counter()
    feature_seconds = 0.0
    inference_seconds = 0.0
    peak_feature_tensor_bytes = 0
    scores: list[float] = []
    numpy_backend = isinstance(model, NumpyInferenceModule)
    vectorized_features = numpy_backend or resolved_device.type == "cpu"
    stability_scores = (
        [[0.0] * len(candidates) for _ in validated_pair_subsets]
        if validated_pair_subsets
        else None
    )

    model.to(resolved_device)
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(candidates), candidate_batch_size):
                candidate_batch = candidates[start : start + candidate_batch_size]
                feature_started = time.perf_counter()
                if vectorized_features:
                    feature_array = vectorized_final_round_features(
                        instance.pairs,
                        candidate_batch,
                    )
                    peak_feature_tensor_bytes = max(
                        peak_feature_tensor_bytes,
                        feature_array.nbytes,
                    )
                    if not numpy_backend:
                        feature_tensor = torch.from_numpy(
                            feature_array.reshape(
                                len(candidate_batch) * len(instance.pairs),
                                4,
                                16,
                            )
                        ).to(resolved_device)
                else:
                    features: list[tuple[int, ...]] = []
                    for candidate in candidate_batch:
                        for pair in instance.pairs:
                            state0 = partial_decrypt_final_round(
                                pair.ciphertext0,
                                candidate,
                            )
                            state1 = partial_decrypt_final_round(
                                pair.ciphertext1,
                                candidate,
                            )
                            features.append(encode_ciphertext_pair(state0, state1))
                    feature_tensor = features_to_tensor(features)
                    peak_feature_tensor_bytes = max(
                        peak_feature_tensor_bytes,
                        feature_tensor.numel() * feature_tensor.element_size(),
                    )
                    feature_tensor = feature_tensor.to(resolved_device)
                feature_seconds += time.perf_counter() - feature_started

                inference_started = time.perf_counter()
                if numpy_backend:
                    logits = torch.from_numpy(
                        model.predict_numpy(  # type: ignore[attr-defined]
                            feature_array.reshape(
                                len(candidate_batch) * len(instance.pairs),
                                4,
                                16,
                            )
                        )
                    )
                else:
                    logits = _model_logits(model, feature_tensor)
                batch_scores = logits.view(len(candidate_batch), len(instance.pairs))
                scores.extend(batch_scores.sum(dim=1).detach().cpu().tolist())
                if stability_scores is not None:
                    for subset_index, pair_indices in enumerate(
                        validated_pair_subsets
                    ):
                        subset_scores = (
                            batch_scores[:, list(pair_indices)]
                            .sum(dim=1)
                            .detach()
                            .cpu()
                            .tolist()
                        )
                        stability_scores[subset_index][
                            start : start + len(candidate_batch)
                        ] = [float(score) for score in subset_scores]
                inference_seconds += time.perf_counter() - inference_started
    finally:
        if was_training:
            model.train()
        else:
            model.eval()
        if original_device is not None and original_device != resolved_device:
            model.to(original_device)

    sorting_started = time.perf_counter()
    ranked_pairs = sorted(
        zip(scores, candidates, strict=True),
        key=lambda item: (-item[0], item[1]),
    )
    candidate_order = tuple(candidate for _, candidate in ranked_pairs)
    pair_subset_stability: PairSubsetStability | None = None
    if stability_scores is not None:
        subset_orders = tuple(
            tuple(
                candidate
                for _, candidate in sorted(
                    zip(subset_score, candidates, strict=True),
                    key=lambda item: (-item[0], item[1]),
                )
            )
            for subset_score in stability_scores
        )
        full_top_k = set(candidate_order[:stability_top_k])
        full_jaccards = tuple(
            len(full_top_k & set(subset_order[:stability_top_k]))
            / len(full_top_k | set(subset_order[:stability_top_k]))
            for subset_order in subset_orders
        )
        pairwise_jaccards: list[float] = []
        for first_index, first_order in enumerate(subset_orders):
            first_set = set(first_order[:stability_top_k])
            for second_order in subset_orders[first_index + 1 :]:
                second_set = set(second_order[:stability_top_k])
                pairwise_jaccards.append(
                    len(first_set & second_set) / len(first_set | second_set)
                )
        pair_subset_stability = PairSubsetStability(
            pair_subsets=validated_pair_subsets,
            top_k=stability_top_k,
            subset_top_k_candidates=tuple(
                subset_order[:stability_top_k] for subset_order in subset_orders
            ),
            full_top_k_jaccard=sum(full_jaccards) / len(full_jaccards),
            pairwise_top_k_jaccard=sum(pairwise_jaccards) / len(pairwise_jaccards),
            top1_full_order_agreement=sum(
                subset_order[0] == candidate_order[0]
                for subset_order in subset_orders
            )
            / len(subset_orders),
        )
    sorting_seconds = time.perf_counter() - sorting_started
    total_seconds = time.perf_counter() - total_started
    return SubkeyRanking(
        candidate_values=candidates,
        candidate_scores=tuple(float(score) for score in scores),
        candidate_order=candidate_order,
        pair_count=len(instance.pairs),
        candidate_batch_size=candidate_batch_size,
        score_aggregation=SCORE_AGGREGATION,
        partial_decryption_seconds=feature_seconds,
        inference_seconds=inference_seconds,
        sorting_seconds=sorting_seconds,
        total_seconds=total_seconds,
        peak_feature_tensor_bytes=peak_feature_tensor_bytes,
        pair_subset_stability=pair_subset_stability,
    )


def evaluate_subkey_ranking(
    ranking: SubkeyRanking,
    true_subkey: int,
    *,
    top_k_values: Sequence[int] = (1, 16, 32, 64, 128, 256, 1024),
) -> dict[str, Any]:
    """Compute post hoc true-key rank and top-k membership metrics."""

    if isinstance(true_subkey, bool) or not isinstance(true_subkey, int):
        raise TypeError("true_subkey must be an integer")
    if not 0 <= true_subkey <= WORD_MASK:
        raise ValueError("true_subkey must be a 16-bit word")
    top_k_membership: dict[str, bool] = {}
    for k in top_k_values:
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("top_k_values must contain integers")
        if not 0 <= k <= len(ranking.candidate_order):
            raise ValueError("top-k value is outside the candidate count")
        top_k_membership[str(k)] = true_subkey in ranking.top_k(k)

    if true_subkey in ranking.candidate_values:
        rank: int | None = ranking.rank(true_subkey)
        rank_interval: tuple[int, int] | None = ranking.rank_interval(true_subkey)
    else:
        rank = None
        rank_interval = None
    return {
        "true_final_round_subkey": true_subkey,
        "true_rank": rank,
        "true_rank_interval": list(rank_interval) if rank_interval else None,
        "top_k_membership": top_k_membership,
        "true_subkey_in_candidate_set": true_subkey in ranking.candidate_values,
    }
