"""Complete-domain ranking for the Phase 6 pair-aware rankers."""

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
from neural_sat_speck.neural.group_model import PairGroupRanker
from neural_sat_speck.neural.numpy_inference import (
    NumpyInferenceModule,
    vectorized_final_round_features,
)
from neural_sat_speck.speck.partial import partial_decrypt_final_round
from neural_sat_speck.speck.primitives import WORD_MASK

from .ranking import _validate_candidate_values

PAIR_GROUP_SCORE_AGGREGATION = "group_logit"


def _resolve_device(model: nn.Module, device: torch.device | str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    parameter = next(model.parameters(), None)
    return parameter.device if parameter is not None else torch.device("cpu")


def _group_logits(model: nn.Module, features: torch.Tensor) -> torch.Tensor:
    output = model(features)
    if not isinstance(output, torch.Tensor):
        raise TypeError("pair-group ranking model must return a torch.Tensor")
    logits = output.reshape(-1)
    if logits.numel() != features.shape[0]:
        raise ValueError(
            "pair-group ranking model must return one scalar logit per group; "
            f"got {tuple(output.shape)} for {features.shape[0]} groups"
        )
    if not torch.isfinite(logits).all():
        raise ValueError("pair-group ranking model returned a non-finite logit")
    return logits


@dataclass(frozen=True)
class PairGroupSubkeyRanking:
    """Scores and deterministic order for one pair-aware ranking run."""

    candidate_values: tuple[int, ...]
    candidate_scores: tuple[float, ...]
    candidate_order: tuple[int, ...]
    pair_count: int
    group_size: int
    candidate_batch_size: int
    score_aggregation: str
    partial_decryption_seconds: float
    inference_seconds: float
    sorting_seconds: float
    total_seconds: float
    peak_feature_tensor_bytes: int

    def __post_init__(self) -> None:
        if len(self.candidate_values) != len(self.candidate_scores):
            raise ValueError("candidate values and scores must have equal length")
        if set(self.candidate_order) != set(self.candidate_values):
            raise ValueError("candidate order must contain every candidate once")
        if self.pair_count != self.group_size:
            raise ValueError("pair count must equal the pair-group size")
        if self.pair_count != 64:
            raise ValueError("pair-aware ranking requires exactly 64 pairs")
        if self.candidate_batch_size <= 0:
            raise ValueError("candidate_batch_size must be positive")
        if self.score_aggregation != PAIR_GROUP_SCORE_AGGREGATION:
            raise ValueError("unsupported pair-group score aggregation")

    def score_for(self, candidate: int) -> float:
        """Return a candidate's group logit."""

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
        """Return a candidate's deterministic one-based rank."""

        if isinstance(candidate, bool) or not isinstance(candidate, int):
            raise TypeError("candidate must be an integer")
        try:
            return self.candidate_order.index(candidate) + 1
        except ValueError as error:
            raise KeyError(f"candidate {candidate:#06x} was not scored") from error

    def rank_interval(self, candidate: int) -> tuple[int, int]:
        """Return the rank interval implied by exact score ties."""

        target = self.score_for(candidate)
        greater = sum(score > target for score in self.candidate_scores)
        greater_or_equal = sum(score >= target for score in self.candidate_scores)
        return greater + 1, greater_or_equal

    def top_k(self, k: int) -> tuple[int, ...]:
        """Return the first ``k`` deterministically ordered candidates."""

        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an integer")
        if not 0 <= k <= len(self.candidate_order):
            raise ValueError("k must be between zero and the candidate count")
        return self.candidate_order[:k]

    def to_dict(self) -> dict[str, Any]:
        """Serialize scores, order, and resource measurements."""

        return {
            "candidate_values": list(self.candidate_values),
            "candidate_scores": list(self.candidate_scores),
            "candidate_order": list(self.candidate_order),
            "pair_count": self.pair_count,
            "group_size": self.group_size,
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


def rank_pair_group_subkeys(
    instance: AttackInstance,
    model: PairGroupRanker,
    *,
    candidate_values: Sequence[int] | None = None,
    candidate_batch_size: int = 128,
    device: torch.device | str | None = None,
) -> PairGroupSubkeyRanking:
    """Rank all candidate final-round keys using one group logit each.

    Every candidate is evaluated on exactly the same 64 ciphertext pairs.
    The known final-round subkey stored in ``instance`` is never consulted
    during scoring; it is used only by the caller for post-hoc evaluation.
    """

    if not isinstance(instance, AttackInstance):
        raise TypeError("instance must be an AttackInstance")
    if len(instance.pairs) != 64:
        raise ValueError("pair-aware ranking requires an instance with 64 pairs")
    if not isinstance(model, (PairGroupRanker, NumpyInferenceModule)):
        raise TypeError("model must be a PairGroupRanker")
    if isinstance(model, NumpyInferenceModule) and not model.is_group_model:
        raise TypeError("NumPy ranking model must wrap a PairGroupRanker")
    if isinstance(candidate_batch_size, bool) or not isinstance(
        candidate_batch_size, int
    ):
        raise TypeError("candidate_batch_size must be an integer")
    if candidate_batch_size <= 0:
        raise ValueError("candidate_batch_size must be positive")
    candidates = _validate_candidate_values(candidate_values)
    resolved_device = _resolve_device(model, device)
    original_parameter = next(model.parameters(), None)
    original_device = original_parameter.device if original_parameter is not None else None
    was_training = model.training
    total_started = time.perf_counter()
    partial_seconds = 0.0
    inference_seconds = 0.0
    peak_feature_tensor_bytes = 0
    scores: list[float] = []
    numpy_backend = isinstance(model, NumpyInferenceModule)
    vectorized_features = numpy_backend or resolved_device.type == "cpu"

    model.to(resolved_device)
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(candidates), candidate_batch_size):
                candidate_batch = candidates[start : start + candidate_batch_size]
                partial_started = time.perf_counter()
                if vectorized_features:
                    feature_array = vectorized_final_round_features(
                        instance.pairs,
                        candidate_batch,
                    )
                    if not numpy_backend:
                        features = torch.from_numpy(feature_array).to(
                            resolved_device
                        )
                else:
                    batch_features: list[tuple[int, ...]] = []
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
                            batch_features.append(
                                encode_ciphertext_pair(state0, state1)
                            )
                partial_seconds += time.perf_counter() - partial_started
                if vectorized_features:
                    feature_bytes = feature_array.nbytes
                else:
                    features = features_to_tensor(batch_features).reshape(
                        len(candidate_batch),
                        64,
                        4,
                        16,
                    )
                    feature_bytes = features.numel() * features.element_size()
                peak_feature_tensor_bytes = max(peak_feature_tensor_bytes, feature_bytes)
                inference_started = time.perf_counter()
                if numpy_backend:
                    logits = torch.from_numpy(
                        model.predict_numpy(feature_array)  # type: ignore[attr-defined]
                    )
                else:
                    logits = _group_logits(
                        model,
                        features.to(resolved_device),
                    )
                inference_seconds += time.perf_counter() - inference_started
                scores.extend(float(value) for value in logits.cpu())

        sorting_started = time.perf_counter()
        candidate_order = tuple(
            candidate
            for candidate, _ in sorted(
                zip(candidates, scores, strict=True),
                key=lambda item: (-item[1], item[0]),
            )
        )
        sorting_seconds = time.perf_counter() - sorting_started
    finally:
        if original_device is not None:
            model.to(original_device)
        if was_training:
            model.train()
        else:
            model.eval()

    return PairGroupSubkeyRanking(
        candidate_values=candidates,
        candidate_scores=tuple(scores),
        candidate_order=candidate_order,
        pair_count=64,
        group_size=64,
        candidate_batch_size=candidate_batch_size,
        score_aggregation=PAIR_GROUP_SCORE_AGGREGATION,
        partial_decryption_seconds=partial_seconds,
        inference_seconds=inference_seconds,
        sorting_seconds=sorting_seconds,
        total_seconds=time.perf_counter() - total_started,
        peak_feature_tensor_bytes=peak_feature_tensor_bytes,
    )


def evaluate_pair_group_subkey_ranking(
    ranking: PairGroupSubkeyRanking,
    true_subkey: int,
    *,
    top_k_values: Sequence[int] = (1, 16, 32, 64),
) -> dict[str, Any]:
    """Return post-hoc rank and top-k membership for one known validation key."""

    if isinstance(true_subkey, bool) or not isinstance(true_subkey, int):
        raise TypeError("true_subkey must be an integer")
    if not 0 <= true_subkey <= WORD_MASK:
        raise ValueError("true_subkey must be a 16-bit word")
    top_k = tuple(top_k_values)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in top_k):
        raise ValueError("top_k_values must contain non-negative integers")
    in_domain = true_subkey in ranking.candidate_values
    true_rank = ranking.rank(true_subkey) if in_domain else None
    return {
        "true_subkey_in_candidate_set": in_domain,
        "true_rank": true_rank,
        "true_rank_interval": (
            list(ranking.rank_interval(true_subkey)) if in_domain else None
        ),
        "top_k_membership": {
            str(k): bool(in_domain and true_rank is not None and true_rank <= k)
            for k in top_k
        },
    }


__all__ = [
    "PAIR_GROUP_SCORE_AGGREGATION",
    "PairGroupSubkeyRanking",
    "evaluate_pair_group_subkey_ranking",
    "rank_pair_group_subkeys",
]
