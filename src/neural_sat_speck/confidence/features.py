"""Interpretable post-hoc features for complete-domain rankings."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol


class _RankingView(Protocol):
    candidate_values: tuple[int, ...]
    candidate_scores: tuple[float, ...]
    candidate_order: tuple[int, ...]


def _validate_top_k(top_k: int, candidate_count: int) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be an integer")
    if not 2 <= top_k <= candidate_count:
        raise ValueError("top_k must be between 2 and the candidate count")
    return top_k


def _ordered_scores(ranking: _RankingView) -> tuple[float, ...]:
    values = tuple(ranking.candidate_values)
    scores = tuple(ranking.candidate_scores)
    order = tuple(ranking.candidate_order)
    if not values or len(values) != len(scores) or len(values) != len(order):
        raise ValueError("ranking values, scores, and order must have equal length")
    if any(
        isinstance(candidate, bool) or not isinstance(candidate, int)
        for candidate in (*values, *order)
    ):
        raise ValueError("ranking candidates must be integers")
    if len(set(values)) != len(values) or set(values) != set(order):
        raise ValueError("ranking candidates must form one complete permutation")
    score_by_candidate = dict(zip(values, scores, strict=True))
    ordered = tuple(score_by_candidate[candidate] for candidate in order)
    if any(
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
        for score in ordered
    ):
        raise ValueError("ranking scores must be finite numbers")
    return tuple(float(score) for score in ordered)


@dataclass(frozen=True)
class RankingConfidenceFeatures:
    """Confidence features computed without using the true subkey.

    ``top_k_candidates`` is retained only to make the later target calculation
    explicit. The feature values themselves depend solely on the candidate
    score vector and deterministic order.
    """

    candidate_count: int
    top_k: int
    top1_candidate: int
    top2_candidate: int
    top_k_candidates: tuple[int, ...]
    top1_score: float
    top2_score: float
    top_k_score: float
    top1_top2_margin: float
    top1_top_k_margin: float
    top1_softmax_concentration: float
    normalized_softmax_entropy: float
    score_mean: float
    score_std: float
    pair_subset_top_k_jaccard: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the feature row for a confidence experiment."""

        return {
            "candidate_count": self.candidate_count,
            "top_k": self.top_k,
            "top1_candidate": self.top1_candidate,
            "top2_candidate": self.top2_candidate,
            "top_k_candidates": list(self.top_k_candidates),
            "top1_score": self.top1_score,
            "top2_score": self.top2_score,
            "top_k_score": self.top_k_score,
            "top1_top2_margin": self.top1_top2_margin,
            "top1_top_k_margin": self.top1_top_k_margin,
            "top1_softmax_concentration": self.top1_softmax_concentration,
            "normalized_softmax_entropy": self.normalized_softmax_entropy,
            "score_mean": self.score_mean,
            "score_std": self.score_std,
            "pair_subset_top_k_jaccard": self.pair_subset_top_k_jaccard,
        }


def _optional_stability_jaccard(ranking: object) -> float | None:
    """Read and validate the optional pair-subset stability statistic."""

    stability = getattr(ranking, "pair_subset_stability", None)
    if stability is not None:
        value = getattr(stability, "full_top_k_jaccard", None)
    else:
        value = getattr(ranking, "pair_subset_top_k_jaccard", None)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("pair-subset stability must be numeric")
    value_float = float(value)
    if not math.isfinite(value_float) or not 0.0 <= value_float <= 1.0:
        raise ValueError("pair-subset stability must be finite and in [0, 1]")
    return value_float


def extract_ranking_confidence(
    ranking: _RankingView,
    *,
    top_k: int,
) -> RankingConfidenceFeatures:
    """Extract margins and stable score-distribution features.

    The softmax is computed with a max subtraction for numerical stability. It
    is a diagnostic distribution over candidate scores, not a claim that the
    classifier's raw logits are calibrated probabilities.
    """

    order = tuple(ranking.candidate_order)
    scores = _ordered_scores(ranking)
    candidate_count = len(order)
    _validate_top_k(top_k, candidate_count)
    stability_jaccard = _optional_stability_jaccard(ranking)

    max_score = scores[0]
    weights = tuple(math.exp(score - max_score) for score in scores)
    normalizer = math.fsum(weights)
    if not math.isfinite(normalizer) or normalizer <= 0.0:
        raise ValueError("ranking scores do not produce a finite distribution")
    probabilities = tuple(weight / normalizer for weight in weights)
    entropy = -math.fsum(
        probability * math.log(probability)
        for probability in probabilities
        if probability > 0.0
    )
    normalized_entropy = entropy / math.log(candidate_count)
    normalized_entropy = min(1.0, max(0.0, normalized_entropy))
    mean = math.fsum(scores) / candidate_count
    variance = math.fsum((score - mean) ** 2 for score in scores) / candidate_count
    return RankingConfidenceFeatures(
        candidate_count=candidate_count,
        top_k=top_k,
        top1_candidate=order[0],
        top2_candidate=order[1],
        top_k_candidates=order[:top_k],
        top1_score=scores[0],
        top2_score=scores[1],
        top_k_score=scores[top_k - 1],
        top1_top2_margin=scores[0] - scores[1],
        top1_top_k_margin=scores[0] - scores[top_k - 1],
        top1_softmax_concentration=probabilities[0],
        normalized_softmax_entropy=normalized_entropy,
        score_mean=mean,
        score_std=math.sqrt(max(0.0, variance)),
        pair_subset_top_k_jaccard=stability_jaccard,
    )


def _serialized_ranking_view(document: Mapping[str, Any]) -> _RankingView:
    raw_ranking = document.get("ranking", document)
    if not isinstance(raw_ranking, Mapping):
        raise ValueError("serialized ranking must contain a ranking object")
    raw_values = raw_ranking.get("candidate_values")
    raw_scores = raw_ranking.get("candidate_scores")
    raw_order = raw_ranking.get("candidate_order")
    if not all(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes))
        for value in (raw_values, raw_scores, raw_order)
    ):
        raise ValueError("serialized ranking must contain candidate arrays")
    values = tuple(raw_values)
    scores = tuple(raw_scores)
    order = tuple(raw_order)
    if any(
        isinstance(candidate, bool) or not isinstance(candidate, int)
        for candidate in (*values, *order)
    ):
        raise ValueError("serialized ranking candidates must be integers")
    raw_stability = raw_ranking.get("pair_subset_stability")
    stability_jaccard: float | None = None
    if raw_stability is not None:
        if not isinstance(raw_stability, Mapping):
            raise ValueError("serialized pair-subset stability must be an object")
        stability_jaccard = raw_stability.get("full_top_k_jaccard")

    @dataclass(frozen=True)
    class SerializedRanking:
        candidate_values: tuple[int, ...]
        candidate_scores: tuple[float, ...]
        candidate_order: tuple[int, ...]
        pair_subset_top_k_jaccard: float | None

    return SerializedRanking(
        candidate_values=values,
        candidate_scores=scores,
        candidate_order=order,
        pair_subset_top_k_jaccard=stability_jaccard,
    )


def extract_serialized_ranking_confidence(
    document: Mapping[str, Any],
    *,
    top_k: int,
) -> RankingConfidenceFeatures:
    """Extract features directly from a Phase 4 result JSON object."""

    return extract_ranking_confidence(
        _serialized_ranking_view(document),
        top_k=top_k,
    )


def candidate_is_in_top_k(
    features: RankingConfidenceFeatures,
    candidate: int,
) -> bool:
    """Return the post-hoc top-k target for one candidate value."""

    if isinstance(candidate, bool) or not isinstance(candidate, int):
        raise TypeError("candidate must be an integer")
    return candidate in features.top_k_candidates
