"""Attack-level ranking stability and post-hoc reliability estimators (G0, Glogit, G1).

This module implements the reliability-gated infrastructure designed in
docs/FALLBACK_CONFIDENCE_REDESIGN.md, moving beyond single scalar score margins
to measure pair-subset stability, normalized logit dispersion (Cattelan & Silva 2024),
and cross-fold rank agreement.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol

import numpy as np

EPSILON: Final[float] = 1e-12


def normalized_logit_margin(
    scores: Sequence[float] | np.ndarray,
    p: float = 2.0,
    top_k: int = 16,
) -> float:
    """Compute Cattelan & Silva (UAI 2024) p-norm normalized logit margin.

    kappa_p(x) = (s_(1) - s_(top_k)) / ||s||_p

    Parameters
    ----------
    scores : Sequence[float] | np.ndarray
        Raw candidate scores or logits over the candidate domain.
    p : float
        Norm order (typically 2.0 for L2 norm).
    top_k : int
        Candidate rank boundary (default: 16).

    Returns
    -------
    float
        Normalized margin in [0.0, inf). Returns 0.0 if scores are degenerate or empty.
    """
    arr = np.asarray(scores, dtype=np.float64)
    if len(arr) < 2:
        return 0.0

    sorted_scores = np.sort(arr)[::-1]
    k_idx = min(top_k - 1, len(sorted_scores) - 1)
    margin = sorted_scores[0] - sorted_scores[k_idx]

    norm = float(np.linalg.norm(arr, ord=p))
    if norm < EPSILON:
        return 0.0

    return float(max(0.0, margin / norm))


def topk_entropy(
    scores: Sequence[float] | np.ndarray,
    top_k: int = 16,
) -> float:
    """Compute Shannon entropy of the softmax distribution over the top-k candidates.

    Lower entropy indicates peakier, more confident candidate separation.
    """
    arr = np.asarray(scores, dtype=np.float64)
    if len(arr) == 0:
        return 0.0

    top_scores = np.sort(arr)[::-1][:top_k]
    # Numerically stable softmax
    shifted = top_scores - np.max(top_scores)
    exp_scores = np.exp(shifted)
    probs = exp_scores / np.sum(exp_scores)

    entropy = -np.sum(probs * np.log(probs + EPSILON))
    return float(max(0.0, entropy))


def score_curvature(
    scores: Sequence[float] | np.ndarray,
    top_k: int = 16,
) -> float:
    """Compute peak curvature ratio: (s_(1) - s_(2)) / (s_(2) - s_(k) + eps)."""
    arr = np.asarray(scores, dtype=np.float64)
    if len(arr) < 3:
        return 0.0

    sorted_scores = np.sort(arr)[::-1]
    k_idx = min(top_k - 1, len(sorted_scores) - 1)
    s1, s2, sk = sorted_scores[0], sorted_scores[1], sorted_scores[k_idx]

    numerator = s1 - s2
    denominator = s2 - sk + EPSILON
    return float(max(0.0, numerator / denominator))


def topk_jaccard_stability(
    order_a: Sequence[int],
    order_b: Sequence[int],
    top_k: int = 16,
) -> float:
    """Compute Jaccard similarity between top-k candidates of two rankings."""
    if not order_a or not order_b:
        return 0.0

    set_a = set(order_a[:top_k])
    set_b = set(order_b[:top_k])
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    if union == 0:
        return 0.0
    return float(intersection / union)


def topk_spearman_correlation(
    order_a: Sequence[int],
    order_b: Sequence[int],
    eval_depth: int = 32,
) -> float:
    """Compute Spearman rank correlation across union of top candidates."""
    union_cands = list(dict.fromkeys(list(order_a[:eval_depth]) + list(order_b[:eval_depth])))
    if len(union_cands) < 2:
        return 1.0

    pos_a = {cand: idx for idx, cand in enumerate(order_a)}
    pos_b = {cand: idx for idx, cand in enumerate(order_b)}

    ranks_a = [pos_a.get(c, len(order_a)) for c in union_cands]
    ranks_b = [pos_b.get(c, len(order_b)) for c in union_cands]

    # Spearman rank correlation: 1 - 6 * sum(d^2) / (n*(n^2-1))
    n = len(union_cands)
    d_sq_sum = sum((ra - rb) ** 2 for ra, rb in zip(ranks_a, ranks_b, strict=True))
    rho = 1.0 - (6.0 * d_sq_sum) / (n * (n * n - 1))
    return float(np.clip(rho, -1.0, 1.0))


@dataclass(frozen=True)
class G1FeatureVector:
    """Seven-dimensional attack-level reliability feature vector (Part E frozen set).

    Features
    --------
    1. top1_top2_margin : float
        Difference between 1st and 2nd ranked candidate scores.
    2. top1_top16_margin : float
        Normalized logit margin between 1st and 16th ranked candidates.
    3. normalized_top16_entropy : float
        Normalized Shannon entropy of softmax over top 16 candidates in [0.0, 1.0].
    4. top16_jaccard_16_32 : float
        Jaccard similarity of top-16 candidate sets between 16-pair and 32-pair rankings.
    5. top16_jaccard_32_64 : float
        Jaccard similarity of top-16 candidate sets between 32-pair and 64-pair rankings.
    6. top1_rank_variance : float
        Empirical variance of the 64-pair top-1 candidate's rank across {16, 32, 64}-pair views.
    7. dispersion_normalized_margin : float
        Score margin divided by subset rank dispersion: margin / (std(ranks) + 1.0).
    """

    top1_top2_margin: float
    top1_top16_margin: float
    normalized_top16_entropy: float
    top16_jaccard_16_32: float
    top16_jaccard_32_64: float
    top1_rank_variance: float
    dispersion_normalized_margin: float

    # Backwards-compatibility properties
    @property
    def normalized_logit_margin(self) -> float:
        return self.top1_top16_margin

    @property
    def top16_entropy(self) -> float:
        return self.normalized_top16_entropy

    @property
    def score_curvature(self) -> float:
        return self.top1_top2_margin

    @property
    def subset_jaccard_stability(self) -> float:
        return self.top16_jaccard_32_64

    @property
    def subset_spearman_rho(self) -> float:
        return self.top16_jaccard_16_32

    @property
    def multifold_rank_dispersion(self) -> float:
        return self.top1_rank_variance

    @property
    def cross_model_agreement(self) -> float:
        return self.dispersion_normalized_margin

    def to_array(self) -> np.ndarray:
        return np.array(
            [
                self.top1_top2_margin,
                self.top1_top16_margin,
                self.normalized_top16_entropy,
                self.top16_jaccard_16_32,
                self.top16_jaccard_32_64,
                self.top1_rank_variance,
                self.dispersion_normalized_margin,
            ],
            dtype=np.float64,
        )


def extract_g1_features_from_rankings(
    ranking_64: Any,
    ranking_32: Any,
    ranking_16: Any,
) -> G1FeatureVector:
    """Extract the frozen 7-dimensional G1 reliability feature vector from multi-subset rankings.

    Parameters
    ----------
    ranking_64 : PairGroupSubkeyRanking
        Ranking evaluated on all 64 ciphertext pairs.
    ranking_32 : PairGroupSubkeyRanking
        Ranking evaluated on 32 ciphertext pairs.
    ranking_16 : PairGroupSubkeyRanking
        Ranking evaluated on 16 ciphertext pairs.
    """
    scores_64 = np.asarray(ranking_64.candidate_scores, dtype=np.float64)
    sorted_64 = np.sort(scores_64)[::-1]

    # 1. top1 - top2 margin
    top1_top2 = float(sorted_64[0] - sorted_64[1]) if len(sorted_64) >= 2 else 0.0

    # 2. top1 - top16 margin (normalized by L2 norm)
    k16_idx = min(15, len(sorted_64) - 1)
    norm = float(np.linalg.norm(scores_64, ord=2))
    top1_top16 = float((sorted_64[0] - sorted_64[k16_idx]) / (norm + EPSILON))

    # 3. normalized top-16 entropy
    ent = topk_entropy(scores_64, top_k=16)
    norm_ent = float(ent / math.log(16.0)) if len(sorted_64) >= 16 else 0.0

    # 4. top-16 Jaccard: 16 vs 32 pairs
    jacc_16_32 = topk_jaccard_stability(
        ranking_16.candidate_order,
        ranking_32.candidate_order,
        top_k=16,
    )

    # 5. top-16 Jaccard: 32 vs 64 pairs
    jacc_32_64 = topk_jaccard_stability(
        ranking_32.candidate_order,
        ranking_64.candidate_order,
        top_k=16,
    )

    # 6. top-candidate rank variance across {16, 32, 64} views
    c_top1 = ranking_64.candidate_order[0]
    r64 = 1
    r32 = (
        ranking_32.candidate_order.index(c_top1) + 1
        if c_top1 in ranking_32.candidate_order
        else len(ranking_32.candidate_order)
    )
    r16 = (
        ranking_16.candidate_order.index(c_top1) + 1
        if c_top1 in ranking_16.candidate_order
        else len(ranking_16.candidate_order)
    )
    rank_var = float(np.var([r16, r32, r64]))

    # 7. dispersion-normalized margin: margin / (std(ranks) + 1.0)
    rank_std = float(np.std([r16, r32, r64]))
    disp_norm_margin = float((sorted_64[0] - sorted_64[k16_idx]) / (rank_std + 1.0))

    return G1FeatureVector(
        top1_top2_margin=top1_top2,
        top1_top16_margin=top1_top16,
        normalized_top16_entropy=norm_ent,
        top16_jaccard_16_32=jacc_16_32,
        top16_jaccard_32_64=jacc_32_64,
        top1_rank_variance=rank_var,
        dispersion_normalized_margin=disp_norm_margin,
    )


class ReliabilityEstimator(Protocol):
    """Protocol for ranking reliability estimators."""

    def predict_reliability(self, features: G1FeatureVector | Sequence[float]) -> float:
        """Predict probability that true subkey is in top-16 (score in [0.0, 1.0])."""
        ...


class G0Estimator:
    """Historical baseline: scalar score margin estimator."""

    def __init__(self, intercept: float = 0.5, slope: float = 0.1) -> None:
        self.intercept = intercept
        self.slope = slope

    def predict_reliability(self, features: G1FeatureVector | Sequence[float]) -> float:
        margin = features.score_curvature if isinstance(features, G1FeatureVector) else features[0]
        z = self.intercept + self.slope * margin
        return float(1.0 / (1.0 + math.exp(-z)))


class GLogitEstimator:
    """Cattelan & Silva (UAI 2024) normalized logit margin estimator."""

    def __init__(self, threshold: float = 0.05) -> None:
        self.threshold = threshold

    def predict_reliability(self, features: G1FeatureVector | Sequence[float]) -> float:
        norm_margin = (
            features.normalized_logit_margin
            if isinstance(features, G1FeatureVector)
            else features[0]
        )
        # Monotonic logistic transform of normalized margin
        z = (norm_margin - self.threshold) * 20.0
        return float(1.0 / (1.0 + math.exp(-z)))


class G1Estimator:
    """Comprehensive attack-level reliability estimator fusing stability and dispersion."""

    def __init__(
        self,
        weights: Sequence[float] | None = None,
        intercept: float = -0.5,
    ) -> None:
        # Default heuristic weights prioritizing stability and normalized margin
        self.weights = (
            np.asarray(weights, dtype=np.float64)
            if weights is not None
            else np.array([2.5, -1.0, 1.0, 3.0, 2.0, -1.5, 2.0], dtype=np.float64)
        )
        self.intercept = intercept

    def predict_reliability(self, features: G1FeatureVector | Sequence[float]) -> float:
        vec = features.to_array() if isinstance(features, G1FeatureVector) else np.asarray(features)
        z = self.intercept + float(np.dot(self.weights, vec))
        z_clamped = max(-20.0, min(20.0, z))
        return float(1.0 / (1.0 + math.exp(-z_clamped)))
