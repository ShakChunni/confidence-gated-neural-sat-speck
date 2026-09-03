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
from typing import Final, Protocol

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
    """Seven-dimensional attack-level reliability feature vector."""

    normalized_logit_margin: float
    top16_entropy: float
    score_curvature: float
    subset_jaccard_stability: float
    subset_spearman_rho: float
    multifold_rank_dispersion: float
    cross_model_agreement: float

    def to_array(self) -> np.ndarray:
        return np.array(
            [
                self.normalized_logit_margin,
                self.top16_entropy,
                self.score_curvature,
                self.subset_jaccard_stability,
                self.subset_spearman_rho,
                self.multifold_rank_dispersion,
                self.cross_model_agreement,
            ],
            dtype=np.float64,
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
