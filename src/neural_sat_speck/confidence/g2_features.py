"""G2 Shift-Invariant Distributional and Partitioned Stability+ Feature Extraction.

Strictly follows the author-approved G2 feature schema:
- 7 Distributional shift-invariant features (D1-D7) from full 64-pair score vector.
- 7 Stability+ features (S1-S7) from deterministic random partition views (seed 200030).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

# Deterministic subset-mask RNG seed (CORRECTION 6)
SUBSET_MASK_SEED: Final[int] = 200030
FEATURE_EPSILON: Final[float] = 1e-12

D_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "D1_top1_robust_margin",
    "D2_top1_normalized_margin",
    "D3_top1_top2_gap",
    "D4_top1_top16_normalized_gap",
    "D5_top1_top32_normalized_gap",
    "D6_top32_linear_slope",
    "D7_near_peak_density",
)

S_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "S1_top1_survival_rate",
    "S2_top1_top8_retention",
    "S3_top1_median_rank",
    "S4_top1_rank_iqr",
    "S5_top1_top2_sign_flip_freq",
    "S6_pairwise_top16_jaccard_16p",
    "S7_min_top16_jaccard_subview_full",
)

G2_FEATURE_NAMES: Final[tuple[str, ...]] = D_FEATURE_NAMES + S_FEATURE_NAMES


@dataclass(frozen=True)
class DeterministicPartitionMasks:
    """Predeclared balanced partition of 64 pair indices generated with seed 200030."""

    s16_0: tuple[int, ...]
    s16_1: tuple[int, ...]
    s16_2: tuple[int, ...]
    s16_3: tuple[int, ...]
    s32_0: tuple[int, ...]
    s32_1: tuple[int, ...]
    s64: tuple[int, ...]

    @classmethod
    def generate(cls, seed: int = SUBSET_MASK_SEED) -> DeterministicPartitionMasks:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(64).tolist()
        return cls(
            s16_0=tuple(perm[0:16]),
            s16_1=tuple(perm[16:32]),
            s16_2=tuple(perm[32:48]),
            s16_3=tuple(perm[48:64]),
            s32_0=tuple(perm[0:32]),
            s32_1=tuple(perm[32:64]),
            s64=tuple(range(64)),
        )


def extract_distributional_features(
    scores_64: Sequence[float] | np.ndarray,
    eps: float = FEATURE_EPSILON,
) -> dict[str, float]:
    """Extract 7 shift-invariant distributional features from full 64-pair scores.

    Approved Distributional Definitions:
    Let:
      m = median(s)
      s_c = s - m
      norm_c = ||s_c||_2 + eps
      robust_scale = 1.4826 * MAD(s) + eps
      Sorted raw logits descending: s_{(1)} >= s_{(2)} >= ... >= s_{(65536)}

    Features:
      D1 = (s_{(1)} - median(s)) / robust_scale
      D2 = (s_{(1)} - median(s)) / norm_c
      D3 = s_{(1)} - s_{(2)}
      D4 = (s_{(1)} - s_{(16)}) / norm_c
      D5 = (s_{(1)} - s_{(32)}) / norm_c
      D6 = linear slope of sorted raw logits at ranks 1..32
      D7 = fraction of all 65536 candidates with s_{(1)} - s <= 2 * robust_scale
    """
    s = np.asarray(scores_64, dtype=np.float64)
    med = float(np.median(s))
    s_c = s - med
    norm_c = float(np.linalg.norm(s_c, ord=2)) + eps
    mad_full = float(np.median(np.abs(s_c)))
    robust_scale = 1.4826 * mad_full + eps

    s_sorted = np.sort(s)[::-1]

    d1 = float((s_sorted[0] - med) / robust_scale)
    d2 = float((s_sorted[0] - med) / norm_c)
    d3 = float(s_sorted[0] - s_sorted[1])
    d4 = float((s_sorted[0] - s_sorted[15]) / norm_c)
    d5 = float((s_sorted[0] - s_sorted[31]) / norm_c)

    # D6: linear slope of sorted raw logits at ranks 1..32
    # y = s_{(1..32)}, x = 1..32
    # linear polyfit slope: Cov(x, y) / Var(x)
    top32 = s_sorted[:32]
    x = np.arange(1, 33, dtype=np.float64)
    x_c = x - np.mean(x)
    top32_c = top32 - np.mean(top32)
    d6 = float(np.sum(x_c * top32_c) / np.sum(x_c**2))

    # D7: fraction of all 65536 candidates with s_{(1)} - s <= 2 * robust_scale
    d7 = float(np.mean(s_sorted[0] - s <= 2.0 * robust_scale))

    return {
        "D1_top1_robust_margin": d1,
        "D2_top1_normalized_margin": d2,
        "D3_top1_top2_gap": d3,
        "D4_top1_top16_normalized_gap": d4,
        "D5_top1_top32_normalized_gap": d5,
        "D6_top32_linear_slope": d6,
        "D7_near_peak_density": d7,
    }


def extract_stability_plus_features(
    scores_7_views: np.ndarray,
    eps: float = FEATURE_EPSILON,
) -> dict[str, float]:
    """Extract 7 Stability+ features from the 7 deterministic subset views.

    Views (seed 200030):
      0: V_{16, 0}
      1: V_{16, 1}
      2: V_{16, 2}
      3: V_{16, 3}
      4: V_{32, 0}
      5: V_{32, 1}
      6: V_{64} (full 64-pair data)

    Let c* = argmax_c s_64(c) be top-1 candidate on full 64 pairs.
    Let r_v(c*) be rank (1..65536) of candidate c* under view v in {0..5}.

    Features:
      S1 = full-data top-1 survival rate across 6 subviews: (1/6) * sum_{v=0}^5 I(r_v(c*) == 1)
      S2 = full-data top-1 retention in top 8 across 6 subviews: (1/6) * sum_{v=0}^5 I(r_v(c*) <= 8)
      S3 = median rank of full-data top-1 across 6 subviews: median_{v=0..5}(r_v(c*))
      S4 = IQR of rank of full-data top-1 across 6 subviews: Q75(r_v(c*)) - Q25(r_v(c*))
      S5 = sign-flip frequency between full-data top 1 and top 2: (1/6) * sum_{v=0}^5 I(s_v(c2*) > s_v(c*))
      S6 = pairwise top-16 Jaccard similarity across the 4 16-pair subsets: mean over all 6 pairs
      S7 = minimum top-16 agreement between any subview and full data: min_{v=0..5} Jaccard(Top16(V_v), Top16(V_64))
    """
    arr = np.asarray(scores_7_views, dtype=np.float64)
    assert arr.shape == (7, 65536), f"Expected shape (7, 65536), got {arr.shape}"

    # Full data view: arr[6]
    s_full = arr[6]
    # Identify top 2 candidates on full 64 pairs
    top2_full_idx = np.argpartition(s_full, -2)[-2:]
    if s_full[top2_full_idx[1]] > s_full[top2_full_idx[0]]:
        c_star = int(top2_full_idx[1])
        c2_star = int(top2_full_idx[0])
    else:
        c_star = int(top2_full_idx[0])
        c2_star = int(top2_full_idx[1])

    # Ranks of c* under each subview v in 0..5 (1..65536)
    ranks_c_star = []
    for v in range(6):
        s_v = arr[v]
        r_v = int(np.sum(s_v > s_v[c_star])) + 1
        ranks_c_star.append(r_v)
    ranks_arr = np.array(ranks_c_star, dtype=np.float64)

    # S1: full-data top-1 survival rate across 6 subviews
    s1 = float(np.mean(ranks_arr == 1.0))

    # S2: full-data top-1 retention in top 8 across 6 subviews
    s2 = float(np.mean(ranks_arr <= 8.0))

    # S3: median rank of full-data top-1 across 6 subviews
    s3 = float(np.median(ranks_arr))

    # S4: IQR of rank of full-data top-1 across 6 subviews
    q75, q25 = np.percentile(ranks_arr, [75, 25])
    s4 = float(q75 - q25)

    # S5: sign-flip frequency between full-data top 1 and top 2 across 6 subviews
    s5 = float(np.mean([arr[v, c2_star] > arr[v, c_star] for v in range(6)]))

    # Top-16 sets for all 7 views
    top16_sets = []
    for v in range(7):
        top16_indices = np.argpartition(arr[v], -16)[-16:]
        top16_sets.append(set(top16_indices.tolist()))

    # S6: pairwise top-16 Jaccard similarity across the 4 16-pair subsets (views 0..3)
    jaccs_16 = []
    for i in range(4):
        for j in range(i + 1, 4):
            inter = len(top16_sets[i] & top16_sets[j])
            union = len(top16_sets[i] | top16_sets[j])
            jaccs_16.append(inter / union if union > 0 else 0.0)
    s6 = float(np.mean(jaccs_16))

    # S7: minimum top-16 agreement between any subview (0..5) and full data (6)
    jaccs_full = []
    for v in range(6):
        inter = len(top16_sets[v] & top16_sets[6])
        union = len(top16_sets[v] | top16_sets[6])
        jaccs_full.append(inter / union if union > 0 else 0.0)
    s7 = float(min(jaccs_full))

    return {
        "S1_top1_survival_rate": s1,
        "S2_top1_top8_retention": s2,
        "S3_top1_median_rank": s3,
        "S4_top1_rank_iqr": s4,
        "S5_top1_top2_sign_flip_freq": s5,
        "S6_pairwise_top16_jaccard_16p": s6,
        "S7_min_top16_jaccard_subview_full": s7,
    }


def extract_g2_feature_vector(
    scores_7_views: np.ndarray,
    eps: float = FEATURE_EPSILON,
) -> dict[str, float]:
    """Extract complete 14-dimensional G2 feature dictionary."""
    d_feats = extract_distributional_features(scores_7_views[6], eps=eps)
    s_feats = extract_stability_plus_features(scores_7_views, eps=eps)
    res = {}
    res.update(d_feats)
    res.update(s_feats)
    return res
