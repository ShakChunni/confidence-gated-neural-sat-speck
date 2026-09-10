"""Unit tests for Phase 10-G2 feature extraction and models."""

from __future__ import annotations

import numpy as np

from neural_sat_speck.confidence.g2_features import (
    D_FEATURE_NAMES,
    G2_FEATURE_NAMES,
    S_FEATURE_NAMES,
    DeterministicPartitionMasks,
    extract_distributional_features,
    extract_g2_feature_vector,
    extract_stability_plus_features,
)
from neural_sat_speck.confidence.g2_models import (
    G2AUCRanker,
    G2LogisticRegression,
    G2OrdinalClassifier,
)


def test_partition_masks_properties() -> None:
    masks = DeterministicPartitionMasks.generate(seed=200030)
    assert len(masks.s16_0) == 16
    assert len(masks.s16_1) == 16
    assert len(masks.s16_2) == 16
    assert len(masks.s16_3) == 16
    assert len(masks.s32_0) == 32
    assert len(masks.s32_1) == 32
    assert len(masks.s64) == 64

    # Disjointness
    all_16 = set(masks.s16_0) | set(masks.s16_1) | set(masks.s16_2) | set(masks.s16_3)
    assert len(all_16) == 64
    assert set(masks.s32_0) == (set(masks.s16_0) | set(masks.s16_1))
    assert set(masks.s32_1) == (set(masks.s16_2) | set(masks.s16_3))


def test_distributional_features_shift_invariance() -> None:
    rng = np.random.RandomState(42)
    scores = rng.randn(65536)
    c = 137.89

    d_orig = extract_distributional_features(scores)
    d_shift = extract_distributional_features(scores + c)

    for k in D_FEATURE_NAMES:
        assert np.isclose(d_orig[k], d_shift[k], atol=1e-9), f"Feature {k} failed shift invariance"


def test_stability_plus_features() -> None:
    rng = np.random.RandomState(42)
    scores_7 = rng.randn(7, 65536)

    s_feats = extract_stability_plus_features(scores_7)
    assert len(s_feats) == 7
    for k in S_FEATURE_NAMES:
        assert k in s_feats
        val = s_feats[k]
        assert not np.isnan(val) and not np.isinf(val)


def test_g2_feature_vector_extraction() -> None:
    rng = np.random.RandomState(42)
    scores_7 = rng.randn(7, 65536)
    vec = extract_g2_feature_vector(scores_7)
    assert len(vec) == 14
    for k in G2_FEATURE_NAMES:
        assert k in vec


def test_g2_models_train_and_predict() -> None:
    rng = np.random.RandomState(42)
    N, D = 100, 14
    X = rng.randn(N, D)
    y = (X[:, 0] + X[:, 1] > 0.0).astype(np.int64)

    # 1. G2-LR14
    lr = G2LogisticRegression(l2_lambda=1.0)
    lr.fit(X, y)
    p_lr = lr.predict_proba(X)
    assert p_lr.shape == (N,)
    assert np.all((p_lr >= 0.0) & (p_lr <= 1.0))

    # 2. G2-AUC14
    auc_m = G2AUCRanker(l2_lambda=1.0)
    auc_m.fit(X, y)
    p_auc = auc_m.predict_proba(X)
    assert p_auc.shape == (N,)
    assert np.all((p_auc >= 0.0) & (p_auc <= 1.0))

    # 3. G2-Ordinal14
    ranks = rng.randint(1, 100, size=N)
    y_bins = G2OrdinalClassifier.map_ranks_to_bins(ranks)
    ord_m = G2OrdinalClassifier(l2_lambda=1.0)
    ord_m.fit(X, y_bins)
    p_ord = ord_m.predict_proba_fail(X)
    assert p_ord.shape == (N,)
    assert np.all((p_ord >= 0.0) & (p_ord <= 1.0))


def test_distributional_exact_arithmetic() -> None:
    # Construct a simple array of size 65536
    # Let 65500 elements be 0.0, and 36 elements have specific values
    scores = np.zeros(65536, dtype=np.float64)
    # Median is 0.0, MAD is 0.0, norm_c is non-zero
    for i in range(36):
        scores[i] = 100.0 - i  # 100, 99, ..., 65
    d = extract_distributional_features(scores)
    # D1: (100.0 - 0.0) / (1.4826 * 0.0 + 1e-12) ~ 100 / 1e-12
    # D3: s_(1) - s_(2) = 100 - 99 = 1.0
    assert np.isclose(d["D3_top1_top2_gap"], 1.0)
    # D6: linear slope of ranks 1..32 (values 100, 99, 98, ..., 69)
    # Consecutive values step down by 1.0, so slope must be exactly -1.0
    assert np.isclose(d["D6_top32_linear_slope"], -1.0)


def test_stability_plus_exact_arithmetic() -> None:
    # 7 views of 65536 candidates
    scores_7 = np.zeros((7, 65536), dtype=np.float64)
    # Full data view (view 6): candidate 42 is top-1, candidate 10 is top-2
    scores_7[6, 42] = 10.0
    scores_7[6, 10] = 9.0

    # Views 0..5:
    # Let candidate 42 be top-1 in views 0, 1, 2, but rank 2 in view 3, rank 9 in view 4, rank 10 in view 5
    scores_7[0, 42] = 10.0
    scores_7[1, 42] = 10.0
    scores_7[2, 42] = 10.0

    # In view 3: candidate 10 is top-1, candidate 42 is rank 2 (c2* > c*)
    scores_7[3, 10] = 10.0
    scores_7[3, 42] = 9.0

    # In view 4: 8 candidates have score 11.0, candidate 42 has score 9.0 -> rank 9
    for c in range(8):
        scores_7[4, 100 + c] = 11.0
    scores_7[4, 42] = 9.0

    # In view 5: 9 candidates have score 11.0, candidate 42 has score 9.0 -> rank 10
    for c in range(9):
        scores_7[5, 200 + c] = 11.0
    scores_7[5, 42] = 9.0

    # Ranks of candidate 42 across views 0..5:
    # v0: 1, v1: 1, v2: 1, v3: 2, v4: 9, v5: 10
    s_feats = extract_stability_plus_features(scores_7)

    # S1: top1 survival: 3/6 = 0.5
    assert np.isclose(s_feats["S1_top1_survival_rate"], 0.5)
    # S2: top8 retention: ranks <= 8 are v0, v1, v2, v3 (4 out of 6) = 4/6 = 2/3
    assert np.isclose(s_feats["S2_top1_top8_retention"], 4.0 / 6.0)
    # S3: median rank: median([1, 1, 1, 2, 9, 10]) = (1 + 2) / 2 = 1.5
    assert np.isclose(s_feats["S3_top1_median_rank"], 1.5)
    # S5: sign flip (c2* > c*): occurs only in view 3 (1 out of 6) = 1/6
    assert np.isclose(s_feats["S5_top1_top2_sign_flip_freq"], 1.0 / 6.0)

