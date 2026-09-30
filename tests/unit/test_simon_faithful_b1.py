"""Unit tests for the faithful SIMON32/64 Gohr ResNet B0->B1 architecture."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from neural_sat_speck.simon.cipher import SIMON32_64_KAT_KEY
from neural_sat_speck.simon.faithful_b1_model import (
    SharedSimonPairEncoder,
    SimonGohrResNet,
    SimonPairGroupRanker,
    _BatchNorm1dWithFallback,
    extract_simon_b1_features,
    generate_b1_hybrid_8parent_queue,
    rank_simon_b1_domain,
)
from neural_sat_speck.simon.key_schedule import expand_key
from neural_sat_speck.simon.ranking_features import candidate_value_words
from tests.fixtures.simon_reference_oracle import oracle_simon32_64_encrypt


class SimonB1ToyModel(nn.Module):
    """Archived toy prototype architecture for architectural contrast audit."""

    def __init__(self) -> None:
        super().__init__()
        self.pair_encoder = nn.Linear(64, 64)
        self.head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pair_encoder(x).mean(dim=1))


def _pairs(count: int, seed: int = 260926) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    rng = np.random.default_rng(seed)
    words = rng.integers(0, 1 << 16, size=(count, 2, 2), dtype=np.uint16)
    return tuple(
        (
            (int(words[i, 0, 0]), int(words[i, 0, 1])),
            (int(words[i, 1, 0]), int(words[i, 1, 1])),
        )
        for i in range(count)
    )


def test_simon_gohr_resnet_shapes_and_param_count() -> None:
    """Verify SimonGohrResNet processes 3D and 2D inputs and matches parameter expectations."""
    model = SimonGohrResNet()
    assert 58_000 <= model.parameter_count <= 65_000

    # 3D input: (batch, 4, 16)
    x3d = torch.randn(8, 4, 16)
    out3d = model(x3d)
    assert out3d.shape == (8, 1)

    # 2D input: (batch, 64)
    x2d = torch.randn(8, 64)
    out2d = model(x2d)
    assert out2d.shape == (8, 1)

    # predict_proba in [0, 1]
    probs = model.predict_proba(x3d)
    assert probs.shape == (8, 1)
    assert torch.all((probs >= 0.0) & (probs <= 1.0))


def test_shared_simon_pair_encoder() -> None:
    """Verify SharedSimonPairEncoder extracts dense 64-dim embeddings from (N, 4, 16)."""
    encoder = SharedSimonPairEncoder()
    x = torch.randn(10, 4, 16)
    emb = encoder(x)
    assert emb.shape == (10, 64)


def test_simon_pair_group_ranker_permutation_invariance() -> None:
    """Verify that permuting pairs along dim 1 yields identical candidate scores."""
    torch.manual_seed(42)
    model = SimonPairGroupRanker()
    model.eval()

    batch_size = 4
    num_pairs = 32
    x = torch.randn(batch_size, num_pairs, 4, 16)

    with torch.no_grad():
        out_orig = model(x)
        perm = torch.randperm(num_pairs)
        x_perm = x[:, perm, :, :]
        out_perm = model(x_perm)

    torch.testing.assert_close(out_orig, out_perm, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("num_pairs", [1, 16, 32, 64, 128])
def test_simon_pair_group_ranker_flexible_pair_count(num_pairs: int) -> None:
    """Verify model can evaluate any pair count without weight or architecture changes."""
    model = SimonPairGroupRanker()
    model.eval()
    x = torch.randn(4, num_pairs, 4, 16)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4,)
    assert torch.all(torch.isfinite(out))


def test_b0_to_b1_weight_transfer() -> None:
    """Verify that initialize_from_b0 copies all encoder weights from SimonGohrResNet."""
    torch.manual_seed(12345)
    baseline = SimonGohrResNet()
    ranker = SimonPairGroupRanker()

    # Before transfer, weights should differ
    assert not torch.equal(ranker.encoder.stem_conv.weight, baseline.stem_conv.weight)

    # Perform transfer
    ranker.initialize_from_b0(baseline)

    # After transfer, weights must be bitwise identical
    assert torch.equal(ranker.encoder.stem_conv.weight, baseline.stem_conv.weight)
    assert torch.equal(ranker.encoder.stem_bn.weight, baseline.stem_bn.weight)
    assert torch.equal(ranker.encoder.dense.weight, baseline.dense.weight)
    assert torch.equal(ranker.encoder.dense_bn.weight, baseline.dense_bn.weight)

    for b_enc, b_base in zip(ranker.encoder.blocks, baseline.blocks):
        assert torch.equal(b_enc.conv1.weight, b_base.conv1.weight)
        assert torch.equal(b_enc.conv2.weight, b_base.conv2.weight)


def test_batch_norm_single_sample_fallback() -> None:
    """Verify fallback BatchNorm does not crash on single-sample training batches."""
    bn = _BatchNorm1dWithFallback(64)
    bn.train()
    # 2D input with single row
    x = torch.randn(1, 64)
    out = bn(x)
    assert out.shape == (1, 64)


def test_faithful_feature_extraction() -> None:
    """Verify feature extractor output shape and values."""
    pairs = _pairs(16)
    candidates = [0, 0x1234, 0xFFFF]

    # 4D format (candidates, pairs, 4, 16)
    feats = extract_simon_b1_features(pairs, candidates)
    assert feats.shape == (3, 16, 4, 16)
    assert feats.dtype == torch.float32
    assert torch.all((feats == 0.0) | (feats == 1.0))

    # Flat bits format (candidates, pairs, 64)
    feats_flat = extract_simon_b1_features(pairs, candidates, flat_bits=True)
    assert feats_flat.shape == (3, 16, 64)


def test_true_r10_key_features_equal_independent_nine_round_oracle() -> None:
    plain_a = (0x1234, 0xABCD)
    plain_b = (plain_a[0], plain_a[1] ^ 0x0040)
    pair = tuple(
        oracle_simon32_64_encrypt(p, SIMON32_64_KAT_KEY, 10)
        for p in (plain_a, plain_b)
    )
    true_k9 = expand_key(SIMON32_64_KAT_KEY, 10)[9]
    words = candidate_value_words((pair,), (true_k9, true_k9 ^ 1))
    nine_round = tuple(
        oracle_simon32_64_encrypt(p, SIMON32_64_KAT_KEY, 9)
        for p in (plain_a, plain_b)
    )
    assert tuple(int(x) for x in words[0, 0]) == (*nine_round[0], *nine_round[1])
    assert tuple(int(x) for x in words[1, 0]) == (
        nine_round[0][0], nine_round[0][1] ^ 1,
        nine_round[1][0], nine_round[1][1] ^ 1,
    )


def test_b1_group_mean_is_invariant_to_fourfold_pair_duplication() -> None:
    torch.manual_seed(1701)
    model = SimonPairGroupRanker().eval()
    sixteen_pairs = torch.randint(0, 2, (2, 16, 4, 16), dtype=torch.float32)
    with torch.no_grad():
        score_16 = model(sixteen_pairs)
        score_64 = model(sixteen_pairs.repeat(1, 4, 1, 1))
    torch.testing.assert_close(score_16, score_64, atol=1e-5, rtol=0)


def test_faithful_ranking_pair_count_enforcement() -> None:
    """Verify ranker correctly uses 64 pairs when given 64, never defaulting to 32."""
    model = SimonPairGroupRanker()
    model.eval()

    pairs_64 = _pairs(64)
    res = rank_simon_b1_domain(pairs_64, model, screen_pairs=16, rescore_top=64, batch_size=4096)
    assert res.screen_pairs_used == 16
    assert res.full_pairs_used == 64
    assert len(res.order) == 64


def test_faithful_deterministic_tie_breaking() -> None:
    """Verify ascending candidate word tie breaking when scores are identical."""
    class FlatRanker(SimonPairGroupRanker):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch_size = x.shape[0] if x.ndim >= 3 else 1
            return torch.zeros(batch_size)

    flat_model = FlatRanker()
    pairs = _pairs(16)
    res = rank_simon_b1_domain(
        pairs, flat_model, screen_pairs=16, full_pairs=16, rescore_top=64, batch_size=4096
    )
    assert res.screen_tie_count == 65536
    assert res.full_tie_count == 64
    # First 64 should be strictly ascending 0, 1, 2, ..., 63
    np.testing.assert_array_equal(res.order, np.arange(64, dtype=np.uint16))


def test_faithful_hybrid_8parent_queue() -> None:
    """Verify queue assembly produces exactly 1,024 tuples (k8, k9)."""
    class MockDdtTable:
        rounds = 7

        def log_score_row(self, left: int) -> np.ndarray:
            return np.zeros(1 << 16, dtype=np.float64)

    model = SimonPairGroupRanker()
    pairs = _pairs(16)

    queue, diag = generate_b1_hybrid_8parent_queue(
        pairs,
        model,
        MockDdtTable(),
        screen_pairs=16,
        full_pairs=16,
        parent_count=8,
        children_per_parent=128,
        return_diagnostics=True,
    )
    assert len(queue) == 1024
    assert diag["candidate_tuples_produced"] == 1024
    assert diag["screen_pairs_used"] == 16
    assert diag["full_pairs_used"] == 16
    for child_k8, parent_k9 in queue:
        assert 0 <= child_k8 <= 0xFFFF
        assert 0 <= parent_k9 <= 0xFFFF


def test_audit_contrast_faithful_vs_toy() -> None:
    """Explicitly verify architectural contrast between toy prototype and faithful Gohr model."""
    toy = SimonB1ToyModel()
    faithful = SimonPairGroupRanker()

    # Toy: 0 Conv1d layers, <20k parameters
    toy_conv_count = sum(1 for m in toy.modules() if isinstance(m, nn.Conv1d))
    assert toy_conv_count == 0
    assert toy.pair_encoder.in_features == 64
    assert sum(p.numel() for p in toy.parameters()) < 20_000

    # Faithful: 9 Conv1d layers and the exact SPECK B1 parameter count.
    faithful_conv_count = sum(1 for m in faithful.modules() if isinstance(m, nn.Conv1d))
    assert faithful_conv_count == 9  # 1 stem + 4*2 blocks
    assert faithful.parameter_count == 60_289
