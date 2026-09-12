"""Tests for sparse empirical classical difference-distribution ranker."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from neural_sat_speck.attacks.classical_ranking import (
    ClassicalDistributionTable,
    SparseEmpiricalDistributionTable,
    compute_empirical_speck_distribution,
    compute_rank_interval,
    rank_classical_subkeys,
)
from neural_sat_speck.attacks.ranking import SubkeyRanking
from neural_sat_speck.data.attack import (
    AttackInstanceConfig,
    generate_attack_instances,
)
from neural_sat_speck.speck.cipher import encrypt_round


def test_round1_analytical_derivation_and_scalar_equivalence() -> None:
    """Verify that Round 1 produces (0x8000, 0x8000) with probability exactly 1.0.

    Analytical derivation:
    Input difference Delta_0 = (0x0040, 0x0000).
    Speck round: x1 = ((ROR(x0, 7) + y0) ^ k).
    ROR(0x0040, 7) = 0x8000.
    Since Delta y0 = 0, the addition input difference is Delta_x = 0x8000 (at MSB).
    Modular addition modulo 2^16 with an MSB difference has carry propagation probability 1.0
    because (x ^ 0x8000) + y = ((x + y) ^ 0x8000) mod 2^16.
    XOR with round key k cancels out for both plaintexts.
    Thus Delta x1 = 0x8000 with probability 1.0.
    Then y1 = ROL(y0, 2) ^ x1. Since Delta y0 = 0, ROL(0, 2) = 0, so Delta y1 = 0x8000.
    Output difference is (0x8000, 0x8000), packed as 0x80008000.
    """
    # 1. Test scalar reference implementation directly
    rng = np.random.default_rng(42)
    for _ in range(100):
        p0_x = int(rng.integers(0, 65536))
        p0_y = int(rng.integers(0, 65536))
        p1_x = p0_x ^ 0x0040
        p1_y = p0_y ^ 0x0000
        k = int(rng.integers(0, 65536))

        c0 = encrypt_round((p0_x, p0_y), k)
        c1 = encrypt_round((p1_x, p1_y), k)

        dx = c0[0] ^ c1[0]
        dy = c0[1] ^ c1[1]
        assert (dx, dy) == (0x8000, 0x8000)

    # 2. Test vectorized distribution generator matches this derivation
    dist = compute_empirical_speck_distribution(
        rounds=1,
        sample_count=20_000,
        min_count=1,
        seed=12345,
    )
    assert dist.rounds == 1
    expected_diff = (0x8000 << 16) | 0x8000
    assert len(dist.weights) == 1
    assert expected_diff in dist.weights
    assert abs(dist.weights[expected_diff] - 32.0) < 1e-6


def test_scalar_vs_vectorized_generator_equivalence() -> None:
    """Verify that vectorized simulation matches scalar cipher encryption step-by-step."""
    sample_count = 50
    rounds = 3
    seed = 999
    rng = np.random.default_rng(seed)

    x0 = rng.integers(0, 65536, size=sample_count, dtype=np.uint16)
    y0 = rng.integers(0, 65536, size=sample_count, dtype=np.uint16)
    x1 = x0 ^ np.uint16(0x0040)
    y1 = y0 ^ np.uint16(0x0000)
    keys = rng.integers(0, 65536, size=(rounds, sample_count), dtype=np.uint16)

    # Vectorized execution
    vx0, vy0, vx1, vy1 = x0.copy(), y0.copy(), x1.copy(), y1.copy()
    for r in range(rounds):
        kr = keys[r]
        vx0 = (((vx0 >> 7) | (vx0 << 9)) + vy0) ^ kr
        vy0 = ((vy0 << 2) | (vy0 >> 14)) ^ vx0
        vx1 = (((vx1 >> 7) | (vx1 << 9)) + vy1) ^ kr
        vy1 = ((vy1 << 2) | (vy1 >> 14)) ^ vx1

    # Scalar execution
    for i in range(sample_count):
        block0 = (int(x0[i]), int(y0[i]))
        block1 = (int(x1[i]), int(y1[i]))
        for r in range(rounds):
            kr = int(keys[r, i])
            block0 = encrypt_round(block0, kr)
            block1 = encrypt_round(block1, kr)

        assert block0 == (int(vx0[i]), int(vy0[i]))
        assert block1 == (int(vx1[i]), int(vy1[i]))


def test_round2_analytical_dyadic_probabilities() -> None:
    """Verify that Round 2 matches the known analytical dyadic probabilities."""
    dist = compute_empirical_speck_distribution(
        rounds=2,
        sample_count=100_000,
        min_count=1,
        seed=54321,
    )
    assert dist.rounds == 2
    expected_diffs_and_probs = {
        (0x8100 << 16) | 0x8102: 0.50,
        (0x8300 << 16) | 0x8302: 0.25,
        (0x8700 << 16) | 0x8702: 0.125,
        (0x8F00 << 16) | 0x8F02: 0.0625,
        (0x9F00 << 16) | 0x9F02: 0.03125,
        (0xBF00 << 16) | 0xBF02: 0.015625,
        (0xFF00 << 16) | 0xFF02: 0.0078125,
        (0x7F00 << 16) | 0x7F02: 0.0078125,
    }

    assert len(dist.weights) == 8
    for diff, expected_p in expected_diffs_and_probs.items():
        assert diff in dist.weights
        expected_weight = 32.0 + math.log2(expected_p)
        assert abs(dist.weights[diff] - expected_weight) < 0.1


def test_msb_symmetry_and_tie_intervals() -> None:
    """Verify MSB symmetry scores, rank intervals, and that ties are properly tracked."""
    config = AttackInstanceConfig(
        instance_count=1,
        rounds=4,
        pair_count=16,
        seed=999999,
        split_role="test",
    )
    dataset = generate_attack_instances(config)
    instance = dataset.instances[0]

    dist = compute_empirical_speck_distribution(
        rounds=3,
        sample_count=50_000,
        min_count=2,
        seed=999999,
    )

    ranking = rank_classical_subkeys(instance, dist)
    assert isinstance(ranking, SubkeyRanking)

    # For any candidate k, score(k) == score(k ^ 0x8000)
    for k in (0x1234, 0xABCD, 0x0001, 0x7FFF):
        score_k = ranking.score_for(k)
        score_k_sym = ranking.score_for(k ^ 0x8000)
        assert abs(score_k - score_k_sym) < 1e-9

        # Rank intervals for k and k ^ 0x8000
        int_k = compute_rank_interval(ranking, k)
        int_sym = compute_rank_interval(ranking, k ^ 0x8000)

        # Since scores are identical, best_rank and worst_rank must match
        assert int_k.best_rank == int_sym.best_rank
        assert int_k.worst_rank == int_sym.worst_rank
        assert int_k.midrank == int_sym.midrank
        assert int_k.tied_count >= 2
        assert int_k.best_rank <= int_k.sorted_rank <= int_k.worst_rank
        assert int_sym.best_rank <= int_sym.sorted_rank <= int_sym.worst_rank


def test_candidate_subset_ranking() -> None:
    """Verify ranking over a declared subkey subset rather than full 65536."""
    config = AttackInstanceConfig(
        instance_count=1,
        rounds=4,
        pair_count=16,
        seed=123,
        split_role="test",
    )
    dataset = generate_attack_instances(config)
    instance = dataset.instances[0]
    true_k = instance.final_round_subkey

    dist = compute_empirical_speck_distribution(
        rounds=3,
        sample_count=20_000,
        min_count=1,
        seed=123,
    )

    subset = [true_k, (true_k ^ 0x8000), 0x0000, 0xFFFF]
    ranking = rank_classical_subkeys(instance, dist, candidate_values=subset)

    assert set(ranking.candidate_values) == set(subset)
    assert len(ranking.candidate_order) == len(subset)
    assert set(ranking.candidate_order) == set(subset)

    interval = compute_rank_interval(ranking, true_k)
    assert interval.best_rank >= 1
    assert interval.worst_rank <= len(subset)


def test_deterministic_replay() -> None:
    """Verify identical distribution weights when re-generated with identical seed."""
    dist1 = compute_empirical_speck_distribution(rounds=2, sample_count=10_000, seed=777)
    dist2 = compute_empirical_speck_distribution(rounds=2, sample_count=10_000, seed=777)
    assert dist1.weights == dist2.weights


def test_distribution_table_validation_and_serialization() -> None:
    """Verify ClassicalDistributionTable validation and serialization."""
    table = ClassicalDistributionTable(
        rounds=3,
        weights={0x80008000: 32.0, 0x81008102: 31.0},
        sample_count=1000,
        min_count=5,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "dist_test.json"
        table.save(path)

        # Refuses overwrite without allow_overwrite
        with pytest.raises(FileExistsError):
            table.save(path, allow_overwrite=False)

        table.save(path, allow_overwrite=True)
        loaded = ClassicalDistributionTable.load(path)
        assert loaded.rounds == table.rounds
        assert loaded.sample_count == table.sample_count
        assert loaded.min_count == table.min_count
        assert loaded.weights == table.weights

    # Alias works
    assert SparseEmpiricalDistributionTable is ClassicalDistributionTable


def test_invalid_table_construction() -> None:
    """Verify defensive validation on invalid table parameters."""
    with pytest.raises(ValueError, match="rounds must be positive"):
        ClassicalDistributionTable(rounds=0, weights={}, sample_count=100, min_count=1)

    with pytest.raises(ValueError, match="sample_count must be positive"):
        ClassicalDistributionTable(rounds=1, weights={}, sample_count=-1, min_count=1)

    with pytest.raises(ValueError, match="min_count must be at least 1"):
        ClassicalDistributionTable(rounds=1, weights={}, sample_count=100, min_count=0)

    with pytest.raises(ValueError, match="must be a finite number"):
        ClassicalDistributionTable(
            rounds=1,
            weights={0x80008000: float("nan")},
            sample_count=100,
            min_count=1,
        )

    with pytest.raises(ValueError, match="invalid 32-bit difference key"):
        ClassicalDistributionTable(
            rounds=1,
            weights={-1: 10.0},
            sample_count=100,
            min_count=1,
        )
