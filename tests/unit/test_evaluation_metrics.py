"""Unit tests for evaluation and solver benchmarking metrics."""

from __future__ import annotations

import math

import pytest

from neural_sat_speck.evaluation.metrics import (
    calculate_cactus_curve,
    calculate_global_par2,
    exact_mcnemar_test,
    wilson_score_interval,
)


def test_calculate_global_par2_basic() -> None:
    # 2 solved instances (10s, 20s), 2 unsolved instances (penalized at 2 * 100s = 200s)
    runtimes = [10.0, 20.0, 50.0, 80.0]
    successes = [True, True, False, False]
    cutoff = 100.0

    par2 = calculate_global_par2(runtimes, successes, cutoff)
    expected = (10.0 + 20.0 + 200.0 + 200.0) / 4.0  # 430 / 4 = 107.5
    assert math.isclose(par2, expected, rel_tol=1e-6)


def test_calculate_global_par2_solved_exceeds_cutoff() -> None:
    # Solved, but runtime (120s) exceeds cutoff (100s) -> must be penalized as 2 * 100 = 200s
    runtimes = [120.0]
    successes = [True]
    cutoff = 100.0

    par2 = calculate_global_par2(runtimes, successes, cutoff)
    assert math.isclose(par2, 200.0, rel_tol=1e-6)


def test_calculate_global_par2_solved_exact_cutoff() -> None:
    # Solved exactly at cutoff -> charged cutoff
    runtimes = [100.0]
    successes = [True]
    cutoff = 100.0

    par2 = calculate_global_par2(runtimes, successes, cutoff)
    assert math.isclose(par2, 100.0, rel_tol=1e-6)


def test_calculate_global_par2_empty_and_errors() -> None:
    assert calculate_global_par2([], [], 100.0) == 0.0

    with pytest.raises(ValueError, match="length"):
        calculate_global_par2([10.0], [True, False], 100.0)

    with pytest.raises(ValueError, match="cutoff must be positive"):
        calculate_global_par2([10.0], [True], 0.0)

    with pytest.raises(ValueError, match="cutoff must be positive"):
        calculate_global_par2([10.0], [True], -10.0)

    with pytest.raises(ValueError, match="non-negative and finite"):
        calculate_global_par2([-5.0], [True], 100.0)


def test_wilson_score_interval_exact() -> None:
    # For 58/64 successes at 95% confidence
    lower, upper = wilson_score_interval(58, 64, confidence=0.95)
    assert 0.810 <= lower <= 0.811
    assert 0.956 <= upper <= 0.957

    # For 0/64
    lower0, upper0 = wilson_score_interval(0, 64, confidence=0.95)
    assert lower0 == 0.0
    assert 0.056 <= upper0 <= 0.057

    # For 64/64
    lower64, upper64 = wilson_score_interval(64, 64, confidence=0.95)
    assert 0.943 <= lower64 <= 0.944
    assert upper64 == 1.0


def test_wilson_score_interval_validation() -> None:
    with pytest.raises(ValueError, match="total trials must be positive"):
        wilson_score_interval(0, 0)

    with pytest.raises(ValueError, match="successes .* must be in"):
        wilson_score_interval(10, 5)

    with pytest.raises(ValueError, match="confidence must be in"):
        wilson_score_interval(5, 10, confidence=1.5)


def test_exact_mcnemar_test_basic() -> None:
    # 12 wins vs 4 losses (Phase 7 B1 vs B0)
    res = exact_mcnemar_test(12, 4)
    assert res["b"] == 12
    assert res["c"] == 4
    assert res["n_discordant"] == 16
    assert res["net_difference"] == 8
    # Exact two-sided binomial p-value is 5034 / 65536 ≈ 0.0768066
    assert math.isclose(res["p_value"], 0.0768066, rel_tol=1e-4)
    assert not res["significant_at_05"]

    # 5 wins vs 0 losses (Phase 7 Fallback vs B1)
    res_fb = exact_mcnemar_test(5, 0)
    assert res_fb["n_discordant"] == 5
    assert math.isclose(res_fb["p_value"], 0.0625, rel_tol=1e-4)
    assert not res_fb["significant_at_05"]

    # 46 wins vs 0 losses (Phase 7 B1 vs SAT-only)
    res_sat = exact_mcnemar_test(46, 0)
    assert res_sat["significant_at_05"]
    assert res_sat["p_value"] < 1e-10

    # 0 discordant pairs
    res_zero = exact_mcnemar_test(0, 0)
    assert res_zero["p_value"] == 1.0


def test_calculate_cactus_curve_monotonic() -> None:
    runtimes = [10.0, 50.0, 80.0, 150.0]
    successes = [True, True, True, False]
    curve = calculate_cactus_curve(runtimes, successes, max_time=200.0, num_points=21)

    assert len(curve) >= 21
    assert curve[0]["time_seconds"] == 0.0
    assert curve[0]["solved_fraction"] == 0.0

    # Check monotonicity
    prev_count = 0.0
    for pt in curve:
        assert pt["solved_count"] >= prev_count
        assert 0.0 <= pt["solved_fraction"] <= 1.0
        prev_count = pt["solved_count"]

    # At 200s, 3 out of 4 instances are solved -> 0.75
    assert math.isclose(curve[-1]["solved_fraction"], 0.75, abs_tol=1e-4)
