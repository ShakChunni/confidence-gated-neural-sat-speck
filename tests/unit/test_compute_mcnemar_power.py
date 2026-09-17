"""Unit tests for exact McNemar power computation."""

from __future__ import annotations

import math

import pytest

from scripts.compute_mcnemar_power import (
    compute_exact_mcnemar_power,
    exact_binomial_twoside_pvalue,
    generate_sensitivity_grid,
    parse_args,
)


def test_exact_binomial_twoside_pvalue_symmetry() -> None:
    """Exact two-sided binomial p-value must be symmetric for p=0.5."""
    assert exact_binomial_twoside_pvalue(0, 0) == 1.0
    assert exact_binomial_twoside_pvalue(5, 10) == 1.0
    assert exact_binomial_twoside_pvalue(2, 10) == exact_binomial_twoside_pvalue(8, 10)
    assert exact_binomial_twoside_pvalue(0, 10) == exact_binomial_twoside_pvalue(10, 10)
    # p-value for 0 out of 10 is 2 * (1/2)^10 = 2 / 1024 = 0.001953125
    expected_0_10 = 2.0 * (0.5**10)
    assert math.isclose(exact_binomial_twoside_pvalue(0, 10), expected_0_10)


def test_compute_exact_mcnemar_power_null_hypothesis() -> None:
    """Under H0 (p10 == p01), power should not exceed alpha."""
    power = compute_exact_mcnemar_power(n=128, p10=0.05, p01=0.05, alpha=0.05)
    # Because of discreteness, exact size is <= nominal alpha
    assert 0.0 < power <= 0.05

    power_zero = compute_exact_mcnemar_power(n=128, p10=0.0, p01=0.0, alpha=0.05)
    assert power_zero == 0.0


def test_compute_exact_mcnemar_power_monotonicity() -> None:
    """Power must increase monotonically with sample size N for fixed effect size."""
    powers = [
        compute_exact_mcnemar_power(n=n, p10=0.12, p01=0.02, alpha=0.05)
        for n in [64, 96, 128, 160, 256]
    ]
    for p_low, p_high in zip(powers, powers[1:]):
        assert p_low <= p_high

    # At N=128 and (0.12, 0.02), power should be ~85.7%
    assert 0.84 < powers[2] < 0.88


def test_compute_exact_mcnemar_power_invalid_inputs() -> None:
    """Invalid input probabilities must be rejected."""
    with pytest.raises(ValueError, match="p10 \\+ p01 must be <= 1.0"):
        compute_exact_mcnemar_power(n=100, p10=0.6, p01=0.5)


def test_generate_sensitivity_grid() -> None:
    """Grid generator returns expected number of cell results with consistent properties."""
    scenarios = [(0.12, 0.02), (0.14, 0.04)]
    sample_sizes = [64, 128]
    grid = generate_sensitivity_grid(scenarios, sample_sizes, alpha=0.05)
    assert len(grid) == 4
    for cell in grid:
        assert cell.n in sample_sizes
        assert math.isclose(cell.marginal_diff, cell.p10 - cell.p01)
        assert math.isclose(cell.expected_discordant, cell.n * (cell.p10 + cell.p01))
        assert 0.0 <= cell.exact_power <= 1.0


def test_parse_args() -> None:
    """CLI argument parsing defaults and overrides."""
    args = parse_args([])
    assert args.alpha == 0.05
    assert 128 in args.sample_sizes

    custom_args = parse_args(["--alpha", "0.01", "--sample-sizes", "64", "128"])
    assert custom_args.alpha == 0.01
    assert custom_args.sample_sizes == [64, 128]
