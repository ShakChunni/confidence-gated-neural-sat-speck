"""Unit tests for validated paired statistical procedures."""

from __future__ import annotations

import math

import pytest

from neural_sat_speck.analysis.paired_stats import (
    compute_2x2_contingency,
    compute_exact_mcnemar_power,
    exact_binomial_twoside_pvalue,
    exact_mcnemar_test,
    holm_bonferroni_adjustment,
    newcombe_method10_paired_ci,
)


class TestContingency2x2:
    def test_compute_2x2_contingency_basic(self) -> None:
        a = [True, True, False, False, True]
        b = [True, False, True, False, True]
        c = compute_2x2_contingency(a, b)
        assert c.n11 == 2  # (T, T): indices 0, 4
        assert c.n10 == 1  # (T, F): index 1
        assert c.n01 == 1  # (F, T): index 2
        assert c.n00 == 1  # (F, F): index 3
        assert c.total == 5
        d = c.to_dict()
        assert d["n_discordant"] == 2
        assert d["net_difference"] == 0

    def test_compute_2x2_contingency_mismatch(self) -> None:
        with pytest.raises(ValueError, match="Length mismatch"):
            compute_2x2_contingency([True], [True, False])


class TestExactMcNemar:
    def test_exact_binomial_twoside_pvalue(self) -> None:
        assert exact_binomial_twoside_pvalue(0, 0) == 1.0
        # 1 out of 10 heads
        p = exact_binomial_twoside_pvalue(1, 10)
        expected = 2.0 * (math.comb(10, 0) + math.comb(10, 1)) * (0.5**10)
        assert math.isclose(p, expected, rel_tol=1e-6)

    def test_exact_mcnemar_literature_values(self) -> None:
        # Phase 7 B1 vs B0: b=12, c=4
        res = exact_mcnemar_test(12, 4)
        assert res["n_discordant"] == 16
        assert res["net_difference"] == 8
        assert math.isclose(res["p_value"], 0.0768066, rel_tol=1e-4)
        assert not res["significant_at_05"]

        # Symmetry: b=4, c=12 gives same two-sided p-value
        res_sym = exact_mcnemar_test(4, 12)
        assert math.isclose(res_sym["p_value"], 0.0768066, rel_tol=1e-4)
        assert res_sym["net_difference"] == -8

        # 5 vs 0
        res_5_0 = exact_mcnemar_test(5, 0)
        assert math.isclose(res_5_0["p_value"], 0.0625, rel_tol=1e-4)

        # 46 vs 0
        res_46_0 = exact_mcnemar_test(46, 0)
        assert res_46_0["significant_at_05"]
        assert res_46_0["p_value"] < 1e-10

        # Zero discordance
        res_zero = exact_mcnemar_test(0, 0)
        assert res_zero["p_value"] == 1.0
        assert not res_zero["significant_at_05"]

    def test_exact_mcnemar_errors(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            exact_mcnemar_test(-1, 5)


class TestNewcombeMethod10PairedCI:
    def test_altman_table_6_2_literature_fixture(self) -> None:
        """Verify against Altman et al. (2000) Table 6.2 worked example under Method 10.

        Table: n11=14, n10=5, n01=0, n00=22, total=41.
        Difference: 5 / 41 = 0.121951...
        Published Method 10 95% CI: [0.01148324, 0.22649268], rounding [0.011, 0.226].
        """
        d, lower, upper = newcombe_method10_paired_ci(14, 5, 0, 22, 41)
        assert math.isclose(d, 5.0 / 41.0, rel_tol=1e-6)
        assert math.isclose(lower, 0.01148324, rel_tol=1e-4)
        assert math.isclose(upper, 0.22649268, rel_tol=1e-4)
        assert round(lower, 3) == 0.011
        assert round(upper, 3) == 0.226

    def test_altman_table_6_2_reversed_symmetry(self) -> None:
        """Verify exact order-invariance on Altman Table 6.2."""
        d, lower, upper = newcombe_method10_paired_ci(14, 0, 5, 22, 41)
        assert math.isclose(d, -5.0 / 41.0, rel_tol=1e-6)
        assert math.isclose(lower, -0.22649268, rel_tol=1e-4)
        assert math.isclose(upper, -0.01148324, rel_tol=1e-4)
        assert round(lower, 3) == -0.226
        assert round(upper, 3) == -0.011

    def test_newcombe_table_3_literature_fixture(self) -> None:
        """Verify against Newcombe (1998) Table III worked example.

        Table: n11=20, n10=12, n01=2, n00=16, total=50.
        Difference: (12 - 2) / 50 = 0.200000.
        Published Method 10 95% CI: [0.0562, 0.3292].
        """
        d, lower, upper = newcombe_method10_paired_ci(20, 12, 2, 16, 50)
        assert math.isclose(d, 0.200000, rel_tol=1e-6)
        assert math.isclose(lower, 0.05615642, rel_tol=1e-4)
        assert math.isclose(upper, 0.32920700, rel_tol=1e-4)
        assert round(lower, 4) == 0.0562
        assert round(upper, 4) == 0.3292

    def test_zero_discordance_ties(self) -> None:
        """Verify handling of zero discordant pairs with mixed concordant (16, 0, 0, 16)."""
        d, lower, upper = newcombe_method10_paired_ci(16, 0, 0, 16, 32)
        assert d == 0.0
        assert math.isclose(lower, -0.05787357, rel_tol=1e-4)
        assert math.isclose(upper, 0.05787357, rel_tol=1e-4)
        assert math.isclose(lower, -upper, abs_tol=1e-9)

    def test_different_concordant_splits(self) -> None:
        """Verify zero discordance across asymmetric concordant splits."""
        # Split 24/8
        d1, l1, u1 = newcombe_method10_paired_ci(24, 0, 0, 8, 32)
        assert d1 == 0.0
        assert math.isclose(l1, -u1, abs_tol=1e-9)
        assert math.isclose(u1, 0.07887455, rel_tol=1e-4)

        # Split 8/24 (reversal of marginal success/failure counts)
        d2, l2, u2 = newcombe_method10_paired_ci(8, 0, 0, 24, 32)
        assert d2 == 0.0
        assert math.isclose(u1, u2, abs_tol=1e-9)
        assert math.isclose(l1, l2, abs_tol=1e-9)

    def test_negative_association(self) -> None:
        """Verify handling of strong negative association between methods."""
        # Discordant only: methods never agree
        d, lower, upper = newcombe_method10_paired_ci(0, 10, 10, 0, 20)
        assert d == 0.0
        assert math.isclose(lower, -upper, abs_tol=1e-9)
        assert lower < -0.2
        assert upper > 0.2

    def test_boundary_proportions_full_success(self) -> None:
        """Boundary case: Method A has 100% success (p1=1.0)."""
        d, lower, upper = newcombe_method10_paired_ci(30, 10, 0, 0, 40)
        assert math.isclose(d, 0.25, rel_tol=1e-6)
        assert 0.110 <= lower <= 0.112
        assert 0.401 <= upper <= 0.403

    def test_boundary_proportions_both_zero(self) -> None:
        """Boundary case: both methods have 0% success (p1=0, p2=0)."""
        d, lower, upper = newcombe_method10_paired_ci(0, 0, 0, 32, 32)
        assert d == 0.0
        assert math.isclose(lower, -upper, abs_tol=1e-9)
        assert 0.1071 <= upper <= 0.1073

    def test_boundary_proportions_both_one(self) -> None:
        """Boundary case: both methods have 100% success (p1=1, p2=1)."""
        d, lower, upper = newcombe_method10_paired_ci(32, 0, 0, 0, 32)
        assert d == 0.0
        assert math.isclose(lower, -upper, abs_tol=1e-9)
        assert 0.1071 <= upper <= 0.1073

    def test_complete_separation(self) -> None:
        """Boundary case: Method A=100%, Method B=0% (n10=32, all others 0)."""
        d, lower, upper = newcombe_method10_paired_ci(0, 32, 0, 0, 32)
        assert d == 1.0
        assert 0.848 <= lower <= 0.849
        assert upper == 1.0

        # Reversed complete separation
        d_rev, lower_rev, upper_rev = newcombe_method10_paired_ci(0, 0, 32, 0, 32)
        assert d_rev == -1.0
        assert lower_rev == -1.0
        assert -0.849 <= upper_rev <= -0.848

    def test_small_sample_sizes_n1(self) -> None:
        """Verify behavior on extreme small samples N=1."""
        d1, l1, u1 = newcombe_method10_paired_ci(0, 1, 0, 0, 1)
        assert d1 == 1.0
        assert -1.0 <= l1 <= 1.0
        assert u1 == 1.0

        d2, l2, u2 = newcombe_method10_paired_ci(0, 0, 1, 0, 1)
        assert d2 == -1.0
        assert l2 == -1.0
        assert -1.0 <= u2 <= 1.0

        d3, l3, u3 = newcombe_method10_paired_ci(1, 0, 0, 0, 1)
        assert d3 == 0.0
        assert math.isclose(l3, -u3, abs_tol=1e-9)

        d4, l4, u4 = newcombe_method10_paired_ci(0, 0, 0, 1, 1)
        assert d4 == 0.0
        assert math.isclose(l4, -u4, abs_tol=1e-9)

    def test_reversal_symmetry_general(self) -> None:
        """Verify d(A, B) = -d(B, A) and CI(A, B) = [-upper(B, A), -lower(B, A)]."""
        cases = [
            (21, 5, 1, 5, 32),
            (14, 5, 0, 22, 41),
            (20, 12, 2, 16, 50),
            (5, 10, 3, 2, 20),
        ]
        for n11, n10, n01, n00, total in cases:
            d_ab, l_ab, u_ab = newcombe_method10_paired_ci(n11, n10, n01, n00, total)
            d_ba, l_ba, u_ba = newcombe_method10_paired_ci(n11, n01, n10, n00, total)
            assert math.isclose(d_ab, -d_ba, abs_tol=1e-9)
            assert math.isclose(l_ab, -u_ba, abs_tol=1e-9)
            assert math.isclose(u_ab, -l_ba, abs_tol=1e-9)

    def test_validation_errors(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            newcombe_method10_paired_ci(-1, 0, 0, 10, 9)

        with pytest.raises(ValueError, match="Total .* must be positive"):
            newcombe_method10_paired_ci(0, 0, 0, 0, 0)

        with pytest.raises(ValueError, match="equal sum of cells"):
            newcombe_method10_paired_ci(5, 5, 0, 0, 20)


class TestHolmBonferroniAdjustment:
    def test_two_primary_contrasts_both_rejected(self) -> None:
        p_vals = [0.01, 0.04]
        res = holm_bonferroni_adjustment(p_vals, alpha=0.05)
        assert len(res) == 2
        # rank 1: p=0.01 -> mult=2 -> adj=0.02 <= 0.05 -> rejected
        assert res[0]["rank"] == 1
        assert math.isclose(res[0]["adjusted_p"], 0.02)
        assert res[0]["rejected"] is True

        # rank 2: p=0.04 -> mult=1 -> adj=max(0.02, 0.04)=0.04 <= 0.05 -> rejected
        assert res[1]["rank"] == 2
        assert math.isclose(res[1]["adjusted_p"], 0.04)
        assert res[1]["rejected"] is True

    def test_two_primary_contrasts_neither_rejected(self) -> None:
        p_vals = [0.03, 0.04]
        res = holm_bonferroni_adjustment(p_vals, alpha=0.05)
        # rank 1: p=0.03 -> mult=2 -> adj=0.06 > 0.05 -> not rejected
        assert res[0]["rank"] == 1
        assert math.isclose(res[0]["adjusted_p"], 0.06)
        assert res[0]["rejected"] is False

        # rank 2: p=0.04 -> adj=max(0.06, 0.04)=0.06 > 0.05 -> not rejected
        assert res[1]["rank"] == 2
        assert math.isclose(res[1]["adjusted_p"], 0.06)
        assert res[1]["rejected"] is False

    def test_two_primary_contrasts_one_rejected(self) -> None:
        p_vals = [0.01, 0.06]
        res = holm_bonferroni_adjustment(p_vals, alpha=0.05)
        assert res[0]["rejected"] is True
        assert math.isclose(res[0]["adjusted_p"], 0.02)
        assert res[1]["rejected"] is False
        assert math.isclose(res[1]["adjusted_p"], 0.06)

    def test_order_invariance(self) -> None:
        res1 = holm_bonferroni_adjustment([0.01, 0.06], alpha=0.05)
        res2 = holm_bonferroni_adjustment([0.06, 0.01], alpha=0.05)
        assert res1[0]["adjusted_p"] == res2[1]["adjusted_p"]
        assert res1[1]["adjusted_p"] == res2[0]["adjusted_p"]

    def test_boundary_cases(self) -> None:
        assert holm_bonferroni_adjustment([]) == []

        res_single = holm_bonferroni_adjustment([0.03], alpha=0.05)
        assert res_single[0]["adjusted_p"] == 0.03
        assert res_single[0]["rejected"] is True

        res_ones = holm_bonferroni_adjustment([1.0, 1.0], alpha=0.05)
        assert res_ones[0]["adjusted_p"] == 1.0
        assert res_ones[1]["adjusted_p"] == 1.0
        assert not res_ones[0]["rejected"]

        with pytest.raises(ValueError, match="P-values must be finite"):
            holm_bonferroni_adjustment([-0.1, 0.5])


class TestMcNemarPowerSensitivity:
    def test_protocol_power_fixtures(self) -> None:
        """Independently validate power figures in protocol Table 4."""
        # N=32, p10=0.50, p01=0.00 at alpha=0.025: power 99.973%
        p1 = compute_exact_mcnemar_power(32, 0.50, 0.00, alpha=0.025)
        assert math.isclose(p1, 0.99973, rel_tol=1e-4)

        # N=32, p10=0.20, p01=0.05 at alpha=0.025: power 17.336%
        p2 = compute_exact_mcnemar_power(32, 0.20, 0.05, alpha=0.025)
        assert math.isclose(p2, 0.17336, rel_tol=1e-4)

        # N=64, p10=0.20, p01=0.05 at alpha=0.025: power 50.463%
        p3 = compute_exact_mcnemar_power(64, 0.20, 0.05, alpha=0.025)
        assert math.isclose(p3, 0.50463, rel_tol=1e-4)

    def test_power_edge_cases(self) -> None:
        assert compute_exact_mcnemar_power(32, 0.0, 0.0) == 0.0

        with pytest.raises(ValueError, match="p10 \\+ p01 must be <= 1.0"):
            compute_exact_mcnemar_power(32, 0.6, 0.6)
