"""Validated paired statistical procedures for cryptographic-SAT verification.

Provides independently verified implementations of:
1. Exact two-sided McNemar test on discordant pairs (b, c) with exact binomial p-values;
2. Holm-Bonferroni step-down multiplicity correction controlling family-wise error rate;
3. Newcombe Method 10 paired risk difference 95% confidence intervals (Newcombe 1998);
4. Unconditional exact McNemar power and sensitivity calculations;
5. Paired 2x2 contingency table computation and structured contrast summaries.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from neural_sat_speck.evaluation.metrics import wilson_score_interval


@dataclass(frozen=True)
class Contingency2x2:
    """2x2 contingency table for paired binary outcomes."""

    n11: int  # Both methods succeed
    n10: int  # Method A succeeds, Method B fails (b)
    n01: int  # Method A fails, Method B succeeds (c)
    n00: int  # Both methods fail
    total: int

    def to_dict(self) -> dict[str, int]:
        return {
            "n11": self.n11,
            "n10": self.n10,
            "n01": self.n01,
            "n00": self.n00,
            "total": self.total,
            "n_discordant": self.n10 + self.n01,
            "net_difference": self.n10 - self.n01,
        }


def compute_2x2_contingency(
    successes_a: Sequence[bool],
    successes_b: Sequence[bool],
) -> Contingency2x2:
    """Compute 2x2 contingency table for paired binary outcomes.

    Parameters
    ----------
    successes_a : Sequence[bool]
        Boolean recovery outcomes for Method A across N instances.
    successes_b : Sequence[bool]
        Boolean recovery outcomes for Method B across the same N instances.

    Returns
    -------
    Contingency2x2
        Counts of joint successes, discordant pairs, joint failures, and total.

    Raises
    -------
    ValueError
        If sequence lengths do not match.
    """
    if len(successes_a) != len(successes_b):
        raise ValueError(
            f"Length mismatch: successes_a ({len(successes_a)}) vs "
            f"successes_b ({len(successes_b)})"
        )

    n11 = 0
    n10 = 0
    n01 = 0
    n00 = 0

    for a, b in zip(successes_a, successes_b, strict=True):
        if a and b:
            n11 += 1
        elif a and not b:
            n10 += 1
        elif not a and b:
            n01 += 1
        else:
            n00 += 1

    return Contingency2x2(
        n11=n11,
        n10=n10,
        n01=n01,
        n00=n00,
        total=len(successes_a),
    )


def exact_binomial_twoside_pvalue(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial p-value for p=0.5 using symmetry.

    Under H0: p = 0.5, the binomial distribution Binomial(n, 0.5) is symmetric
    around n / 2. The exact two-sided p-value is 2 * P(X <= min(k, n - k)).
    """
    if n == 0:
        return 1.0
    tail_k = min(k, n - k)
    lower_tail_prob = sum(math.comb(n, i) for i in range(tail_k + 1)) * (0.5**n)
    return min(1.0, 2.0 * lower_tail_prob)


def exact_mcnemar_test(b: int, c: int) -> dict[str, Any]:
    """Perform exact two-sided McNemar test on discordant pairs (b, c).

    Under H0 (marginal homogeneity / equal recovery rates), conditional on the
    number of discordant pairs n_d = b + c, b follows Binomial(n_d, 0.5).

    Parameters
    ----------
    b : int
        Count of discordant pairs favoring Method A (Method A=1, Method B=0).
    c : int
        Count of discordant pairs favoring Method B (Method A=0, Method B=1).

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - 'b': int
        - 'c': int
        - 'n_discordant': int
        - 'net_difference': int (b - c)
        - 'p_value': float (exact two-sided p-value)
        - 'significant_at_05': bool (p_value <= 0.05)
    """
    if b < 0 or c < 0:
        raise ValueError(f"Discordant counts must be non-negative, got b={b}, c={c}")

    n_d = b + c
    if n_d == 0:
        return {
            "b": b,
            "c": c,
            "n_discordant": 0,
            "net_difference": 0,
            "p_value": 1.0,
            "significant_at_05": False,
        }

    p_value = exact_binomial_twoside_pvalue(min(b, c), n_d, 0.5)
    return {
        "b": b,
        "c": c,
        "n_discordant": n_d,
        "net_difference": b - c,
        "p_value": float(p_value),
        "significant_at_05": p_value <= 0.05,
    }


def holm_bonferroni_adjustment(
    p_values: Sequence[float],
    alpha: float = 0.05,
) -> list[dict[str, Any]]:
    """Apply Holm-Bonferroni step-down procedure to control family-wise error rate.

    Given m hypothesis tests with raw p-values P_1, ..., P_m:
    1. Sort raw p-values: P_(1) <= P_(2) <= ... <= P_(m).
    2. Step-down adjusted p-values:
       P_adj_(k) = max_{j <= k} min(1.0, (m - j + 1) * P_(j))
    3. H_(k) is rejected at family-wise alpha iff P_adj_(k) <= alpha.

    Parameters
    ----------
    p_values : Sequence[float]
        Sequence of raw p-values.
    alpha : float
        Nominal family-wise significance level (default: 0.05).

    Returns
    -------
    list[dict[str, Any]]
        List of adjustment records matching the input order, each containing:
        - 'index': original index (0 to m-1)
        - 'raw_p': float
        - 'adjusted_p': float
        - 'rank': 1-based order rank
        - 'threshold': float (alpha / (m - rank + 1))
        - 'rejected': bool
    """
    m = len(p_values)
    if m == 0:
        return []

    for p in p_values:
        if not math.isfinite(p) or not (0.0 <= p <= 1.0):
            raise ValueError(f"P-values must be finite in [0.0, 1.0], got {p}")

    # Sort indices by raw p-value ascending
    sorted_indices = sorted(range(m), key=lambda i: p_values[i])

    adj_p: list[float] = [0.0] * m
    ranks: list[int] = [0] * m
    thresholds: list[float] = [0.0] * m
    cum_max = 0.0

    for rank_idx, orig_idx in enumerate(sorted_indices):
        rank = rank_idx + 1
        multiplier = m - rank_idx
        raw = p_values[orig_idx]
        val = min(1.0, multiplier * raw)
        cum_max = max(cum_max, val)
        adj_p[orig_idx] = min(1.0, cum_max)
        ranks[orig_idx] = rank
        thresholds[orig_idx] = alpha / float(multiplier)

    results: list[dict[str, Any]] = []
    for i in range(m):
        results.append({
            "index": i,
            "raw_p": float(p_values[i]),
            "adjusted_p": float(adj_p[i]),
            "rank": ranks[i],
            "threshold": float(thresholds[i]),
            "rejected": adj_p[i] <= alpha,
        })
    return results


def newcombe_method10_paired_ci(
    n11: int,
    n10: int,
    n01: int,
    n00: int,
    total: int,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Compute paired risk difference and Wilson-score-based confidence interval.

    Implements Method 10 from Newcombe (1998), "Improved confidence intervals
    for the difference between binomial proportions based on paired data",
    Statistics in Medicine, 17(22): 2635-2650.

    Given a 2x2 contingency table:
    - n11: both succeed
    - n10: Method A succeeds, Method B fails (b)
    - n01: Method A fails, Method B succeeds (c)
    - n00: both fail
    Total must equal n11 + n10 + n01 + n00.

    Risk difference: d = (n10 - n01) / total.

    Parameters
    ----------
    n11, n10, n01, n00 : int
        Non-negative cell counts.
    total : int
        Total number of paired trials.
    confidence : float
        Confidence level in (0, 1), default 0.95.

    Returns
    -------
    tuple[float, float, float]
        (risk_difference, lower_bound, upper_bound) clamped to [-1.0, 1.0].
    """
    if n11 < 0 or n10 < 0 or n01 < 0 or n00 < 0:
        raise ValueError("Contingency table cell counts must be non-negative")
    cell_sum = n11 + n10 + n01 + n00
    if total <= 0 or cell_sum != total:
        raise ValueError(
            f"Total ({total}) must be positive and equal sum of cells ({cell_sum})"
        )

    d = (n10 - n01) / float(total)

    p1 = (n11 + n10) / float(total)
    p2 = (n11 + n01) / float(total)
    l1, u1 = wilson_score_interval(n11 + n10, total, confidence=confidence)
    l2, u2 = wilson_score_interval(n11 + n01, total, confidence=confidence)

    # Newcombe (1998) Method 10 continuity-corrected association:
    # B = e*h - f*g where e=n11, f=n10, g=n01, h=n00.
    # If B > 0, numerator is max(B - total / 2, 0); otherwise B.
    # Zero denominator yields phi = 0.0.
    b_val = n11 * n00 - n10 * n01
    if b_val > 0:
        num = max(float(b_val) - float(total) / 2.0, 0.0)
    else:
        num = float(b_val)

    denom = math.sqrt(
        (n11 + n10) * (total - (n11 + n10)) * (n11 + n01) * (total - (n11 + n01))
    )
    if denom > 0.0:
        phi = num / denom
    else:
        phi = 0.0
    phi = max(-1.0, min(1.0, phi))

    term1 = (p1 - l1) ** 2 - 2.0 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2
    delta1 = math.sqrt(max(0.0, term1))

    term2 = (u1 - p1) ** 2 - 2.0 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2
    delta2 = math.sqrt(max(0.0, term2))

    lower = max(-1.0, d - delta1)
    upper = min(1.0, d + delta2)
    return float(d), float(lower), float(upper)


def compute_exact_mcnemar_power(
    n: int,
    p10: float,
    p01: float,
    alpha: float = 0.05,
) -> float:
    """Compute unconditional exact two-sided McNemar power.

    Enumerates the exact binomial mixture over all possible discordant counts
    D ~ Binomial(n, p10 + p01), and conditional power under
    X | D=d ~ Binomial(d, p10 / (p10 + p01)).

    Parameters
    ----------
    n : int
        Sample size (number of independent attack keys).
    p10 : float
        P(Method A succeeds, Method B fails).
    p01 : float
        P(Method A fails, Method B succeeds).
    alpha : float
        Two-sided nominal significance level (default: 0.05).

    Returns
    -------
    float
        Unconditional probability of rejecting H0: p10 = p01.
    """
    pd = p10 + p01
    if pd <= 0.0:
        return 0.0
    if pd > 1.0:
        raise ValueError(f"p10 + p01 must be <= 1.0, got {pd}")

    pi = p10 / pd
    total_power = 0.0

    for d in range(1, n + 1):
        prob_d = math.comb(n, d) * (pd**d) * ((1.0 - pd) ** (n - d))
        cond_power = 0.0
        for x in range(d + 1):
            p_val = exact_binomial_twoside_pvalue(x, d, 0.5)
            if p_val <= alpha:
                prob_x_given_d = math.comb(d, x) * (pi**x) * ((1.0 - pi) ** (d - x))
                cond_power += prob_x_given_d
        total_power += prob_d * cond_power

    return float(total_power)


@dataclass(frozen=True)
class ContrastResult:
    """Structured summary of a single paired contrast."""

    contrast_id: str
    contrast_name: str
    method_a: str
    method_b: str
    contrast_role: str  # "primary" or "secondary"
    contingency: Contingency2x2
    risk_difference: float
    ci_95: tuple[float, float]
    mcnemar_raw_p: float
    mcnemar_adjusted_p: float | None
    mcnemar_rejected: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "contrast_id": self.contrast_id,
            "contrast_name": self.contrast_name,
            "method_a": self.method_a,
            "method_b": self.method_b,
            "contrast_role": self.contrast_role,
            "contingency": self.contingency.to_dict(),
            "risk_difference": self.risk_difference,
            "ci_95": list(self.ci_95),
            "mcnemar_raw_p": self.mcnemar_raw_p,
            "mcnemar_adjusted_p": self.mcnemar_adjusted_p,
            "mcnemar_rejected": self.mcnemar_rejected,
        }
