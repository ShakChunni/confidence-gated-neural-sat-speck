#!/usr/bin/env python3
"""Compute exact two-sided McNemar power and sensitivity across sample sizes.

Computes exact binomial rejection probabilities for paired binary cryptanalytic
evaluations under specified alternative discordant probabilities (p10, p01)
and sample sizes N in {64, 96, 128, 160, 192, 256}.
"""

from __future__ import annotations

import argparse
import math
from typing import NamedTuple


class CellResult(NamedTuple):
    n: int
    p10: float
    p01: float
    marginal_diff: float
    expected_discordant: float
    exact_power: float
    alpha: float


def exact_binomial_twoside_pvalue(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial p-value for p=0.5 using symmetry."""
    if n == 0:
        return 1.0
    # For p = 0.5, distribution is symmetric around n / 2
    tail_k = min(k, n - k)
    # Sum lower tail: sum_{i=0}^{tail_k} math.comb(n, i) * 0.5^n
    lower_tail_prob = sum(math.comb(n, i) for i in range(tail_k + 1)) * (0.5**n)
    return min(1.0, 2.0 * lower_tail_prob)


def compute_exact_mcnemar_power(
    n: int,
    p10: float,
    p01: float,
    alpha: float = 0.05,
) -> float:
    """Compute unconditional exact two-sided McNemar power.

    Parameters
    ----------
    n : int
        Sample size (number of independent attack keys).
    p10 : float
        P(method 1 succeeds, method 2 fails).
    p01 : float
        P(method 1 fails, method 2 succeeds).
    alpha : float
        Two-sided nominal significance level.

    Returns
    -------
    float
        Unconditional probability of rejecting H0: p10 = p01.
    """
    pd = p10 + p01
    if pd <= 0.0:
        return 0.0
    if pd >= 1.0:
        raise ValueError("p10 + p01 must be <= 1.0")

    pi = p10 / pd
    total_power = 0.0

    # Sum over all possible number of discordant pairs d ~ Binomial(n, pd)
    for d in range(1, n + 1):
        prob_d = math.comb(n, d) * (pd**d) * ((1.0 - pd) ** (n - d))
        # Conditional power given d discordant pairs
        cond_power = 0.0
        for x in range(d + 1):
            p_val = exact_binomial_twoside_pvalue(x, d, 0.5)
            if p_val <= alpha:
                prob_x_given_d = math.comb(d, x) * (pi**x) * ((1.0 - pi) ** (d - x))
                cond_power += prob_x_given_d
        total_power += prob_d * cond_power

    return total_power


def generate_sensitivity_grid(
    scenarios: list[tuple[float, float]],
    sample_sizes: list[int],
    alpha: float = 0.05,
) -> list[CellResult]:
    """Generate sensitivity grid for all scenario x sample size pairs."""
    results: list[CellResult] = []
    for p10, p01 in scenarios:
        diff = p10 - p01
        for n in sample_sizes:
            exp_d = n * (p10 + p01)
            power = compute_exact_mcnemar_power(n, p10, p01, alpha=alpha)
            results.append(
                CellResult(
                    n=n,
                    p10=p10,
                    p01=p01,
                    marginal_diff=round(diff, 4),
                    expected_discordant=round(exp_d, 2),
                    exact_power=round(power, 4),
                    alpha=alpha,
                )
            )
    return results


def format_markdown_table(results: list[CellResult]) -> str:
    """Format results into a publication-ready Markdown table."""
    lines = [
        "| $p_{10}$ (S/F) | $p_{01}$ (F/S) | Marginal $\\Delta$ | $N=64$ | $N=96$ | $N=128$ | $N=160$ | $N=192$ | $N=256$ |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    # Group by scenario
    by_scenario: dict[tuple[float, float], dict[int, CellResult]] = {}
    for r in results:
        by_scenario.setdefault((r.p10, r.p01), {})[r.n] = r

    for (p10, p01), n_map in by_scenario.items():
        diff = p10 - p01
        diff_str = f"{diff*100:+.1f}%" if diff != 0 else "0.0%"
        row = [f"{p10:.2f}", f"{p01:.2f}", diff_str]
        for n in [64, 96, 128, 160, 192, 256]:
            res = n_map[n]
            # Format power as percentage and expected discordance in parens
            power_pct = f"{res.exact_power*100:.1f}%"
            row.append(f"{power_pct} ($E[D]={res.expected_discordant:.1f}$)")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alpha", type=float, default=0.05, help="Two-sided alpha (default: 0.05)"
    )
    parser.add_argument(
        "--sample-sizes",
        type=int,
        nargs="+",
        default=[64, 96, 128, 160, 192, 256],
        help="Sample sizes N to evaluate",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    scenarios = [
        (0.12, 0.02),  # +10% diff, low discordance (14%)
        (0.14, 0.04),  # +10% diff, moderate discordance (18%)
        (0.15, 0.03),  # +12% diff, moderate discordance (18%)
        (0.18, 0.06),  # +12% diff, higher discordance (24%)
        (0.05, 0.05),  # null case: symmetric low discordance (10%)
        (0.10, 0.10),  # null case: symmetric moderate discordance (20%)
    ]
    sample_sizes = [64, 96, 128, 160, 192, 256]

    results = generate_sensitivity_grid(scenarios, sample_sizes, alpha=args.alpha)
    table_md = format_markdown_table(results)

    print("\n# Exact Two-Sided McNemar Power Sensitivity Grid (alpha = 0.05)\n")
    print(table_md)
    print("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
