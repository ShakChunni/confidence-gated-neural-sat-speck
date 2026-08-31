"""Statistical metrics and solver benchmarking functions.

This module provides verified mathematical metrics for solver benchmarking:
1. Global Penalized Average Runtime (PAR-2) with strict cutoff semantics;
2. Wilson score confidence intervals for binomial success rates;
3. Exact two-sided McNemar paired hypothesis testing;
4. Continuous Empirical Cumulative Distribution (ECDF) / Cactus curve generation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def calculate_global_par2(
    runtimes: Sequence[float],
    successes: Sequence[bool],
    global_cutoff: float,
) -> float:
    """Calculate Penalized Average Runtime (PAR-2) under a single global cutoff.

    In solver competitions (SAT Competition Standard), failures (timeouts,
    unsolved, memory-out, or runs that exceed global_cutoff) are penalized as
    exactly 2 * global_cutoff. Successful runs that completed within the cutoff
    are charged their actual wall-clock execution time.

    Args:
        runtimes: Sequence of non-negative wall-clock execution times (seconds).
        successes: Sequence of boolean indicators whether recovery was verified.
        global_cutoff: Positive global wall-clock cutoff threshold (seconds).

    Returns:
        The mean penalized runtime across all instances.

    Raises:
        ValueError: If lengths mismatch, runtimes are negative/non-finite,
            or global_cutoff is non-positive.
    """
    if len(runtimes) != len(successes):
        raise ValueError(
            f"runtimes length ({len(runtimes)}) does not match successes length ({len(successes)})"
        )
    if not runtimes:
        return 0.0
    if not math.isfinite(global_cutoff) or global_cutoff <= 0:
        raise ValueError(f"global_cutoff must be positive and finite, got {global_cutoff}")

    penalized_times: list[float] = []
    for t, succ in zip(runtimes, successes, strict=True):
        if not math.isfinite(t) or t < 0:
            raise ValueError(f"runtime values must be non-negative and finite, got {t}")
        if succ and t <= global_cutoff:
            penalized_times.append(float(t))
        else:
            penalized_times.append(2.0 * float(global_cutoff))

    return float(sum(penalized_times) / len(penalized_times))


def wilson_score_interval(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Calculate the exact Wilson score confidence interval for a proportion.

    Args:
        successes: Number of successful outcomes (0 <= successes <= total).
        total: Total number of independent trials (total > 0).
        confidence: Confidence level in (0, 1), default 0.95 (z ≈ 1.95996).

    Returns:
        A tuple (lower_bound, upper_bound) clamped to [0.0, 1.0].

    Raises:
        ValueError: If inputs are invalid or out of bounds.
    """
    if total <= 0:
        raise ValueError(f"total trials must be positive, got {total}")
    if not (0 <= successes <= total):
        raise ValueError(f"successes ({successes}) must be in [0, {total}]")
    if not (0 < confidence < 1):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    # Standard normal quantile z for two-sided interval
    # Approximation for common confidence levels with scipy fallback or standard values
    if math.isclose(confidence, 0.95, abs_tol=1e-4):
        z = 1.959963984540054
    elif math.isclose(confidence, 0.99, abs_tol=1e-4):
        z = 2.5758293035489004
    elif math.isclose(confidence, 0.90, abs_tol=1e-4):
        z = 1.6448536269514722
    else:
        # Standard inverse error function approximation
        z = math.sqrt(2.0) * _erfinv(confidence)

    n = float(total)
    p = float(successes) / n
    z2 = z * z

    center = (p + z2 / (2.0 * n)) / (1.0 + z2 / n)
    margin = (z / (1.0 + z2 / n)) * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))

    if successes == 0:
        lower = 0.0
    else:
        lower = max(0.0, center - margin)

    if successes == total:
        upper = 1.0
    else:
        upper = min(1.0, center + margin)

    return (float(lower), float(upper))


def exact_mcnemar_test(b: int, c: int) -> dict[str, Any]:
    """Perform exact two-sided McNemar test on discordant pairs (b, c).

    Given paired binary outcomes between Model 1 and Model 2:
    - b: instances where Model 1 succeeded and Model 2 failed.
    - c: instances where Model 1 failed and Model 2 succeeded.

    Under H0 (marginal homogeneity), b ~ Binomial(b + c, 0.5).

    Args:
        b: Non-negative count of discordant pairs (M1=1, M2=0).
        c: Non-negative count of discordant pairs (M1=0, M2=1).

    Returns:
        Dictionary containing discordant counts, net difference, and exact p-value.
    """
    if b < 0 or c < 0:
        raise ValueError(f"discordant counts must be non-negative, got b={b}, c={c}")

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

    k_min = min(b, c)
    # Compute cumulative binomial probability P(X <= k_min) for X ~ Binomial(n_d, 0.5)
    cum_prob = 0.0
    for k in range(k_min + 1):
        cum_prob += math.comb(n_d, k) * (0.5 ** n_d)

    p_value = min(1.0, 2.0 * cum_prob)

    return {
        "b": b,
        "c": c,
        "n_discordant": n_d,
        "net_difference": b - c,
        "p_value": float(p_value),
        "significant_at_05": p_value < 0.05,
    }


def calculate_cactus_curve(
    runtimes: Sequence[float],
    successes: Sequence[bool],
    max_time: float = 300.0,
    num_points: int = 150,
) -> list[dict[str, float]]:
    """Compute solved fraction over time for continuous Cactus / ECDF plots.

    Args:
        runtimes: Sequence of execution times for each instance.
        successes: Sequence of success indicators for each instance.
        max_time: Upper time limit for evaluation curve.
        num_points: Number of discrete evaluation points on the curve.

    Returns:
        List of dicts with 'time', 'solved_count', and 'solved_fraction'.
    """
    if len(runtimes) != len(successes):
        raise ValueError("runtimes and successes must have the same length")
    if not runtimes:
        return []

    total_instances = len(runtimes)
    # Extract solved times
    solved_times = sorted([
        t for t, succ in zip(runtimes, successes, strict=True)
        if succ and math.isfinite(t) and t >= 0
    ])

    time_grid = [max_time * (i / float(num_points - 1)) for i in range(num_points)]
    # Ensure 0.0 and max_time are included
    if time_grid[0] != 0.0:
        time_grid.insert(0, 0.0)

    points: list[dict[str, float]] = []
    for t_eval in time_grid:
        count = sum(1 for st in solved_times if st <= t_eval)
        points.append({
            "time_seconds": float(t_eval),
            "solved_count": float(count),
            "solved_fraction": float(count / total_instances),
        })

    return points


def _erfinv(y: float) -> float:
    """Winitzki approximation of inverse error function."""
    a = 0.147
    sgn = 1.0 if y >= 0 else -1.0
    y_abs = abs(y)
    if y_abs >= 1.0:
        return sgn * float("inf")
    term1 = 2.0 / (math.pi * a) + math.log(1.0 - y_abs * y_abs) / 2.0
    term2 = math.log(1.0 - y_abs * y_abs) / a
    inner = term1 * term1 - term2
    return sgn * math.sqrt(math.sqrt(inner) - term1)
