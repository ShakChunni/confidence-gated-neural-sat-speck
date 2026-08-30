"""The frozen Phase 6C bounded diversity fallback.

The fallback is an order composition, not a fused score.  It alternates the
first sixteen candidates from the pair-aware B1 and independent-pair B0
orders, keeps the first occurrence of each candidate, and fills any remaining
budget from the B1 order starting at position seventeen.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

PHASE6C_FALLBACK_HEAD_SIZE = 16
PHASE6C_FALLBACK_BUDGET = 32
PHASE6C_TRUSTED_BUDGET = 8
Phase6CRoute = Literal["trusted_top8", "diversity_fallback"]


def _validate_order(order: Sequence[int], name: str) -> tuple[int, ...]:
    if isinstance(order, (str, bytes)):
        raise TypeError(f"{name} must be an integer sequence")
    values = tuple(order)
    if len(values) < PHASE6C_FALLBACK_BUDGET:
        raise ValueError(
            f"{name} must contain at least {PHASE6C_FALLBACK_BUDGET} candidates"
        )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError(f"{name} must contain integers")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must be a complete permutation without duplicates")
    return values


def compose_diversity_fallback(
    pair_aware_order: Sequence[int],
    baseline_order: Sequence[int],
) -> tuple[int, ...]:
    """Compose exactly 32 candidates under the frozen B1/B0 policy.

    Both inputs must be complete permutations of the same candidate domain.
    The first occurrence in the alternating B1-then-B0 scan wins; the B1
    suffix is used only to fill candidates not already seen.
    """

    b1 = _validate_order(pair_aware_order, "pair_aware_order")
    b0 = _validate_order(baseline_order, "baseline_order")
    if set(b1) != set(b0):
        raise ValueError("pair-aware and baseline orders must share one domain")
    result: list[int] = []
    seen: set[int] = set()
    for rank in range(PHASE6C_FALLBACK_HEAD_SIZE):
        for candidate in (b1[rank], b0[rank]):
            if candidate not in seen:
                seen.add(candidate)
                result.append(candidate)
    for candidate in b1[PHASE6C_FALLBACK_HEAD_SIZE:]:
        if candidate not in seen:
            seen.add(candidate)
            result.append(candidate)
            if len(result) == PHASE6C_FALLBACK_BUDGET:
                break
    if len(result) != PHASE6C_FALLBACK_BUDGET:
        raise RuntimeError("diversity fallback could not fill its frozen budget")
    return tuple(result)


def choose_phase6c_route(probability: float, threshold: float) -> Phase6CRoute:
    """Choose the trusted top-8 route or bounded fallback without rescue."""

    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        raise TypeError("probability must be numeric")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError("threshold must be numeric")
    if not 0.0 <= float(probability) <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    return "trusted_top8" if probability >= threshold else "diversity_fallback"


__all__ = [
    "PHASE6C_FALLBACK_BUDGET",
    "PHASE6C_FALLBACK_HEAD_SIZE",
    "PHASE6C_TRUSTED_BUDGET",
    "Phase6CRoute",
    "choose_phase6c_route",
    "compose_diversity_fallback",
]
