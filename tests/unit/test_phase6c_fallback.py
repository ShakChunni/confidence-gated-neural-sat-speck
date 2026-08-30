"""Tests for the bounded Phase 6C diversity route."""

from __future__ import annotations

import pytest

from neural_sat_speck.attacks.diversity_fallback import (
    choose_phase6c_route,
    compose_diversity_fallback,
)


def test_fallback_alternates_and_fills_exactly_32_unique_candidates() -> None:
    b1 = tuple(range(40))
    b0 = tuple(range(8)) + tuple(range(20, 40)) + tuple(range(8, 20))
    result = compose_diversity_fallback(b1, b0)

    assert len(result) == 32
    assert len(set(result)) == 32
    assert result[:10] == (0, 1, 2, 3, 4, 5, 6, 7, 8, 20)
    assert result[10:24] == (9, 21, 10, 22, 11, 23, 12, 24, 13, 25, 14, 26, 15, 27)
    assert set(result) == set(range(32))
    assert result[-1] not in result[:-1]


def test_fallback_first_occurrence_and_duplicate_fill_are_deterministic() -> None:
    b1 = tuple(range(40))
    b0 = b1
    assert compose_diversity_fallback(b1, b0) == tuple(range(32))
    assert compose_diversity_fallback(b1, b0) == compose_diversity_fallback(b1, b0)


def test_route_boundary_uses_trusted_route_at_threshold() -> None:
    assert choose_phase6c_route(0.7, 0.7) == "trusted_top8"
    assert choose_phase6c_route(0.699, 0.7) == "diversity_fallback"


@pytest.mark.parametrize(
    ("b1", "b0"),
    [
        (tuple(range(31)), tuple(range(31))),
        (tuple(range(40)), tuple(range(39)) + (39, 39)),
        (tuple(range(40)), tuple(range(1, 41))),
    ],
)
def test_fallback_rejects_non_complete_inputs(
    b1: tuple[int, ...], b0: tuple[int, ...]
) -> None:
    with pytest.raises((TypeError, ValueError)):
        compose_diversity_fallback(b1, b0)
