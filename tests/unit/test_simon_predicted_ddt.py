"""Falsification tests for the SIMON Markov-predicted DDT reference algebra."""

from __future__ import annotations

from collections import Counter, defaultdict

import pytest

from neural_sat_speck.simon.predicted_ddt import (
    affine_derivative_basis,
    affine_derivative_outputs,
    predicted_distribution_toy,
    simon_f_width,
)


def _naive_scalar_distribution(
    word_bits: int, rounds: int, delta: tuple[int, int]
) -> tuple[int, dict[tuple[int, int], int]]:
    """Count all round inputs x independently, without affine-basis logic."""
    denominator = 1 << (rounds * word_bits)
    states = {delta: denominator}
    for _ in range(rounds):
        next_states: defaultdict[tuple[int, int], int] = defaultdict(int)
        for (a, b), mass in states.items():
            assert mass % (1 << word_bits) == 0
            share = mass >> word_bits
            for x in range(1 << word_bits):
                derivative = simon_f_width(x ^ a, word_bits) ^ simon_f_width(
                    x, word_bits
                )
                next_states[(b ^ derivative, a)] += share
        states = dict(next_states)
        assert sum(states.values()) == denominator
    return denominator, states


@pytest.mark.parametrize("word_bits", [3, 4, 6])
@pytest.mark.parametrize("rounds", [0, 1, 3, 7])
def test_full_toy_distribution_matches_exhaustive_scalar_enumeration(
    word_bits: int, rounds: int
) -> None:
    delta = (0, 1)
    assert predicted_distribution_toy(word_bits, rounds, delta) == (
        _naive_scalar_distribution(word_bits, rounds, delta)
    )


@pytest.mark.parametrize("a", [0, 1, 0x0040, 0x0080, 0x1234, 0xFFFF])
def test_full_width_affine_support_matches_exhaustive_scalar_histogram(a: int) -> None:
    offset, basis = affine_derivative_basis(a, 16)
    outputs = affine_derivative_outputs(a, 16)
    histogram = Counter(
        simon_f_width(x ^ a, 16) ^ simon_f_width(x, 16)
        for x in range(1 << 16)
    )
    assert len(outputs) == 1 << len(basis)
    assert len(set(outputs)) == len(outputs)
    assert set(outputs) == set(histogram)
    assert set(histogram.values()) == {1 << (16 - len(basis))}
    assert offset in histogram


def _bao_source_probability(a: int, b: int, c: int) -> float:
    """Independent translation of pinned simon_ddt.cpp::diff_prob's conditions."""
    mask = 0xFFFF

    def rol(value: int, shift: int) -> int:
        return ((value << shift) | (value >> (16 - shift))) & mask

    gamma = rol(a, 2) ^ b ^ c
    if a == mask:
        return 2.0 ** -15 if gamma.bit_count() % 2 == 0 else 0.0
    rola, rolb = rol(a, 8), rol(a, 1)
    variable_bits = rola | rolb
    if gamma & (variable_bits ^ mask):
        return 0.0
    double_bits = rol(a, 15) & (rola ^ mask) & rolb
    if (gamma ^ rol(gamma, 7)) & double_bits:
        return 0.0
    return 2.0 ** -(variable_bits ^ double_bits).bit_count()


@pytest.mark.parametrize("a", [0, 1, 0x0040, 0x1234, 0xFFFF])
def test_affine_kernel_matches_pinned_source_probability_for_all_outputs(a: int) -> None:
    _, basis = affine_derivative_basis(a, 16)
    valid = set(affine_derivative_outputs(a, 16))
    expected_probability = 2.0 ** -len(basis)
    for c in range(1 << 16):
        probability = _bao_source_probability(a, 0, c)
        assert probability == (expected_probability if c in valid else 0.0)


def test_reference_rejects_full_width_generation_and_malformed_inputs() -> None:
    with pytest.raises(ValueError, match="toy oracle"):
        predicted_distribution_toy(16, 7, (0, 0x40))
    with pytest.raises(ValueError, match="fit"):
        affine_derivative_basis(0x10000, 16)
    with pytest.raises(TypeError, match="integer"):
        affine_derivative_basis(True, 16)
