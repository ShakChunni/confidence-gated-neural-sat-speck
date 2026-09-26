"""Reference algebra for Bao's SIMON Markov-predicted difference distribution.

The full-width distribution is deliberately not generated here. The small-width
integer implementation is a correctness oracle for a future bounded generator.
"""

from __future__ import annotations

from collections import defaultdict
from numbers import Integral


def _validate_word_bits(word_bits: int) -> int:
    if isinstance(word_bits, bool) or not isinstance(word_bits, Integral):
        raise TypeError("word_bits must be an integer")
    if not 3 <= word_bits <= 16:
        raise ValueError("word_bits must be between 3 and 16")
    return int(word_bits)


def _validate_word(value: int, word_bits: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if not 0 <= value < (1 << word_bits):
        raise ValueError(f"{name} must fit in {word_bits} bits")
    return int(value)


def simon_f_width(value: int, word_bits: int) -> int:
    """SIMON f with rotations reduced modulo an explicit toy word width."""
    word_bits = _validate_word_bits(word_bits)
    value = _validate_word(value, word_bits, "value")
    mask = (1 << word_bits) - 1

    def rol(shift: int) -> int:
        shift %= word_bits
        if shift == 0:
            return value
        return ((value << shift) | (value >> (word_bits - shift))) & mask

    return (rol(1) & rol(8)) ^ rol(2)


def affine_derivative_basis(input_left: int, word_bits: int) -> tuple[int, tuple[int, ...]]:
    """Return offset and independent image basis of ``f(x^a)^f(x)``.

    The quadratic AND derivative is affine in ``x``. The returned basis is
    reduced by descending bit pivots; its span, not the basis order, matters.
    """
    word_bits = _validate_word_bits(word_bits)
    input_left = _validate_word(input_left, word_bits, "input_left")
    offset = simon_f_width(input_left, word_bits)
    pivots = [0] * word_bits
    for bit in range(word_bits):
        one = 1 << bit
        column = (
            simon_f_width(input_left ^ one, word_bits)
            ^ simon_f_width(one, word_bits)
            ^ offset
        )
        for pivot in range(word_bits - 1, -1, -1):
            if column & (1 << pivot):
                if pivots[pivot]:
                    column ^= pivots[pivot]
                else:
                    pivots[pivot] = column
                    break
    return offset, tuple(pivot for pivot in pivots if pivot)


def affine_derivative_outputs(input_left: int, word_bits: int) -> tuple[int, ...]:
    """List every nonzero-probability derivative output once, without sampling."""
    offset, basis = affine_derivative_basis(input_left, word_bits)
    outputs = [offset]
    for vector in basis:
        outputs.extend(value ^ vector for value in tuple(outputs))
    return tuple(outputs)


def predicted_distribution_toy(
    word_bits: int,
    rounds: int,
    input_difference: tuple[int, int],
) -> tuple[int, dict[tuple[int, int], int]]:
    """Compute a complete toy-width predicted DDT with exact integer mass.

    Returns ``(denominator, {(left, right): numerator})``. This reference is
    limited to ``word_bits <= 6`` so it cannot accidentally run a full-width
    distribution in Python. No zero-mass state is stored.
    """
    word_bits = _validate_word_bits(word_bits)
    if word_bits > 6:
        raise ValueError("toy oracle is limited to at most six-bit words")
    if isinstance(rounds, bool) or not isinstance(rounds, int):
        raise TypeError("rounds must be an integer")
    if not 0 <= rounds <= 7:
        raise ValueError("rounds must be between zero and seven")
    if len(input_difference) != 2:
        raise ValueError("input_difference must have two words")
    left = _validate_word(input_difference[0], word_bits, "input left")
    right = _validate_word(input_difference[1], word_bits, "input right")

    denominator = 1 << (rounds * word_bits)
    states = {(left, right): denominator}
    for _ in range(rounds):
        next_states: defaultdict[tuple[int, int], int] = defaultdict(int)
        for (a, b), mass in states.items():
            outputs = affine_derivative_outputs(a, word_bits)
            if mass % len(outputs):
                raise ArithmeticError("fixed denominator cannot represent transition")
            share = mass // len(outputs)
            for derivative in outputs:
                next_states[(b ^ derivative, a)] += share
        states = dict(next_states)
        if sum(states.values()) != denominator:
            raise ArithmeticError("predicted distribution lost probability mass")
    return denominator, states
