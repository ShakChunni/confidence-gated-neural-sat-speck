"""Tests for the explicit 16-bit SPECK primitive layer."""

import pytest

from neural_sat_speck.speck.primitives import (
    WORD_MASK,
    add_words,
    mask_word,
    rotate_left,
    rotate_right,
    subtract_words,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0),
        (1, 1),
        (WORD_MASK, WORD_MASK),
        (WORD_MASK + 1, 0),
        (-1, WORD_MASK),
    ],
)
def test_mask_word_wraps_to_16_bits(value: int, expected: int) -> None:
    assert mask_word(value) == expected


@pytest.mark.parametrize(
    ("value", "amount", "expected_left", "expected_right"),
    [
        (0x1234, 0, 0x1234, 0x1234),
        (0x1234, 4, 0x2341, 0x4123),
        (0x1234, 16, 0x1234, 0x1234),
        (0xFFFF, 7, 0xFFFF, 0xFFFF),
    ],
)
def test_rotations_use_16_bit_words(
    value: int,
    amount: int,
    expected_left: int,
    expected_right: int,
) -> None:
    assert rotate_left(value, amount) == expected_left
    assert rotate_right(value, amount) == expected_right


@pytest.mark.parametrize("amount", [0, 1, 2, 7, 15, 16, 31])
@pytest.mark.parametrize("value", [0, 1, 0x1234, WORD_MASK])
def test_rotations_are_inverses(value: int, amount: int) -> None:
    assert rotate_right(rotate_left(value, amount), amount) == mask_word(value)
    assert rotate_left(rotate_right(value, amount), amount) == mask_word(value)


def test_rotations_normalise_negative_counts() -> None:
    assert rotate_left(0x1234, -1) == rotate_right(0x1234, 1)
    assert rotate_right(0x1234, -1) == rotate_left(0x1234, 1)


def test_add_words_wraps() -> None:
    assert add_words(WORD_MASK, 1) == 0
    assert add_words(0xFFFF, 0xFFFF) == 0xFFFE
    assert add_words(0x10000, 1) == 1


def test_subtract_words_wraps() -> None:
    assert subtract_words(0, 1) == WORD_MASK
    assert subtract_words(0x10000, 1) == WORD_MASK


@pytest.mark.parametrize(
    ("left", "right"),
    [(0, 0), (1, 2), (0x1234, 0xFEDC), (WORD_MASK, WORD_MASK)],
)
def test_addition_and_subtraction_are_inverses(left: int, right: int) -> None:
    assert subtract_words(add_words(left, right), right) == mask_word(left)


@pytest.mark.parametrize("function", [mask_word, rotate_left, rotate_right])
def test_non_integer_word_inputs_are_rejected(function) -> None:
    if function is mask_word:
        with pytest.raises(TypeError):
            function(1.0)
    else:
        with pytest.raises(TypeError):
            function(1.0, 1)


def test_boolean_inputs_are_rejected() -> None:
    with pytest.raises(TypeError):
        mask_word(True)
    with pytest.raises(TypeError):
        rotate_left(1, False)
