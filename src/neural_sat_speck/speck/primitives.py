"""Explicit 16-bit arithmetic primitives used by the scalar SPECK reference.

The functions in this module deliberately operate on Python integers and return
values reduced to the SPECK32/64 word range. They do not perform byte conversion
or choose a serialized block layout.
"""

from __future__ import annotations

from typing import Final

WORD_BITS: Final[int] = 16
WORD_MODULUS: Final[int] = 1 << WORD_BITS
WORD_MASK: Final[int] = WORD_MODULUS - 1


def _require_integer(value: int, *, name: str) -> int:
    """Return an integer input or raise a clear type error.

    Boolean values are rejected even though bool is an int subclass, because a
    boolean is not a meaningful cryptographic word or rotation count.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def mask_word(value: int) -> int:
    """Reduce an integer modulo 2^16.

    Values outside the word range intentionally wrap, matching the modular
    arithmetic used by SPECK. Non-integer inputs are rejected.
    """

    value = _require_integer(value, name="value")
    return value & WORD_MASK


def _normalise_rotation(amount: int) -> int:
    """Normalise a rotation count to one word period."""

    amount = _require_integer(amount, name="amount")
    return amount % WORD_BITS


def rotate_left(value: int, amount: int) -> int:
    """Rotate a 16-bit word left by amount bits."""

    word = mask_word(value)
    shift = _normalise_rotation(amount)
    if shift == 0:
        return word
    return ((word << shift) | (word >> (WORD_BITS - shift))) & WORD_MASK


def rotate_right(value: int, amount: int) -> int:
    """Rotate a 16-bit word right by amount bits."""

    word = mask_word(value)
    shift = _normalise_rotation(amount)
    if shift == 0:
        return word
    return ((word >> shift) | (word << (WORD_BITS - shift))) & WORD_MASK


def add_words(left: int, right: int) -> int:
    """Add two words modulo 2^16."""

    return mask_word(mask_word(left) + mask_word(right))


def subtract_words(left: int, right: int) -> int:
    """Subtract two words modulo 2^16."""

    return mask_word(mask_word(left) - mask_word(right))
