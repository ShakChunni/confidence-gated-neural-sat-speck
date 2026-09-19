"""Explicit 16-bit arithmetic and bitwise primitives for SIMON32/64.

Operates on Python integers and reduces values to the 16-bit word range
modulo 2^16. Rejects non-integer types and booleans.
"""

from __future__ import annotations

from typing import Final

WORD_BITS: Final[int] = 16
WORD_MODULUS: Final[int] = 1 << WORD_BITS
WORD_MASK: Final[int] = WORD_MODULUS - 1


def _require_integer(value: int, *, name: str) -> int:
    """Return an integer input or raise a clear type error.

    Boolean values are rejected even though bool is an int subclass.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def mask_word(value: int) -> int:
    """Reduce an integer modulo 2^16."""
    value = _require_integer(value, name="value")
    return value & WORD_MASK


def _normalise_rotation(amount: int) -> int:
    """Normalise a rotation count to one 16-bit word period."""
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


def simon_f(x: int) -> int:
    """Nonlinear Feistel branch function: f(x) = (x <<< 1 & x <<< 8) ^ (x <<< 2)."""
    word = mask_word(x)
    return ((rotate_left(word, 1) & rotate_left(word, 8)) ^ rotate_left(word, 2)) & WORD_MASK
