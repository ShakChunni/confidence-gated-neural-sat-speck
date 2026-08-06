"""Scalar SPECK32/64 reference components."""

from .primitives import (
    WORD_BITS,
    WORD_MASK,
    WORD_MODULUS,
    add_words,
    mask_word,
    rotate_left,
    rotate_right,
    subtract_words,
)

__all__ = [
    "WORD_BITS",
    "WORD_MASK",
    "WORD_MODULUS",
    "add_words",
    "mask_word",
    "rotate_left",
    "rotate_right",
    "subtract_words",
]
