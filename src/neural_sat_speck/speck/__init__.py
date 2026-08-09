"""Scalar SPECK32/64 reference components."""

from .cipher import (
    SPECK32_64_BLOCK_WORD_COUNT,
    Block,
    decrypt_block,
    decrypt_round,
    encrypt_block,
    encrypt_round,
)
from .key_schedule import (
    SPECK32_64_ALPHA,
    SPECK32_64_BETA,
    SPECK32_64_KEY_WORD_COUNT,
    SPECK32_64_ROUND_COUNT,
    expand_key,
)
from .partial import partial_decrypt_final_round
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
    "Block",
    "SPECK32_64_ALPHA",
    "SPECK32_64_BETA",
    "SPECK32_64_BLOCK_WORD_COUNT",
    "SPECK32_64_KEY_WORD_COUNT",
    "SPECK32_64_ROUND_COUNT",
    "WORD_BITS",
    "WORD_MASK",
    "WORD_MODULUS",
    "add_words",
    "decrypt_block",
    "decrypt_round",
    "expand_key",
    "encrypt_block",
    "encrypt_round",
    "mask_word",
    "partial_decrypt_final_round",
    "rotate_left",
    "rotate_right",
    "subtract_words",
]
