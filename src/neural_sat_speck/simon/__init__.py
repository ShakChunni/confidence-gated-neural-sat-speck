"""Scalar SIMON32/64 reference components."""

from .cipher import (
    SIMON32_64_BLOCK_WORD_COUNT,
    SIMON32_64_KAT_CIPHERTEXT,
    SIMON32_64_KAT_KEY,
    SIMON32_64_KAT_PLAINTEXT,
    Block,
    decrypt_block,
    decrypt_round,
    encrypt_block,
    encrypt_round,
)
from .key_schedule import (
    SIMON32_64_CONSTANT_C,
    SIMON32_64_KEY_WORD_COUNT,
    SIMON32_64_ROUND_COUNT,
    SIMON32_64_Z0,
    SIMON32_64_Z0_PERIOD,
    expand_key,
)
from .partial import partial_decrypt_final_round
from .primitives import (
    WORD_BITS,
    WORD_MASK,
    WORD_MODULUS,
    mask_word,
    rotate_left,
    rotate_right,
    simon_f,
)

__all__ = [
    "Block",
    "SIMON32_64_BLOCK_WORD_COUNT",
    "SIMON32_64_CONSTANT_C",
    "SIMON32_64_KAT_CIPHERTEXT",
    "SIMON32_64_KAT_KEY",
    "SIMON32_64_KAT_PLAINTEXT",
    "SIMON32_64_KEY_WORD_COUNT",
    "SIMON32_64_ROUND_COUNT",
    "SIMON32_64_Z0",
    "SIMON32_64_Z0_PERIOD",
    "WORD_BITS",
    "WORD_MASK",
    "WORD_MODULUS",
    "decrypt_block",
    "decrypt_round",
    "encrypt_block",
    "encrypt_round",
    "expand_key",
    "mask_word",
    "partial_decrypt_final_round",
    "rotate_left",
    "rotate_right",
    "simon_f",
]
