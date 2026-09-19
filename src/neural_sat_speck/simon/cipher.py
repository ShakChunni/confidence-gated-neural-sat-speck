"""Scalar word-level SIMON32/64 round and block operations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, TypeAlias

from .key_schedule import (
    SIMON32_64_ROUND_COUNT,
    expand_key,
)
from .primitives import (
    mask_word,
    simon_f,
)

Block: TypeAlias = tuple[int, int]
SIMON32_64_BLOCK_WORD_COUNT: Final[int] = 2

# Official Beaulieu et al. (NSA) Known-Answer Test (KAT) vectors
# Master key in (k3, k2, k1, k0) order:
SIMON32_64_KAT_KEY: Final[tuple[int, int, int, int]] = (0x1918, 0x1110, 0x0908, 0x0100)
# Plaintext (L0, R0):
SIMON32_64_KAT_PLAINTEXT: Final[Block] = (0x6565, 0x6877)
# Ciphertext after 32 rounds (L32, R32):
SIMON32_64_KAT_CIPHERTEXT: Final[Block] = (0xC69B, 0xE9BB)


def _normalise_block(block: Sequence[int]) -> Block:
    """Validate and mask a two-word block."""
    try:
        words = tuple(block)
    except TypeError as error:
        raise TypeError("block must be a sequence of two integers") from error

    if len(words) != SIMON32_64_BLOCK_WORD_COUNT:
        raise ValueError(
            "block must contain exactly "
            f"{SIMON32_64_BLOCK_WORD_COUNT} words"
        )
    return mask_word(words[0]), mask_word(words[1])


def encrypt_round(state: Sequence[int], round_key: int) -> Block:
    """Apply one SIMON32/64 encryption round to a word-level state (L, R).

    L_{i+1} = R_i ^ f(L_i) ^ k_i
    R_{i+1} = L_i
    where f(x) = (x <<< 1 & x <<< 8) ^ (x <<< 2).
    """
    l_val, r_val = _normalise_block(state)
    key = mask_word(round_key)

    next_l = mask_word(r_val ^ simon_f(l_val) ^ key)
    next_r = l_val
    return next_l, next_r


def decrypt_round(state: Sequence[int], round_key: int) -> Block:
    """Apply the inverse of one SIMON32/64 encryption round.

    L_i = R_{i+1}
    R_i = L_{i+1} ^ f(R_{i+1}) ^ k_i
    """
    next_l, next_r = _normalise_block(state)
    key = mask_word(round_key)

    l_val = next_r
    r_val = mask_word(next_l ^ simon_f(next_r) ^ key)
    return l_val, r_val


def encrypt_block(
    block: Sequence[int],
    master_key: Sequence[int],
    rounds: int = SIMON32_64_ROUND_COUNT,
) -> Block:
    """Encrypt a two-word block with the first rounds of SIMON round keys."""
    round_keys = expand_key(master_key, rounds)
    state = _normalise_block(block)
    for round_key in round_keys:
        state = encrypt_round(state, round_key)
    return state


def decrypt_block(
    block: Sequence[int],
    master_key: Sequence[int],
    rounds: int = SIMON32_64_ROUND_COUNT,
) -> Block:
    """Decrypt a two-word block using the matching reduced-round prefix."""
    round_keys = expand_key(master_key, rounds)
    state = _normalise_block(block)
    for round_key in reversed(round_keys):
        state = decrypt_round(state, round_key)
    return state
