"""Scalar word-level SPECK32/64 round and block operations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, TypeAlias

from .key_schedule import (
    SPECK32_64_ALPHA,
    SPECK32_64_BETA,
    SPECK32_64_ROUND_COUNT,
    expand_key,
)
from .primitives import (
    add_words,
    mask_word,
    rotate_left,
    rotate_right,
    subtract_words,
)

Block: TypeAlias = tuple[int, int]
SPECK32_64_BLOCK_WORD_COUNT: Final[int] = 2


def _normalise_block(block: Sequence[int]) -> Block:
    """Validate and mask a two-word block."""

    try:
        words = tuple(block)
    except TypeError as error:
        raise TypeError("block must be a sequence of two integers") from error

    if len(words) != SPECK32_64_BLOCK_WORD_COUNT:
        raise ValueError(
            "block must contain exactly "
            f"{SPECK32_64_BLOCK_WORD_COUNT} words"
        )
    return mask_word(words[0]), mask_word(words[1])


def encrypt_round(state: Sequence[int], round_key: int) -> Block:
    """Apply one SPECK32/64 encryption round to a word-level state."""

    x, y = _normalise_block(state)
    key = mask_word(round_key)

    next_x = mask_word(
        add_words(rotate_right(x, SPECK32_64_ALPHA), y) ^ key
    )
    next_y = mask_word(rotate_left(y, SPECK32_64_BETA) ^ next_x)
    return next_x, next_y


def decrypt_round(state: Sequence[int], round_key: int) -> Block:
    """Apply the inverse of one SPECK32/64 encryption round."""

    next_x, next_y = _normalise_block(state)
    key = mask_word(round_key)

    y = rotate_right(
        mask_word(next_y ^ next_x),
        SPECK32_64_BETA,
    )
    x = rotate_left(
        subtract_words(mask_word(next_x ^ key), y),
        SPECK32_64_ALPHA,
    )
    return x, y


def encrypt_block(
    block: Sequence[int],
    master_key: Sequence[int],
    rounds: int = SPECK32_64_ROUND_COUNT,
) -> Block:
    """Encrypt a two-word block with the first rounds of SPECK round keys."""

    round_keys = expand_key(master_key, rounds)
    state = _normalise_block(block)
    for round_key in round_keys:
        state = encrypt_round(state, round_key)
    return state


def decrypt_block(
    block: Sequence[int],
    master_key: Sequence[int],
    rounds: int = SPECK32_64_ROUND_COUNT,
) -> Block:
    """Decrypt a two-word block using the matching reduced-round prefix."""

    round_keys = expand_key(master_key, rounds)
    state = _normalise_block(block)
    for round_key in reversed(round_keys):
        state = decrypt_round(state, round_key)
    return state
