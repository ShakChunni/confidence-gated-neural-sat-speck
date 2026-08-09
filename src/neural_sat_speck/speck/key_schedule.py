"""Scalar SPECK32/64 key expansion.

The public master-key order is the word-level order used by the verified test
vector: (l[2], l[1], l[0], k[0]). The returned tuple contains the requested
prefix of round keys, beginning with k[0].
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from .primitives import add_words, mask_word, rotate_left, rotate_right

SPECK32_64_KEY_WORD_COUNT: Final[int] = 4
SPECK32_64_ROUND_COUNT: Final[int] = 22
SPECK32_64_ALPHA: Final[int] = 7
SPECK32_64_BETA: Final[int] = 2


def _validate_round_count(rounds: int) -> int:
    """Validate a reduced-round key-prefix length."""

    if isinstance(rounds, bool) or not isinstance(rounds, int):
        raise TypeError("rounds must be an integer")
    if not 1 <= rounds <= SPECK32_64_ROUND_COUNT:
        raise ValueError(
            f"rounds must be between 1 and {SPECK32_64_ROUND_COUNT}"
        )
    return rounds


def _normalise_master_key(master_key: Sequence[int]) -> tuple[int, ...]:
    """Validate and mask a four-word master key."""

    try:
        words = tuple(master_key)
    except TypeError as error:
        raise TypeError("master_key must be a sequence of four integers") from error

    if len(words) != SPECK32_64_KEY_WORD_COUNT:
        raise ValueError(
            "master_key must contain exactly "
            f"{SPECK32_64_KEY_WORD_COUNT} words"
        )
    return tuple(mask_word(word) for word in words)


def expand_key(
    master_key: Sequence[int],
    rounds: int = SPECK32_64_ROUND_COUNT,
) -> tuple[int, ...]:
    """Expand a SPECK32/64 master key into a round-key prefix.

    Args:
        master_key: Four 16-bit words in the order
            (l[2], l[1], l[0], k[0]).
        rounds: Number of round keys to return, from 1 through 22.

    Returns:
        A tuple containing k[0] through k[rounds - 1].

    Raises:
        TypeError: If the key or round count has an invalid type.
        ValueError: If the key has the wrong length or rounds is outside the
            SPECK32/64 range.
    """

    round_count = _validate_round_count(rounds)
    words = _normalise_master_key(master_key)

    # The source notation gives (l[2], l[1], l[0], k[0]); the recurrence
    # consumes l[0], l[1], l[2], ... in ascending index order.
    l_words = [words[2], words[1], words[0]]
    round_keys = [words[3]]

    for index in range(round_count - 1):
        next_l = mask_word(
            add_words(
                round_keys[index],
                rotate_right(l_words[index], SPECK32_64_ALPHA),
            )
            ^ index
        )
        next_key = mask_word(
            rotate_left(round_keys[index], SPECK32_64_BETA) ^ next_l
        )
        l_words.append(next_l)
        round_keys.append(next_key)

    return tuple(round_keys)
