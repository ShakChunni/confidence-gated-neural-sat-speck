"""Scalar SIMON32/64 key expansion.

Governed by Beaulieu et al. (2013), Cryptology ePrint Report 2013/404.
The public master-key word order matches the verified test vector:
(k[3], k[2], k[1], k[0]), where k[0] is the least significant word and
first round key.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from .primitives import WORD_MASK, mask_word, rotate_right

SIMON32_64_KEY_WORD_COUNT: Final[int] = 4
SIMON32_64_ROUND_COUNT: Final[int] = 32
SIMON32_64_CONSTANT_C: Final[int] = 0xFFFC
SIMON32_64_Z0: Final[int] = 0b01100111000011010100100010111110110011100001101010010001011111
SIMON32_64_Z0_PERIOD: Final[int] = 62


def _validate_round_count(rounds: int) -> int:
    """Validate a reduced-round key-prefix length."""
    if isinstance(rounds, bool) or not isinstance(rounds, int):
        raise TypeError("rounds must be an integer")
    if not 1 <= rounds <= SIMON32_64_ROUND_COUNT:
        raise ValueError(
            f"rounds must be between 1 and {SIMON32_64_ROUND_COUNT}"
        )
    return rounds


def _normalise_master_key(master_key: Sequence[int]) -> tuple[int, int, int, int]:
    """Validate and mask a four-word master key in (k3, k2, k1, k0) order."""
    try:
        words = tuple(master_key)
    except TypeError as error:
        raise TypeError("master_key must be a sequence of four integers") from error

    if len(words) != SIMON32_64_KEY_WORD_COUNT:
        raise ValueError(
            f"master_key must contain exactly {SIMON32_64_KEY_WORD_COUNT} words"
        )
    return (
        mask_word(words[0]),
        mask_word(words[1]),
        mask_word(words[2]),
        mask_word(words[3]),
    )


def expand_key(
    master_key: Sequence[int],
    rounds: int = SIMON32_64_ROUND_COUNT,
) -> tuple[int, ...]:
    """Expand a SIMON32/64 master key into a sequence of round keys.

    Args:
        master_key: Four 16-bit words in order (k[3], k[2], k[1], k[0]).
        rounds: Number of round keys to return, from 1 through 32.

    Returns:
        Tuple of round keys (k[0], k[1], ..., k[rounds - 1]).
    """
    valid_rounds = _validate_round_count(rounds)
    k3, k2, k1, k0 = _normalise_master_key(master_key)

    keys: list[int] = [k0, k1, k2, k3]

    for i in range(valid_rounds - 4):
        # tmp = (k_{i+3} >>> 3) ^ k_{i+1}
        tmp = rotate_right(keys[i + 3], 3) ^ keys[i + 1]
        # (I ^ S^{-1})(tmp)
        tmp = tmp ^ rotate_right(tmp, 1)
        z_bit = (SIMON32_64_Z0 >> (i % SIMON32_64_Z0_PERIOD)) & 1
        c_i = SIMON32_64_CONSTANT_C ^ z_bit
        k_next = (keys[i] ^ tmp ^ c_i) & WORD_MASK
        keys.append(k_next)

    return tuple(keys[:valid_rounds])
