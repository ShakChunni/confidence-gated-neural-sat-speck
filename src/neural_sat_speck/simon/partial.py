"""Scalar partial operations used by later final-round candidate testing."""

from __future__ import annotations

from collections.abc import Sequence

from .cipher import Block, decrypt_round


def partial_decrypt_final_round(
    ciphertext: Sequence[int],
    final_round_key: int,
) -> Block:
    """Remove one candidate final round from a ciphertext block.

    The operation is the inverse of one encryption round. It does not expand
    the master key and therefore accepts only the candidate final-round key.
    """
    return decrypt_round(ciphertext, final_round_key)
