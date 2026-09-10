"""Instance descent for hierarchical multi-round-key ranking.

A ranker that targets a single final-round subkey cannot reach the residual
depths the exact solver needs at seven rounds or more (see
:mod:`neural_sat_speck.attacks.multi_round_recovery`). The scaling architecture
instead ranks one trailing round key at a time: fix a candidate for the last
round key, partially decrypt the ciphertexts, and let the next stage rank the
round key below it on the resulting shorter instance.

This module provides the descent primitive only. It adds no neural decision
logic, and the ground-truth ``final_round_subkey`` it stores is used exclusively
by post-hoc evaluation helpers, exactly as for the top-level attack instances.
"""

from __future__ import annotations

from neural_sat_speck.data.attack import AttackInstance, AttackPair
from neural_sat_speck.speck.cipher import decrypt_round
from neural_sat_speck.speck.primitives import WORD_MASK


def partial_decrypt_instance(
    instance: AttackInstance,
    round_key: int,
    *,
    expected_next_subkey: int,
    label: str | None = None,
) -> AttackInstance:
    """Return the instance reached by decrypting one round under ``round_key``.

    Every ciphertext is decrypted by exactly one round using the candidate round
    key. ``expected_next_subkey`` is the round key below it, stored for post-hoc
    evaluation only and never consulted while ranking. The returned instance
    keeps the same master key, plaintexts, and pair count, so a pair-group ranker
    can score the next stage on it unchanged.
    """

    if not isinstance(instance, AttackInstance):
        raise TypeError("instance must be an AttackInstance")
    if isinstance(round_key, bool) or not isinstance(round_key, int):
        raise TypeError("round_key must be an integer")
    if not 0 <= round_key <= WORD_MASK:
        raise ValueError("round_key must be a 16-bit word")
    if isinstance(expected_next_subkey, bool) or not isinstance(
        expected_next_subkey, int
    ):
        raise TypeError("expected_next_subkey must be an integer")
    if not 0 <= expected_next_subkey <= WORD_MASK:
        raise ValueError("expected_next_subkey must be a 16-bit word")

    pairs = tuple(
        AttackPair(
            plaintext0=pair.plaintext0,
            plaintext1=pair.plaintext1,
            ciphertext0=decrypt_round(pair.ciphertext0, round_key),
            ciphertext1=decrypt_round(pair.ciphertext1, round_key),
        )
        for pair in instance.pairs
    )
    return AttackInstance(
        instance_id=label if label is not None else f"{instance.instance_id}-descended",
        master_key=instance.master_key,
        final_round_subkey=expected_next_subkey,
        pairs=pairs,
    )
