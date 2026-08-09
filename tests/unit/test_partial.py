"""Tests for scalar final-round partial decryption."""

import random

from neural_sat_speck.speck.cipher import encrypt_block, encrypt_round
from neural_sat_speck.speck.partial import partial_decrypt_final_round
from tests.fixtures.speck32_64_vectors import (
    CIPHERTEXT,
    EXPECTED_ROUND_KEYS,
    MASTER_KEY,
    PLAINTEXT,
)


def test_true_final_round_key_recovers_the_pre_final_state() -> None:
    pre_final_state = encrypt_block(PLAINTEXT, MASTER_KEY, rounds=21)
    ciphertext = encrypt_round(pre_final_state, EXPECTED_ROUND_KEYS[-1])

    assert ciphertext == CIPHERTEXT
    assert (
        partial_decrypt_final_round(ciphertext, EXPECTED_ROUND_KEYS[-1])
        == pre_final_state
    )


def test_wrong_final_round_key_does_not_recover_the_pre_final_state() -> None:
    pre_final_state = encrypt_block(PLAINTEXT, MASTER_KEY, rounds=21)

    assert (
        partial_decrypt_final_round(CIPHERTEXT, EXPECTED_ROUND_KEYS[-1] ^ 1)
        != pre_final_state
    )


def test_partial_decryption_matches_inverse_round_for_seeded_states() -> None:
    rng = random.Random(0x6002)
    for _ in range(16):
        state = (rng.randrange(1 << 16), rng.randrange(1 << 16))
        round_key = rng.randrange(1 << 16)
        ciphertext = encrypt_round(state, round_key)
        assert partial_decrypt_final_round(ciphertext, round_key) == state
