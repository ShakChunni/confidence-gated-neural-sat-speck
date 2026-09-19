"""Tests for scalar word-level SIMON32/64 cipher, key schedule, and primitives."""

from __future__ import annotations

import random

import pytest

from neural_sat_speck.simon.cipher import (
    SIMON32_64_KAT_CIPHERTEXT,
    SIMON32_64_KAT_KEY,
    SIMON32_64_KAT_PLAINTEXT,
    decrypt_block,
    decrypt_round,
    encrypt_block,
    encrypt_round,
)
from neural_sat_speck.simon.key_schedule import (
    SIMON32_64_KEY_WORD_COUNT,
    SIMON32_64_ROUND_COUNT,
    expand_key,
)
from neural_sat_speck.simon.partial import partial_decrypt_final_round
from neural_sat_speck.simon.primitives import (
    WORD_BITS,
    WORD_MASK,
    mask_word,
    rotate_left,
    rotate_right,
    simon_f,
)
from tests.fixtures.simon32_64_vectors import (
    CIPHERTEXT,
    EXPECTED_REDUCED_CIPHERTEXTS,
    EXPECTED_ROUND_KEYS,
    MASTER_KEY,
    PLAINTEXT,
)
from tests.fixtures.simon_reference_oracle import (
    oracle_simon32_64_decrypt,
    oracle_simon32_64_encrypt,
    oracle_simon32_64_key_schedule,
)


def test_one_round_encryption_matches_regression_state() -> None:
    assert encrypt_round(PLAINTEXT, EXPECTED_ROUND_KEYS[0]) == (
        0xBCA2,
        0x6565,
    )


def test_one_round_decryption_is_the_inverse() -> None:
    state = (0xBCA2, 0x6565)
    assert decrypt_round(state, EXPECTED_ROUND_KEYS[0]) == PLAINTEXT


def test_full_known_answer_vector() -> None:
    assert encrypt_block(PLAINTEXT, MASTER_KEY) == CIPHERTEXT
    assert decrypt_block(CIPHERTEXT, MASTER_KEY) == PLAINTEXT
    assert encrypt_block(SIMON32_64_KAT_PLAINTEXT, SIMON32_64_KAT_KEY) == SIMON32_64_KAT_CIPHERTEXT
    assert decrypt_block(SIMON32_64_KAT_CIPHERTEXT, SIMON32_64_KAT_KEY) == SIMON32_64_KAT_PLAINTEXT


@pytest.mark.parametrize(("rounds", "expected"), EXPECTED_REDUCED_CIPHERTEXTS)
def test_reduced_round_outputs_use_the_round_key_prefix(
    rounds: int,
    expected: tuple[int, int],
) -> None:
    assert encrypt_block(PLAINTEXT, MASTER_KEY, rounds) == expected
    assert decrypt_block(expected, MASTER_KEY, rounds) == PLAINTEXT


def test_key_schedule_matches_all_expected_round_keys() -> None:
    expanded = expand_key(MASTER_KEY, SIMON32_64_ROUND_COUNT)
    assert expanded == EXPECTED_ROUND_KEYS


@pytest.mark.parametrize("rounds", range(1, SIMON32_64_ROUND_COUNT + 1))
def test_key_schedule_prefix_consistency(rounds: int) -> None:
    assert expand_key(MASTER_KEY, rounds) == EXPECTED_ROUND_KEYS[:rounds]


def test_partial_decrypt_final_round_matches_prior_round() -> None:
    c7 = encrypt_block(PLAINTEXT, MASTER_KEY, 7)
    c6 = encrypt_block(PLAINTEXT, MASTER_KEY, 6)
    k6 = EXPECTED_ROUND_KEYS[6]
    assert partial_decrypt_final_round(c7, k6) == c6


def test_deterministic_random_round_trips() -> None:
    rng = random.Random(0x6002)
    for _ in range(32):
        block = (rng.randrange(1 << 16), rng.randrange(1 << 16))
        master_key = tuple(rng.randrange(1 << 16) for _ in range(4))
        for rounds in (1, 2, 6, 7, 16, 32):
            ciphertext = encrypt_block(block, master_key, rounds)
            assert decrypt_block(ciphertext, master_key, rounds) == block


def test_independent_reference_oracle_kat() -> None:
    """The clean-room independent reference oracle must reproduce the official KAT independently."""
    assert oracle_simon32_64_key_schedule(MASTER_KEY, 32) == EXPECTED_ROUND_KEYS
    assert oracle_simon32_64_encrypt(PLAINTEXT, MASTER_KEY, 32) == CIPHERTEXT
    assert oracle_simon32_64_decrypt(CIPHERTEXT, MASTER_KEY, 32) == PLAINTEXT


def test_independent_reference_oracle_differential_random_comparison() -> None:
    """Production SIMON32/64 implementation must match clean-room reference oracle bitwise."""
    rng = random.Random(0x1337)
    for _ in range(32):
        block = (rng.randrange(1 << 16), rng.randrange(1 << 16))
        master_key = tuple(rng.randrange(1 << 16) for _ in range(4))
        for rounds in (1, 2, 4, 7, 16, 32):
            # Key expansion equivalence
            assert expand_key(master_key, rounds) == oracle_simon32_64_key_schedule(master_key, rounds)

            # Encryption equivalence
            prod_ct = encrypt_block(block, master_key, rounds)
            oracle_ct = oracle_simon32_64_encrypt(block, master_key, rounds)
            assert prod_ct == oracle_ct

            # Decryption equivalence
            prod_pt = decrypt_block(prod_ct, master_key, rounds)
            oracle_pt = oracle_simon32_64_decrypt(oracle_ct, master_key, rounds)
            assert prod_pt == oracle_pt == block


def test_round_and_block_inputs_are_explicitly_masked() -> None:
    wide_block = (PLAINTEXT[0] + (1 << 16), PLAINTEXT[1] - (1 << 16))
    wide_key = EXPECTED_ROUND_KEYS[0] + (1 << 16)

    assert encrypt_round(wide_block, wide_key) == encrypt_round(
        PLAINTEXT,
        EXPECTED_ROUND_KEYS[0],
    )
    assert encrypt_block(
        wide_block,
        (MASTER_KEY[0] + (1 << 16), *MASTER_KEY[1:]),
    ) == encrypt_block(PLAINTEXT, MASTER_KEY)


@pytest.mark.parametrize("rounds", [0, -1, 33, 100])
@pytest.mark.parametrize("function", [encrypt_block, decrypt_block])
def test_invalid_round_counts_are_rejected(function, rounds: int) -> None:
    with pytest.raises(ValueError):
        function(PLAINTEXT, MASTER_KEY, rounds)


@pytest.mark.parametrize("rounds", [1.0, True, "32", None])
@pytest.mark.parametrize("function", [encrypt_block, decrypt_block])
def test_non_integer_round_counts_are_rejected(function, rounds) -> None:
    with pytest.raises(TypeError):
        function(PLAINTEXT, MASTER_KEY, rounds)


def test_malformed_round_state_is_rejected() -> None:
    with pytest.raises(ValueError):
        encrypt_round((0x0000,), EXPECTED_ROUND_KEYS[0])
    with pytest.raises(ValueError):
        encrypt_round((0x0000, 0x0000, 0x0000), EXPECTED_ROUND_KEYS[0])
    with pytest.raises(TypeError):
        encrypt_round(0x0000, EXPECTED_ROUND_KEYS[0])
    with pytest.raises(TypeError):
        encrypt_round((True, 0x0000), EXPECTED_ROUND_KEYS[0])
    with pytest.raises(TypeError):
        encrypt_round(PLAINTEXT, True)


def test_key_schedule_validation() -> None:
    assert SIMON32_64_KEY_WORD_COUNT == 4
    with pytest.raises(ValueError):
        expand_key((0x0100, 0x0200, 0x0300), 32)
    with pytest.raises(ValueError):
        expand_key((0x0100, 0x0200, 0x0300, 0x0400, 0x0500), 32)
    with pytest.raises(TypeError):
        expand_key((0x0100, 0x0200, 0x0300, True), 32)
    with pytest.raises(TypeError):
        expand_key(12345, 32)
    with pytest.raises(ValueError):
        expand_key(MASTER_KEY, 0)
    with pytest.raises(ValueError):
        expand_key(MASTER_KEY, 33)
    with pytest.raises(TypeError):
        expand_key(MASTER_KEY, True)


def test_primitives_properties() -> None:
    assert WORD_BITS == 16
    assert WORD_MASK == 0xFFFF
    assert mask_word(0x12345) == 0x2345
    assert mask_word(-1) == 0xFFFF
    with pytest.raises(TypeError):
        mask_word(True)
    with pytest.raises(TypeError):
        mask_word(3.14)

    # Rotation invariants
    x = 0x1234
    assert rotate_left(x, 0) == x
    assert rotate_left(x, 16) == x
    assert rotate_left(x, 32) == x
    assert rotate_right(x, 0) == x
    assert rotate_right(x, 16) == x
    assert rotate_right(rotate_left(x, 5), 5) == x
    assert rotate_left(rotate_right(x, 7), 7) == x

    # Feistel branch function invariants
    assert simon_f(0) == 0
    with pytest.raises(TypeError):
        simon_f(False)
    with pytest.raises(TypeError):
        simon_f("invalid")
