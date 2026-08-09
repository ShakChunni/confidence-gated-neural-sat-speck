"""Tests for the scalar word-level SPECK32/64 cipher."""

import random

import pytest

from neural_sat_speck.speck.cipher import (
    decrypt_block,
    decrypt_round,
    encrypt_block,
    encrypt_round,
)
from tests.fixtures.speck32_64_vectors import (
    CIPHERTEXT,
    EXPECTED_REDUCED_CIPHERTEXTS,
    EXPECTED_ROUND_KEYS,
    MASTER_KEY,
    PLAINTEXT,
)


def test_one_round_encryption_matches_regression_state() -> None:
    assert encrypt_round(PLAINTEXT, EXPECTED_ROUND_KEYS[0]) == (
        0x5316,
        0xF627,
    )


def test_one_round_decryption_is_the_inverse() -> None:
    state = (0x5316, 0xF627)
    assert decrypt_round(state, EXPECTED_ROUND_KEYS[0]) == PLAINTEXT


def test_full_known_answer_vector() -> None:
    assert encrypt_block(PLAINTEXT, MASTER_KEY) == CIPHERTEXT
    assert decrypt_block(CIPHERTEXT, MASTER_KEY) == PLAINTEXT


@pytest.mark.parametrize(("rounds", "expected"), EXPECTED_REDUCED_CIPHERTEXTS)
def test_reduced_round_outputs_use_the_round_key_prefix(
    rounds: int,
    expected: tuple[int, int],
) -> None:
    assert encrypt_block(PLAINTEXT, MASTER_KEY, rounds) == expected
    assert decrypt_block(expected, MASTER_KEY, rounds) == PLAINTEXT


def test_deterministic_random_round_trips() -> None:
    rng = random.Random(0x6002)
    for _ in range(16):
        block = tuple(rng.randrange(1 << 16) for _ in range(2))
        master_key = tuple(rng.randrange(1 << 16) for _ in range(4))
        for rounds in (1, 2, 7, 11, 22):
            ciphertext = encrypt_block(block, master_key, rounds)
            assert decrypt_block(ciphertext, master_key, rounds) == block


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


@pytest.mark.parametrize("rounds", [0, -1, 23])
@pytest.mark.parametrize("function", [encrypt_block, decrypt_block])
def test_invalid_round_counts_are_rejected(function, rounds: int) -> None:
    with pytest.raises(ValueError):
        function(PLAINTEXT, MASTER_KEY, rounds)


@pytest.mark.parametrize("rounds", [1.0, True])
@pytest.mark.parametrize("function", [encrypt_block, decrypt_block])
def test_non_integer_round_counts_are_rejected(function, rounds) -> None:
    with pytest.raises(TypeError):
        function(PLAINTEXT, MASTER_KEY, rounds)


def test_malformed_round_state_is_rejected() -> None:
    with pytest.raises(ValueError):
        encrypt_round((0x0000,), EXPECTED_ROUND_KEYS[0])
    with pytest.raises(TypeError):
        encrypt_round(0x0000, EXPECTED_ROUND_KEYS[0])
    with pytest.raises(TypeError):
        encrypt_round((True, 0x0000), EXPECTED_ROUND_KEYS[0])
    with pytest.raises(TypeError):
        encrypt_round(PLAINTEXT, True)
