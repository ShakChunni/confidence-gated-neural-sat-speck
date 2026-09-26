"""Independent scalar, oracle, and published-layout checks for SIMON features."""

from __future__ import annotations

import random

import numpy as np
import pytest

from neural_sat_speck.simon.cipher import (
    SIMON32_64_KAT_KEY,
    decrypt_round,
    encrypt_block,
)
from neural_sat_speck.simon.key_schedule import expand_key
from neural_sat_speck.simon.ranking_features import (
    bao_full_value_bits,
    bao_value_difference_bits,
    candidate_difference_words,
    candidate_transformed_words,
    candidate_value_words,
)
from tests.fixtures.simon_reference_oracle import (
    oracle_simon32_64_decrypt,
    oracle_simon32_64_encrypt,
)


def _direct_ciphertext_pairs(count: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    rng = random.Random(260926)
    pairs = []
    for _ in range(count):
        plain_a = (rng.randrange(1 << 16), rng.randrange(1 << 16))
        plain_b = (plain_a[0], plain_a[1] ^ 0x0040)
        pairs.append(
            (
                encrypt_block(plain_a, SIMON32_64_KAT_KEY, 9),
                encrypt_block(plain_b, SIMON32_64_KAT_KEY, 9),
            )
        )
    return tuple(pairs)


def test_source_word_and_msb_bit_order() -> None:
    # Bao key_rank.py uses [c0a, c0b, c1a xor c1b] after the zero inverse;
    # simon.py::convert_to_binary emits bits 15 down to 0 in each word.
    states = np.array([[[0x8001, 0x1234, 0x4002, 0x5678]]], dtype=np.uint16)
    actual = bao_value_difference_bits(states)
    expected_words = (0x8001, 0x4002, 0x1234 ^ 0x5678)
    expected = np.array(
        [[[list(map(int, f"{word:016b}")) for word in expected_words]]],
        dtype=np.uint8,
    )
    assert actual.shape == (1, 1, 3, 16)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        candidate_difference_words(states),
        np.array([[[0x8001 ^ 0x4002, 0x1234 ^ 0x5678]]], dtype=np.uint16),
    )
    # Bao simon.py::make_train_data uses four full words in this order for VV.
    expected_vv = np.array(
        [[[list(map(int, f"{word:016b}")) for word in states[0, 0]]]],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(bao_full_value_bits(states), expected_vv)


@pytest.mark.parametrize("final_key_candidate", [None, 0, 0xFFFF])
def test_vectorized_candidate_states_match_scalar_and_independent_oracle(
    final_key_candidate: int | None,
) -> None:
    ciphertext_pairs = _direct_ciphertext_pairs(3)
    candidates = (0, 1, 0x1234, 0xFFFF)
    value_states = candidate_value_words(
        ciphertext_pairs, candidates, final_key_candidate=final_key_candidate
    )
    states = candidate_transformed_words(
        ciphertext_pairs, candidates, final_key_candidate=final_key_candidate
    )
    assert states.shape == (len(candidates), len(ciphertext_pairs), 4)
    assert states.dtype == np.uint16

    for key_index, candidate in enumerate(candidates):
        for pair_index, pair in enumerate(ciphertext_pairs):
            expected_blocks = []
            for ciphertext in pair:
                starting_state = (
                    decrypt_round(ciphertext, final_key_candidate)
                    if final_key_candidate is not None
                    else ciphertext
                )
                scalar = decrypt_round(decrypt_round(starting_state, candidate), 0)

                oracle_state = (
                    oracle_simon32_64_decrypt(
                        ciphertext, (0, 0, 0, final_key_candidate), 1
                    )
                    if final_key_candidate is not None
                    else ciphertext
                )
                oracle_once = oracle_simon32_64_decrypt(
                    oracle_state, (0, 0, 0, candidate), 1
                )
                oracle = oracle_simon32_64_decrypt(
                    oracle_once,
                    (0, 0, 0, 0),
                    1,
                )
                assert scalar == oracle
                expected_blocks.append(scalar)
            assert tuple(int(word) for word in value_states[key_index, pair_index]) == (
                *oracle_simon32_64_decrypt(
                    oracle_simon32_64_decrypt(pair[0], (0, 0, 0, final_key_candidate), 1)
                    if final_key_candidate is not None
                    else pair[0],
                    (0, 0, 0, candidate), 1,
                ),
                *oracle_simon32_64_decrypt(
                    oracle_simon32_64_decrypt(pair[1], (0, 0, 0, final_key_candidate), 1)
                    if final_key_candidate is not None
                    else pair[1],
                    (0, 0, 0, candidate), 1,
                ),
            )
            expected = (*expected_blocks[0], *expected_blocks[1])
            assert tuple(int(word) for word in states[key_index, pair_index]) == expected

            left_a, right_a = expected_blocks[0]
            left_b, right_b = expected_blocks[1]
            assert tuple(
                int(word) for word in candidate_difference_words(states)[key_index, pair_index]
            ) == (left_a ^ left_b, right_a ^ right_b)
            source_words = (left_a, left_b, right_a ^ right_b)
            expected_bits = tuple(
                tuple(int(bit) for bit in f"{word:016b}") for word in source_words
            )
            assert tuple(
                tuple(int(bit) for bit in plane)
                for plane in bao_value_difference_bits(states)[key_index, pair_index]
            ) == expected_bits


def test_correct_keys_have_independent_seven_and_six_round_differences() -> None:
    rng = random.Random(60926)
    keys = expand_key(SIMON32_64_KAT_KEY, 9)
    pairs = []
    plaintext_pairs = []
    for _ in range(4):
        a = (rng.randrange(1 << 16), rng.randrange(1 << 16))
        b = (a[0], a[1] ^ 0x0040)
        plaintext_pairs.append((a, b))
        pairs.append(tuple(oracle_simon32_64_encrypt(p, SIMON32_64_KAT_KEY, 9) for p in (a, b)))

    for candidate, final_candidate, depth in (
        (keys[8], None, 7),
        (keys[7], keys[8], 6),
    ):
        states = candidate_transformed_words(
            pairs, (candidate,), final_key_candidate=final_candidate
        )
        value_states = candidate_value_words(
            pairs, (candidate,), final_key_candidate=final_candidate
        )
        differences = candidate_difference_words(states)[0]
        for index, plaintext_pair in enumerate(plaintext_pairs):
            value_a, value_b = (
                oracle_simon32_64_encrypt(p, SIMON32_64_KAT_KEY, depth + 1)
                for p in plaintext_pair
            )
            assert tuple(int(word) for word in value_states[0, index]) == (
                *value_a, *value_b
            )
            a, b = (
                oracle_simon32_64_encrypt(p, SIMON32_64_KAT_KEY, depth)
                for p in plaintext_pair
            )
            assert tuple(int(word) for word in differences[index]) == (
                a[0] ^ b[0], a[1] ^ b[1]
            )


def test_direct_difference_expression_and_candidate_collision() -> None:
    pair = _direct_ciphertext_pairs(1)[0]
    states = candidate_transformed_words((pair,), tuple(range(1 << 16)))
    differences = candidate_difference_words(states)[:, 0, :]
    # The left difference is candidate-invariant. With one direct pair the
    # right difference is affine in the 16-bit key and may collide exactly.
    assert np.all(differences[:, 0] == differences[0, 0])
    assert len(np.unique(differences[:, 1])) < 1 << 16
    assert len(np.unique(differences[:, 1])) > 1

    def f(value: int) -> int:
        def rot(n: int) -> int:
            return ((value << n) | (value >> (16 - n))) & 0xFFFF

        return (rot(1) & rot(8)) ^ rot(2)

    (u_a, r_a), (u_b, r_b) = pair
    v_a = u_a ^ f(r_a)
    v_b = u_b ^ f(r_b)
    for key in (0, 1, 0x1234, 0xFFFF):
        assert tuple(int(word) for word in differences[key]) == (
            v_a ^ v_b,
            r_a ^ r_b ^ f(v_a ^ key) ^ f(v_b ^ key),
        )


@pytest.mark.parametrize(
    ("pairs", "candidates", "final_key", "error"),
    [
        ((), (0,), None, ValueError),
        (((0, 0), (1, 1)), (), None, ValueError),
        ((((0, 0), (1, 1)),), (0x10000,), None, ValueError),
        ((((0, 0), (1, 1)),), (True,), None, TypeError),
        ((((0, 0), (1, 1)),), (0,), -1, ValueError),
        ((((0, 0), (1,)),), (0,), None, ValueError),
    ],
)
def test_malformed_candidate_feature_inputs_are_rejected(
    pairs, candidates, final_key, error
) -> None:
    with pytest.raises(error):
        candidate_transformed_words(
            pairs, candidates, final_key_candidate=final_key
        )
