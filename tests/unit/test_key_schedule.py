"""Tests for the scalar SPECK32/64 key schedule."""

import pytest

from neural_sat_speck.speck.key_schedule import (
    SPECK32_64_ALPHA,
    SPECK32_64_BETA,
    SPECK32_64_KEY_WORD_COUNT,
    SPECK32_64_ROUND_COUNT,
    expand_key,
)
from neural_sat_speck.speck.primitives import WORD_MASK
from tests.fixtures.speck32_64_vectors import (
    EXPECTED_ROUND_KEYS,
    MASTER_KEY,
)


def test_full_key_schedule_matches_regression_values() -> None:
    assert expand_key(MASTER_KEY) == EXPECTED_ROUND_KEYS


@pytest.mark.parametrize("rounds", [1, 2, 7, 11, SPECK32_64_ROUND_COUNT])
def test_reduced_key_schedule_is_a_prefix(rounds: int) -> None:
    assert expand_key(MASTER_KEY, rounds) == EXPECTED_ROUND_KEYS[:rounds]


def test_key_schedule_is_deterministic() -> None:
    assert expand_key(MASTER_KEY) == expand_key(list(MASTER_KEY))


def test_key_schedule_outputs_16_bit_words() -> None:
    round_keys = expand_key(MASTER_KEY)
    assert len(round_keys) == SPECK32_64_ROUND_COUNT
    assert all(0 <= word <= WORD_MASK for word in round_keys)


def test_key_schedule_constants_match_verified_parameters() -> None:
    assert SPECK32_64_KEY_WORD_COUNT == 4
    assert SPECK32_64_ROUND_COUNT == 22
    assert SPECK32_64_ALPHA == 7
    assert SPECK32_64_BETA == 2


def test_master_key_words_are_explicitly_masked() -> None:
    wide_key = (MASTER_KEY[0] + 0x10000, *MASTER_KEY[1:])
    assert expand_key(wide_key) == EXPECTED_ROUND_KEYS


@pytest.mark.parametrize("rounds", [0, -1, SPECK32_64_ROUND_COUNT + 1])
def test_invalid_round_counts_are_rejected(rounds: int) -> None:
    with pytest.raises(ValueError):
        expand_key(MASTER_KEY, rounds)


def test_non_integer_round_count_is_rejected() -> None:
    with pytest.raises(TypeError):
        expand_key(MASTER_KEY, 1.0)


def test_boolean_round_count_is_rejected() -> None:
    with pytest.raises(TypeError):
        expand_key(MASTER_KEY, True)


@pytest.mark.parametrize(
    "master_key",
    [
        (),
        MASTER_KEY[:3],
        MASTER_KEY + (0x0000,),
    ],
)
def test_wrong_master_key_length_is_rejected(master_key: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        expand_key(master_key)


def test_non_integer_master_key_word_is_rejected() -> None:
    invalid_key = (MASTER_KEY[0], 1.0, MASTER_KEY[2], MASTER_KEY[3])
    with pytest.raises(TypeError):
        expand_key(invalid_key)
