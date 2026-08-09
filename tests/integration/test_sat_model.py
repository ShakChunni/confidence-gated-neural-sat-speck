"""Integration tests for the exact CryptoMiniSat SPECK model."""

import pytest

from neural_sat_speck.sat.model import SpeckSatModel
from neural_sat_speck.speck.cipher import encrypt_block, encrypt_round
from neural_sat_speck.speck.key_schedule import expand_key
from tests.fixtures.speck32_64_vectors import MASTER_KEY

pytest.importorskip("pycryptosat")


PLAINTEXTS = (
    (0x6574, 0x694C),
    (0x0000, 0x0000),
    (0x1234, 0xABCD),
    (0xDEAD, 0xBEEF),
)


def _known_pairs(rounds: int) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    return tuple(
        (plaintext, encrypt_block(plaintext, MASTER_KEY, rounds))
        for plaintext in PLAINTEXTS
    )


def test_fixed_key_model_states_match_the_python_reference() -> None:
    rounds = 4
    pairs = _known_pairs(rounds)
    model = SpeckSatModel.from_known_pairs(pairs, rounds)
    result = model.solve(
        model.assumptions_for_master_key(MASTER_KEY),
        time_limit=5.0,
    )

    assert result.satisfiable
    assert not result.timed_out
    assert result.solution is not None
    round_keys = expand_key(MASTER_KEY, rounds)
    for pair_index, (plaintext, _) in enumerate(pairs):
        state = plaintext
        assert model.state_words(result.solution, pair_index, 0) == state
        for round_index, round_key in enumerate(round_keys, start=1):
            state = encrypt_round(state, round_key)
            assert model.state_words(
                result.solution,
                pair_index,
                round_index,
            ) == state


def test_unknown_key_recovery_produces_a_verifiable_candidate() -> None:
    rounds = 4
    pairs = _known_pairs(rounds)
    model = SpeckSatModel.from_known_pairs(pairs, rounds)
    result = model.solve(time_limit=10.0)

    assert result.satisfiable
    assert not result.timed_out
    assert result.solution is not None
    recovered_key = model.master_key_from_solution(result.solution)
    assert all(
        encrypt_block(plaintext, recovered_key, rounds) == ciphertext
        for plaintext, ciphertext in pairs
    )


def test_modified_ciphertext_is_unsatisfiable_under_the_true_key() -> None:
    rounds = 4
    pairs = _known_pairs(rounds)
    plaintext, ciphertext = pairs[0]
    modified_pairs = ((plaintext, (ciphertext[0] ^ 1, ciphertext[1])),)
    model = SpeckSatModel.from_known_pairs(modified_pairs, rounds)

    result = model.solve(
        model.assumptions_for_master_key(MASTER_KEY),
        time_limit=5.0,
    )
    assert not result.satisfiable
    assert not result.timed_out


def test_blocking_clause_exposes_multiple_solutions_when_key_is_underdetermined() -> None:
    rounds = 1
    pairs = _known_pairs(rounds)[:1]
    model = SpeckSatModel.from_known_pairs(pairs, rounds)

    first = model.solve(time_limit=5.0)
    assert first.satisfiable
    assert not first.timed_out
    assert first.solution is not None
    model.add_blocking_clause(first.solution)

    second = model.solve(time_limit=5.0)
    assert second.satisfiable
    assert not second.timed_out
    assert second.solution is not None
