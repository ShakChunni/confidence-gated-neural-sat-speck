"""Unit tests for SIMON32/64 exact SAT model, solver isolation, and verification."""

from __future__ import annotations

import subprocess

import pytest

from neural_sat_speck.simon.cipher import (
    SIMON32_64_KAT_KEY,
    SIMON32_64_KAT_PLAINTEXT,
    encrypt_block,
)
from neural_sat_speck.simon.key_schedule import expand_key
from neural_sat_speck.simon.sat_model import SimonSatModel
from neural_sat_speck.simon.verification import (
    verify_simon_master_key,
)


@pytest.mark.parametrize("rounds", [1, 2, 4, 6, 8])
def test_simon_sat_model_matches_reference_cipher_on_kat(rounds: int) -> None:
    """SAT model satisfies exact known-answer encryption round by round."""
    pt = SIMON32_64_KAT_PLAINTEXT
    key = SIMON32_64_KAT_KEY
    ct = encrypt_block(pt, key, rounds)
    expected_round_keys = expand_key(key, rounds)

    model = SimonSatModel.from_known_pairs([(pt, ct)], rounds=rounds)
    assumptions = model.assumptions_for_master_key(key)
    result = model.solve(assumptions)

    assert result.satisfiable is True
    assert result.solution is not None

    # Check that all round keys match expected
    decoded_keys = model.round_keys_from_solution(result.solution)
    assert decoded_keys == expected_round_keys

    # Check decoded master key matches
    decoded_master = model.master_key_from_solution(result.solution)
    assert decoded_master == key


def test_simon_sat_exact_master_key_recovery_with_trailing_assumptions() -> None:
    """Exact master key recovery at 7 rounds given trailing subkey assumptions."""
    key = (0x1234, 0x5678, 0x9ABC, 0xDEF0)
    rounds = 7
    # 4 known plaintext/ciphertext pairs
    plaintexts = [(0x1111, 0x2222), (0x3333, 0x4444), (0x5555, 0x6666), (0x7777, 0x8888)]
    known_pairs = [(pt, encrypt_block(pt, key, rounds)) for pt in plaintexts]

    model = SimonSatModel.from_known_pairs(known_pairs, rounds=rounds)
    expected_round_keys = expand_key(key, rounds)

    # Condition on A=2 trailing subkeys: (k5, k6)
    trailing_keys = (expected_round_keys[-2], expected_round_keys[-1])
    assumptions = model.assumptions_for_trailing_round_subkeys(trailing_keys)

    result = model.solve(assumptions, time_limit=10.0)
    assert result.satisfiable is True
    assert result.solution is not None

    rec_key = model.master_key_from_solution(result.solution)
    assert rec_key == key

    # Verification against fresh pairs
    fresh_pairs = [
        ((0xAAAA, 0xBBBB), encrypt_block((0xAAAA, 0xBBBB), key, rounds)),
        ((0xCCCC, 0xDDDD), encrypt_block((0xCCCC, 0xDDDD), key, rounds)),
    ]
    verif = verify_simon_master_key(rec_key, known_pairs, fresh_pairs, rounds)
    assert verif.verified is True
    assert verif.known_pairs_verified is True
    assert verif.independent_pairs_verified is True


def test_solver_isolation_between_models() -> None:
    """Ensure two SimonSatModel instances have independent state and solver objects."""
    key1 = (0x1111, 0x2222, 0x3333, 0x4444)
    key2 = (0x5555, 0x6666, 0x7777, 0x8888)
    pt = (0x0123, 0x4567)

    model1 = SimonSatModel.from_known_pairs([(pt, encrypt_block(pt, key1, 5))], rounds=5)
    model2 = SimonSatModel.from_known_pairs([(pt, encrypt_block(pt, key2, 5))], rounds=5)

    assert model1.solver is not model2.solver
    assert model1.pool is not model2.pool
    assert model1.recorder is not model2.recorder

    # Solving model1 with key1 assumptions does not solve or affect model2
    res1 = model1.solve(model1.assumptions_for_master_key(key1))
    assert res1.satisfiable is True

    # model2 solved with key1 assumptions should be UNSAT (different ciphertexts)
    res2 = model2.solve(model2.assumptions_for_master_key(key1))
    assert res2.satisfiable is False
    assert res2.status == "unsat"


def test_conflicting_assumptions_yield_unsat() -> None:
    """Contradictory assumptions correctly yield UNSAT, not timeout."""
    key = (0x1234, 0x5678, 0x9ABC, 0xDEF0)
    pt = (0x1111, 0x2222)
    ct = encrypt_block(pt, key, 4)

    model = SimonSatModel.from_known_pairs([(pt, ct)], rounds=4)
    # Assume round key 0 is both 0x0000 and 0xFFFF (conflicting assumptions on bit 0)
    k0_var = model.round_keys[0][0]
    conflict_assumptions = (k0_var, -k0_var)

    res = model.solve(conflict_assumptions)
    assert res.satisfiable is False
    assert res.status == "unsat"
    assert res.timed_out is False


def test_timeout_state_classification() -> None:
    """An extremely small time_limit on an unsolved model properly classifies timeout."""
    key = (0x1234, 0x5678, 0x9ABC, 0xDEF0)
    pairs = [((i, i + 1), encrypt_block((i, i + 1), key, 8)) for i in range(4)]
    model = SimonSatModel.from_known_pairs(pairs, rounds=8)

    # 8 rounds unconditioned is a hard instance; 0.0001s cutoff triggers timeout
    res = model.solve(time_limit=0.0001)
    assert res.satisfiable is False
    assert res.timed_out is True or res.status in ("solver_timeout", "unsat_or_timeout")


def test_master_key_verification_fails_closed() -> None:
    """Wrong master key correctly fails verification."""
    true_key = (0x1234, 0x5678, 0x9ABC, 0xDEF0)
    wrong_key = (0x1234, 0x5678, 0x9ABC, 0xDEF1)
    rounds = 6

    known = [((0x1000, 0x2000), encrypt_block((0x1000, 0x2000), true_key, rounds))]
    fresh = [((0x3000, 0x4000), encrypt_block((0x3000, 0x4000), true_key, rounds))]

    verif = verify_simon_master_key(wrong_key, known, fresh, rounds)
    assert verif.verified is False
    assert verif.known_pairs_verified is False or verif.independent_pairs_verified is False


def test_dimacs_export_and_native_cli_solver() -> None:
    """Export to DIMACS produces valid format and native CLI solves correctly."""
    key = (0x2468, 0x1357, 0x9BDF, 0xACE0)
    rounds = 6
    known = [
        ((0x0100, 0x0200), encrypt_block((0x0100, 0x0200), key, rounds)),
        ((0x0300, 0x0400), encrypt_block((0x0300, 0x0400), key, rounds)),
        ((0x0500, 0x0600), encrypt_block((0x0500, 0x0600), key, rounds)),
        ((0x0700, 0x0800), encrypt_block((0x0700, 0x0800), key, rounds)),
    ]
    model = SimonSatModel.from_known_pairs(known, rounds=rounds)

    dimacs_text = model.to_dimacs()
    lines = dimacs_text.splitlines()
    assert lines[0].startswith("p cnf ")
    assert any(line.startswith("x ") for line in lines)

    # Check clause counts
    assert model.cnf_clause_count > 0
    assert model.xor_clause_count > 0
    assert model.variable_count > 0

    # Solve using native cryptominisat5 CLI with Gaussian elimination
    try:
        subprocess.run(["cryptominisat5", "--version"], capture_output=True, check=True)
        has_cms_cli = True
    except (FileNotFoundError, subprocess.CalledProcessError):
        has_cms_cli = False

    if has_cms_cli:
        cli_res = model.solve_native_cli(enable_gauss=True, time_limit=10.0)
        assert cli_res.satisfiable is True
        assert cli_res.solution is not None
        rec_key = model.master_key_from_solution(cli_res.solution)
        assert rec_key == key
