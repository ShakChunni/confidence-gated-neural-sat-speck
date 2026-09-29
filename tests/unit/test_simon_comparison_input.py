"""Unit tests for SimonComparisonInput, input validator, and random control streams.

Tests cover:
- Strict validation: R=10, exactly 32 attack pairs, 64 known observations, 2 verification pairs.
- Malformed inputs (invalid types, empty strings, non-integers).
- Mismatched inputs (rounds != 10).
- Shortened and lengthened inputs (<32 or >32 attack pairs, <2 or >2 verification pairs).
- Duplicate plaintext detection (within attack, within verification, and cross-set).
- Chronological SAT assumptions validation (k8, k9).
- Strict isolation of verification observations from rankers.
- Deterministic random control stream derivation.
"""

from __future__ import annotations

import pytest

from neural_sat_speck.simon.attack_instance import (
    SimonAttackInstanceConfig,
    SimonAttackPair,
    generate_simon_attack_instances,
)
from neural_sat_speck.simon.comparison_input import (
    EXPECTED_ATTACK_PAIR_COUNT,
    EXPECTED_COMPARISON_ROUNDS,
    EXPECTED_KNOWN_OBSERVATION_COUNT,
    EXPECTED_VERIFICATION_PAIR_COUNT,
    SimonComparisonInput,
    derive_random_control_seed,
)
from neural_sat_speck.simon.key_schedule import expand_key
from neural_sat_speck.simon.method_worker import run_method_worker
from neural_sat_speck.simon.sat_model import SatSolveResult, SimonSatModel


@pytest.fixture(scope="module")
def valid_demo_instance():
    cfg = SimonAttackInstanceConfig(
        instance_count=2,
        pair_count=32,
        rounds=10,
        seed=999999,
        split_role="demo",
        verification_pair_count=2,
    )
    return generate_simon_attack_instances(cfg)[1]


@pytest.fixture(scope="module")
def valid_comparison_input(valid_demo_instance):
    return SimonComparisonInput.from_attack_instance(
        instance=valid_demo_instance,
        manifest_seed=999999,
    )


# 1. Valid Input Creation and Properties
def test_valid_comparison_input_properties(valid_comparison_input):
    inp = valid_comparison_input
    assert inp.rounds == EXPECTED_COMPARISON_ROUNDS
    assert len(inp.attack_pairs) == EXPECTED_ATTACK_PAIR_COUNT
    assert len(inp.known_pairs) == EXPECTED_KNOWN_OBSERVATION_COUNT
    assert len(inp.verification_pairs) == EXPECTED_VERIFICATION_PAIR_COUNT
    assert len(inp.ciphertext_pairs) == EXPECTED_ATTACK_PAIR_COUNT
    assert len(inp.candidate_sat_pairs) == 4
    assert len(inp.all_sat_pairs) == 64


def test_wrong_attack_input_difference_rejected(valid_comparison_input):
    original = valid_comparison_input.attack_pairs[0]
    bad_pair = SimonAttackPair(
        plaintext0=original.plaintext0,
        plaintext1=(original.plaintext1[0], original.plaintext1[1] ^ 1),
        ciphertext0=original.ciphertext0,
        ciphertext1=original.ciphertext1,
    )
    tampered = list(valid_comparison_input.attack_pairs)
    tampered[0] = bad_pair
    with pytest.raises(ValueError, match="input difference"):
        SimonComparisonInput(
            instance_id=valid_comparison_input.instance_id,
            rounds=10,
            attack_pairs=tuple(tampered),
            verification_pairs=valid_comparison_input.verification_pairs,
            manifest_seed=999999,
            pair_count=32,
        )


def test_comparison_input_serialization_round_trip(valid_comparison_input):
    d = valid_comparison_input.to_dict()
    restored = SimonComparisonInput.from_dict(d)
    assert restored.instance_id == valid_comparison_input.instance_id
    assert restored.rounds == valid_comparison_input.rounds
    assert restored.manifest_seed == valid_comparison_input.manifest_seed
    assert restored.attack_pairs == valid_comparison_input.attack_pairs
    assert restored.verification_pairs == valid_comparison_input.verification_pairs


# 2. Malformed Inputs
@pytest.mark.parametrize("bad_id", ["", "   ", None, 12345])
def test_malformed_instance_id_rejected(valid_comparison_input, bad_id):
    with pytest.raises((ValueError, TypeError)):
        SimonComparisonInput(
            instance_id=bad_id,  # type: ignore
            rounds=10,
            attack_pairs=valid_comparison_input.attack_pairs,
            verification_pairs=valid_comparison_input.verification_pairs,
            manifest_seed=999999,
        )


@pytest.mark.parametrize("bad_rounds", ["10", 10.0, True, False, None])
def test_malformed_rounds_rejected(valid_comparison_input, bad_rounds):
    with pytest.raises(TypeError):
        SimonComparisonInput(
            instance_id="simon_test_inst",
            rounds=bad_rounds,  # type: ignore
            attack_pairs=valid_comparison_input.attack_pairs,
            verification_pairs=valid_comparison_input.verification_pairs,
            manifest_seed=999999,
        )


@pytest.mark.parametrize("bad_seed", ["999999", 999999.0, True, False, None])
def test_malformed_manifest_seed_rejected(valid_comparison_input, bad_seed):
    with pytest.raises(TypeError):
        SimonComparisonInput(
            instance_id="simon_test_inst",
            rounds=10,
            attack_pairs=valid_comparison_input.attack_pairs,
            verification_pairs=valid_comparison_input.verification_pairs,
            manifest_seed=bad_seed,  # type: ignore
        )


# 3. Mismatched Inputs
@pytest.mark.parametrize("mismatched_rounds", [0, 8, 9, 11, 16, -1])
def test_mismatched_rounds_rejected(valid_comparison_input, mismatched_rounds):
    with pytest.raises(ValueError, match="Rounds must be exactly 10"):
        SimonComparisonInput(
            instance_id="simon_test_inst",
            rounds=mismatched_rounds,
            attack_pairs=valid_comparison_input.attack_pairs,
            verification_pairs=valid_comparison_input.verification_pairs,
            manifest_seed=999999,
        )


# 4. Shortened and Lengthened Inputs
def test_shortened_attack_pairs_rejected(valid_comparison_input):
    short_pairs = valid_comparison_input.attack_pairs[:31]
    with pytest.raises(ValueError, match="Expected exactly 32 attack pairs"):
        SimonComparisonInput(
            instance_id="simon_test_inst",
            rounds=10,
            attack_pairs=short_pairs,
            verification_pairs=valid_comparison_input.verification_pairs,
            manifest_seed=999999,
        )


def test_lengthened_attack_pairs_rejected(valid_comparison_input):
    long_pairs = valid_comparison_input.attack_pairs + (valid_comparison_input.attack_pairs[0],)
    with pytest.raises(ValueError, match="Expected exactly 32 attack pairs|Duplicate plaintexts"):
        SimonComparisonInput(
            instance_id="simon_test_inst",
            rounds=10,
            attack_pairs=long_pairs,
            verification_pairs=valid_comparison_input.verification_pairs,
            manifest_seed=999999,
        )


def test_shortened_verification_pairs_rejected(valid_comparison_input):
    short_verif = valid_comparison_input.verification_pairs[:1]
    with pytest.raises(ValueError, match="Expected exactly 2 verification pairs"):
        SimonComparisonInput(
            instance_id="simon_test_inst",
            rounds=10,
            attack_pairs=valid_comparison_input.attack_pairs,
            verification_pairs=short_verif,
            manifest_seed=999999,
        )


def test_lengthened_verification_pairs_rejected(valid_comparison_input):
    long_verif = valid_comparison_input.verification_pairs + (valid_comparison_input.verification_pairs[0],)
    with pytest.raises(ValueError, match="Expected exactly 2 verification pairs"):
        SimonComparisonInput(
            instance_id="simon_test_inst",
            rounds=10,
            attack_pairs=valid_comparison_input.attack_pairs,
            verification_pairs=long_verif,
            manifest_seed=999999,
        )


# 5. Plaintext Duplicate Collision Tests
def test_duplicate_plaintexts_within_attack_pairs_rejected(valid_comparison_input):
    # Construct attack pairs with a duplicate plaintext in pair 1
    dup_pair = SimonAttackPair(
        plaintext0=valid_comparison_input.attack_pairs[0].plaintext0,
        plaintext1=valid_comparison_input.attack_pairs[1].plaintext1,
        ciphertext0=valid_comparison_input.attack_pairs[0].ciphertext0,
        ciphertext1=valid_comparison_input.attack_pairs[1].ciphertext1,
    )
    tampered_pairs = list(valid_comparison_input.attack_pairs)
    tampered_pairs[1] = dup_pair
    with pytest.raises(ValueError, match="Duplicate plaintexts found within attack pairs"):
        SimonComparisonInput(
            instance_id="simon_test_inst",
            rounds=10,
            attack_pairs=tuple(tampered_pairs),
            verification_pairs=valid_comparison_input.verification_pairs,
            manifest_seed=999999,
        )


def test_duplicate_plaintexts_within_verification_pairs_rejected(valid_comparison_input):
    dup_verif = (
        valid_comparison_input.verification_pairs[0],
        valid_comparison_input.verification_pairs[0],
    )
    with pytest.raises(ValueError, match="Duplicate plaintexts found within verification pairs"):
        SimonComparisonInput(
            instance_id="simon_test_inst",
            rounds=10,
            attack_pairs=valid_comparison_input.attack_pairs,
            verification_pairs=dup_verif,
            manifest_seed=999999,
        )


def test_duplicate_plaintexts_between_attack_and_verification_rejected(valid_comparison_input):
    # Take an attack plaintext and use it as a verification plaintext
    colliding_pt = valid_comparison_input.attack_pairs[0].plaintext0
    colliding_verif = (
        (colliding_pt, (0x1111, 0x2222)),
        valid_comparison_input.verification_pairs[1],
    )
    with pytest.raises(ValueError, match="Duplicate plaintexts found between attack and verification"):
        SimonComparisonInput(
            instance_id="simon_test_inst",
            rounds=10,
            attack_pairs=valid_comparison_input.attack_pairs,
            verification_pairs=colliding_verif,
            manifest_seed=999999,
        )


# 6. Chronological SAT Assumptions Validation
def test_validate_sat_assumptions_chronological(valid_comparison_input):
    assumptions = valid_comparison_input.validate_sat_assumptions((0x1234, 0x5678))
    assert assumptions == (0x1234, 0x5678)
    assert valid_comparison_input.format_chronological_assumptions(0x1234, 0x5678) == (0x1234, 0x5678)


@pytest.mark.parametrize("bad_assumptions", [
    (0x1234,),  # 1 subkey
    (0x1234, 0x5678, 0x9ABC),  # 3 subkeys
    (-1, 0x5678),  # negative
    (0x1234, 0x10000),  # > 16-bit
])
def test_malformed_sat_assumptions_rejected(valid_comparison_input, bad_assumptions):
    with pytest.raises(ValueError):
        valid_comparison_input.validate_sat_assumptions(bad_assumptions)


def test_reversed_sat_assumptions_fail_to_recover(valid_demo_instance, valid_comparison_input):
    round_keys = expand_key(valid_demo_instance.master_key, 10)
    true_k8, true_k9 = round_keys[8], round_keys[9]
    assert true_k8 != true_k9, "Trailing subkeys must be distinct for reversal test"

    # Chronological (k8, k9) assumptions satisfy model
    model = SimonSatModel.from_known_pairs(valid_comparison_input.candidate_sat_pairs, rounds=10)
    chronological_assumptions = model.assumptions_for_trailing_round_subkeys([true_k8, true_k9])
    res_chronological = model.solve(chronological_assumptions, time_limit=45.0)
    assert res_chronological.status == "sat"

    # Reversed (k9, k8) assumptions yield rapid UNSAT on residual 8 rounds
    reversed_assumptions = model.assumptions_for_trailing_round_subkeys([true_k9, true_k8])
    res_reversed = model.solve(reversed_assumptions, time_limit=10.0)
    assert res_reversed.status == "unsat"


# 7. Verification Isolation: Verification Observations Never Exposed to Rankers
def test_verification_observations_never_in_ciphertext_pairs(valid_comparison_input):
    verif_pts = {v[0] for v in valid_comparison_input.verification_pairs}
    verif_cts = {v[1] for v in valid_comparison_input.verification_pairs}

    # Ranker input is strictly ciphertext_pairs
    ranker_input = valid_comparison_input.ciphertext_pairs
    assert len(ranker_input) == EXPECTED_ATTACK_PAIR_COUNT

    for ct0, ct1 in ranker_input:
        assert ct0 not in verif_pts
        assert ct1 not in verif_pts
        assert ct0 not in verif_cts
        assert ct1 not in verif_cts


def test_method_worker_rejects_verification_leaked_into_attack_pairs(valid_demo_instance):
    # Deliberately substitute a verification pair into attack pairs
    leaked_attack = list(tuple((p.ciphertext0, p.ciphertext1) for p in valid_demo_instance.pairs))
    leaked_attack[0] = (valid_demo_instance.verification_pairs[0][1], (0x9999, 0x8888))

    with pytest.raises(ValueError, match="Verification observation leaked into ranking attack pairs"):
        run_method_worker(
            method="test_mock_queue",
            rounds=10,
            attack_pairs=tuple(leaked_attack),
            known_pairs=valid_demo_instance.known_pairs,
            verification_pairs=valid_demo_instance.verification_pairs,
            mock_candidate_queue=[(0x1234, 0x5678)],
            use_parent_watchdog=False,
        )


# 8. Deterministic Random Control Stream Derivation
def test_deterministic_random_control_stream_derivation():
    s1 = derive_random_control_seed(999999, "simon32_64_demo_r10_0000", "random_control_128")
    s1_dup = derive_random_control_seed(999999, "simon32_64_demo_r10_0000", "random_control_128")
    assert s1 == s1_dup
    assert isinstance(s1, int)
    assert 0 <= s1 < (1 << 64)

    # Different method gives different stream
    s2 = derive_random_control_seed(999999, "simon32_64_demo_r10_0000", "random_control_1024")
    assert s1 != s2

    # Different instance gives different stream
    s3 = derive_random_control_seed(999999, "simon32_64_demo_r10_0001", "random_control_128")
    assert s1 != s3

    # Different manifest seed gives different stream
    s4 = derive_random_control_seed(260926, "simon32_64_demo_r10_0000", "random_control_128")
    assert s1 != s4


def test_comparison_input_integrates_with_method_worker(valid_comparison_input):
    outcome = run_method_worker(
        method="test_mock_queue",
        comparison_input=valid_comparison_input,
        mock_candidate_queue=[(0x1234, 0x5678)],
        solver_stub=lambda model, assumptions, time_limit: SatSolveResult(
            False, None, 0.0, False, "unsat"
        ),
        use_parent_watchdog=False,
    )
    assert outcome.outcome == "queue_exhausted"
    assert outcome.candidates_tested_count == 1
    assert outcome.candidate_attempts[0].status == "unsat"


# 9. Support for 64 Attack Pairs and 128 Known Observations
@pytest.fixture(scope="module")
def valid_demo_instance_64_pairs():
    cfg = SimonAttackInstanceConfig(
        instance_count=1,
        pair_count=64,
        rounds=10,
        seed=999999,
        split_role="demo",
        verification_pair_count=2,
    )
    return generate_simon_attack_instances(cfg)[0]


def test_valid_comparison_input_64_pairs(valid_demo_instance_64_pairs):
    inp = SimonComparisonInput.from_attack_instance(
        instance=valid_demo_instance_64_pairs,
        manifest_seed=999999,
    )
    assert inp.pair_count == 64
    assert inp.rounds == EXPECTED_COMPARISON_ROUNDS
    assert len(inp.attack_pairs) == 64
    assert len(inp.known_pairs) == 128
    assert len(inp.verification_pairs) == EXPECTED_VERIFICATION_PAIR_COUNT
    assert len(inp.ciphertext_pairs) == 64
    assert len(inp.candidate_sat_pairs) == 4
    assert len(inp.all_sat_pairs) == 128

    # Serialization round trip
    d = inp.to_dict()
    assert d["pair_count"] == 64
    restored = SimonComparisonInput.from_dict(d)
    assert restored.pair_count == 64
    assert len(restored.attack_pairs) == 64
    assert len(restored.known_pairs) == 128


@pytest.mark.parametrize("bad_pair_count", [16, 48, 128, 0, -1])
def test_unsupported_pair_counts_rejected(valid_comparison_input, bad_pair_count):
    with pytest.raises(ValueError, match="pair_count must be in"):
        SimonComparisonInput(
            instance_id="simon_test_inst",
            rounds=10,
            attack_pairs=valid_comparison_input.attack_pairs,
            verification_pairs=valid_comparison_input.verification_pairs,
            manifest_seed=999999,
            pair_count=bad_pair_count,
        )


@pytest.mark.parametrize("bad_type", ["32", 32.0, True, False, None])
def test_malformed_pair_count_rejected(valid_comparison_input, bad_type):
    with pytest.raises(TypeError, match="pair_count must be an integer"):
        SimonComparisonInput(
            instance_id="simon_test_inst",
            rounds=10,
            attack_pairs=valid_comparison_input.attack_pairs,
            verification_pairs=valid_comparison_input.verification_pairs,
            manifest_seed=999999,
            pair_count=bad_type,  # type: ignore
        )


def test_mismatched_pair_count_and_attack_pairs_rejected(valid_comparison_input, valid_demo_instance_64_pairs):
    # pair_count=64 but only 32 pairs passed
    with pytest.raises(ValueError, match="Expected exactly 64 attack pairs, got 32"):
        SimonComparisonInput(
            instance_id="simon_test_inst",
            rounds=10,
            attack_pairs=valid_comparison_input.attack_pairs,
            verification_pairs=valid_comparison_input.verification_pairs,
            manifest_seed=999999,
            pair_count=64,
        )

    # pair_count=32 but 64 pairs passed
    with pytest.raises(ValueError, match="Expected exactly 32 attack pairs, got 64"):
        SimonComparisonInput(
            instance_id="simon_test_inst",
            rounds=10,
            attack_pairs=valid_demo_instance_64_pairs.pairs,
            verification_pairs=valid_comparison_input.verification_pairs,
            manifest_seed=999999,
            pair_count=32,
        )


# 10. Non-16-bit word rejection
def test_attack_pairs_non_16bit_words_rejected(valid_comparison_input):
    # Word > 0xFFFF in plaintext
    bad_pair = SimonAttackPair(
        plaintext0=(0x10000, 0x1234),
        plaintext1=valid_comparison_input.attack_pairs[0].plaintext1,
        ciphertext0=valid_comparison_input.attack_pairs[0].ciphertext0,
        ciphertext1=valid_comparison_input.attack_pairs[0].ciphertext1,
    )
    tampered = list(valid_comparison_input.attack_pairs)
    tampered[0] = bad_pair
    with pytest.raises(ValueError, match="must be a 16-bit word"):
        SimonComparisonInput(
            instance_id="simon_test_inst",
            rounds=10,
            attack_pairs=tuple(tampered),
            verification_pairs=valid_comparison_input.verification_pairs,
            manifest_seed=999999,
            pair_count=32,
        )

    # Negative word in ciphertext
    bad_pair_neg = SimonAttackPair(
        plaintext0=valid_comparison_input.attack_pairs[0].plaintext0,
        plaintext1=valid_comparison_input.attack_pairs[0].plaintext1,
        ciphertext0=(-1, 0x1234),
        ciphertext1=valid_comparison_input.attack_pairs[0].ciphertext1,
    )
    tampered[0] = bad_pair_neg
    with pytest.raises(ValueError, match="must be a 16-bit word"):
        SimonComparisonInput(
            instance_id="simon_test_inst",
            rounds=10,
            attack_pairs=tuple(tampered),
            verification_pairs=valid_comparison_input.verification_pairs,
            manifest_seed=999999,
            pair_count=32,
        )


def test_verification_pairs_non_16bit_words_rejected(valid_comparison_input):
    bad_verif = (
        ((0x1234, 0x20000), (0x5678, 0x9ABC)),
        valid_comparison_input.verification_pairs[1],
    )
    with pytest.raises(ValueError, match="must be a 16-bit word"):
        SimonComparisonInput(
            instance_id="simon_test_inst",
            rounds=10,
            attack_pairs=valid_comparison_input.attack_pairs,
            verification_pairs=bad_verif,  # type: ignore
            manifest_seed=999999,
            pair_count=32,
        )


# 11. Formula Size Control for 32 vs 64 Pairs (Task 5)
def test_formula_size_control_32_vs_64_pairs(valid_comparison_input, valid_demo_instance_64_pairs):
    """Verify exact formula variable, CNF clause, and XOR clause counts on public synthetic data.

    Invariants (R=10):
    - Candidate formula (first 2 pairs, 4 observations): 1,856 vars, 2,176 CNF, 1,024 XOR.
    - 32-pair all-observation control (64 observations): 22,976 vars, 34,816 CNF, 10,624 XOR.
    - 64-pair all-observation control (128 observations): 45,504 vars, 69,632 CNF, 20,864 XOR.

    NOTE: Does NOT solve or inspect key ranks!
    """
    # 1. Candidate formula: first 2 pairs (4 observations)
    cand_model = SimonSatModel.from_known_pairs(valid_comparison_input.candidate_sat_pairs, rounds=10)
    assert cand_model.variable_count == 1856
    assert cand_model.cnf_clause_count == 2176
    assert cand_model.xor_clause_count == 1024

    # 2. 32-pair all-observation control (64 observations)
    control_32_model = SimonSatModel.from_known_pairs(valid_comparison_input.all_sat_pairs, rounds=10)
    assert control_32_model.variable_count == 22976
    assert control_32_model.cnf_clause_count == 34816
    assert control_32_model.xor_clause_count == 10624

    # 3. 64-pair all-observation control (128 observations)
    inp_64 = SimonComparisonInput.from_attack_instance(valid_demo_instance_64_pairs, manifest_seed=999999)
    control_64_model = SimonSatModel.from_known_pairs(inp_64.all_sat_pairs, rounds=10)
    assert control_64_model.variable_count == 45504
    assert control_64_model.cnf_clause_count == 69632
    assert control_64_model.xor_clause_count == 20864
