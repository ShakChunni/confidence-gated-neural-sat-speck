"""Unit tests for multi-round-key conditioning and ordered recovery."""

from __future__ import annotations

import pytest

from neural_sat_speck.attacks.multi_round_instance import partial_decrypt_instance
from neural_sat_speck.attacks.multi_round_recovery import run_multi_round_key_recovery
from neural_sat_speck.attacks.recovery import (
    independent_verification_pairs,
    known_pairs_for_attack_instance,
)
from neural_sat_speck.data.attack import (
    AttackInstance,
    AttackInstanceConfig,
    generate_attack_instances,
)
from neural_sat_speck.sat.isolation import IsolatedSolverFactory
from neural_sat_speck.sat.model import SpeckSatModel
from neural_sat_speck.sat.multi_round import (
    assumptions_for_round_key,
    assumptions_for_round_keys,
    trailing_round_key_indices,
)
from neural_sat_speck.speck.cipher import encrypt_block
from neural_sat_speck.speck.key_schedule import expand_key

ROUNDS = 4
KNOWN_PAIRS = 4
VERIFICATION_PAIRS = 2
SEED = 4001


def build_instance() -> AttackInstance:
    config = AttackInstanceConfig(
        instance_count=1,
        pair_count=8,
        rounds=ROUNDS,
        seed=SEED,
        split_role="train",
    )
    return generate_attack_instances(config).instances[0]


def build_model(instance: AttackInstance) -> SpeckSatModel:
    pairs = known_pairs_for_attack_instance(instance)[:KNOWN_PAIRS]
    factory = IsolatedSolverFactory.from_pairs(
        pairs,
        rounds=ROUNDS,
        instance_id=instance.instance_id,
    )
    return factory.build_isolated_model("multi_round_test")


def true_round_keys(instance: AttackInstance) -> tuple[int, ...]:
    return expand_key(instance.master_key, rounds=ROUNDS)


def test_assumptions_for_round_key_pins_every_bit() -> None:
    instance = build_instance()
    model = build_model(instance)
    value = true_round_keys(instance)[-1]

    assumptions = assumptions_for_round_key(model, ROUNDS - 1, value)

    assert len(assumptions) == 16
    for bit, literal in enumerate(assumptions):
        expected_variable = model.round_keys[ROUNDS - 1][bit]
        assert abs(literal) == expected_variable
        assert (literal > 0) == bool((value >> bit) & 1)


def test_assumptions_reject_invalid_inputs() -> None:
    instance = build_instance()
    model = build_model(instance)

    with pytest.raises(TypeError):
        assumptions_for_round_key(model, True, 0)
    with pytest.raises(ValueError):
        assumptions_for_round_key(model, ROUNDS, 0)
    with pytest.raises(ValueError):
        assumptions_for_round_key(model, 0, 0x1_0000)
    with pytest.raises(TypeError):
        assumptions_for_round_key("model", 0, 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        assumptions_for_round_keys(model, {})
    with pytest.raises(TypeError):
        assumptions_for_round_keys(model, [0, 1])  # type: ignore[arg-type]


def test_assumptions_for_round_keys_are_order_independent() -> None:
    instance = build_instance()
    model = build_model(instance)
    schedule = true_round_keys(instance)

    ascending = assumptions_for_round_keys(model, {ROUNDS - 2: schedule[-2], ROUNDS - 1: schedule[-1]})
    descending = assumptions_for_round_keys(model, {ROUNDS - 1: schedule[-1], ROUNDS - 2: schedule[-2]})

    assert ascending == descending
    assert len(ascending) == 32


def test_trailing_round_key_indices() -> None:
    assert trailing_round_key_indices(7, 2) == (5, 6)
    assert trailing_round_key_indices(7, 0) == ()
    with pytest.raises(ValueError):
        trailing_round_key_indices(7, 8)
    with pytest.raises(TypeError):
        trailing_round_key_indices(7, True)


def test_multi_round_recovery_verifies_true_candidate() -> None:
    instance = build_instance()
    model = build_model(instance)
    schedule = true_round_keys(instance)
    known = known_pairs_for_attack_instance(instance)[:KNOWN_PAIRS]
    verification = independent_verification_pairs(
        instance.master_key,
        rounds=ROUNDS,
        pair_count=VERIFICATION_PAIRS,
        seed=999999,
    )

    result = run_multi_round_key_recovery(
        model,
        [{ROUNDS - 1: schedule[-1]}],
        known,
        verification,
        rounds=ROUNDS,
        solver_timeout=30.0,
    )

    assert result.success is True
    assert result.status == "verified_recovery"
    assert result.recovered_master_key == tuple(instance.master_key)
    assert result.assumed_round_keys == 1
    assert result.residual_rounds == ROUNDS - 1
    assert result.candidates_attempted == 1
    assert result.attempts[0].independent_pairs_verified is True


def test_multi_round_recovery_skips_wrong_candidate() -> None:
    instance = build_instance()
    model = build_model(instance)
    schedule = true_round_keys(instance)
    known = known_pairs_for_attack_instance(instance)[:KNOWN_PAIRS]
    verification = independent_verification_pairs(
        instance.master_key,
        rounds=ROUNDS,
        pair_count=VERIFICATION_PAIRS,
        seed=999999,
    )
    wrong = (schedule[-1] + 1) & 0xFFFF

    result = run_multi_round_key_recovery(
        model,
        [{ROUNDS - 1: wrong}, {ROUNDS - 1: schedule[-1]}],
        known,
        verification,
        rounds=ROUNDS,
        solver_timeout=30.0,
    )

    assert result.success is True
    assert result.candidates_attempted == 2
    assert result.attempts[0].solver_status == "unsat"
    assert result.attempts[1].recovered_master_key == tuple(instance.master_key)


def test_multi_round_recovery_reports_exhausted_candidates() -> None:
    instance = build_instance()
    model = build_model(instance)
    schedule = true_round_keys(instance)
    known = known_pairs_for_attack_instance(instance)[:KNOWN_PAIRS]
    verification = independent_verification_pairs(
        instance.master_key,
        rounds=ROUNDS,
        pair_count=VERIFICATION_PAIRS,
        seed=999999,
    )

    result = run_multi_round_key_recovery(
        model,
        [{ROUNDS - 1: (schedule[-1] + 1) & 0xFFFF}],
        known,
        verification,
        rounds=ROUNDS,
        solver_timeout=30.0,
    )

    assert result.success is False
    assert result.status == "exhausted_candidates"
    assert result.attempts[0].solver_status == "unsat"
    assert result.recovered_master_key is None


def test_partial_decrypt_instance_matches_shorter_encryption() -> None:
    instance = build_instance()
    schedule = true_round_keys(instance)
    next_subkey = schedule[-2]

    descended = partial_decrypt_instance(
        instance,
        schedule[-1],
        expected_next_subkey=next_subkey,
        label="descended",
    )

    assert descended.instance_id == "descended"
    assert descended.master_key == instance.master_key
    assert descended.final_round_subkey == next_subkey
    assert len(descended.pairs) == len(instance.pairs)
    for original, pair in zip(instance.pairs, descended.pairs, strict=True):
        assert pair.plaintext0 == original.plaintext0
        assert pair.ciphertext0 == encrypt_block(
            original.plaintext0,
            instance.master_key,
            rounds=ROUNDS - 1,
        )
        assert pair.ciphertext1 == encrypt_block(
            original.plaintext1,
            instance.master_key,
            rounds=ROUNDS - 1,
        )


def test_partial_decrypt_instance_rejects_invalid_inputs() -> None:
    instance = build_instance()
    with pytest.raises(ValueError):
        partial_decrypt_instance(instance, 0x1_0000, expected_next_subkey=0)
    with pytest.raises(TypeError):
        partial_decrypt_instance(instance, True, expected_next_subkey=0)
    with pytest.raises(TypeError):
        partial_decrypt_instance(instance, 0, expected_next_subkey=True)
    with pytest.raises(TypeError):
        partial_decrypt_instance("instance", 0, expected_next_subkey=0)  # type: ignore[arg-type]


def test_multi_round_recovery_requires_uniform_candidate_width() -> None:
    instance = build_instance()
    model = build_model(instance)
    schedule = true_round_keys(instance)
    known = known_pairs_for_attack_instance(instance)[:KNOWN_PAIRS]
    verification = independent_verification_pairs(
        instance.master_key,
        rounds=ROUNDS,
        pair_count=VERIFICATION_PAIRS,
        seed=999999,
    )

    with pytest.raises(ValueError):
        run_multi_round_key_recovery(
            model,
            [{ROUNDS - 1: schedule[-1]}, {ROUNDS - 2: schedule[-2], ROUNDS - 1: schedule[-1]}],
            known,
            verification,
            rounds=ROUNDS,
            solver_timeout=30.0,
        )


def test_seven_round_recovery_needs_two_assumed_round_keys() -> None:
    """The scaling capability the seven-round architecture depends on.

    At seven rounds the exact backend cannot complete an instance conditioned on
    the final round key alone; assuming the final two round keys brings the
    residual depth to five and the instance solves. This is the measured boundary
    recorded in docs/PHASE11_SCALING_STRATEGY.md and must not silently regress.
    """

    rounds = 7
    config = AttackInstanceConfig(
        instance_count=1,
        pair_count=64,
        rounds=rounds,
        seed=SEED,
        split_role="train",
    )
    instance = generate_attack_instances(config).instances[0]
    schedule = expand_key(instance.master_key, rounds=rounds)
    known = known_pairs_for_attack_instance(instance)[:KNOWN_PAIRS]
    verification = independent_verification_pairs(
        instance.master_key,
        rounds=rounds,
        pair_count=VERIFICATION_PAIRS,
        seed=999999,
    )
    factory = IsolatedSolverFactory.from_pairs(
        known,
        rounds=rounds,
        instance_id=instance.instance_id,
    )
    model = factory.build_isolated_model("seven_round_scaling")

    result = run_multi_round_key_recovery(
        model,
        [{rounds - 2: schedule[-2], rounds - 1: schedule[-1]}],
        known,
        verification,
        rounds=rounds,
        solver_timeout=30.0,
    )

    assert result.success is True
    assert result.residual_rounds == rounds - 2
    assert result.recovered_master_key == tuple(instance.master_key)
