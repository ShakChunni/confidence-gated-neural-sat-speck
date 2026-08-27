"""Integration tests for the Phase 5 exact candidate-recovery boundary."""

import pytest

from neural_sat_speck.attacks.recovery import (
    independent_verification_pairs,
    known_pairs_for_attack_instance,
    random_candidate_order,
    run_candidate_recovery,
    run_sat_only_recovery,
)
from neural_sat_speck.data.attack import AttackInstanceConfig, generate_attack_instances
from neural_sat_speck.sat.model import SpeckSatModel
from neural_sat_speck.speck.key_schedule import expand_key

pytest.importorskip("pycryptosat")


def _attack_instance():
    return generate_attack_instances(
        AttackInstanceConfig(
            instance_count=1,
            pair_count=4,
            rounds=4,
            seed=71001,
            split_role="test",
        ),
        generated_at="2026-08-27T00:00:00+00:00",
    ).instances[0]


def test_candidate_recovery_verifies_the_decoded_master_key() -> None:
    instance = _attack_instance()
    known_pairs = known_pairs_for_attack_instance(instance)
    verification_pairs = independent_verification_pairs(
        instance.master_key,
        rounds=4,
        pair_count=2,
        seed=72001,
    )
    model = SpeckSatModel.from_known_pairs(known_pairs, rounds=4)
    true_subkey = expand_key(instance.master_key, 4)[-1]

    result = run_candidate_recovery(
        model,
        (true_subkey, true_subkey ^ 1),
        known_pairs,
        verification_pairs,
        candidate_budget=2,
        solver_timeout=5.0,
        max_solutions_per_candidate=2,
        strategy="neural",
    )

    assert result.success
    assert result.status == "recovered"
    assert result.candidates_attempted == 1
    assert result.winning_candidate == true_subkey
    assert result.winning_candidate_position == 1
    assert result.recovered_master_key == instance.master_key
    assert result.attempts[0].outcome == "recovered"
    assert result.attempts[0].final_round_subkey_matches
    assert result.attempts[0].known_pairs_verified
    assert result.attempts[0].independent_pairs_verified
    assert result.to_dict()["attempts"][0]["candidate"] == true_subkey


def test_sat_only_recovery_requires_independent_verification() -> None:
    instance = _attack_instance()
    known_pairs = known_pairs_for_attack_instance(instance)
    verification_pairs = independent_verification_pairs(
        instance.master_key,
        rounds=4,
        pair_count=2,
        seed=72002,
    )
    model = SpeckSatModel.from_known_pairs(known_pairs, rounds=4)

    result = run_sat_only_recovery(
        model,
        known_pairs,
        verification_pairs,
        solver_timeout=5.0,
    )

    assert result.success
    assert result.status == "recovered"
    assert result.solver_calls == 1
    assert result.recovered_master_key == instance.master_key
    assert result.decoded_master_key == instance.master_key
    assert result.winning_candidate == instance.final_round_subkey


def test_random_candidate_order_is_seeded_without_hidden_state() -> None:
    first = random_candidate_order(73001, range(32))
    second = random_candidate_order(73001, range(32))

    assert first == second
    assert set(first) == set(range(32))
    assert first != random_candidate_order(73002, range(32))


@pytest.mark.parametrize(
    "candidate_order",
    [(), (1, 1), (-1, 2), (0, 1 << 16)],
)
def test_candidate_recovery_rejects_invalid_orders(candidate_order: tuple[int, ...]) -> None:
    instance = _attack_instance()
    known_pairs = known_pairs_for_attack_instance(instance)
    verification_pairs = independent_verification_pairs(
        instance.master_key,
        rounds=4,
        pair_count=1,
        seed=72003,
    )
    model = SpeckSatModel.from_known_pairs(known_pairs, rounds=4)

    with pytest.raises((TypeError, ValueError)):
        run_candidate_recovery(
            model,
            candidate_order,
            known_pairs,
            verification_pairs,
            candidate_budget=1,
        )
