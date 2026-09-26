"""Unit tests for SIMON32/64 attack instances, seed governance, and role labelling."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from neural_sat_speck.data.seed_policy import (
    CLI_DEMO_SEED,
    HISTORICAL_PHASE7_SEED,
    PHASE10_DEVELOPMENT_SEED,
    PHASE10_PROSPECTIVE_SEED,
    PHASE11A_ORACLE_SEED,
    QUARANTINED_PHASE10_SEED,
    RETIRED_BLIND_SEED,
)
from neural_sat_speck.simon.attack_instance import (
    DEFAULT_SIMON_INPUT_DIFFERENCE,
    SimonAttackInstanceConfig,
    generate_simon_attack_instances,
    load_simon_attack_instances,
    save_simon_attack_instances,
)
from neural_sat_speck.simon.cipher import encrypt_block
from neural_sat_speck.simon.key_schedule import expand_key
from tests.fixtures.simon_reference_oracle import oracle_simon32_64_encrypt


def test_demo_instance_generation_is_deterministic() -> None:
    config = SimonAttackInstanceConfig(
        instance_count=5,
        pair_count=8,
        rounds=6,
        seed=CLI_DEMO_SEED,
        split_role="demo",
    )
    run1 = generate_simon_attack_instances(config)
    run2 = generate_simon_attack_instances(config)

    assert len(run1) == 5
    assert len(run2) == 5
    for inst1, inst2 in zip(run1, run2, strict=True):
        assert inst1.instance_id == inst2.instance_id
        assert inst1.master_key == inst2.master_key
        assert inst1.rounds == inst2.rounds
        assert len(inst1.pairs) == 8
        for p1, p2 in zip(inst1.pairs, inst2.pairs, strict=True):
            assert p1.plaintext0 == p2.plaintext0
            assert p1.plaintext1 == p2.plaintext1
            assert p1.ciphertext0 == p2.ciphertext0
            assert p1.ciphertext1 == p2.ciphertext1


def test_input_difference_structure_and_encryption() -> None:
    config = SimonAttackInstanceConfig(
        instance_count=2,
        pair_count=4,
        rounds=7,
        seed=CLI_DEMO_SEED,
        split_role="demo",
        input_difference=DEFAULT_SIMON_INPUT_DIFFERENCE,
    )
    instances = generate_simon_attack_instances(config)
    for inst in instances:
        for pair in inst.pairs:
            diff_l = pair.plaintext0[0] ^ pair.plaintext1[0]
            diff_r = pair.plaintext0[1] ^ pair.plaintext1[1]
            assert (diff_l, diff_r) == DEFAULT_SIMON_INPUT_DIFFERENCE
            assert encrypt_block(pair.plaintext0, inst.master_key, inst.rounds) == pair.ciphertext0
            assert encrypt_block(pair.plaintext1, inst.master_key, inst.rounds) == pair.ciphertext1


def test_fresh_verification_pairs_are_independently_valid() -> None:
    config = SimonAttackInstanceConfig(
        instance_count=3,
        pair_count=4,
        rounds=6,
        seed=CLI_DEMO_SEED,
        split_role="demo",
        verification_pair_count=3,
    )
    instances = generate_simon_attack_instances(config)
    for inst in instances:
        assert len(inst.verification_pairs) == 3
        all_plaintexts = [pt for pt, _ in inst.known_pairs + inst.verification_pairs]
        assert len(all_plaintexts) == len(set(all_plaintexts))
        # Ensure verification plaintexts are encrypted under the master key
        for pt, ct in inst.verification_pairs:
            assert encrypt_block(pt, inst.master_key, inst.rounds) == ct


def test_generator_rejects_attack_and_verification_plaintext_collisions(monkeypatch) -> None:
    """Scripted RNG forces both kinds of collision before distinct draws."""
    from neural_sat_speck.simon import attack_instance as module

    draws = iter((
        10, 20, 30, 40,  # master key
        1, 2,             # accepted pair: (1,2), (1,66)
        1, 66,            # rejected second pair, duplicates first pair
        3, 4,             # accepted pair: (3,4), (3,68)
        1, 2,             # rejected verification plaintext from first pair
        3, 68,            # rejected verification plaintext from second pair
        5, 6,             # accepted first verification plaintext
        5, 6,             # rejected duplicate verification plaintext
        7, 8,             # accepted second verification plaintext
    ))

    class ScriptedRandom:
        def __init__(self, seed: int) -> None:
            assert seed == CLI_DEMO_SEED

        def randrange(self, stop: int) -> int:
            assert stop == 1 << 16
            return next(draws)

    monkeypatch.setattr(module, "random", SimpleNamespace(Random=ScriptedRandom))
    instance, = generate_simon_attack_instances(SimonAttackInstanceConfig(
        instance_count=1,
        pair_count=2,
        rounds=9,
        seed=CLI_DEMO_SEED,
        verification_pair_count=2,
    ))
    plaintexts = [pt for pt, _ in instance.known_pairs + instance.verification_pairs]
    assert plaintexts == [(1, 2), (1, 66), (3, 4), (3, 68), (5, 6), (7, 8)]
    assert len(plaintexts) == len(set(plaintexts))
    for pt, ct in instance.known_pairs + instance.verification_pairs:
        assert ct == oracle_simon32_64_encrypt(pt, instance.master_key, instance.rounds)
    with pytest.raises(StopIteration):
        next(draws)


def test_instance_rejects_overlap_after_deserialization_or_manual_construction() -> None:
    instance, = generate_simon_attack_instances(SimonAttackInstanceConfig(
        instance_count=1, pair_count=1, rounds=9, seed=CLI_DEMO_SEED,
    ))
    with pytest.raises(ValueError, match="plaintexts must be distinct"):
        replace(instance, verification_pairs=(instance.known_pairs[0],))
    with pytest.raises(ValueError, match="plaintexts must be distinct"):
        replace(instance, pairs=(instance.pairs[0], instance.pairs[0]))


@pytest.mark.parametrize("input_difference", [(0, 0), (1 << 16, 0)])
def test_zero_effective_input_difference_is_rejected(input_difference) -> None:
    with pytest.raises(ValueError, match="input_difference must be nonzero"):
        SimonAttackInstanceConfig(
            instance_count=1, pair_count=1, rounds=9, seed=CLI_DEMO_SEED,
            input_difference=input_difference,
        )


def test_impossible_unique_plaintext_request_is_rejected() -> None:
    with pytest.raises(ValueError, match="exceed the 32-bit block domain"):
        SimonAttackInstanceConfig(
            instance_count=1, pair_count=1 << 31, rounds=9, seed=CLI_DEMO_SEED,
        )


@pytest.mark.parametrize(
    ("rounds", "expected_sha256"),
    [
        (8, "8a762dee8ab2f4d0e8e96c10f9f3fb3e00a681fa419bfd18104570cadb1f16e8"),
        (9, "b294ee5fed9e0b6ae79437ef976da67b5e061b5f3318858948fd9bfcea98bca4"),
        (10, "e4e9b03e21a67c28d7764ab8e759f6c7bb2d8377f32c89e164946e45b3357718"),
    ],
)
def test_opened_demo_instances_replay_unchanged(rounds: int, expected_sha256: str) -> None:
    instances = generate_simon_attack_instances(SimonAttackInstanceConfig(
        instance_count=8, pair_count=8, rounds=rounds, seed=CLI_DEMO_SEED,
    ))
    payload = json.dumps(
        [instance.to_dict() for instance in instances],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert hashlib.sha256(payload).hexdigest() == expected_sha256


@pytest.mark.parametrize(
    "bad_seed",
    [
        HISTORICAL_PHASE7_SEED,
        QUARANTINED_PHASE10_SEED,
        RETIRED_BLIND_SEED,
        PHASE10_DEVELOPMENT_SEED,
        PHASE10_PROSPECTIVE_SEED,
        PHASE11A_ORACLE_SEED,
        406737,  # Phase 11 beam adjudication seed
        669981,  # Phase 11 prospective evaluation seed
    ],
)
def test_seed_governance_rejects_protected_seeds(bad_seed: int) -> None:
    with pytest.raises(ValueError, match="protected/quarantined research seed"):
        SimonAttackInstanceConfig(
            instance_count=2,
            pair_count=4,
            rounds=6,
            seed=bad_seed,
            split_role="development",
        )


def test_role_governance_rejects_unauthorized_roles() -> None:
    with pytest.raises(ValueError, match="not in permitted"):
        SimonAttackInstanceConfig(
            instance_count=2,
            pair_count=4,
            rounds=6,
            seed=1234567,
            split_role="prospective_validation",
        )


def test_demo_role_requires_cli_demo_seed() -> None:
    with pytest.raises(ValueError, match="Demo split role requires CLI_DEMO_SEED"):
        SimonAttackInstanceConfig(
            instance_count=2,
            pair_count=4,
            rounds=6,
            seed=1234567,
            split_role="demo",
        )


def test_development_role_cannot_relabel_opened_demo_seed() -> None:
    with pytest.raises(ValueError, match="cannot reuse the demo seed"):
        SimonAttackInstanceConfig(
            instance_count=2,
            pair_count=4,
            rounds=9,
            seed=CLI_DEMO_SEED,
            split_role="development",
        )


@pytest.mark.parametrize("count", [0, -1])
def test_zero_or_negative_fresh_verification_count_is_rejected(count: int) -> None:
    with pytest.raises(ValueError, match="verification_pair_count must be positive"):
        SimonAttackInstanceConfig(
            instance_count=1,
            pair_count=2,
            rounds=9,
            seed=CLI_DEMO_SEED,
            verification_pair_count=count,
        )


@pytest.mark.parametrize("count", [True, 1.5, "2"])
def test_noninteger_verification_count_is_rejected(count) -> None:
    with pytest.raises(TypeError, match="verification_pair_count must be an integer"):
        SimonAttackInstanceConfig(
            instance_count=1,
            pair_count=2,
            rounds=9,
            seed=CLI_DEMO_SEED,
            verification_pair_count=count,
        )


def test_ground_truth_isolation_properties() -> None:
    config = SimonAttackInstanceConfig(
        instance_count=2,
        pair_count=4,
        rounds=7,
        seed=CLI_DEMO_SEED,
        split_role="demo",
    )
    instances = generate_simon_attack_instances(config)
    for inst in instances:
        expected_keys = expand_key(inst.master_key, inst.rounds)
        assert inst.expected_final_round_subkey == expected_keys[-1]
        assert inst.expected_trailing_subkeys(1) == (expected_keys[-1],)
        assert inst.expected_trailing_subkeys(2) == (expected_keys[-2], expected_keys[-1])
        assert inst.expected_trailing_subkeys(3) == (expected_keys[-3], expected_keys[-2], expected_keys[-1])


def test_serialization_round_trip() -> None:
    config = SimonAttackInstanceConfig(
        instance_count=3,
        pair_count=4,
        rounds=6,
        seed=CLI_DEMO_SEED,
        split_role="demo",
    )
    instances = generate_simon_attack_instances(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "simon_instances.json"
        save_simon_attack_instances(instances, json_path, metadata={"format": "test"})

        loaded = load_simon_attack_instances(json_path)
        assert len(loaded) == len(instances)
        for orig, reloaded in zip(instances, loaded, strict=True):
            assert orig.instance_id == reloaded.instance_id
            assert orig.split_role == reloaded.split_role
            assert orig.master_key == reloaded.master_key
            assert orig.rounds == reloaded.rounds
            assert orig.pairs == reloaded.pairs
            assert orig.verification_pairs == reloaded.verification_pairs
