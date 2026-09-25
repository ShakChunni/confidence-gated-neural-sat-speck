"""Unit tests for SIMON32/64 attack instances, seed governance, and role labelling."""

from __future__ import annotations

import tempfile
from pathlib import Path

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
        # Ensure verification plaintexts are encrypted under the master key
        for pt, ct in inst.verification_pairs:
            assert encrypt_block(pt, inst.master_key, inst.rounds) == ct


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
