"""Focused tests for the Phase 11B' six-round stage-1 pair-group family.

Covers the narrow six-round extension of the Phase 6A pair-group format:

1. six-round generation is deterministic and really uses six-round encryption;
2. six-round shard metadata and the manifest record ``rounds == 6``;
3. a six-round manifest round-trips through export, verification, and loading;
4. the frozen five-round path still rejects six rounds and still produces its
   frozen identifiers;
5. the new six-round stage-1 seed role accepts its own seeds and rejects every
   historical, protected, quarantined, or retired seed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neural_sat_speck.data.generator import DEFAULT_INPUT_DIFFERENCE
from neural_sat_speck.data.pair_groups import (
    PAIR_GROUP_DATASET_ID,
    PAIR_GROUP_DATASET_ROUNDS,
    PAIR_GROUP_GENERATOR_VERSION,
    PAIR_GROUP_MANIFEST_SCHEMA,
    PAIR_GROUP_ROUNDS,
    PAIR_GROUP_ROUNDS_SIX,
    PAIR_GROUP_SHARD_SCHEMA,
    PAIR_GROUP_SIX_ROUND_DATASET_ID,
    PAIR_GROUP_SIZE,
    PAIR_GROUP_SPLIT_ROLES,
    PAIR_GROUP_TASK,
    PairGroupConfig,
    SixRoundPairGroupConfig,
    export_pair_group_benchmark,
    generate_pair_groups,
    load_pair_group_manifest,
    load_pair_group_shard,
    verify_pair_group_benchmark,
)
from neural_sat_speck.data.seed_policy import (
    HISTORICAL_PHASE7_SEED,
    HISTORICAL_SEEDS,
    PHASE10_DEVELOPMENT_SEED,
    PHASE10_G2_DEVELOPMENT_SEED,
    PHASE10_G2_PROSPECTIVE_SEED,
    PHASE11A_ORACLE_SEED,
    PHASE11B_PRIME_SIX_ROUND_TRAIN_SEED,
    PHASE11B_PRIME_SIX_ROUND_VALIDATION_SEED,
    PROTECTED_RESEARCH_SEEDS,
    QUARANTINED_PHASE10_SEED,
    RETIRED_BLIND_SEED,
    SeedRole,
    validate_seed_for_role,
)
from neural_sat_speck.speck.cipher import encrypt_block

FIXED_TIMESTAMP = "2026-09-10T00:00:00+08:00"
FIXED_COMMIT = "0" * 40
STAGE1_ROLE = SeedRole.PHASE11B_PRIME_STAGE1_DEVELOPMENT
OWNED_SEEDS = (
    PHASE11B_PRIME_SIX_ROUND_TRAIN_SEED,
    PHASE11B_PRIME_SIX_ROUND_VALIDATION_SEED,
)


def _six_round_config(
    seed: int,
    role: str = "train",
    count: int = 2,
) -> SixRoundPairGroupConfig:
    return SixRoundPairGroupConfig(
        group_count=count,
        group_size=PAIR_GROUP_SIZE,
        rounds=PAIR_GROUP_ROUNDS_SIX,
        seed=seed,
        split_role=role,
    )


def _five_round_config(
    seed: int,
    role: str = "train",
    count: int = 2,
) -> PairGroupConfig:
    return PairGroupConfig(
        group_count=count,
        group_size=PAIR_GROUP_SIZE,
        rounds=PAIR_GROUP_ROUNDS,
        seed=seed,
        split_role=role,
    )


def _six_round_splits() -> tuple[SixRoundPairGroupConfig, SixRoundPairGroupConfig]:
    return (
        _six_round_config(PHASE11B_PRIME_SIX_ROUND_TRAIN_SEED, "train"),
        _six_round_config(PHASE11B_PRIME_SIX_ROUND_VALIDATION_SEED, "validation"),
    )


def test_six_round_generation_is_deterministic() -> None:
    first = generate_pair_groups(
        _six_round_config(OWNED_SEEDS[0]),
        generated_at=FIXED_TIMESTAMP,
    )
    second = generate_pair_groups(
        _six_round_config(OWNED_SEEDS[0]),
        generated_at=FIXED_TIMESTAMP,
    )

    assert np.array_equal(first.features, second.features)
    assert np.array_equal(first.labels, second.labels)
    assert np.array_equal(first.master_keys, second.master_keys)
    assert first.metadata == second.metadata
    assert first.features.shape == (2, PAIR_GROUP_SIZE, 4, 16)
    assert first.features.dtype == np.uint8
    assert first.master_keys.dtype == np.uint16
    assert first.labels.tolist().count(0) == first.labels.tolist().count(1) == 1


def test_six_round_metadata_records_six_rounds_and_owned_dataset_id() -> None:
    dataset = generate_pair_groups(_six_round_config(OWNED_SEEDS[0]))

    metadata = dataset.metadata
    assert metadata["rounds"] == PAIR_GROUP_ROUNDS_SIX == 6
    assert metadata["dataset_id"] == PAIR_GROUP_SIX_ROUND_DATASET_ID
    assert metadata["group_size"] == PAIR_GROUP_SIZE == 64
    assert metadata["input_difference"] == list(DEFAULT_INPUT_DIFFERENCE)
    assert metadata["task"] == PAIR_GROUP_TASK
    assert metadata["generator_version"] == PAIR_GROUP_GENERATOR_VERSION
    assert metadata["seed"] == OWNED_SEEDS[0]


def test_six_round_features_use_six_round_encryption() -> None:
    config = _six_round_config(OWNED_SEEDS[0])
    dataset = generate_pair_groups(config, include_debug_pairs=True)
    assert dataset.plaintext_pairs is not None
    assert dataset.ciphertext_pairs is not None

    for pair_index in range(PAIR_GROUP_SIZE):
        for side in (0, 1):
            for group_index in range(config.group_count):
                plaintext = tuple(
                    int(word)
                    for word in dataset.plaintext_pairs[group_index, pair_index, side]
                )
                key = tuple(int(word) for word in dataset.master_keys[group_index])
                ciphertext = tuple(
                    int(word)
                    for word in dataset.ciphertext_pairs[group_index, pair_index, side]
                )
                assert ciphertext == encrypt_block(
                    plaintext,
                    key,
                    rounds=PAIR_GROUP_ROUNDS_SIX,
                )


def test_six_round_config_rejects_five_rounds() -> None:
    with pytest.raises(ValueError, match="rounds must be exactly 6"):
        SixRoundPairGroupConfig(
            group_count=2,
            group_size=PAIR_GROUP_SIZE,
            rounds=PAIR_GROUP_ROUNDS,
            seed=OWNED_SEEDS[0],
            split_role="train",
        )


def test_frozen_five_round_config_still_rejects_six_rounds() -> None:
    with pytest.raises(ValueError, match="rounds must be exactly 5"):
        PairGroupConfig(
            group_count=2,
            group_size=PAIR_GROUP_SIZE,
            rounds=PAIR_GROUP_ROUNDS_SIX,
            seed=61001,
            split_role="train",
        )


def test_frozen_five_round_identifiers_are_unchanged() -> None:
    assert PAIR_GROUP_ROUNDS == 5
    assert PAIR_GROUP_ROUNDS_SIX == 6
    assert PAIR_GROUP_SIZE == 64
    assert PAIR_GROUP_DATASET_ID == "speck32_64_pair_group_phase6a_r5_g64"
    assert PAIR_GROUP_SIX_ROUND_DATASET_ID == (
        "speck32_64_pair_group_phase11b_prime_r6_g64"
    )
    assert PAIR_GROUP_TASK == "group_related_vs_random"
    assert PAIR_GROUP_SPLIT_ROLES == ("train", "validation")
    assert PAIR_GROUP_SHARD_SCHEMA == "neural_sat_speck.pair_group_shard.v1"
    assert PAIR_GROUP_MANIFEST_SCHEMA == "neural_sat_speck.pair_group_manifest.v1"
    assert PAIR_GROUP_GENERATOR_VERSION == "0.1.0"
    assert PAIR_GROUP_DATASET_ROUNDS == {
        PAIR_GROUP_DATASET_ID: 5,
        PAIR_GROUP_SIX_ROUND_DATASET_ID: 6,
    }

    frozen = generate_pair_groups(
        _five_round_config(61001),
        generated_at=FIXED_TIMESTAMP,
    )
    assert frozen.metadata["rounds"] == PAIR_GROUP_ROUNDS
    assert frozen.metadata["dataset_id"] == PAIR_GROUP_DATASET_ID
    assert frozen.metadata["input_difference"] == [0x0040, 0x0000]
    assert frozen.metadata["generator_version"] == PAIR_GROUP_GENERATOR_VERSION


def test_six_round_export_manifest_and_shard_round_trip(tmp_path: Path) -> None:
    output_dir = tmp_path / "phase11b_prime_pair_group_r6_g64"
    manifest = export_pair_group_benchmark(
        _six_round_splits(),
        output_dir,
        dataset_id=PAIR_GROUP_SIX_ROUND_DATASET_ID,
        generated_at=FIXED_TIMESTAMP,
        repository_commit=FIXED_COMMIT,
        rounds=PAIR_GROUP_ROUNDS_SIX,
    )

    assert manifest["rounds"] == 6
    assert manifest["dataset_id"] == PAIR_GROUP_SIX_ROUND_DATASET_ID
    assert manifest["group_size"] == 64
    assert manifest["generated_at"] == FIXED_TIMESTAMP
    assert manifest["repository_commit"] == FIXED_COMMIT
    assert manifest["input_difference"] == [0x0040, 0x0000]
    splits = manifest["splits"]
    assert set(splits) == {"train", "validation"}
    assert splits["train"]["seed"] == OWNED_SEEDS[0]
    assert splits["validation"]["seed"] == OWNED_SEEDS[1]
    assert splits["train"]["split_role"] == "train"
    assert splits["validation"]["split_role"] == "validation"
    assert splits["train"]["group_count"] == 2
    assert splits["train"]["group_size"] == 64

    # Mirrors how the training/evaluation scripts verify the family: by id alone.
    verified = verify_pair_group_benchmark(
        output_dir / "manifest.json",
        expected_dataset_id=PAIR_GROUP_SIX_ROUND_DATASET_ID,
    )
    assert verified["rounds"] == 6
    assert verified["splits"] == splits

    loaded = load_pair_group_shard(
        output_dir / "train.npz",
        expected_dataset_id=PAIR_GROUP_SIX_ROUND_DATASET_ID,
    )
    assert loaded.metadata["rounds"] == 6
    assert loaded.metadata["seed"] == OWNED_SEEDS[0]
    assert loaded.metadata["split_role"] == "train"
    assert loaded.features.shape == (2, PAIR_GROUP_SIZE, 4, 16)

    # An explicit round expectation must agree with the dataset id's family.
    explicit = load_pair_group_manifest(
        output_dir / "manifest.json",
        expected_dataset_id=PAIR_GROUP_SIX_ROUND_DATASET_ID,
        expected_rounds=PAIR_GROUP_ROUNDS_SIX,
    )
    assert explicit["rounds"] == PAIR_GROUP_ROUNDS_SIX


def test_five_round_path_rejects_six_round_artefacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "groups"
    export_pair_group_benchmark(
        _six_round_splits(),
        output_dir,
        dataset_id=PAIR_GROUP_SIX_ROUND_DATASET_ID,
        generated_at=FIXED_TIMESTAMP,
        repository_commit=FIXED_COMMIT,
        rounds=PAIR_GROUP_ROUNDS_SIX,
    )

    with pytest.raises(ValueError, match="dataset_id"):
        load_pair_group_shard(output_dir / "train.npz")
    with pytest.raises(ValueError, match="dataset_id"):
        verify_pair_group_benchmark(output_dir / "manifest.json")
    with pytest.raises(ValueError, match="rounds are unsupported"):
        load_pair_group_manifest(
            output_dir / "manifest.json",
            expected_dataset_id=PAIR_GROUP_SIX_ROUND_DATASET_ID,
            expected_rounds=PAIR_GROUP_ROUNDS,
        )
    # An unregistered dataset id keeps the frozen five-round expectation, so a
    # six-round manifest cannot be loaded under an unknown id without an
    # explicit round opt-in.
    unregistered_id = "speck32_64_pair_group_unregistered_r6"
    unregistered_dir = tmp_path / "unregistered"
    export_pair_group_benchmark(
        _six_round_splits(),
        unregistered_dir,
        dataset_id=unregistered_id,
        generated_at=FIXED_TIMESTAMP,
        repository_commit=FIXED_COMMIT,
        rounds=PAIR_GROUP_ROUNDS_SIX,
    )
    with pytest.raises(ValueError, match="rounds are unsupported"):
        load_pair_group_manifest(
            unregistered_dir / "manifest.json",
            expected_dataset_id=unregistered_id,
        )
    assert (
        load_pair_group_manifest(
            unregistered_dir / "manifest.json",
            expected_dataset_id=unregistered_id,
            expected_rounds=PAIR_GROUP_ROUNDS_SIX,
        )["rounds"]
        == PAIR_GROUP_ROUNDS_SIX
    )


def test_five_round_export_cannot_emit_six_round_configurations(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must use 5 rounds"):
        export_pair_group_benchmark(
            _six_round_splits(),
            tmp_path / "groups",
            generated_at=FIXED_TIMESTAMP,
        )


def test_stage1_seed_role_accepts_its_own_seeds() -> None:
    assert OWNED_SEEDS == (400001, 400002)
    for seed in OWNED_SEEDS:
        validate_seed_for_role(seed, STAGE1_ROLE)
        validate_seed_for_role(seed, STAGE1_ROLE.value)
    assert STAGE1_ROLE.value == "phase11b_prime_stage1_development"
    # Registered as protected so no other role or CLI default can adopt them.
    assert set(OWNED_SEEDS) <= set(PROTECTED_RESEARCH_SEEDS)


def test_stage1_seed_role_rejects_history_and_other_protected_seeds() -> None:
    prohibited = {
        HISTORICAL_PHASE7_SEED,
        QUARANTINED_PHASE10_SEED,
        RETIRED_BLIND_SEED,
        PHASE10_DEVELOPMENT_SEED,
        101001,
        PHASE10_G2_DEVELOPMENT_SEED,
        PHASE10_G2_PROSPECTIVE_SEED,
        PHASE11A_ORACLE_SEED,
        61001,
        62001,
        *sorted(HISTORICAL_SEEDS),
    }
    assert prohibited.isdisjoint(OWNED_SEEDS)
    for seed in sorted(prohibited):
        with pytest.raises(ValueError):
            validate_seed_for_role(seed, STAGE1_ROLE)

    with pytest.raises(ValueError, match="Unknown seed role"):
        validate_seed_for_role(OWNED_SEEDS[0], "not_a_seed_role")


def test_protected_stage1_seeds_are_rejected_by_other_roles() -> None:
    for seed in OWNED_SEEDS:
        with pytest.raises(ValueError):
            validate_seed_for_role(seed, SeedRole.DEMO)
        with pytest.raises(ValueError):
            validate_seed_for_role(seed, SeedRole.PROSPECTIVE_VALIDATION)
        with pytest.raises(ValueError):
            validate_seed_for_role(seed, SeedRole.BLIND_CONFIRMATION)
        with pytest.raises(ValueError):
            validate_seed_for_role(seed, SeedRole.PHASE11A_ORACLE_EVALUATION)
