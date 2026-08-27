"""Tests for the isolated Phase 6 pair-group data format."""

from pathlib import Path

import numpy as np
import pytest

from neural_sat_speck.data.generator import DEFAULT_INPUT_DIFFERENCE
from neural_sat_speck.data.pair_groups import (
    PAIR_GROUP_DATASET_ID,
    PairGroupConfig,
    assert_disjoint_pair_group_keys,
    export_pair_group_benchmark,
    generate_pair_groups,
    load_pair_group_shard,
    verify_pair_group_benchmark,
)
from neural_sat_speck.data.phase6a import (
    Phase6AValidationAttackConfig,
    export_phase6a_validation_attack,
    verify_phase6a_validation_attack,
)
from neural_sat_speck.speck.cipher import encrypt_block

FIXED_TIMESTAMP = "2026-08-27T00:00:00+08:00"


def _config(seed: int, role: str = "train", count: int = 8) -> PairGroupConfig:
    return PairGroupConfig(count, 64, 5, seed, role)


def test_pair_group_generation_is_deterministic_and_typed() -> None:
    first = generate_pair_groups(
        _config(61001),
        generated_at=FIXED_TIMESTAMP,
        include_debug_pairs=True,
    )
    second = generate_pair_groups(
        _config(61001),
        generated_at=FIXED_TIMESTAMP,
        include_debug_pairs=True,
    )

    assert np.array_equal(first.features, second.features)
    assert np.array_equal(first.labels, second.labels)
    assert np.array_equal(first.master_keys, second.master_keys)
    assert first.metadata == second.metadata
    assert first.features.shape == (8, 64, 4, 16)
    assert first.features.dtype == np.uint8
    assert first.labels.shape == (8,)
    assert first.labels.dtype == np.uint8
    assert first.master_keys.shape == (8, 4)
    assert first.master_keys.dtype == np.uint16
    assert first.labels.tolist().count(0) == 4
    assert first.labels.tolist().count(1) == 4
    assert first.metadata["dataset_id"] == PAIR_GROUP_DATASET_ID


def test_different_pair_group_seed_changes_content() -> None:
    first = generate_pair_groups(_config(61001))
    second = generate_pair_groups(_config(61002))
    assert not np.array_equal(first.features, second.features)
    assert first.metadata["data_checksum"] != second.metadata["data_checksum"]


def test_group_keys_and_positive_negative_construction_are_correct() -> None:
    dataset = generate_pair_groups(
        _config(61001),
        generated_at=FIXED_TIMESTAMP,
        include_debug_pairs=True,
    )
    assert dataset.plaintext_pairs is not None
    assert dataset.ciphertext_pairs is not None
    for group_index, label in enumerate(dataset.labels.tolist()):
        key = tuple(int(word) for word in dataset.master_keys[group_index])
        for pair_index in range(64):
            plaintext0 = tuple(
                int(word) for word in dataset.plaintext_pairs[group_index, pair_index, 0]
            )
            plaintext1 = tuple(
                int(word) for word in dataset.plaintext_pairs[group_index, pair_index, 1]
            )
            ciphertext0 = tuple(
                int(word) for word in dataset.ciphertext_pairs[group_index, pair_index, 0]
            )
            ciphertext1 = tuple(
                int(word) for word in dataset.ciphertext_pairs[group_index, pair_index, 1]
            )
            assert ciphertext0 == encrypt_block(plaintext0, key, rounds=5)
            assert ciphertext1 == encrypt_block(plaintext1, key, rounds=5)
            if label == 1:
                assert plaintext1 == (
                    plaintext0[0] ^ DEFAULT_INPUT_DIFFERENCE[0],
                    plaintext0[1] ^ DEFAULT_INPUT_DIFFERENCE[1],
                )
            else:
                assert plaintext1 != (
                    plaintext0[0] ^ DEFAULT_INPUT_DIFFERENCE[0],
                    plaintext0[1] ^ DEFAULT_INPUT_DIFFERENCE[1],
                )


def test_group_split_keys_are_disjoint() -> None:
    train = generate_pair_groups(_config(61001, "train"))
    validation = generate_pair_groups(_config(62001, "validation"))
    assert_disjoint_pair_group_keys((train, validation))
    with pytest.raises(ValueError):
        assert_disjoint_pair_group_keys((train, train))


def test_pair_group_export_load_and_manifest_verification(tmp_path: Path) -> None:
    output_dir = tmp_path / "groups"
    manifest = export_pair_group_benchmark(
        (_config(61001, "train"), _config(62001, "validation")),
        output_dir,
        generated_at=FIXED_TIMESTAMP,
    )
    assert manifest["dataset_id"] == PAIR_GROUP_DATASET_ID
    verified = verify_pair_group_benchmark(
        output_dir / "manifest.json",
        expected_dataset_id=PAIR_GROUP_DATASET_ID,
    )
    assert set(verified["splits"]) == {"train", "validation"}
    loaded = load_pair_group_shard(
        output_dir / "train.npz",
        expected_dataset_id=PAIR_GROUP_DATASET_ID,
    )
    assert loaded.features.shape == (8, 64, 4, 16)
    assert loaded.metadata["class_balance"] == {"0": 4, "1": 4}


def test_phase6a_attack_export_is_validation_only(tmp_path: Path) -> None:
    output_dir = tmp_path / "attack"
    manifest = export_phase6a_validation_attack(
        Phase6AValidationAttackConfig(instance_count=2, seed=64001),
        output_dir,
        dataset_id="speck32_64_attack_phase6a_test_r6",
        generated_at=FIXED_TIMESTAMP,
    )
    assert manifest["validation_only"] is True
    assert manifest["held_out_test_generated"] is False
    assert set(manifest["splits"]) == {"validation"}
    assert (output_dir / "validation.json").is_file()
    assert not (output_dir / "test.json").exists()
    verified = verify_phase6a_validation_attack(
        output_dir / "manifest.json",
        expected_dataset_id="speck32_64_attack_phase6a_test_r6",
    )
    assert verified["splits"]["validation"]["pair_count"] == 64
