"""Tests for Dataset v1 single-key attack instances."""

import json
from pathlib import Path

import pytest

from neural_sat_speck.data.attack import (
    AttackInstanceConfig,
    assert_disjoint_attack_keys,
    export_attack_benchmark,
    generate_attack_instances,
    load_attack_manifest,
    load_attack_shard,
    validate_attack_split_configs,
    verify_attack_benchmark,
    verify_attack_shard,
)
from neural_sat_speck.data.format import ATTACK_DATASET_ID, ATTACK_TASK
from neural_sat_speck.data.generator import DEFAULT_INPUT_DIFFERENCE
from neural_sat_speck.speck.cipher import encrypt_block
from neural_sat_speck.speck.key_schedule import expand_key

FIXED_TIMESTAMP = "2026-08-13T00:00:00+00:00"


def _split_configs() -> tuple[
    AttackInstanceConfig, AttackInstanceConfig, AttackInstanceConfig
]:
    return (
        AttackInstanceConfig(2, 2, 7, 4001, "train"),
        AttackInstanceConfig(2, 2, 7, 6001, "validation"),
        AttackInstanceConfig(2, 2, 7, 5001, "test"),
    )


def test_attack_generation_is_identical_for_a_fixed_seed() -> None:
    config = AttackInstanceConfig(2, 2, 7, 4001, "train")
    first = generate_attack_instances(config, generated_at=FIXED_TIMESTAMP)
    second = generate_attack_instances(config, generated_at=FIXED_TIMESTAMP)
    assert first == second


def test_different_attack_seeds_produce_different_instances() -> None:
    first = generate_attack_instances(
        AttackInstanceConfig(2, 2, 7, 4001, "train"),
        generated_at=FIXED_TIMESTAMP,
    )
    second = generate_attack_instances(
        AttackInstanceConfig(2, 2, 7, 4002, "train"),
        generated_at=FIXED_TIMESTAMP,
    )
    assert first.instances != second.instances
    assert first.metadata["data_checksum"] != second.metadata["data_checksum"]


def test_attack_pairs_share_one_key_and_the_selected_difference() -> None:
    dataset = generate_attack_instances(
        AttackInstanceConfig(2, 2, 7, 4001, "train"),
        generated_at=FIXED_TIMESTAMP,
    )
    assert len(dataset.instances) == 2
    for instance in dataset.instances:
        assert instance.final_round_subkey == expand_key(
            instance.master_key, 7
        )[-1]
        assert len(instance.pairs) == 2
        for pair in instance.pairs:
            assert pair.plaintext1 == (
                pair.plaintext0[0] ^ DEFAULT_INPUT_DIFFERENCE[0],
                pair.plaintext0[1] ^ DEFAULT_INPUT_DIFFERENCE[1],
            )
            assert pair.ciphertext0 == encrypt_block(
                pair.plaintext0, instance.master_key, 7
            )
            assert pair.ciphertext1 == encrypt_block(
                pair.plaintext1, instance.master_key, 7
            )


def test_attack_split_keys_do_not_overlap() -> None:
    configs = _split_configs()
    validate_attack_split_configs(configs)
    datasets = [
        generate_attack_instances(config, generated_at=FIXED_TIMESTAMP)
        for config in configs
    ]
    assert_disjoint_attack_keys(datasets)

    with pytest.raises(ValueError):
        validate_attack_split_configs(
            (
                configs[0],
                AttackInstanceConfig(2, 2, 7, 4001, "validation"),
                configs[2],
            )
        )


def test_attack_export_checksums_and_independent_load(
    tmp_path: Path,
) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = export_attack_benchmark(
        _split_configs(),
        first_dir,
        generated_at=FIXED_TIMESTAMP,
    )
    second = export_attack_benchmark(
        _split_configs(),
        second_dir,
        generated_at=FIXED_TIMESTAMP,
    )
    assert first == second
    assert first["dataset_id"] == ATTACK_DATASET_ID
    assert first["task"] == ATTACK_TASK

    raw = json.loads((first_dir / "test.json").read_text(encoding="utf-8"))
    assert raw["schema"].startswith("neural_sat_speck.attack_shard.")
    assert len(raw["instances"]) == 2
    assert raw["instances"][0]["master_key"]
    assert raw["metadata"]["pair_construction"] == "plaintext_difference"

    loaded = load_attack_shard(first_dir / "test.json")
    generated = generate_attack_instances(
        AttackInstanceConfig(2, 2, 7, 5001, "test"),
        generated_at=FIXED_TIMESTAMP,
    )
    assert loaded == generated

    assert verify_attack_shard(first_dir / "test.json") == loaded
    assert load_attack_manifest(first_dir / "manifest.json")["dataset_id"] == (
        ATTACK_DATASET_ID
    )
    assert verify_attack_benchmark(first_dir / "manifest.json")["task"] == (
        ATTACK_TASK
    )


def test_attack_loader_rejects_tampered_ciphertext(tmp_path: Path) -> None:
    output_dir = tmp_path / "attack"
    export_attack_benchmark(
        _split_configs(),
        output_dir,
        generated_at=FIXED_TIMESTAMP,
    )
    shard_path = output_dir / "test.json"
    document = json.loads(shard_path.read_text(encoding="utf-8"))
    current = int(document["instances"][0]["pairs"][0]["ciphertext0"][0], 16)
    document["instances"][0]["pairs"][0]["ciphertext0"][0] = f"{current ^ 1:04x}"
    shard_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError):
        load_attack_shard(shard_path)


@pytest.mark.parametrize(
    ("instance_count", "pair_count", "rounds", "split_role"),
    [
        (0, 2, 7, "train"),
        (2, 0, 7, "train"),
        (2, 2, 0, "train"),
        (2, 2, 23, "train"),
        (2, 2, 7, "holdout"),
    ],
)
def test_invalid_attack_configuration_is_rejected(
    instance_count: int,
    pair_count: int,
    rounds: int,
    split_role: str,
) -> None:
    with pytest.raises(ValueError):
        AttackInstanceConfig(instance_count, pair_count, rounds, 4001, split_role)
