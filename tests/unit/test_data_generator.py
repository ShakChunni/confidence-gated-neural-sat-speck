"""Tests for deterministic Phase 1 synthetic data generation."""

import pytest

from neural_sat_speck.data.generator import (
    DEFAULT_INPUT_DIFFERENCE,
    DatasetConfig,
    assert_disjoint_master_keys,
    encode_ciphertext_pair,
    generate_dataset,
    validate_split_configs,
    word_to_bits_msb,
)
from neural_sat_speck.data.metadata import KEY_GENERATION_POLICY

FIXED_TIMESTAMP = "2026-08-09T00:00:00+00:00"


def test_word_encoding_is_most_significant_bit_first() -> None:
    assert word_to_bits_msb(0x8001) == (
        1,
        *([0] * 14),
        1,
    )
    assert word_to_bits_msb(0x0002) == (
        *([0] * 14),
        1,
        0,
    )

    bits = encode_ciphertext_pair(
        (0x8001, 0x0002),
        (0x8004, 0x0008),
    )
    assert len(bits) == 64
    assert bits[:16] == word_to_bits_msb(0x8001)
    assert bits[16:32] == word_to_bits_msb(0x0002)
    assert bits[32:48] == word_to_bits_msb(0x8004)
    assert bits[48:] == word_to_bits_msb(0x0008)


def test_generation_is_identical_for_a_fixed_configuration_and_seed() -> None:
    config = DatasetConfig(
        sample_count=8,
        rounds=7,
        seed=1001,
        split_role="train",
    )
    first = generate_dataset(
        config,
        include_debug_pairs=True,
        generated_at=FIXED_TIMESTAMP,
    )
    second = generate_dataset(
        config,
        include_debug_pairs=True,
        generated_at=FIXED_TIMESTAMP,
    )

    assert first == second


def test_different_seeds_produce_different_samples() -> None:
    first_config = DatasetConfig(8, 7, 1001, "train")
    second_config = DatasetConfig(8, 7, 1002, "train")

    first = generate_dataset(first_config, generated_at=FIXED_TIMESTAMP)
    second = generate_dataset(second_config, generated_at=FIXED_TIMESTAMP)

    assert first.features != second.features


def test_generated_shapes_labels_and_debug_pairs_are_consistent() -> None:
    config = DatasetConfig(8, 7, 1001, "train")
    dataset = generate_dataset(
        config,
        include_debug_pairs=True,
        generated_at=FIXED_TIMESTAMP,
    )

    assert len(dataset.features) == 8
    assert len(dataset.labels) == 8
    assert all(len(feature) == 64 for feature in dataset.features)
    assert all(bit in (0, 1) for feature in dataset.features for bit in feature)
    assert dataset.labels.count(0) == 4
    assert dataset.labels.count(1) == 4
    assert dataset.plaintext_pairs is not None
    assert dataset.ciphertext_pairs is not None
    assert all(
        feature == encode_ciphertext_pair(*ciphertext_pair)
        for feature, ciphertext_pair in zip(
            dataset.features,
            dataset.ciphertext_pairs,
            strict=True,
        )
    )

    metadata = dataset.metadata
    assert metadata.sample_count == 8
    assert metadata.class_balance == (4, 4)
    assert metadata.rounds == 7
    assert metadata.input_difference == DEFAULT_INPUT_DIFFERENCE
    assert metadata.key_generation_policy == KEY_GENERATION_POLICY
    assert metadata.feature_dtype == "uint8"
    assert metadata.feature_shape == (8, 64)
    assert len(metadata.data_checksum) == 64
    assert metadata.to_dict()["class_balance"] == {"0": 4, "1": 4}
    assert all(
        0 <= word < (1 << 16)
        for pair_collection in (
            dataset.plaintext_pairs,
            dataset.ciphertext_pairs,
        )
        for pair in pair_collection
        for block in pair
        for word in block
    )


def test_positive_debug_pairs_use_the_selected_difference() -> None:
    config = DatasetConfig(8, 7, 1001, "train")
    dataset = generate_dataset(
        config,
        include_debug_pairs=True,
        generated_at=FIXED_TIMESTAMP,
    )
    assert dataset.plaintext_pairs is not None

    for label, (first, second) in zip(
        dataset.labels,
        dataset.plaintext_pairs,
        strict=True,
    ):
        if label == 1:
            assert second == (
                first[0] ^ DEFAULT_INPUT_DIFFERENCE[0],
                first[1] ^ DEFAULT_INPUT_DIFFERENCE[1],
            )


def test_split_configs_require_distinct_seeds_and_shared_protocol() -> None:
    configs = (
        DatasetConfig(8, 7, 1001, "train"),
        DatasetConfig(8, 7, 2001, "validation"),
        DatasetConfig(8, 7, 3001, "test"),
    )
    validate_split_configs(configs)
    datasets = [
        generate_dataset(config, generated_at=FIXED_TIMESTAMP)
        for config in configs
    ]
    assert len({dataset.metadata.data_checksum for dataset in datasets}) == 3

    keyed = [
        generate_dataset(
            config,
            include_master_keys=True,
            generated_at=FIXED_TIMESTAMP,
        )
        for config in configs
    ]
    assert_disjoint_master_keys(keyed)
    assert keyed[0].features == datasets[0].features

    with pytest.raises(ValueError):
        validate_split_configs(
            (
                configs[0],
                DatasetConfig(8, 7, 1001, "validation"),
                configs[2],
            )
        )

    with pytest.raises(ValueError):
        validate_split_configs(
            (
                configs[0],
                DatasetConfig(8, 11, 2001, "validation"),
                configs[2],
            )
        )


@pytest.mark.parametrize(
    ("sample_count", "rounds", "seed", "split_role"),
    [
        (0, 7, 1001, "train"),
        (3, 7, 1001, "train"),
        (8, 0, 1001, "train"),
        (8, 23, 1001, "train"),
        (8, 7, 1001, "holdout"),
    ],
)
def test_invalid_dataset_configuration_is_rejected(
    sample_count: int,
    rounds: int,
    seed: int,
    split_role: str,
) -> None:
    with pytest.raises(ValueError):
        DatasetConfig(sample_count, rounds, seed, split_role)
