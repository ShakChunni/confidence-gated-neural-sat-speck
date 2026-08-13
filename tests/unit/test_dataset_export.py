"""Tests for Dataset v1 classifier export and checksums."""

import json
from pathlib import Path

import pytest

from neural_sat_speck.data.export import (
    export_classifier_benchmark,
    load_classifier_manifest,
    load_classifier_shard,
    verify_classifier_shard,
)
from neural_sat_speck.data.format import (
    CLASSIFIER_DATASET_ID,
    CLASSIFIER_TASK,
    DATASET_FORMAT_VERSION,
)
from neural_sat_speck.data.generator import (
    DatasetConfig,
    generate_dataset,
    validate_split_configs,
)

FIXED_TIMESTAMP = "2026-08-13T00:00:00+00:00"


def _split_configs() -> tuple[DatasetConfig, DatasetConfig, DatasetConfig]:
    return (
        DatasetConfig(8, 7, 1001, "train"),
        DatasetConfig(8, 7, 2001, "validation"),
        DatasetConfig(8, 7, 3001, "test"),
    )


def test_classifier_export_is_identical_for_a_fixed_seed(
    tmp_path: Path,
) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    configs = _split_configs()

    first = export_classifier_benchmark(
        configs, first_dir, generated_at=FIXED_TIMESTAMP
    )
    second = export_classifier_benchmark(
        configs, second_dir, generated_at=FIXED_TIMESTAMP
    )

    assert first == second
    for name in ("manifest.json", "train.json", "validation.json", "test.json"):
        assert (first_dir / name).read_bytes() == (second_dir / name).read_bytes()


def test_different_seeds_change_exported_classifier_data(
    tmp_path: Path,
) -> None:
    first = export_classifier_benchmark(
        _split_configs(),
        tmp_path / "first",
        generated_at=FIXED_TIMESTAMP,
    )
    shifted = (
        DatasetConfig(8, 7, 1101, "train"),
        DatasetConfig(8, 7, 2101, "validation"),
        DatasetConfig(8, 7, 3101, "test"),
    )
    second = export_classifier_benchmark(
        shifted,
        tmp_path / "second",
        generated_at=FIXED_TIMESTAMP,
    )

    assert first["splits"]["train"]["data_checksum"] != (
        second["splits"]["train"]["data_checksum"]
    )


def test_exported_classifier_metadata_and_balance_are_complete(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "classifier"
    manifest = export_classifier_benchmark(
        _split_configs(),
        output_dir,
        generated_at=FIXED_TIMESTAMP,
    )

    assert manifest["dataset_id"] == CLASSIFIER_DATASET_ID
    assert manifest["dataset_format_version"] == DATASET_FORMAT_VERSION
    assert manifest["task"] == CLASSIFIER_TASK
    assert manifest["rounds"] == 7
    assert manifest["input_difference"] == [64, 0]
    assert set(manifest["splits"]) == {"train", "validation", "test"}

    loaded_manifest = load_classifier_manifest(output_dir / "manifest.json")
    assert loaded_manifest["splits"]["train"]["file_sha256"] == (
        manifest["splits"]["train"]["file_sha256"]
    )

    dataset = verify_classifier_shard(output_dir / "train.json")
    assert dataset.metadata.feature_shape == (8, 64)
    assert dataset.metadata.feature_dtype == "uint8"
    assert dataset.labels.count(0) == 4
    assert dataset.labels.count(1) == 4
    assert all(len(feature) == 64 for feature in dataset.features)
    assert all(bit in (0, 1) for feature in dataset.features for bit in feature)


def test_exported_classifier_loads_without_the_generator(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "classifier"
    export_classifier_benchmark(
        _split_configs(),
        output_dir,
        generated_at=FIXED_TIMESTAMP,
    )
    document = json.loads((output_dir / "test.json").read_text(encoding="utf-8"))

    features = document["features"]
    labels = document["labels"]
    assert document["schema"].startswith("neural_sat_speck.classifier_shard.")
    assert len(features) == len(labels) == 8
    assert document["metadata"]["feature_dtype"] == "uint8"
    assert [len(row) for row in features] == [64] * 8


def test_classifier_checksums_detect_a_tampered_shard(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "classifier"
    export_classifier_benchmark(
        _split_configs(),
        output_dir,
        generated_at=FIXED_TIMESTAMP,
    )
    shard_path = output_dir / "validation.json"
    document = json.loads(shard_path.read_text(encoding="utf-8"))
    document["labels"][0] = 1 - int(document["labels"][0])
    shard_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError):
        verify_classifier_shard(shard_path)


def test_loaded_classifier_shard_matches_in_memory_generation(
    tmp_path: Path,
) -> None:
    config = DatasetConfig(8, 7, 1001, "train")
    export_classifier_benchmark(
        _split_configs(),
        tmp_path,
        generated_at=FIXED_TIMESTAMP,
    )
    exported = load_classifier_shard(tmp_path / "train.json")
    generated = generate_dataset(config, generated_at=FIXED_TIMESTAMP)

    assert exported.features == generated.features
    assert exported.labels == generated.labels
    assert exported.metadata == generated.metadata


def test_split_validation_still_requires_one_role_each() -> None:
    with pytest.raises(ValueError):
        validate_split_configs(
            (
                DatasetConfig(8, 7, 1001, "train"),
                DatasetConfig(8, 7, 2001, "train"),
                DatasetConfig(8, 7, 3001, "test"),
            )
        )
