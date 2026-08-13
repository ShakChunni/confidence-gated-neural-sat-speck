"""Integration tests for the committed Dataset v1 fixtures."""

import json
from pathlib import Path

from neural_sat_speck.data.attack import (
    AttackInstanceConfig,
    generate_attack_instances,
    load_attack_shard,
)
from neural_sat_speck.data.export import (
    load_classifier_manifest,
    load_classifier_shard,
    verify_classifier_shard,
)
from neural_sat_speck.data.generator import (
    DatasetConfig,
    assert_disjoint_master_keys,
    generate_dataset,
)

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "dataset_v1"
FIXED_TIMESTAMP = "2026-08-13T00:00:00+00:00"


def test_classifier_fixture_matches_the_generator() -> None:
    manifest = load_classifier_manifest(FIXTURE_ROOT / "classifier" / "manifest.json")
    datasets = []
    for role, record in manifest["splits"].items():
        path = FIXTURE_ROOT / "classifier" / record["path"]
        dataset = verify_classifier_shard(path)
        expected = generate_dataset(
            DatasetConfig(
                sample_count=int(record["sample_count"]),
                rounds=int(manifest["rounds"]),
                seed=int(record["seed"]),
                split_role=role,
            ),
            include_master_keys=True,
            generated_at=FIXED_TIMESTAMP,
        )
        assert dataset.features == expected.features
        assert dataset.labels == expected.labels
        assert dataset.metadata.data_checksum == record["data_checksum"]
        datasets.append(expected)

    assert_disjoint_master_keys(datasets)


def test_attack_fixture_matches_the_generator() -> None:
    manifest = json.loads(
        (FIXTURE_ROOT / "attack" / "manifest.json").read_text(encoding="utf-8")
    )
    keys: list[tuple[int, int, int, int]] = []
    for role, record in manifest["splits"].items():
        loaded = load_attack_shard(FIXTURE_ROOT / "attack" / record["path"])
        expected = generate_attack_instances(
            AttackInstanceConfig(
                instance_count=int(record["instance_count"]),
                pair_count=int(record["pair_count"]),
                rounds=int(manifest["rounds"]),
                seed=int(record["seed"]),
                split_role=role,
            ),
            generated_at=FIXED_TIMESTAMP,
        )
        assert loaded == expected
        keys.extend(instance.master_key for instance in loaded.instances)

    assert len(keys) == len(set(keys))


def test_classifier_and_attack_fixtures_are_separate_schemas() -> None:
    classifier = json.loads(
        (FIXTURE_ROOT / "classifier" / "manifest.json").read_text(encoding="utf-8")
    )
    attack = json.loads(
        (FIXTURE_ROOT / "attack" / "manifest.json").read_text(encoding="utf-8")
    )
    assert classifier["task"] == "related_vs_random"
    assert attack["task"] == "single_key_related_pairs"
    assert classifier["dataset_id"] != attack["dataset_id"]
    assert "features" in json.loads(
        (FIXTURE_ROOT / "classifier" / "train.json").read_text(encoding="utf-8")
    )
    assert "instances" in json.loads(
        (FIXTURE_ROOT / "attack" / "train.json").read_text(encoding="utf-8")
    )


def test_classifier_fixture_can_be_loaded_as_plain_json() -> None:
    shard = json.loads(
        (FIXTURE_ROOT / "classifier" / "test.json").read_text(encoding="utf-8")
    )
    loaded = load_classifier_shard(FIXTURE_ROOT / "classifier" / "test.json")
    assert [list(feature) for feature in loaded.features] == shard["features"]
    assert list(loaded.labels) == shard["labels"]
