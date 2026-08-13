"""Tests that the frozen Dataset v1 contract stays consistent."""

import json
from pathlib import Path

from neural_sat_speck.data.format import (
    ATTACK_DATASET_ID,
    CLASSIFIER_DATASET_ID,
    DATASET_FORMAT_VERSION,
    DATASET_LICENSE,
    PLAINTEXT_DIFFERENCE_POLICY,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CLASSIFIER_CONFIG = REPOSITORY_ROOT / "configs" / "data" / "dataset_v1_classifier.json"
ATTACK_CONFIG = REPOSITORY_ROOT / "configs" / "data" / "dataset_v1_attack.json"
CLASSIFIER_MANIFEST = (
    REPOSITORY_ROOT / "tests" / "fixtures" / "dataset_v1" / "classifier" / "manifest.json"
)
ATTACK_MANIFEST = (
    REPOSITORY_ROOT / "tests" / "fixtures" / "dataset_v1" / "attack" / "manifest.json"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_classifier_freeze_matches_code_and_fixture() -> None:
    config = _load(CLASSIFIER_CONFIG)
    manifest = _load(CLASSIFIER_MANIFEST)

    assert config["status"] == "frozen"
    assert config["dataset_id"] == CLASSIFIER_DATASET_ID
    assert config["dataset_format_version"] == DATASET_FORMAT_VERSION
    assert config["license"] == DATASET_LICENSE
    assert config["task"] == "related_vs_random"
    assert config["rounds"] == 7
    assert config["input_difference"] == [64, 0]
    assert manifest["dataset_id"] == config["dataset_id"]
    assert manifest["rounds"] == config["rounds"]
    assert manifest["input_difference"] == config["input_difference"]
    for role in ("train", "validation", "test"):
        assert manifest["splits"][role]["seed"] == config["splits"][role]["seed"]
        assert manifest["splits"][role]["sample_count"] == (
            config["splits"][role]["sample_count"]
        )


def test_attack_freeze_uses_plaintext_difference() -> None:
    config = _load(ATTACK_CONFIG)
    manifest = _load(ATTACK_MANIFEST)

    assert config["status"] == "frozen"
    assert config["dataset_id"] == ATTACK_DATASET_ID
    assert config["pair_construction"] == PLAINTEXT_DIFFERENCE_POLICY
    assert config["task"] == "single_key_related_pairs"
    assert config["rounds"] == 7
    assert config["input_difference"] == [64, 0]
    assert manifest["dataset_id"] == config["dataset_id"]
    assert manifest["pair_construction"] == PLAINTEXT_DIFFERENCE_POLICY
    assert manifest["task"] != "related_vs_random"
    for role in ("train", "validation", "test"):
        assert manifest["splits"][role]["seed"] == config["splits"][role]["seed"]


def test_classifier_and_attack_remain_separate_frozen_datasets() -> None:
    classifier = _load(CLASSIFIER_CONFIG)
    attack = _load(ATTACK_CONFIG)
    assert classifier["dataset_id"] != attack["dataset_id"]
    assert classifier["task"] != attack["task"]
    classifier_seeds = {
        classifier["splits"][role]["seed"]
        for role in ("train", "validation", "test")
    }
    attack_seeds = {
        attack["splits"][role]["seed"] for role in ("train", "validation", "test")
    }
    assert classifier_seeds.isdisjoint(attack_seeds)
