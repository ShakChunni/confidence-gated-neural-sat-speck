"""Integration test for the committed tiny deterministic data fixture."""

import json
from pathlib import Path

from neural_sat_speck.data.generator import DatasetConfig, generate_dataset

FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "tiny_dataset.json"


def test_tiny_fixture_matches_the_generator() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text())
    dataset = generate_dataset(
        DatasetConfig(
            sample_count=4,
            rounds=7,
            seed=1001,
            split_role="train",
        ),
        include_debug_pairs=True,
        generated_at="2026-08-09T00:00:00+00:00",
    )

    assert dataset.metadata.to_dict() == fixture["metadata"]
    assert list(dataset.labels) == fixture["labels"]
    assert [
        "".join(str(bit) for bit in feature)
        for feature in dataset.features
    ] == fixture["features"]
    assert dataset.plaintext_pairs is not None
    assert dataset.ciphertext_pairs is not None
    assert [
        [[f"{word:04x}" for word in block] for block in pair]
        for pair in dataset.plaintext_pairs
    ] == fixture["plaintext_pairs"]
    assert [
        [[f"{word:04x}" for word in block] for block in pair]
        for pair in dataset.ciphertext_pairs
    ] == fixture["ciphertext_pairs"]
