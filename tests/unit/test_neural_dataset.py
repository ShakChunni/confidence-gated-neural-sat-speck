"""Tests for the PyTorch dataset interface."""

from pathlib import Path

from neural_sat_speck.data.generator import DatasetConfig, generate_dataset
from neural_sat_speck.neural.dataset import (
    SpeckClassifierDataset,
    create_dataloader,
    features_to_tensor,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "fixtures"
    / "dataset_v1"
    / "classifier"
    / "train.json"
)


def test_features_to_tensor_reshapes_64_bits_to_4_words() -> None:
    raw_64 = list(range(64))
    tensor = features_to_tensor(raw_64)
    assert tensor.shape == (4, 16)
    assert tensor[0, 0].item() == 0.0
    assert tensor[3, 15].item() == 63.0

    batch_64 = [raw_64, raw_64]
    batch_tensor = features_to_tensor(batch_64)
    assert batch_tensor.shape == (2, 4, 16)


def test_dataset_from_fixture_json() -> None:
    dataset = SpeckClassifierDataset.from_json_path(FIXTURE_PATH)
    assert len(dataset) == 4
    features, label = dataset[0]
    assert features.shape == (4, 16)
    assert label.shape == (1,)
    assert label.item() in (0.0, 1.0)


def test_dataset_from_generated_in_memory() -> None:
    config = DatasetConfig(
        sample_count=8,
        rounds=7,
        seed=1001,
        split_role="train",
    )
    generated = generate_dataset(config)
    dataset = SpeckClassifierDataset.from_generated_dataset(generated)
    assert len(dataset) == 8

    loader = create_dataloader(dataset, batch_size=4, shuffle=False)
    batches = list(loader)
    assert len(batches) == 2
    batch_features, batch_labels = batches[0]
    assert batch_features.shape == (4, 4, 16)
    assert batch_labels.shape == (4, 1)
