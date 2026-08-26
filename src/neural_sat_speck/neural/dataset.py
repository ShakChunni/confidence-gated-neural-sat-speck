"""PyTorch dataset interface for Dataset v1 SPECK32/64 classifier shards."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from neural_sat_speck.data.export import load_classifier_shard
from neural_sat_speck.data.generator import GeneratedDataset


def features_to_tensor(features: Any) -> torch.Tensor:
    """Convert a sequence of 64-bit binary features into a float tensor of shape (N, 4, 16).

    Channel order (dim 1):
      0: First ciphertext x (bits 0..15, MSB to LSB)
      1: First ciphertext y (bits 16..31, MSB to LSB)
      2: Second ciphertext x (bits 32..47, MSB to LSB)
      3: Second ciphertext y (bits 48..63, MSB to LSB)
    """
    tensor_64 = torch.as_tensor(features, dtype=torch.float32)
    if tensor_64.ndim == 1:
        if tensor_64.shape[0] != 64:
            raise ValueError(f"expected 64 bits per sample, got {tensor_64.shape[0]}")
        return tensor_64.view(4, 16)
    if tensor_64.ndim == 2:
        if tensor_64.shape[1] != 64:
            raise ValueError(f"expected 64 bits per sample, got {tensor_64.shape[1]}")
        return tensor_64.view(-1, 4, 16)
    raise ValueError(f"unsupported feature tensor dimensions: {tensor_64.shape}")


class SpeckClassifierDataset(Dataset):
    """PyTorch Dataset consuming frozen Dataset v1 classifier benchmarks."""

    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if len(features) != len(labels):
            raise ValueError("features and labels must have the same length")
        if features.ndim != 3 or features.shape[1:] != (4, 16):
            raise ValueError(
                f"features must have shape (N, 4, 16), got {features.shape}"
            )

        self.features = features.float()
        self.labels = labels.view(-1, 1).float()
        self.metadata = metadata or {}

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]

    @classmethod
    def from_json_path(cls, path: Path) -> SpeckClassifierDataset:
        """Load a classifier shard from a canonical Dataset v1 JSON file."""
        dataset = load_classifier_shard(path)
        features_tensor = features_to_tensor(dataset.features)
        labels_tensor = torch.tensor(dataset.labels, dtype=torch.float32)
        return cls(
            features=features_tensor,
            labels=labels_tensor,
            metadata=dataset.metadata.to_dict(),
        )

    @classmethod
    def from_generated_dataset(cls, dataset: GeneratedDataset) -> SpeckClassifierDataset:
        """Create a dataset from an in-memory GeneratedDataset instance."""
        features_tensor = features_to_tensor(dataset.features)
        labels_tensor = torch.tensor(dataset.labels, dtype=torch.float32)
        return cls(
            features=features_tensor,
            labels=labels_tensor,
            metadata=dataset.metadata.to_dict(),
        )


def create_dataloader(
    dataset: SpeckClassifierDataset,
    batch_size: int = 64,
    shuffle: bool = False,
    seed: int | None = None,
) -> DataLoader:
    """Create a deterministic PyTorch DataLoader."""
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )
