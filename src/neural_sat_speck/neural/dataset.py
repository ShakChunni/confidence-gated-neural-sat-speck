"""PyTorch dataset interface for Dataset v1 SPECK32/64 classifier shards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from neural_sat_speck.data.export import (
    CLASSIFIER_DATASET_ID,
    verify_classifier_benchmark,
    verify_classifier_shard,
)
from neural_sat_speck.data.generator import GeneratedDataset


def features_to_tensor(features: Any) -> torch.Tensor:
    """Convert a sequence of 64-bit binary features into a float tensor of shape (N, 4, 16).

    Channel order (dim 1):
      0: First ciphertext x (bits 0..15, MSB to LSB)
      1: First ciphertext y (bits 16..31, MSB to LSB)
      2: Second ciphertext x (bits 32..47, MSB to LSB)
      3: Second ciphertext y (bits 48..63, MSB to LSB)
    """
    try:
        raw_tensor = torch.as_tensor(features)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("features must be a numeric sequence") from error
    if raw_tensor.ndim not in (1, 2):
        raise ValueError(f"unsupported feature tensor dimensions: {raw_tensor.shape}")
    if raw_tensor.numel() == 0:
        raise ValueError("features cannot be empty")
    if raw_tensor.is_floating_point() and not torch.isfinite(raw_tensor).all():
        raise ValueError("features must contain finite values")
    if ((raw_tensor != 0) & (raw_tensor != 1)).any():
        raise ValueError("features must contain only binary 0/1 values")
    tensor_64 = raw_tensor.to(dtype=torch.float32)
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
        if not isinstance(features, torch.Tensor) or not isinstance(
            labels, torch.Tensor
        ):
            raise TypeError("features and labels must be torch tensors")
        if len(features) != len(labels):
            raise ValueError("features and labels must have the same length")
        if len(features) == 0:
            raise ValueError("classifier datasets cannot be empty")
        if features.ndim != 3 or features.shape[1:] != (4, 16):
            raise ValueError(
                f"features must have shape (N, 4, 16), got {features.shape}"
            )
        if not torch.isfinite(features).all():
            raise ValueError("features must contain finite values")
        if ((features != 0) & (features != 1)).any():
            raise ValueError("features must contain only binary 0/1 values")

        flat_labels = labels.detach().view(-1)
        if flat_labels.numel() != len(features):
            raise ValueError("labels must contain one value per feature row")
        if flat_labels.is_floating_point() and not torch.isfinite(flat_labels).all():
            raise ValueError("labels must contain finite values")
        if ((flat_labels != 0) & (flat_labels != 1)).any():
            raise ValueError("labels must contain only binary 0/1 values")

        self.features = features.float()
        self.labels = flat_labels.view(-1, 1).float()
        self.metadata = metadata or {}

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]

    @classmethod
    def from_json_path(
        cls,
        path: Path,
        *,
        expected_dataset_id: str | None = CLASSIFIER_DATASET_ID,
    ) -> SpeckClassifierDataset:
        """Load and verify a classifier shard from canonical JSON."""
        dataset = verify_classifier_shard(
            path,
            expected_dataset_id=expected_dataset_id,
        )
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


@dataclass(frozen=True)
class ClassifierSplits:
    """The three verified classifier splits used by an official run."""

    train: SpeckClassifierDataset
    validation: SpeckClassifierDataset
    test: SpeckClassifierDataset
    manifest: dict[str, Any]

    def as_dict(self) -> dict[str, SpeckClassifierDataset]:
        """Return splits by their stable role names."""

        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }


def load_classifier_splits(
    root: Path,
    *,
    expected_dataset_id: str | None = CLASSIFIER_DATASET_ID,
    expected_rounds: int | None = None,
    expected_input_difference: tuple[int, int] | None = None,
) -> ClassifierSplits:
    """Verify and load all train/validation/test shards from one manifest.

    This is the entry point for official Phase 3 training. It deliberately
    accepts an existing export directory and never calls the data generator.
    """

    manifest_path = root / "manifest.json"
    manifest = verify_classifier_benchmark(
        manifest_path,
        expected_dataset_id=expected_dataset_id,
        expected_rounds=expected_rounds,
        expected_input_difference=expected_input_difference,
    )
    dataset_id = str(manifest["dataset_id"])
    splits: dict[str, SpeckClassifierDataset] = {}
    manifest_splits = manifest["splits"]
    for role in ("train", "validation", "test"):
        record = manifest_splits[role]
        splits[role] = SpeckClassifierDataset.from_json_path(
            root / str(record["path"]),
            expected_dataset_id=dataset_id,
        )
    return ClassifierSplits(
        train=splits["train"],
        validation=splits["validation"],
        test=splits["test"],
        manifest=manifest,
    )


def create_dataloader(
    dataset: SpeckClassifierDataset,
    batch_size: int = 64,
    shuffle: bool = False,
    seed: int | None = None,
) -> DataLoader:
    """Create a deterministic PyTorch DataLoader."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
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
