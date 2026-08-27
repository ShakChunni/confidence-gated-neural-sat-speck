"""PyTorch dataset helpers for Phase 6 pair groups."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from neural_sat_speck.data.pair_groups import (
    GeneratedPairGroups,
    load_pair_group_shard,
)


class PairGroupDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """A float32 tensor view of binary ``(N,64,4,16)`` group features."""

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        if not isinstance(features, np.ndarray) or not isinstance(labels, np.ndarray):
            raise TypeError("features and labels must be NumPy arrays")
        if features.dtype != np.dtype(np.uint8):
            raise ValueError("pair-group features must have dtype uint8")
        if labels.dtype != np.dtype(np.uint8):
            raise ValueError("pair-group labels must have dtype uint8")
        if features.ndim != 4 or features.shape[1:] != (64, 4, 16):
            raise ValueError("pair-group features must have shape (N,64,4,16)")
        if labels.shape != (features.shape[0],):
            raise ValueError("pair-group labels must have shape (N,)")
        if not np.isin(features, (0, 1)).all():
            raise ValueError("pair-group features must be binary")
        if not np.isin(labels, (0, 1)).all():
            raise ValueError("pair-group labels must be binary")

        contiguous_features = np.ascontiguousarray(features)
        contiguous_labels = np.ascontiguousarray(labels)
        self.features = torch.from_numpy(contiguous_features).to(torch.float32)
        self.labels = torch.from_numpy(contiguous_labels).to(torch.float32).view(-1, 1)

    @classmethod
    def from_generated_dataset(cls, dataset: GeneratedPairGroups) -> PairGroupDataset:
        """Construct a dataset from generated in-memory arrays."""

        return cls(dataset.features, dataset.labels)

    @classmethod
    def from_shard(
        cls,
        path: Path,
        *,
        expected_dataset_id: str | None = None,
    ) -> PairGroupDataset:
        """Load a verified group shard without exposing its audit keys."""

        generated = load_pair_group_shard(
            path,
            expected_dataset_id=expected_dataset_id,
        )
        return cls(generated.features, generated.labels)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


def create_pair_group_dataloader(
    dataset: PairGroupDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int | None = None,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Create a single-worker loader with an explicit optional shuffle seed."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not isinstance(shuffle, bool):
        raise TypeError("shuffle must be a boolean")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("seed must be an integer or None")
    generator = None
    if shuffle:
        if seed is None:
            raise ValueError("a seed is required when shuffling pair groups")
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


__all__ = ["PairGroupDataset", "create_pair_group_dataloader"]
