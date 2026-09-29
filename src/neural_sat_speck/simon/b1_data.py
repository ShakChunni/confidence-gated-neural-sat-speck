"""Faithful SIMON32/64 B0/B1 dataset generator, PyTorch datasets, and training infrastructure.

Invariants and Scientific Rules (AGENTS.md, D-029, D-034, D-035, D-037):
========================================================================
1. Fixed 9-round direct-pair construction:
   - Input difference: (0x0000, 0x0040) matching the Beaulieu / Bao et al. differential trail.
   - Positive pairs: encrypted under the group's true master key with P1 = P0 ^ (0x0000, 0x0040).
   - Negative pairs: encrypted under the group's true master key with independent random
     second plaintexts (or random ciphertexts/random differences).
2. Group construction:
   - All 64 pairs in a B1 group share the exact same master-key role.
   - Master key is generated per group; no pair within a group uses a divergent key.
3. Strict master-key splitting:
   - Training and validation sets are partitioned strictly by master key, NEVER by pair.
   - Distinct seeds do not prove key disjointness; the training runner must
     check exact train/validation intersections and within-split duplicates
     before any optimizer step.
4. Feature layout:
   - Bit-sliced shape (4, 16) for (L0, R0, L1, R1) encoded MSB-first in {0, 1}.
   - Group feature tensor shape: (N, 64, 4, 16).
   - B0 pair feature tensor shape: (N, 4, 16).
5. Outcome-independent stop and promotion rule:
   - Models are retained and reported descriptively even if validation distinguishing
     accuracy is weak (e.g. 50.8%).
   - Rejection occurs only for an unusable/non-finite checkpoint or execution
     defect. Finite flat-loss models remain reportable negative evidence.
   - Weak models are never hidden or silently dropped.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from neural_sat_speck.neural.metrics import binary_accuracy, roc_auc_score
from neural_sat_speck.simon.faithful_b1_model import (
    SimonGohrResNet,
    SimonGohrResNetConfig,
    SimonPairGroupRanker,
)
from neural_sat_speck.simon.key_schedule import expand_key

# Authoritative default constants for 9-round SIMON32/64 B0/B1
DEFAULT_SIMON_INPUT_DIFFERENCE: Final[tuple[int, int]] = (0x0000, 0x0040)
SIMON_B1_GROUP_SIZE: Final[int] = 64
SIMON_DEFAULT_ROUNDS: Final[int] = 9
WORD_MASK: Final[int] = 0xFFFF


# -----------------------------------------------------------------------------
# Data Containers
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SimonGeneratedPairGroups:
    """In-memory generated group features, labels, master keys, and audit metadata.

    Features shape: (group_count, group_size, 4, 16), uint8 in {0, 1}.
    Labels shape: (group_count,), uint8 in {0, 1}.
    Master keys shape: (group_count, 4), uint16.
    """

    features: np.ndarray
    labels: np.ndarray
    master_keys: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if self.features.ndim != 4 or self.features.shape[2:] != (4, 16):
            raise ValueError(
                f"features must have shape (N, G, 4, 16), got {self.features.shape}"
            )
        if self.labels.ndim != 1 or len(self.labels) != len(self.features):
            raise ValueError("labels must be 1D with length matching features")
        if self.master_keys.ndim != 2 or self.master_keys.shape != (len(self.features), 4):
            raise ValueError("master_keys must have shape (N, 4)")
        if self.features.dtype != np.uint8 or self.labels.dtype != np.uint8:
            raise TypeError("features and labels must have dtype uint8")
        if self.master_keys.dtype != np.uint16:
            raise TypeError("master_keys must have dtype uint16")

    @property
    def group_count(self) -> int:
        return len(self.labels)

    @property
    def group_size(self) -> int:
        return self.features.shape[1]

    @property
    def pair_count(self) -> int:
        return self.group_count * self.group_size

    def to_b0_data(self) -> SimonGeneratedB0Data:
        """Flatten group structure into individual pair samples (for B0 pretraining).

        Replicates group labels and master keys across the group_size pairs.
        Maintains group integrity: all pairs from one group inherit the group's master key.
        """
        n, g, c, w = self.features.shape
        flat_features = self.features.reshape(n * g, c, w)
        flat_labels = np.repeat(self.labels, g)
        flat_keys = np.repeat(self.master_keys, g, axis=0)
        meta = dict(self.metadata)
        meta["derived_from_groups"] = True
        meta["source_group_count"] = n
        meta["source_group_size"] = g
        return SimonGeneratedB0Data(
            features=flat_features,
            labels=flat_labels,
            master_keys=flat_keys,
            metadata=meta,
        )


@dataclass(frozen=True)
class SimonGeneratedB0Data:
    """In-memory generated single-pair features, labels, master keys, and metadata.

    Features shape: (pair_count, 4, 16), uint8 in {0, 1}.
    Labels shape: (pair_count,), uint8 in {0, 1}.
    Master keys shape: (pair_count, 4), uint16.
    """

    features: np.ndarray
    labels: np.ndarray
    master_keys: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if self.features.ndim != 3 or self.features.shape[1:] != (4, 16):
            raise ValueError(
                f"features must have shape (N, 4, 16), got {self.features.shape}"
            )
        if self.labels.ndim != 1 or len(self.labels) != len(self.features):
            raise ValueError("labels must be 1D with length matching features")
        if self.master_keys.ndim != 2 or self.master_keys.shape != (len(self.features), 4):
            raise ValueError("master_keys must have shape (N, 4)")
        if self.features.dtype != np.uint8 or self.labels.dtype != np.uint8:
            raise TypeError("features and labels must have dtype uint8")
        if self.master_keys.dtype != np.uint16:
            raise TypeError("master_keys must have dtype uint16")

    @property
    def sample_count(self) -> int:
        return len(self.labels)


# -----------------------------------------------------------------------------
# Bit & Checksum Utilities
# -----------------------------------------------------------------------------

def encode_words_to_bits(words: np.ndarray) -> np.ndarray:
    """Convert (..., 4) uint16 words to (..., 4, 16) MSB-first uint8 bits."""
    if words.shape[-1] != 4:
        raise ValueError(f"expected last dimension to be 4 words, got {words.shape}")
    shifts = np.arange(15, -1, -1, dtype=np.uint16)
    return ((words[..., None] >> shifts) & 1).astype(np.uint8)


def compute_content_checksum(
    features: np.ndarray,
    labels: np.ndarray,
    master_keys: np.ndarray,
) -> str:
    """Compute deterministic SHA-256 digest over array contents and shapes."""
    digest = hashlib.sha256()
    for name, arr in (
        ("features", features),
        ("labels", labels),
        ("master_keys", master_keys),
    ):
        contiguous = np.ascontiguousarray(arr)
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


# -----------------------------------------------------------------------------
# Core SIMON Vectorised Encryption for Batch Generation
# -----------------------------------------------------------------------------

def _encrypt_simon_arrays(
    l_pt: np.ndarray,
    r_pt: np.ndarray,
    round_keys: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Encrypt 16-bit word arrays through SIMON rounds using NumPy bitwise operations."""
    left_arr = l_pt.astype(np.uint16, copy=True)
    right_arr = r_pt.astype(np.uint16, copy=True)
    for rk in round_keys:
        f = (((left_arr << 1) | (left_arr >> 15)) & ((left_arr << 8) | (left_arr >> 8))) ^ ((left_arr << 2) | (left_arr >> 14))
        new_left = (right_arr ^ f ^ np.uint16(rk)) & np.uint16(0xFFFF)
        right_arr = left_arr
        left_arr = new_left
    return left_arr, right_arr


# -----------------------------------------------------------------------------
# Data Generators
# -----------------------------------------------------------------------------

def audit_training_key_roles(
    b0_train_keys: np.ndarray,
    b1_train_keys: np.ndarray,
    b0_validation_keys: np.ndarray,
    b1_validation_keys: np.ndarray,
) -> dict[str, int]:
    """Fail closed on duplicate dataset keys or train/validation key overlap.

    B0 and B1 may intentionally reuse keys *within the same data role*.
    Every individual dataset must have unique keys, and the union of training
    keys must be disjoint from the union of validation keys.
    """
    matrices = {
        "b0_train": b0_train_keys,
        "b1_train": b1_train_keys,
        "b0_validation": b0_validation_keys,
        "b1_validation": b1_validation_keys,
    }
    unique: dict[str, np.ndarray] = {}
    for role, keys in matrices.items():
        if not isinstance(keys, np.ndarray) or keys.ndim != 2 or keys.shape[1] != 4:
            raise ValueError(f"{role} master keys must have shape (N, 4)")
        if keys.dtype != np.uint16:
            raise TypeError(f"{role} master keys must be uint16")
        packed = np.ascontiguousarray(keys).view("V8").reshape(-1)
        unique[role] = np.unique(packed)
        if len(unique[role]) != len(packed):
            raise ValueError(f"duplicate master key in {role}")

    train = np.unique(np.concatenate((unique["b0_train"], unique["b1_train"])))
    validation = np.unique(
        np.concatenate((unique["b0_validation"], unique["b1_validation"]))
    )
    if np.intersect1d(train, validation, assume_unique=True).size:
        raise ValueError("training and validation master keys overlap")
    return {
        "train_unique_keys": int(len(train)),
        "validation_unique_keys": int(len(validation)),
        "cross_role_key_overlap": 0,
    }

def generate_simon_b1_groups(
    group_count: int,
    seed: int,
    *,
    rounds: int = SIMON_DEFAULT_ROUNDS,
    group_size: int = SIMON_B1_GROUP_SIZE,
    input_difference: tuple[int, int] = DEFAULT_SIMON_INPUT_DIFFERENCE,
    negative_strategy: str = "random_plaintext",
) -> SimonGeneratedPairGroups:
    """Generate balanced, deterministic same-key SIMON32/64 ciphertext-pair groups.

    Args:
        group_count: Positive even integer number of groups.
        seed: Random seed for deterministic replay.
        rounds: Cipher rounds (default 9).
        group_size: Pairs per group (default 64).
        input_difference: Plaintext XOR difference (L, R) for positive pairs.
        negative_strategy: 'random_plaintext', 'random_ciphertext', or 'random_difference'.

    Returns:
        SimonGeneratedPairGroups with features of shape (group_count, group_size, 4, 16).
    """
    if group_count <= 0:
        raise ValueError("group_count must be a positive integer")
    if group_size <= 0:
        raise ValueError("group_size must be a positive integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if negative_strategy not in ("random_plaintext", "random_ciphertext", "random_difference"):
        raise ValueError(f"unknown negative_strategy: {negative_strategy}")

    rng = np.random.default_rng(seed)
    diff_l, diff_r = int(input_difference[0]) & WORD_MASK, int(input_difference[1]) & WORD_MASK

    # Generate master keys: 1 independent key per group
    master_keys = rng.integers(0, WORD_MASK + 1, size=(group_count, 4), dtype=np.uint16)

    # Balanced labels: pos_count = (group_count + 1) // 2, neg_count = group_count - pos_count
    pos_count = (group_count + 1) // 2
    neg_count = group_count - pos_count
    labels = np.zeros(group_count, dtype=np.uint8)
    labels[:pos_count] = 1

    # Plaintexts for positive and negative groups
    # P0: (group_count, group_size, 2)
    p0 = rng.integers(0, WORD_MASK + 1, size=(group_count, group_size, 2), dtype=np.uint16)
    p1 = np.empty_like(p0)

    # Positive groups: P1 = P0 ^ input_difference
    p1[:pos_count, :, 0] = p0[:pos_count, :, 0] ^ np.uint16(diff_l)
    p1[:pos_count, :, 1] = p0[:pos_count, :, 1] ^ np.uint16(diff_r)

    # Negative groups
    if negative_strategy == "random_plaintext":
        p1[pos_count:] = rng.integers(0, WORD_MASK + 1, size=(neg_count, group_size, 2), dtype=np.uint16)
    elif negative_strategy == "random_difference":
        rand_diff = rng.integers(1, WORD_MASK + 1, size=(neg_count, group_size, 2), dtype=np.uint16)
        # Avoid exact collision with target input difference
        collision = (rand_diff[:, :, 0] == diff_l) & (rand_diff[:, :, 1] == diff_r)
        rand_diff[collision, 1] ^= 0x0001
        p1[pos_count:] = p0[pos_count:] ^ rand_diff
    elif negative_strategy == "random_ciphertext":
        pass  # Handled below

    # Deterministic group-level permutation to interleave classes
    perm = rng.permutation(group_count)
    master_keys = master_keys[perm]
    labels = labels[perm]
    p0 = p0[perm]
    p1 = p1[perm]

    # Encrypt all groups under their respective master keys
    c0 = np.empty_like(p0)
    c1 = np.empty_like(p1)

    for g_idx in range(group_count):
        mk = tuple(int(x) for x in master_keys[g_idx])
        rks = expand_key(mk, rounds)

        # Encrypt P0
        c0_l, c0_r = _encrypt_simon_arrays(p0[g_idx, :, 0], p0[g_idx, :, 1], rks)
        c0[g_idx, :, 0] = c0_l
        c0[g_idx, :, 1] = c0_r

        if negative_strategy == "random_ciphertext" and labels[g_idx] == 0:
            c1[g_idx] = rng.integers(0, WORD_MASK + 1, size=(group_size, 2), dtype=np.uint16)
        else:
            c1_l, c1_r = _encrypt_simon_arrays(p1[g_idx, :, 0], p1[g_idx, :, 1], rks)
            c1[g_idx, :, 0] = c1_l
            c1[g_idx, :, 1] = c1_r

    # Layout: (L0, R0, L1, R1)
    words = np.stack(
        (c0[:, :, 0], c0[:, :, 1], c1[:, :, 0], c1[:, :, 1]),
        axis=2,
    )  # shape: (group_count, group_size, 4)

    features = encode_words_to_bits(words)  # shape: (group_count, group_size, 4, 16)

    checksum = compute_content_checksum(features, labels, master_keys)
    metadata: dict[str, Any] = {
        "cipher": "SIMON32/64",
        "rounds": rounds,
        "group_count": group_count,
        "group_size": group_size,
        "pair_count": group_count * group_size,
        "seed": seed,
        "input_difference": [diff_l, diff_r],
        "negative_strategy": negative_strategy,
        "class_balance": {
            "0": int(np.count_nonzero(labels == 0)),
            "1": int(np.count_nonzero(labels == 1)),
        },
        "content_checksum": checksum,
    }

    return SimonGeneratedPairGroups(
        features=features,
        labels=labels,
        master_keys=master_keys,
        metadata=metadata,
    )


def generate_simon_b0_pairs(
    pair_count: int,
    seed: int,
    *,
    rounds: int = SIMON_DEFAULT_ROUNDS,
    input_difference: tuple[int, int] = DEFAULT_SIMON_INPUT_DIFFERENCE,
    negative_strategy: str = "random_plaintext",
) -> SimonGeneratedB0Data:
    """Generate balanced, deterministic single-pair SIMON32/64 samples for B0 training.

    Each pair has its own independently generated master key.
    """
    if pair_count <= 0:
        raise ValueError("pair_count must be a positive integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    rng = np.random.default_rng(seed)
    diff_l, diff_r = int(input_difference[0]) & WORD_MASK, int(input_difference[1]) & WORD_MASK

    master_keys = rng.integers(0, WORD_MASK + 1, size=(pair_count, 4), dtype=np.uint16)
    pos_count = (pair_count + 1) // 2
    neg_count = pair_count - pos_count
    labels = np.zeros(pair_count, dtype=np.uint8)
    labels[:pos_count] = 1

    p0 = rng.integers(0, WORD_MASK + 1, size=(pair_count, 2), dtype=np.uint16)
    p1 = np.empty_like(p0)

    p1[:pos_count, 0] = p0[:pos_count, 0] ^ np.uint16(diff_l)
    p1[:pos_count, 1] = p0[:pos_count, 1] ^ np.uint16(diff_r)

    if negative_strategy == "random_plaintext":
        p1[pos_count:] = rng.integers(0, WORD_MASK + 1, size=(neg_count, 2), dtype=np.uint16)
    elif negative_strategy == "random_difference":
        rand_diff = rng.integers(1, WORD_MASK + 1, size=(neg_count, 2), dtype=np.uint16)
        collision = (rand_diff[:, 0] == diff_l) & (rand_diff[:, 1] == diff_r)
        rand_diff[collision, 1] ^= 0x0001
        p1[pos_count:] = p0[pos_count:] ^ rand_diff

    perm = rng.permutation(pair_count)
    master_keys = master_keys[perm]
    labels = labels[perm]
    p0 = p0[perm]
    p1 = p1[perm]

    c0 = np.empty_like(p0)
    c1 = np.empty_like(p1)

    for i in range(pair_count):
        mk = tuple(int(x) for x in master_keys[i])
        rks = expand_key(mk, rounds)
        c0_l, c0_r = _encrypt_simon_arrays(p0[i : i + 1, 0], p0[i : i + 1, 1], rks)
        c0[i, 0] = c0_l[0]
        c0[i, 1] = c0_r[0]
        if negative_strategy == "random_ciphertext" and labels[i] == 0:
            c1[i] = rng.integers(0, WORD_MASK + 1, size=2, dtype=np.uint16)
        else:
            c1_l, c1_r = _encrypt_simon_arrays(p1[i : i + 1, 0], p1[i : i + 1, 1], rks)
            c1[i, 0] = c1_l[0]
            c1[i, 1] = c1_r[0]

    words = np.stack((c0[:, 0], c0[:, 1], c1[:, 0], c1[:, 1]), axis=1)  # shape: (pair_count, 4)
    features = encode_words_to_bits(words)  # shape: (pair_count, 4, 16)

    checksum = compute_content_checksum(features, labels, master_keys)
    metadata: dict[str, Any] = {
        "cipher": "SIMON32/64",
        "rounds": rounds,
        "pair_count": pair_count,
        "seed": seed,
        "input_difference": [diff_l, diff_r],
        "negative_strategy": negative_strategy,
        "class_balance": {
            "0": int(np.count_nonzero(labels == 0)),
            "1": int(np.count_nonzero(labels == 1)),
        },
        "content_checksum": checksum,
    }

    return SimonGeneratedB0Data(
        features=features,
        labels=labels,
        master_keys=master_keys,
        metadata=metadata,
    )


# -----------------------------------------------------------------------------
# PyTorch Datasets & DataLoaders
# -----------------------------------------------------------------------------

class SimonB1GroupDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch Dataset yielding float32 group features (64, 4, 16) and label (1,)."""

    def __init__(self, features: np.ndarray | torch.Tensor, labels: np.ndarray | torch.Tensor) -> None:
        if isinstance(features, np.ndarray):
            if features.dtype != np.uint8:
                raise TypeError("features numpy array must have dtype uint8")
            if features.ndim != 4 or features.shape[2:] != (4, 16):
                raise ValueError(f"features must have shape (N, G, 4, 16), got {features.shape}")
            self.features = torch.from_numpy(np.ascontiguousarray(features)).to(torch.float32)
        elif isinstance(features, torch.Tensor):
            if features.ndim != 4 or features.shape[2:] != (4, 16):
                raise ValueError(f"features must have shape (N, G, 4, 16), got {features.shape}")
            self.features = features.float()
        else:
            raise TypeError("features must be a numpy array or torch Tensor")

        if isinstance(labels, np.ndarray):
            self.labels = torch.from_numpy(np.ascontiguousarray(labels)).to(torch.float32).view(-1, 1)
        elif isinstance(labels, torch.Tensor):
            self.labels = labels.float().view(-1, 1)
        else:
            raise TypeError("labels must be a numpy array or torch Tensor")

        if len(self.features) != len(self.labels):
            raise ValueError("features and labels must have the same length")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]

    @classmethod
    def from_generated(cls, data: SimonGeneratedPairGroups) -> SimonB1GroupDataset:
        return cls(data.features, data.labels)


class SimonB0Dataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch Dataset yielding float32 pair features (4, 16) and label (1,)."""

    def __init__(self, features: np.ndarray | torch.Tensor, labels: np.ndarray | torch.Tensor) -> None:
        if isinstance(features, np.ndarray):
            if features.dtype != np.uint8:
                raise TypeError("features numpy array must have dtype uint8")
            if features.ndim != 3 or features.shape[1:] != (4, 16):
                raise ValueError(f"features must have shape (N, 4, 16), got {features.shape}")
            self.features = torch.from_numpy(np.ascontiguousarray(features)).to(torch.float32)
        elif isinstance(features, torch.Tensor):
            if features.ndim != 3 or features.shape[1:] != (4, 16):
                raise ValueError(f"features must have shape (N, 4, 16), got {features.shape}")
            self.features = features.float()
        else:
            raise TypeError("features must be a numpy array or torch Tensor")

        if isinstance(labels, np.ndarray):
            self.labels = torch.from_numpy(np.ascontiguousarray(labels)).to(torch.float32).view(-1, 1)
        elif isinstance(labels, torch.Tensor):
            self.labels = labels.float().view(-1, 1)
        else:
            raise TypeError("labels must be a numpy array or torch Tensor")

        if len(self.features) != len(self.labels):
            raise ValueError("features and labels must have the same length")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]

    @classmethod
    def from_generated(cls, data: SimonGeneratedB0Data) -> SimonB0Dataset:
        return cls(data.features, data.labels)


def create_simon_b1_dataloader(
    dataset: SimonB1GroupDataset,
    *,
    batch_size: int = 64,
    shuffle: bool = True,
    seed: int | None = None,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Create deterministic PyTorch DataLoader for SimonB1GroupDataset."""
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
        num_workers=0,
    )


def create_simon_b0_dataloader(
    dataset: SimonB0Dataset,
    *,
    batch_size: int = 64,
    shuffle: bool = True,
    seed: int | None = None,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Create deterministic PyTorch DataLoader for SimonB0Dataset."""
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
        num_workers=0,
    )


# -----------------------------------------------------------------------------
# Train and Evaluation Step Helpers
# -----------------------------------------------------------------------------

def train_step_b0(
    model: SimonGohrResNet,
    batch: tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """Execute one training step for SimonGohrResNet."""
    model.train()
    features, targets = batch
    features = features.to(device)
    targets = targets.to(device).view(-1, 1)

    optimizer.zero_grad()
    logits = model(features)
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        acc = binary_accuracy(torch.sigmoid(logits), targets)

    return {"loss": float(loss.item()), "accuracy": acc}


def eval_step_b0(
    model: SimonGohrResNet,
    batch: tuple[torch.Tensor, torch.Tensor],
    criterion: nn.Module,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """Execute one evaluation step for SimonGohrResNet."""
    model.eval()
    features, targets = batch
    features = features.to(device)
    targets = targets.to(device).view(-1, 1)

    with torch.no_grad():
        logits = model(features)
        loss = criterion(logits, targets)
        acc = binary_accuracy(torch.sigmoid(logits), targets)

    return {"loss": float(loss.item()), "accuracy": acc}


def train_step_b1(
    model: SimonPairGroupRanker,
    batch: tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """Execute one training step for SimonPairGroupRanker."""
    model.train()
    features, targets = batch
    features = features.to(device)
    targets = targets.to(device).view(-1, 1)

    optimizer.zero_grad()
    logits = model(features).view(-1, 1)
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        acc = binary_accuracy(torch.sigmoid(logits), targets)

    return {"loss": float(loss.item()), "accuracy": acc}


def eval_step_b1(
    model: SimonPairGroupRanker,
    batch: tuple[torch.Tensor, torch.Tensor],
    criterion: nn.Module,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """Execute one evaluation step for SimonPairGroupRanker."""
    model.eval()
    features, targets = batch
    features = features.to(device)
    targets = targets.to(device).view(-1, 1)

    with torch.no_grad():
        logits = model(features).view(-1, 1)
        loss = criterion(logits, targets)
        acc = binary_accuracy(torch.sigmoid(logits), targets)

    return {"loss": float(loss.item()), "accuracy": acc}


def evaluate_simon_b1(
    model: SimonPairGroupRanker,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Full evaluation of SimonPairGroupRanker across a dataloader."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_probs: list[float] = []
    all_targets: list[int] = []

    with torch.no_grad():
        for features, targets in dataloader:
            features = features.to(device)
            targets = targets.to(device).view(-1, 1)
            logits = model(features).view(-1, 1)
            loss = criterion(logits, targets)
            bs = int(features.shape[0])
            total_loss += float(loss.item()) * bs
            total_samples += bs
            probs = torch.sigmoid(logits).view(-1).cpu().tolist()
            targs = targets.view(-1).int().cpu().tolist()
            all_probs.extend(probs)
            all_targets.extend(targs)

    avg_loss = total_loss / max(1, total_samples)
    probs_t = torch.tensor(all_probs, dtype=torch.float32)
    targs_t = torch.tensor(all_targets, dtype=torch.int64)
    acc = binary_accuracy(probs_t, targs_t)
    auc = roc_auc_score(all_probs, all_targets) if len(set(all_targets)) > 1 else 0.5

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "auc": auc,
        "sample_count": total_samples,
        "class_balance": {
            "0": all_targets.count(0),
            "1": all_targets.count(1),
        },
    }


# -----------------------------------------------------------------------------
# Weight Transfer & Checkpoint Persistence
# -----------------------------------------------------------------------------

def transfer_b0_to_b1(
    b0_model: SimonGohrResNet,
    b1_model: SimonPairGroupRanker,
) -> None:
    """Transfer weights from trained B0 SimonGohrResNet to B1 SimonPairGroupRanker encoder."""
    b1_model.initialize_from_b0(b0_model)


def save_simon_checkpoint(
    model: nn.Module,
    output_path: Path | str,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save SimonGohrResNet or SimonPairGroupRanker checkpoint."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    is_b1 = isinstance(model, SimonPairGroupRanker)
    payload = {
        "checkpoint_format_version": 1,
        "model_type": "simon_pair_group_ranker" if is_b1 else "simon_gohr_resnet",
        "model_config": model.config.to_dict(),
        "state_dict": model.state_dict(),
        "metadata": metadata or {},
        "torch_version": str(torch.__version__),
    }
    if is_b1:
        payload["variant"] = getattr(model, "variant", "b1_mean")
        payload["head_variant"] = getattr(model, "head_variant", "speck_b1")
    else:
        payload["head_type"] = getattr(model, "head_type", "direct")

    torch.save(payload, path)


def load_simon_b0_checkpoint(
    checkpoint_path: Path | str,
    device: torch.device | str = "cpu",
) -> tuple[SimonGohrResNet, dict[str, Any]]:
    """Load SimonGohrResNet checkpoint."""
    path = Path(checkpoint_path)
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a dictionary")
    if payload.get("model_type") != "simon_gohr_resnet":
        raise ValueError(f"expected simon_gohr_resnet, got {payload.get('model_type')}")

    config = SimonGohrResNetConfig.from_dict(payload["model_config"])
    head_type = payload.get("head_type", "direct")
    model = SimonGohrResNet(config, head_type=head_type)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload.get("metadata", {})


def load_simon_b1_checkpoint(
    checkpoint_path: Path | str,
    device: torch.device | str = "cpu",
) -> tuple[SimonPairGroupRanker, dict[str, Any]]:
    """Load SimonPairGroupRanker checkpoint."""
    path = Path(checkpoint_path)
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a dictionary")
    if payload.get("model_type") != "simon_pair_group_ranker":
        raise ValueError(f"expected simon_pair_group_ranker, got {payload.get('model_type')}")

    config = SimonGohrResNetConfig.from_dict(payload["model_config"])
    variant = payload.get("variant", "b1_mean")
    # Legacy pre-freeze checkpoints lacked head_variant and used b1_mlp_bn.
    head_variant = payload.get("head_variant", "b1_mlp_bn")
    model = SimonPairGroupRanker(config, variant=variant, head_variant=head_variant)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload.get("metadata", {})


# -----------------------------------------------------------------------------
# Synthetic Overfit Verification
# -----------------------------------------------------------------------------

def run_synthetic_overfit_check(
    model_type: str = "b1",
    *,
    num_batches: int = 10,
    batch_size: int = 4,
    device: torch.device | str = "cpu",
    seed: int = 42,
) -> dict[str, Any]:
    """Verify that gradients flow and loss strictly decreases on tiny synthetic batches.

    Used as an engineering smoke test before initiating full dataset training.
    """
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)

    criterion = nn.BCEWithLogitsLoss()

    if model_type == "b1":
        model: nn.Module = SimonPairGroupRanker()
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Fixed small synthetic batch of groups: (B, 64, 4, 16)
        x_data = torch.from_numpy(
            np_rng.integers(0, 2, size=(batch_size, 64, 4, 16), dtype=np.uint8)
        ).float()
        y_data = torch.tensor([1.0, 0.0] * (batch_size // 2), dtype=torch.float32).view(-1, 1)
        batch = (x_data, y_data)

        losses: list[float] = []
        for _ in range(num_batches):
            res = train_step_b1(model, batch, optimizer, criterion, device=device)
            losses.append(res["loss"])

    elif model_type == "b0":
        model = SimonGohrResNet()
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        x_data = torch.from_numpy(
            np_rng.integers(0, 2, size=(batch_size, 4, 16), dtype=np.uint8)
        ).float()
        y_data = torch.tensor([1.0, 0.0] * (batch_size // 2), dtype=torch.float32).view(-1, 1)
        batch = (x_data, y_data)

        losses = []
        for _ in range(num_batches):
            res = train_step_b0(model, batch, optimizer, criterion, device=device)
            losses.append(res["loss"])
    else:
        raise ValueError(f"unknown model_type: {model_type}")

    # Verify gradients flowed to all trainable parameters
    grad_norms: dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                raise RuntimeError(f"parameter {name} received no gradient")
            grad_norm = float(param.grad.norm().item())
            grad_norms[name] = grad_norm

    loss_decreased = losses[-1] < losses[0]

    return {
        "model_type": model_type,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_decreased": loss_decreased,
        "loss_history": losses,
        "all_parameters_received_gradients": all(g > 0 for g in grad_norms.values()),
        "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


__all__ = [
    "audit_training_key_roles",
    "DEFAULT_SIMON_INPUT_DIFFERENCE",
    "SIMON_B1_GROUP_SIZE",
    "SIMON_DEFAULT_ROUNDS",
    "SimonB0Dataset",
    "SimonB1GroupDataset",
    "SimonGeneratedB0Data",
    "SimonGeneratedPairGroups",
    "compute_content_checksum",
    "create_simon_b0_dataloader",
    "create_simon_b1_dataloader",
    "encode_words_to_bits",
    "eval_step_b0",
    "eval_step_b1",
    "evaluate_simon_b1",
    "generate_simon_b0_pairs",
    "generate_simon_b1_groups",
    "load_simon_b0_checkpoint",
    "load_simon_b1_checkpoint",
    "run_synthetic_overfit_check",
    "save_simon_checkpoint",
    "train_step_b0",
    "train_step_b1",
    "transfer_b0_to_b1",
]
