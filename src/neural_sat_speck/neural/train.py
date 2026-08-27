"""Training and validation pipeline for neural differential distinguishers."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import SpeckClassifierDataset, create_dataloader
from .metrics import binary_accuracy, roc_auc_score
from .model import GohrResNet, GohrResNetConfig


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters for training GohrResNet."""

    epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lr_decay: float = 0.95
    seed: int = 1001
    device: str = "auto"
    deterministic: bool = True

    def __post_init__(self) -> None:
        """Validate training values before an experiment starts."""

        integer_fields = (
            ("epochs", self.epochs),
            ("batch_size", self.batch_size),
            ("seed", self.seed),
        )
        for name, value in integer_fields:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay cannot be negative")
        if self.lr_decay <= 0.0:
            raise ValueError("lr_decay must be positive")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a non-empty string")
        if not isinstance(self.deterministic, bool):
            raise TypeError("deterministic must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingConfig:
        return cls(
            epochs=int(data.get("epochs", 10)),
            batch_size=int(data.get("batch_size", 128)),
            learning_rate=float(data.get("learning_rate", 1e-3)),
            weight_decay=float(data.get("weight_decay", 1e-4)),
            lr_decay=float(data.get("lr_decay", 0.95)),
            seed=int(data.get("seed", 1001)),
            device=str(data.get("device", "auto")),
            deterministic=bool(data.get("deterministic", True)),
        )


def select_device(preferred: str = "auto") -> torch.device:
    """Select the best available compute device (MPS, CUDA, or CPU)."""
    preferred = preferred.lower()
    if preferred == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if preferred == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return torch.device("mps")
    if preferred == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if preferred != "cpu":
        raise ValueError("device must be one of auto, cpu, mps, or cuda")
    return torch.device("cpu")


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Set process RNGs and PyTorch deterministic-operation preferences."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = False


def build_model(
    model_config: GohrResNetConfig | None = None,
    *,
    seed: int,
    deterministic: bool = True,
) -> GohrResNet:
    """Construct a model after seeding its parameter initialisation."""

    set_seed(seed, deterministic=deterministic)
    return GohrResNet(model_config)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch and return the average loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = features.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(1, total_samples)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate loss, accuracy, and ROC-AUC on a dataset."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_probs: list[float] = []
    all_targets: list[int] = []

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)

            batch_size = features.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_probs.extend(probs.view(-1).cpu().tolist())
            all_targets.extend(labels.view(-1).int().cpu().tolist())

    avg_loss = total_loss / max(1, total_samples)
    probs_tensor = torch.tensor(all_probs)
    targets_tensor = torch.tensor(all_targets)

    acc = binary_accuracy(probs_tensor, targets_tensor)
    auc = roc_auc_score(all_probs, all_targets)
    positive_count = sum(all_targets)
    negative_count = len(all_targets) - positive_count

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "auc": auc,
        "sample_count": len(all_targets),
        "class_balance": {
            "0": negative_count,
            "1": positive_count,
        },
    }


def train_distinguisher(
    model: GohrResNet,
    train_dataset: SpeckClassifierDataset,
    val_dataset: SpeckClassifierDataset,
    config: TrainingConfig | None = None,
) -> dict[str, Any]:
    """Train on train_dataset and select the checkpoint by validation AUC.

    The caller must construct a new model with :func:`build_model` when exact
    parameter-initialisation reproducibility is required. The function still
    seeds all subsequent RNG use before creating loaders and the optimiser.
    """
    config = config or TrainingConfig()
    set_seed(config.seed, deterministic=config.deterministic)
    device = select_device(config.device)
    model = model.to(device)

    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=config.lr_decay,
    )

    history: list[dict[str, Any]] = []
    best_val_auc = float("-inf")
    best_epoch = 0
    best_weights: dict[str, torch.Tensor] | None = None

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_auc": val_metrics["auc"],
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(epoch_record)

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_epoch = epoch
            best_weights = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }

    if best_weights is not None:
        model.load_state_dict(best_weights)
        model.to(device)

    return {
        "config": config.to_dict(),
        "model_config": model.config.to_dict(),
        "best_val_auc": best_val_auc,
        "best_epoch": best_epoch,
        "device": str(device),
        "history": history,
    }


def save_checkpoint(
    model: GohrResNet,
    output_path: Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save model weights and metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_format_version": 1,
        "model_config": model.config.to_dict(),
        "state_dict": model.state_dict(),
        "metadata": metadata or {},
        "torch_version": str(torch.__version__),
    }
    torch.save(payload, output_path)


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device | str = "cpu",
) -> tuple[GohrResNet, dict[str, Any]]:
    """Load model weights and metadata from checkpoint."""
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_config = GohrResNetConfig.from_dict(payload["model_config"])
    model = GohrResNet(model_config)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload.get("metadata", {})
