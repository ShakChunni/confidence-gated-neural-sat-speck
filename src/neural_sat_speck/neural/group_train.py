"""Training and checkpointing for the Phase 6 pair-group rankers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .group_dataset import PairGroupDataset, create_pair_group_dataloader
from .group_model import PairGroupRanker, build_pair_group_model
from .metrics import binary_accuracy, roc_auc_score
from .model import GohrResNet, GohrResNetConfig
from .train import select_device, set_seed


@dataclass(frozen=True)
class PairGroupTrainingConfig:
    """Frozen training settings shared by B1, B2, and B3."""

    epochs: int = 8
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lr_decay: float = 0.95
    seed: int = 63001
    device: str = "cpu"
    deterministic: bool = True

    def __post_init__(self) -> None:
        for name, value in (
            ("epochs", self.epochs),
            ("batch_size", self.batch_size),
            ("seed", self.seed),
        ):
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


def build_transferred_pair_group_model(
    baseline: GohrResNet,
    *,
    variant: str,
    config: PairGroupTrainingConfig,
) -> PairGroupRanker:
    """Build one candidate with deterministic B0-transfer initialization."""

    return build_pair_group_model(
        baseline,
        variant=variant,
        seed=config.seed,
        deterministic=config.deterministic,
    )


def evaluate_pair_group(
    model: PairGroupRanker,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate group BCE, accuracy, and ROC-AUC without changing weights."""

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
            batch_size = int(features.shape[0])
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            all_probs.extend(torch.sigmoid(logits).view(-1).cpu().tolist())
            all_targets.extend(labels.view(-1).int().cpu().tolist())
    probabilities = torch.tensor(all_probs, dtype=torch.float32)
    targets = torch.tensor(all_targets, dtype=torch.int64)
    negative_count = all_targets.count(0)
    positive_count = all_targets.count(1)
    return {
        "loss": total_loss / max(1, total_samples),
        "accuracy": binary_accuracy(probabilities, targets),
        "auc": roc_auc_score(all_probs, all_targets),
        "sample_count": len(all_targets),
        "class_balance": {"0": negative_count, "1": positive_count},
    }


def _train_one_epoch(
    model: PairGroupRanker,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(features), labels)
        loss.backward()
        optimizer.step()
        batch_size = int(features.shape[0])
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    return total_loss / max(1, total_samples)


def train_pair_group_ranker(
    model: PairGroupRanker,
    train_dataset: PairGroupDataset,
    validation_dataset: PairGroupDataset,
    config: PairGroupTrainingConfig | None = None,
) -> dict[str, Any]:
    """Train one candidate and select its weights by minimum validation BCE."""

    config = config or PairGroupTrainingConfig()
    set_seed(config.seed, deterministic=config.deterministic)
    device = select_device(config.device)
    model.to(device)
    train_loader = create_pair_group_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
    )
    validation_loader = create_pair_group_dataloader(
        validation_dataset,
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
    best_val_loss = float("inf")
    best_epoch = 0
    best_weights: dict[str, torch.Tensor] | None = None
    for epoch in range(1, config.epochs + 1):
        train_loss = _train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )
        validation = evaluate_pair_group(
            model,
            validation_loader,
            criterion,
            device,
        )
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": validation["loss"],
                "val_accuracy": validation["accuracy"],
                "val_auc": validation["auc"],
                "lr": scheduler.get_last_lr()[0],
            }
        )
        # Strict inequality is the deterministic earliest-epoch tie rule.
        if validation["loss"] < best_val_loss:
            best_val_loss = validation["loss"]
            best_epoch = epoch
            best_weights = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if best_weights is None:
        raise RuntimeError("pair-group training did not produce a checkpoint")
    model.load_state_dict(best_weights)
    model.to(device)
    return {
        "config": config.to_dict(),
        "model_id": model.model_id,
        "variant": model.variant,
        "model_config": model.config.to_dict(),
        "best_val_bce": best_val_loss,
        "best_epoch": best_epoch,
        "device": str(device),
        "history": history,
    }


def save_pair_group_checkpoint(
    model: PairGroupRanker,
    output_path: Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a B1/B2/B3 checkpoint without touching the B0 format."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_format_version": 1,
        "model_id": model.model_id,
        "variant": model.variant,
        "group_size": model.group_size,
        "model_config": model.config.to_dict(),
        "state_dict": model.state_dict(),
        "metadata": metadata or {},
        "torch_version": str(torch.__version__),
    }
    torch.save(payload, output_path)


def load_pair_group_checkpoint(
    checkpoint_path: Path,
    device: torch.device | str = "cpu",
) -> tuple[PairGroupRanker, dict[str, Any]]:
    """Load a B1/B2/B3 checkpoint and reject ambiguous model identities."""

    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("pair-group checkpoint must contain a mapping")
    variant = payload.get("variant")
    model_id = payload.get("model_id")
    if not isinstance(variant, str) or not isinstance(model_id, str):
        raise ValueError("pair-group checkpoint is missing its model identity")
    model = PairGroupRanker(
        GohrResNetConfig.from_dict(payload["model_config"]),
        variant=variant,
        group_size=int(payload.get("group_size", 0)),
    )
    if model.model_id != model_id:
        raise ValueError("pair-group checkpoint model_id does not match variant")
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("pair-group checkpoint metadata must be a mapping")
    return model, metadata


__all__ = [
    "PairGroupTrainingConfig",
    "build_transferred_pair_group_model",
    "evaluate_pair_group",
    "load_pair_group_checkpoint",
    "save_pair_group_checkpoint",
    "train_pair_group_ranker",
]
