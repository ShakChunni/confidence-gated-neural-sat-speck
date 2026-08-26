"""Neural distinguishing and differential cryptanalysis components."""

from .dataset import SpeckClassifierDataset, create_dataloader, features_to_tensor
from .metrics import binary_accuracy, roc_auc_score
from .model import GohrResNet, GohrResNetConfig, ResBlock1D
from .train import (
    TrainingConfig,
    evaluate,
    load_checkpoint,
    save_checkpoint,
    train_distinguisher,
)

__all__ = [
    "GohrResNet",
    "GohrResNetConfig",
    "ResBlock1D",
    "SpeckClassifierDataset",
    "TrainingConfig",
    "binary_accuracy",
    "create_dataloader",
    "evaluate",
    "features_to_tensor",
    "load_checkpoint",
    "roc_auc_score",
    "save_checkpoint",
    "train_distinguisher",
]
