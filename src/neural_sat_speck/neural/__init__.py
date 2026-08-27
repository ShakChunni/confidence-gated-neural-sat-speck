"""Neural distinguishing and differential cryptanalysis components."""

from .dataset import (
    ClassifierSplits,
    SpeckClassifierDataset,
    create_dataloader,
    features_to_tensor,
    load_classifier_splits,
)
from .metrics import binary_accuracy, roc_auc_score
from .model import GohrResNet, GohrResNetConfig, ResBlock1D
from .train import (
    TrainingConfig,
    build_model,
    evaluate,
    load_checkpoint,
    save_checkpoint,
    train_distinguisher,
)

__all__ = [
    "GohrResNet",
    "GohrResNetConfig",
    "ResBlock1D",
    "ClassifierSplits",
    "SpeckClassifierDataset",
    "TrainingConfig",
    "binary_accuracy",
    "build_model",
    "create_dataloader",
    "evaluate",
    "features_to_tensor",
    "load_classifier_splits",
    "load_checkpoint",
    "roc_auc_score",
    "save_checkpoint",
    "train_distinguisher",
]
