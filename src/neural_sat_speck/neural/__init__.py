"""Neural distinguishing and differential cryptanalysis components."""

from .dataset import (
    ClassifierSplits,
    SpeckClassifierDataset,
    create_dataloader,
    features_to_tensor,
    load_classifier_splits,
)
from .group_dataset import PairGroupDataset, create_pair_group_dataloader
from .group_model import (
    PAIR_GROUP_MODEL_IDS,
    PAIR_GROUP_VARIANTS,
    PairGroupRanker,
    SharedGohrPairEncoder,
    build_pair_group_model,
)
from .group_train import (
    PairGroupTrainingConfig,
    build_transferred_pair_group_model,
    evaluate_pair_group,
    load_pair_group_checkpoint,
    save_pair_group_checkpoint,
    train_pair_group_ranker,
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
    "PairGroupDataset",
    "create_pair_group_dataloader",
    "PAIR_GROUP_MODEL_IDS",
    "PAIR_GROUP_VARIANTS",
    "PairGroupRanker",
    "SharedGohrPairEncoder",
    "build_pair_group_model",
    "PairGroupTrainingConfig",
    "build_transferred_pair_group_model",
    "evaluate_pair_group",
    "load_pair_group_checkpoint",
    "save_pair_group_checkpoint",
    "train_pair_group_ranker",
]
