"""Tests for the neural training, metrics, and checkpointing pipeline."""

import tempfile
from pathlib import Path

import torch

from neural_sat_speck.data.generator import DatasetConfig, generate_dataset
from neural_sat_speck.neural.dataset import SpeckClassifierDataset
from neural_sat_speck.neural.metrics import binary_accuracy, roc_auc_score
from neural_sat_speck.neural.model import GohrResNet, GohrResNetConfig
from neural_sat_speck.neural.train import (
    TrainingConfig,
    load_checkpoint,
    save_checkpoint,
    train_distinguisher,
)


def test_roc_auc_metric_perfect_and_random() -> None:
    # Perfect ranking
    probs_perfect = [0.1, 0.2, 0.8, 0.9]
    labels_perfect = [0, 0, 1, 1]
    assert roc_auc_score(probs_perfect, labels_perfect) == 1.0

    # Inverted ranking
    probs_inverted = [0.9, 0.8, 0.2, 0.1]
    labels_inverted = [0, 0, 1, 1]
    assert roc_auc_score(probs_inverted, labels_inverted) == 0.0

    # Accuracy
    assert binary_accuracy(torch.tensor(probs_perfect), torch.tensor(labels_perfect)) == 1.0


def test_train_distinguisher_runs_and_improves_loss() -> None:
    train_gen = generate_dataset(DatasetConfig(sample_count=64, rounds=4, seed=1001, split_role="train"))
    val_gen = generate_dataset(DatasetConfig(sample_count=32, rounds=4, seed=2001, split_role="validation"))

    train_ds = SpeckClassifierDataset.from_generated_dataset(train_gen)
    val_ds = SpeckClassifierDataset.from_generated_dataset(val_gen)

    model = GohrResNet(GohrResNetConfig(num_filters=16, num_blocks=2, dense_dim=32))
    config = TrainingConfig(epochs=3, batch_size=16, learning_rate=1e-3, device="cpu")

    result = train_distinguisher(model, train_ds, val_ds, config)
    assert len(result["history"]) == 3
    assert "best_val_auc" in result
    assert result["best_val_auc"] >= 0.0


def test_save_and_load_checkpoint() -> None:
    model = GohrResNet(GohrResNetConfig(num_filters=16, num_blocks=2, dense_dim=32))
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "checkpoint.pt"
        save_checkpoint(model, ckpt_path, metadata={"epoch": 5, "auc": 0.85})
        assert ckpt_path.exists()

        loaded_model, metadata = load_checkpoint(ckpt_path)
        assert metadata["epoch"] == 5
        assert metadata["auc"] == 0.85
        assert loaded_model.config.num_filters == 16
        assert loaded_model.config.num_blocks == 2
