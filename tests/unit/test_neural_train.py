"""Tests for the neural training, metrics, and checkpointing pipeline."""

import tempfile
from pathlib import Path

import pytest
import torch

from neural_sat_speck.data.generator import DatasetConfig, generate_dataset
from neural_sat_speck.neural.dataset import SpeckClassifierDataset
from neural_sat_speck.neural.metrics import binary_accuracy, roc_auc_score
from neural_sat_speck.neural.model import GohrResNet, GohrResNetConfig
from neural_sat_speck.neural.train import (
    TrainingConfig,
    build_model,
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


def test_metrics_reject_nonbinary_labels_and_invalid_probabilities() -> None:
    with pytest.raises(ValueError, match="0/1"):
        roc_auc_score([0.1, 0.9], [0, 2])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        binary_accuracy(torch.tensor([0.1, 1.2]), torch.tensor([0, 1]))


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


def test_seeded_model_training_is_reproducible() -> None:
    train_gen = generate_dataset(
        DatasetConfig(sample_count=32, rounds=4, seed=1001, split_role="train")
    )
    val_gen = generate_dataset(
        DatasetConfig(sample_count=16, rounds=4, seed=2001, split_role="validation")
    )
    train_ds = SpeckClassifierDataset.from_generated_dataset(train_gen)
    val_ds = SpeckClassifierDataset.from_generated_dataset(val_gen)
    model_config = GohrResNetConfig(num_filters=8, num_blocks=1, dense_dim=16)
    config = TrainingConfig(
        epochs=2,
        batch_size=8,
        learning_rate=1e-3,
        device="cpu",
        seed=8123,
    )

    first_model = build_model(model_config, seed=config.seed)
    first = train_distinguisher(first_model, train_ds, val_ds, config)
    first_state = {
        name: value.detach().clone() for name, value in first_model.state_dict().items()
    }

    second_model = build_model(model_config, seed=config.seed)
    second = train_distinguisher(second_model, train_ds, val_ds, config)

    assert first["history"] == second["history"]
    assert first["best_val_auc"] == second["best_val_auc"]
    for name, value in second_model.state_dict().items():
        assert torch.equal(first_state[name], value)


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
