"""Tests for B1/B2/B3 model construction and invariants."""

from pathlib import Path

import numpy as np
import pytest
import torch

from neural_sat_speck.data.pair_groups import PairGroupConfig, generate_pair_groups
from neural_sat_speck.neural.group_dataset import PairGroupDataset
from neural_sat_speck.neural.group_model import (
    PairGroupRanker,
    build_pair_group_model,
)
from neural_sat_speck.neural.group_train import (
    PairGroupTrainingConfig,
    load_pair_group_checkpoint,
    save_pair_group_checkpoint,
    train_pair_group_ranker,
)
from neural_sat_speck.neural.model import GohrResNet, GohrResNetConfig


def _baseline() -> GohrResNet:
    return GohrResNet(
        GohrResNetConfig(num_filters=8, num_blocks=1, dense_dim=16)
    )


def _features() -> torch.Tensor:
    generator = torch.Generator().manual_seed(1234)
    return torch.randint(
        0,
        2,
        (3, 64, 4, 16),
        generator=generator,
        dtype=torch.int64,
    ).to(torch.float32)


@pytest.mark.parametrize("variant", ("b1_mean", "b2_moments", "b3_weighted"))
def test_pair_group_forward_shapes_and_permutation_invariance(variant: str) -> None:
    model = build_pair_group_model(_baseline(), variant=variant, seed=63001)
    model.eval()
    features = _features()
    with torch.no_grad():
        batch_output = model(features)
        single_output = model(features[0])
        permutation = torch.tensor(list(reversed(range(64))))
        permuted_output = model(features[:, permutation])
    assert batch_output.shape == (3, 1)
    assert single_output.shape == (1, 1)
    assert torch.allclose(batch_output[0:1], single_output, atol=1e-6, rtol=0.0)
    assert torch.allclose(batch_output, permuted_output, atol=1e-6, rtol=0.0)
    assert torch.isfinite(batch_output).all()


def test_pair_group_initialization_is_deterministic_and_transfers_b0_encoder() -> None:
    baseline_for_initialization = _baseline()
    first = build_pair_group_model(
        baseline_for_initialization,
        variant="b1_mean",
        seed=63001,
    )
    second = build_pair_group_model(
        baseline_for_initialization,
        variant="b1_mean",
        seed=63001,
    )
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name]), name

    baseline = _baseline().eval()
    transferred = build_pair_group_model(
        baseline,
        variant="b1_mean",
        seed=63001,
    ).eval()
    pair_features = _features().reshape(-1, 4, 16)
    with torch.no_grad():
        baseline_state = baseline.stem_conv(pair_features)
        baseline_state = baseline.stem_bn(baseline_state)
        baseline_state = baseline.relu(baseline_state)
        for block in baseline.blocks:
            baseline_state = block(baseline_state)
        baseline_embedding = baseline.head[3](
            baseline.head[2](baseline.head[1](baseline.head[0](baseline_state)))
        )
        transferred_embedding = transferred.encoder(pair_features)
    assert torch.equal(baseline_embedding, transferred_embedding)


def test_pair_group_rejects_wrong_dtype_and_shape() -> None:
    model = PairGroupRanker(variant="b1_mean")
    with pytest.raises(TypeError, match="float32"):
        model(torch.zeros(1, 64, 4, 16, dtype=torch.float64))
    with pytest.raises(ValueError, match="group input shape"):
        model(torch.zeros(1, 32, 4, 16, dtype=torch.float32))


def test_pair_group_gradients_are_finite() -> None:
    model = build_pair_group_model(_baseline(), variant="b3_weighted", seed=63001)
    features = _features()
    labels = torch.tensor([[1.0], [0.0], [1.0]])
    loss = torch.nn.BCEWithLogitsLoss()(model(features), labels)
    loss.backward()
    assert torch.isfinite(loss)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_pair_group_checkpoint_round_trip(tmp_path: Path) -> None:
    model = build_pair_group_model(_baseline(), variant="b2_moments", seed=63001)
    model.eval()
    checkpoint = tmp_path / "b2.pt"
    save_pair_group_checkpoint(model, checkpoint, metadata={"phase": 6})
    loaded, metadata = load_pair_group_checkpoint(checkpoint)
    loaded.eval()
    with torch.no_grad():
        assert torch.equal(model(_features()), loaded(_features()))
    assert metadata["phase"] == 6
    assert loaded.model_id == model.model_id


def test_pair_group_training_records_finite_validation_bce() -> None:
    train_generated = generate_pair_groups(
        PairGroupConfig(8, 64, 5, 61001, "train"),
        generated_at="2026-08-27T00:00:00+08:00",
    )
    validation_generated = generate_pair_groups(
        PairGroupConfig(8, 64, 5, 62001, "validation"),
        generated_at="2026-08-27T00:00:00+08:00",
    )
    model = build_pair_group_model(_baseline(), variant="b1_mean", seed=63001)
    result = train_pair_group_ranker(
        model,
        PairGroupDataset.from_generated_dataset(train_generated),
        PairGroupDataset.from_generated_dataset(validation_generated),
        PairGroupTrainingConfig(epochs=2, batch_size=4, device="cpu"),
    )
    assert result["best_epoch"] in (1, 2)
    assert np.isfinite(result["best_val_bce"])
    assert len(result["history"]) == 2
