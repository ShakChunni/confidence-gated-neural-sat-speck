"""Tests for the compact Gohr-style ResNet architecture."""

import pytest
import torch

from neural_sat_speck.neural.model import GohrResNet, GohrResNetConfig


def test_gohr_resnet_forward_shapes() -> None:
    config = GohrResNetConfig(num_filters=16, num_blocks=2, dense_dim=32)
    model = GohrResNet(config)

    # 3D tensor: (batch_size, 4, 16)
    x_3d = torch.randn(8, 4, 16)
    out_3d = model(x_3d)
    assert out_3d.shape == (8, 1)

    # 2D tensor: (batch_size, 64)
    x_2d = torch.randn(8, 64)
    out_2d = model(x_2d)
    assert out_2d.shape == (8, 1)


def test_gohr_resnet_probabilities_in_range() -> None:
    model = GohrResNet()
    x = torch.randn(4, 4, 16)
    probs = model.predict_proba(x)
    assert probs.shape == (4, 1)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_predict_proba_supports_one_sample_without_changing_training_mode() -> None:
    model = GohrResNet()
    assert model.training
    probs = model.predict_proba(torch.randn(1, 4, 16))
    assert probs.shape == (1, 1)
    assert model.training


def test_single_sample_training_batch_is_supported() -> None:
    model = GohrResNet()
    logits = model(torch.randn(1, 4, 16))
    assert logits.shape == (1, 1)


def test_invalid_resnet_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="odd"):
        GohrResNetConfig(kernel_size=2)
    with pytest.raises(ValueError, match="dropout"):
        GohrResNetConfig(dropout_rate=1.0)


def test_batched_vs_unbatched_inference_equivalence() -> None:
    """Guarantee that evaluating a sample in a batch produces identical output to evaluating alone."""
    model = GohrResNet(GohrResNetConfig(num_filters=16, num_blocks=2))
    model.eval()

    batch_x = torch.randn(10, 4, 16)
    with torch.no_grad():
        batch_out = model(batch_x)

        for i in range(len(batch_x)):
            single_x = batch_x[i : i + 1]
            single_out = model(single_x)
            assert torch.allclose(single_out, batch_out[i : i + 1], atol=1e-5)


def test_gradient_flows_through_all_parameters() -> None:
    model = GohrResNet(GohrResNetConfig(num_filters=16, num_blocks=2))
    x = torch.randn(4, 4, 16)
    y = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    criterion = torch.nn.BCEWithLogitsLoss()

    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"no gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
