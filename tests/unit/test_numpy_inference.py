"""Equivalence checks for the opt-in NumPy inference backend."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from neural_sat_speck.data.attack import AttackPair
from neural_sat_speck.data.generator import encode_ciphertext_pair
from neural_sat_speck.neural.group_model import PairGroupRanker
from neural_sat_speck.neural.model import GohrResNet, GohrResNetConfig
from neural_sat_speck.neural.numpy_inference import (
    NumpyInferenceModule,
    NumpyRankerInference,
    vectorized_final_round_features,
)
from neural_sat_speck.speck.partial import partial_decrypt_final_round


def _baseline() -> GohrResNet:
    model = GohrResNet(GohrResNetConfig(num_filters=4, num_blocks=1, dense_dim=8))
    model.eval()
    return model


@pytest.mark.parametrize("variant", ["b1_mean", "b2_moments", "b3_weighted"])
def test_numpy_pair_group_outputs_match_pytorch(variant: str) -> None:
    baseline = _baseline()
    model = PairGroupRanker(baseline.config, variant=variant)
    model.initialize_from_b0(baseline)
    model.eval()
    features = torch.randint(
        0,
        2,
        (3, 64, 4, 16),
        dtype=torch.float32,
    )

    expected = model(features)
    predictor = NumpyRankerInference(model)
    actual = torch.from_numpy(predictor.predict(features.numpy()))

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_numpy_b0_outputs_match_pytorch() -> None:
    model = _baseline()
    features = torch.randint(0, 2, (17, 4, 16), dtype=torch.float32)

    expected = model(features)
    predictor = NumpyRankerInference(model)
    actual = torch.from_numpy(predictor.predict(features.numpy()))

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_numpy_adapter_returns_torch_logits() -> None:
    model = _baseline()
    adapter = NumpyInferenceModule(NumpyRankerInference(model))
    features = torch.randint(0, 2, (5, 4, 16), dtype=torch.float32)

    result = adapter(features)

    assert result.shape == (5, 1)
    assert result.dtype == torch.float32
    assert torch.isfinite(result).all()


def test_numpy_backend_rejects_wrong_dtype() -> None:
    model = _baseline()
    predictor = NumpyRankerInference(model)

    with pytest.raises(TypeError, match="dtype float32"):
        predictor.predict(np.zeros((1, 4, 16), dtype=np.float64))


def test_numpy_backend_rejects_wrong_group_shape() -> None:
    baseline = _baseline()
    model = PairGroupRanker(baseline.config, variant="b1_mean")
    model.initialize_from_b0(baseline)
    model.eval()
    predictor = NumpyRankerInference(model)

    with pytest.raises(ValueError, match="group features"):
        predictor.predict(np.zeros((1, 32, 4, 16), dtype=np.float32))


def test_vectorized_final_round_features_match_scalar_reference() -> None:
    pairs = (
        AttackPair(
            plaintext0=(0, 0),
            plaintext1=(1, 1),
            ciphertext0=(0x1234, 0x5678),
            ciphertext1=(0x9ABC, 0xDEF0),
        ),
        AttackPair(
            plaintext0=(2, 3),
            plaintext1=(4, 5),
            ciphertext0=(0x0000, 0xFFFF),
            ciphertext1=(0x1357, 0x2468),
        ),
    )
    candidates = (0, 1, 0x1234, 0xFFFF)

    actual = vectorized_final_round_features(pairs, candidates)
    expected = np.asarray(
        [
            [
                encode_ciphertext_pair(
                    partial_decrypt_final_round(pair.ciphertext0, candidate),
                    partial_decrypt_final_round(pair.ciphertext1, candidate),
                )
                for pair in pairs
            ]
            for candidate in candidates
        ],
        dtype=np.float32,
    ).reshape(len(candidates), len(pairs), 4, 16)

    np.testing.assert_array_equal(actual, expected)
