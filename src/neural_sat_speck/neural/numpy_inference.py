"""NumPy inference for the frozen SPECK ranker architectures.

The PyTorch models remain the reference implementation.  This module only
provides an opt-in inference path for large CPU ranking runs, using the same
evaluation-mode parameters and equations.  It intentionally does not train
models or replace the public PyTorch model definitions.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from neural_sat_speck.data.attack import AttackPair
from neural_sat_speck.speck.key_schedule import (
    SPECK32_64_ALPHA,
    SPECK32_64_BETA,
)

from .group_model import PairGroupRanker
from .model import GohrResNet


def _as_float32(value: torch.Tensor) -> np.ndarray:
    """Copy a tensor to a contiguous CPU float32 NumPy array."""

    return np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32)


def _state_arrays(model: nn.Module) -> dict[str, np.ndarray]:
    """Extract floating-point model parameters for inference."""

    return {
        name: _as_float32(value)
        for name, value in model.state_dict().items()
        if value.is_floating_point()
    }


def _conv1d(
    values: np.ndarray,
    weight: np.ndarray,
) -> np.ndarray:
    """Apply a same-padded bias-free 1-D convolution in float32."""

    if values.ndim != 3 or weight.ndim != 3:
        raise ValueError("convolution inputs must be rank-three arrays")
    batch, input_channels, width = values.shape
    output_channels, weight_channels, kernel_size = weight.shape
    if input_channels != weight_channels:
        raise ValueError("convolution channel dimensions do not match")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("convolution kernel must be a positive odd size")

    padding = kernel_size // 2
    output = np.zeros((batch, output_channels, width), dtype=np.float32)
    for kernel_index in range(kernel_size):
        offset = kernel_index - padding
        source_start = max(0, offset)
        source_end = min(width, width + offset)
        destination_start = max(0, -offset)
        destination_end = min(width, width - offset)
        shifted = np.zeros_like(values)
        shifted[:, :, destination_start:destination_end] = values[
            :, :, source_start:source_end
        ]
        output += np.einsum(
            "ncl,oc->nol",
            shifted,
            weight[:, :, kernel_index],
            optimize=True,
        )
    return output


def _batch_norm(
    values: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    running_mean: np.ndarray,
    running_var: np.ndarray,
    *,
    eps: float = 1e-5,
) -> np.ndarray:
    """Apply evaluation-mode BatchNorm1d with PyTorch's default epsilon."""

    scale = weight / np.sqrt(running_var + np.float32(eps))
    offset = bias - running_mean * scale
    return values * scale[None, :, None] + offset[None, :, None]


def _linear(
    values: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a dense layer with PyTorch-compatible weight orientation."""

    output = values @ weight.T
    if bias is not None:
        output = output + bias
    return np.asarray(output, dtype=np.float32)


def _relu(values: np.ndarray) -> np.ndarray:
    """Apply ReLU while retaining float32 output."""

    return np.maximum(values, np.float32(0.0))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Apply a numerically stable float32 sigmoid."""

    result = np.empty_like(values, dtype=np.float32)
    nonnegative = values >= 0.0
    result[nonnegative] = 1.0 / (
        1.0 + np.exp(-values[nonnegative], dtype=np.float32)
    )
    negative_values = values[~nonnegative]
    exponent = np.exp(negative_values, dtype=np.float32)
    result[~nonnegative] = exponent / (1.0 + exponent)
    return result


def _rotate_left_u16(values: np.ndarray, amount: int) -> np.ndarray:
    """Rotate uint32 values left in a 16-bit word domain."""

    values = values & np.uint32(0xFFFF)
    if amount == 0:
        return values
    return (
        (values << np.uint32(amount))
        | (values >> np.uint32(16 - amount))
    ) & np.uint32(0xFFFF)


def _rotate_right_u16(values: np.ndarray, amount: int) -> np.ndarray:
    """Rotate uint32 values right in a 16-bit word domain."""

    values = values & np.uint32(0xFFFF)
    if amount == 0:
        return values
    return (
        (values >> np.uint32(amount))
        | (values << np.uint32(16 - amount))
    ) & np.uint32(0xFFFF)


def vectorized_final_round_features(
    pairs: Sequence[AttackPair],
    candidates: Sequence[int],
) -> np.ndarray:
    """Build candidate/pair features with vectorized 16-bit inverse rounds.

    The output has shape ``(candidate_count, pair_count, 4, 16)`` and uses
    the same MSB-first word layout as :func:`encode_ciphertext_pair`.  It is
    intended for the NumPy execution backend; scalar partial decryption is
    still the reference path used by the ordinary rankers.
    """

    pair_tuple = tuple(pairs)
    candidate_tuple = tuple(candidates)
    if not pair_tuple or not candidate_tuple:
        raise ValueError("pairs and candidates cannot be empty")
    if any(not isinstance(pair, AttackPair) for pair in pair_tuple):
        raise TypeError("pairs must contain AttackPair values")
    if any(
        isinstance(candidate, bool)
        or not isinstance(candidate, int)
        or not 0 <= candidate <= 0xFFFF
        for candidate in candidate_tuple
    ):
        raise ValueError("candidates must be 16-bit integers")

    candidate_words = np.asarray(candidate_tuple, dtype=np.uint32)[:, None]
    ciphertext0_x = np.asarray(
        [pair.ciphertext0[0] for pair in pair_tuple],
        dtype=np.uint32,
    )[None, :]
    ciphertext0_y = np.asarray(
        [pair.ciphertext0[1] for pair in pair_tuple],
        dtype=np.uint32,
    )[None, :]
    ciphertext1_x = np.asarray(
        [pair.ciphertext1[0] for pair in pair_tuple],
        dtype=np.uint32,
    )[None, :]
    ciphertext1_y = np.asarray(
        [pair.ciphertext1[1] for pair in pair_tuple],
        dtype=np.uint32,
    )[None, :]

    def inverse_x(
        ciphertext_x: np.ndarray,
        ciphertext_y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        state_y = _rotate_right_u16(
            ciphertext_y ^ ciphertext_x,
            SPECK32_64_BETA,
        )
        state_x = _rotate_left_u16(
            (ciphertext_x ^ candidate_words) - state_y,
            SPECK32_64_ALPHA,
        )
        return state_x, np.broadcast_to(state_y, state_x.shape)

    state0_x, state0_y = inverse_x(ciphertext0_x, ciphertext0_y)
    state1_x, state1_y = inverse_x(ciphertext1_x, ciphertext1_y)
    words = np.stack((state0_x, state0_y, state1_x, state1_y), axis=2)
    shifts = np.arange(15, -1, -1, dtype=np.uint32)
    return ((words[:, :, :, None] >> shifts) & np.uint32(1)).astype(
        np.float32,
        copy=False,
    )


class NumpyRankerInference:
    """Evaluate one frozen B0 or pair-group checkpoint with NumPy.

    The constructor accepts an already loaded PyTorch model so checkpoint
    validation remains owned by the existing loaders.  The model is copied to
    NumPy arrays and is never mutated by this class.
    """

    def __init__(self, model: GohrResNet | PairGroupRanker) -> None:
        if not isinstance(model, (GohrResNet, PairGroupRanker)):
            raise TypeError("model must be a GohrResNet or PairGroupRanker")
        if model.training:
            raise ValueError("NumPy inference requires an evaluation-mode model")
        self._arrays = _state_arrays(model)
        self._is_group_model = isinstance(model, PairGroupRanker)
        self.variant = model.variant if self._is_group_model else "b0"
        self.group_size = model.group_size if self._is_group_model else None
        self.embedding_dim = (
            model.embedding_dim if self._is_group_model else None
        )
        self.config = model.config
        self._prefix = "encoder." if self._is_group_model else ""

    def _array(self, name: str, *, encoder: bool = False) -> np.ndarray:
        if encoder and not self._is_group_model:
            name = {
                "dense.weight": "head.1.weight",
                "dense_bn.weight": "head.2.weight",
                "dense_bn.bias": "head.2.bias",
                "dense_bn.running_mean": "head.2.running_mean",
                "dense_bn.running_var": "head.2.running_var",
            }.get(name, name)
        key = self._prefix + name if encoder else name
        try:
            return self._arrays[key]
        except KeyError as error:
            raise ValueError(f"checkpoint is missing parameter {key}") from error

    def _encode(self, features: np.ndarray) -> np.ndarray:
        """Apply the shared B0 feature extractor through its first dense layer."""

        expected_shape = (self.config.in_channels, self.config.word_bits)
        if features.ndim != 3 or tuple(features.shape[1:]) != expected_shape:
            raise ValueError(
                "expected pair features with shape "
                f"(N,{expected_shape[0]},{expected_shape[1]}), got {features.shape}"
            )
        values = np.asarray(features, dtype=np.float32)
        values = _conv1d(values, self._array("stem_conv.weight", encoder=True))
        values = _batch_norm(
            values,
            self._array("stem_bn.weight", encoder=True),
            self._array("stem_bn.bias", encoder=True),
            self._array("stem_bn.running_mean", encoder=True),
            self._array("stem_bn.running_var", encoder=True),
        )
        values = _relu(values)
        for block_index in range(self.config.num_blocks):
            residual = values
            values = _conv1d(
                values,
                self._array(
                    f"blocks.{block_index}.conv1.weight",
                    encoder=True,
                ),
            )
            values = _batch_norm(
                values,
                self._array(f"blocks.{block_index}.bn1.weight", encoder=True),
                self._array(f"blocks.{block_index}.bn1.bias", encoder=True),
                self._array(
                    f"blocks.{block_index}.bn1.running_mean",
                    encoder=True,
                ),
                self._array(
                    f"blocks.{block_index}.bn1.running_var",
                    encoder=True,
                ),
            )
            values = _relu(values)
            values = _conv1d(
                values,
                self._array(
                    f"blocks.{block_index}.conv2.weight",
                    encoder=True,
                ),
            )
            values = _batch_norm(
                values,
                self._array(f"blocks.{block_index}.bn2.weight", encoder=True),
                self._array(f"blocks.{block_index}.bn2.bias", encoder=True),
                self._array(
                    f"blocks.{block_index}.bn2.running_mean",
                    encoder=True,
                ),
                self._array(
                    f"blocks.{block_index}.bn2.running_var",
                    encoder=True,
                ),
            )
            values = values + residual
        flattened = values.reshape(values.shape[0], -1)
        if flattened.shape[1] != self.config.num_filters * self.config.word_bits:
            raise ValueError("checkpoint feature width does not match its config")
        values = _linear(
            flattened,
            self._array("dense.weight", encoder=True),
        )
        values = _batch_norm(
            values[:, :, None],
            self._array("dense_bn.weight", encoder=True),
            self._array("dense_bn.bias", encoder=True),
            self._array("dense_bn.running_mean", encoder=True),
            self._array("dense_bn.running_var", encoder=True),
        )[:, :, 0]
        return _relu(values)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return one raw logit per B0 sample or pair group."""

        values = np.asarray(features)
        if values.dtype != np.float32:
            raise TypeError("NumPy ranker inputs must have dtype float32")
        if not np.isfinite(values).all():
            raise ValueError("NumPy ranker inputs must be finite")

        if not self._is_group_model:
            embeddings = self._encode(values)
            hidden = _relu(_linear(embeddings, self._array("head.4.weight")))
            return _linear(
                hidden,
                self._array("head.6.weight"),
                self._array("head.6.bias"),
            )

        if values.ndim != 4 or tuple(values.shape[1:]) != (
            self.group_size,
            self.config.in_channels,
            self.config.word_bits,
        ):
            raise ValueError(
                "expected group features with shape "
                f"(N,{self.group_size},{self.config.in_channels},"
                f"{self.config.word_bits}), got {values.shape}"
            )
        batch_size = values.shape[0]
        flattened = values.reshape(
            batch_size * int(self.group_size),
            self.config.in_channels,
            self.config.word_bits,
        )
        embeddings = self._encode(flattened).reshape(
            batch_size,
            int(self.group_size),
            int(self.embedding_dim),
        )
        mean = embeddings.mean(axis=1)
        if self.variant == "b1_mean":
            aggregate = mean
        elif self.variant == "b2_moments":
            variance = ((embeddings - mean[:, None, :]) ** 2).mean(axis=1)
            aggregate = np.concatenate((mean, variance), axis=1)
        elif self.variant == "b3_weighted":
            reliability_logits = _linear(
                embeddings,
                self._array("reliability.weight"),
                self._array("reliability.bias"),
            )[:, :, 0]
            weights = _sigmoid(reliability_logits)
            denominator = np.maximum(
                weights.sum(axis=1, keepdims=True),
                np.finfo(np.float32).eps,
            )
            aggregate = (embeddings * weights[:, :, None]).sum(axis=1) / denominator
        else:
            raise ValueError(f"unsupported pair-group variant {self.variant!r}")
        hidden = _relu(
            _linear(
                aggregate,
                self._array("head.0.weight"),
                self._array("head.0.bias"),
            )
        )
        return _linear(
            hidden,
            self._array("head.2.weight"),
            self._array("head.2.bias"),
        )


class NumpyInferenceModule(nn.Module):
    """Adapt :class:`NumpyRankerInference` to the ranking interfaces."""

    def __init__(self, predictor: NumpyRankerInference) -> None:
        super().__init__()
        if not isinstance(predictor, NumpyRankerInference):
            raise TypeError("predictor must be NumpyRankerInference")
        self._predictor = predictor

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if not isinstance(features, torch.Tensor):
            raise TypeError("features must be a torch.Tensor")
        values = self._predictor.predict(features.detach().cpu().numpy())
        return torch.from_numpy(values).to(device=features.device)

    def predict_numpy(self, features: np.ndarray) -> np.ndarray:
        """Evaluate a float32 NumPy feature array without a tensor round-trip."""

        return self._predictor.predict(features)


__all__ = [
    "NumpyInferenceModule",
    "NumpyRankerInference",
    "vectorized_final_round_features",
]
