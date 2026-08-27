"""Compact Gohr-style residual neural network for SPECK32/64 cryptanalysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class GohrResNetConfig:
    """Configuration hyperparameters for GohrResNet."""

    in_channels: int = 4
    word_bits: int = 16
    num_filters: int = 32
    num_blocks: int = 4
    dense_dim: int = 64
    kernel_size: int = 3
    dropout_rate: float = 0.0

    def __post_init__(self) -> None:
        """Reject configurations that cannot preserve the residual shape."""

        integer_fields = (
            ("in_channels", self.in_channels),
            ("word_bits", self.word_bits),
            ("num_filters", self.num_filters),
            ("num_blocks", self.num_blocks),
            ("dense_dim", self.dense_dim),
            ("kernel_size", self.kernel_size),
        )
        for name, value in integer_fields:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve residual shape")
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in [0.0, 1.0)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GohrResNetConfig:
        return cls(
            in_channels=int(data.get("in_channels", 4)),
            word_bits=int(data.get("word_bits", 16)),
            num_filters=int(data.get("num_filters", 32)),
            num_blocks=int(data.get("num_blocks", 4)),
            dense_dim=int(data.get("dense_dim", 64)),
            kernel_size=int(data.get("kernel_size", 3)),
            dropout_rate=float(data.get("dropout_rate", 0.0)),
        )


class _BatchNorm1dWithSingleSampleFallback(nn.BatchNorm1d):
    """Use stored statistics for a one-row training batch.

    The convolutional layers still see sixteen bit positions and can update
    batch statistics with one sample. The dense head has only one statistic per
    row, so ordinary BatchNorm1d raises for a final batch of size one. Using
    the stored statistics in that degenerate case keeps the architecture and
    normal multi-row behaviour unchanged.
    """

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.training and input_tensor.ndim == 2 and input_tensor.shape[0] == 1:
            return F.batch_norm(
                input_tensor,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                momentum=0.0,
                eps=self.eps,
            )
        return super().forward(input_tensor)


class ResBlock1D(nn.Module):
    """Residual block containing two 1D convolutions with batch normalisation."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out + residual


class GohrResNet(nn.Module):
    """Compact Gohr-style 1D convolutional residual distinguisher."""

    def __init__(self, config: GohrResNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or GohrResNetConfig()

        # Initial stem layer mapping 4 word channels to num_filters
        self.stem_conv = nn.Conv1d(
            self.config.in_channels,
            self.config.num_filters,
            kernel_size=1,
            bias=False,
        )
        self.stem_bn = nn.BatchNorm1d(self.config.num_filters)
        self.relu = nn.ReLU(inplace=True)

        # Residual tower
        self.blocks = nn.ModuleList(
            [
                ResBlock1D(
                    self.config.num_filters,
                    kernel_size=self.config.kernel_size,
                )
                for _ in range(self.config.num_blocks)
            ]
        )

        # Classifier head
        flattened_dim = self.config.num_filters * self.config.word_bits
        head_layers: list[nn.Module] = [
            nn.Flatten(),
            nn.Linear(flattened_dim, self.config.dense_dim, bias=False),
            _BatchNorm1dWithSingleSampleFallback(self.config.dense_dim),
            nn.ReLU(inplace=True),
        ]
        if self.config.dropout_rate > 0.0:
            head_layers.append(nn.Dropout(p=self.config.dropout_rate))

        head_layers.extend(
            [
                nn.Linear(
                    self.config.dense_dim,
                    self.config.dense_dim,
                    bias=False,
                ),
                nn.ReLU(inplace=True),
                nn.Linear(self.config.dense_dim, 1),
            ]
        )
        self.head = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 4, 16) or (batch_size, 64).

        Returns:
            Raw logit predictions of shape (batch_size, 1).
        """
        input_width = self.config.in_channels * self.config.word_bits
        if x.ndim == 2 and x.shape[1] == input_width:
            x = x.view(-1, self.config.in_channels, self.config.word_bits)
        elif x.ndim != 3 or x.shape[1:] != (
            self.config.in_channels,
            self.config.word_bits,
        ):
            raise ValueError(
                f"expected input shape (N, {self.config.in_channels}, "
                f"{self.config.word_bits}) or (N, {input_width}), got {x.shape}"
            )

        out = self.stem_conv(x)
        out = self.stem_bn(out)
        out = self.relu(out)

        for block in self.blocks:
            out = block(out)

        logits = self.head(out)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute sigmoid probability scores in [0, 1].

        Inference always uses evaluation mode so a single candidate or sample
        does not depend on BatchNorm batch statistics. The caller's training
        mode is restored afterwards.
        """
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                return torch.sigmoid(self.forward(x))
        finally:
            if was_training:
                self.train()

    @property
    def input_shape(self) -> tuple[int, int]:
        """Return the channel-first shape expected for one input example."""

        return self.config.in_channels, self.config.word_bits

    @property
    def parameter_count(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(param.numel() for param in self.parameters() if param.requires_grad)
