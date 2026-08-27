"""Metrics for evaluating neural differential distinguishers."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def _validated_probabilities(values: torch.Tensor) -> torch.Tensor:
    """Return flattened finite probabilities in the closed unit interval."""

    probabilities = values.detach().view(-1).to(dtype=torch.float64)
    if not torch.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("probabilities must be in the interval [0, 1]")
    return probabilities


def _validated_targets(values: torch.Tensor) -> torch.Tensor:
    """Return flattened binary targets."""

    targets = values.detach().view(-1)
    if targets.is_floating_point() and not torch.isfinite(targets).all():
        raise ValueError("targets must be finite")
    if ((targets != 0) & (targets != 1)).any():
        raise ValueError("targets must contain only 0/1 values")
    return targets.to(dtype=torch.int64)


def binary_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Compute classification accuracy for binary predictions.

    Args:
        predictions: Predicted probabilities in [0, 1] (shape [N, 1] or [N]).
        targets: Ground truth binary labels in {0, 1} (shape [N, 1] or [N]).
        threshold: Decision threshold for positive classification (default 0.5).

    Returns:
        Classification accuracy as a float in [0.0, 1.0].
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in the interval [0, 1]")
    preds = _validated_probabilities(predictions)
    targs = _validated_targets(targets)
    if len(preds) != len(targs):
        raise ValueError("predictions and targets must have the same length")
    if len(preds) == 0:
        return 0.0
    correct = (preds >= threshold) == (targs == 1)
    return float(correct.sum().item() / len(preds))


def roc_auc_score(
    probabilities: Sequence[float] | torch.Tensor,
    targets: Sequence[int] | torch.Tensor,
) -> float:
    """Compute the Area Under the Receiver Operating Characteristic Curve (ROC-AUC).

    Uses the exact Mann-Whitney U / Wilcoxon rank-sum statistic.
    """
    if isinstance(probabilities, torch.Tensor):
        probability_tensor = probabilities
    else:
        probability_tensor = torch.tensor(list(probabilities), dtype=torch.float64)
    if isinstance(targets, torch.Tensor):
        target_tensor = targets
    else:
        target_tensor = torch.tensor(list(targets))

    probs = _validated_probabilities(probability_tensor).tolist()
    targs = _validated_targets(target_tensor).tolist()

    if len(probs) != len(targs):
        raise ValueError("probabilities and targets must have the same length")

    n_samples = len(probs)
    if n_samples == 0:
        return 0.5

    positives = sum(1 for t in targs if t == 1)
    negatives = n_samples - positives

    if positives == 0 or negatives == 0:
        return 0.5

    paired = sorted(zip(probs, targs, strict=True), key=lambda item: item[0])

    ranks = [0.0] * n_samples
    i = 0
    while i < n_samples:
        j = i
        while j < n_samples and paired[j][0] == paired[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    sum_pos_ranks = sum(
        rank for rank, (_, label) in zip(ranks, paired, strict=True) if label == 1
    )

    u_pos = sum_pos_ranks - (positives * (positives + 1)) / 2.0
    return float(u_pos / (positives * negatives))
