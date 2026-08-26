"""Metrics for evaluating neural differential distinguishers."""

from __future__ import annotations

from collections.abc import Sequence

import torch


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
    preds = predictions.detach().view(-1)
    targs = targets.detach().view(-1)
    if len(preds) != len(targs):
        raise ValueError("predictions and targets must have the same length")
    if len(preds) == 0:
        return 0.0
    correct = (preds >= threshold) == (targs >= 0.5)
    return float(correct.sum().item() / len(preds))


def roc_auc_score(
    probabilities: Sequence[float] | torch.Tensor,
    targets: Sequence[int] | torch.Tensor,
) -> float:
    """Compute the Area Under the Receiver Operating Characteristic Curve (ROC-AUC).

    Uses the exact Mann-Whitney U / Wilcoxon rank-sum statistic.
    """
    if isinstance(probabilities, torch.Tensor):
        probs = probabilities.detach().cpu().view(-1).tolist()
    else:
        probs = list(probabilities)

    if isinstance(targets, torch.Tensor):
        targs = targets.detach().cpu().view(-1).int().tolist()
    else:
        targs = [int(t) for t in targets]

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
