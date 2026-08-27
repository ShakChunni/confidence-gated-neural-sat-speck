"""Descriptive validation metrics for post-hoc ranking confidence.

The functions in this module deliberately stop short of fitting a threshold.
They describe the risk--coverage behaviour of a frozen confidence feature on
validation instances.  Threshold selection belongs to a later, separately
reviewed phase and must not use the held-out test split.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .features import RankingConfidenceFeatures

ConfidenceDirection = Literal["higher", "lower"]

CONFIDENCE_FEATURE_NAMES: tuple[str, ...] = (
    "top1_top2_margin",
    "top1_top_k_margin",
    "top1_softmax_concentration",
    "normalized_softmax_entropy",
)
PAIR_SUBSET_STABILITY_FEATURE_NAME = "pair_subset_top_k_jaccard"
EXTENDED_CONFIDENCE_FEATURE_NAMES: tuple[str, ...] = (
    *CONFIDENCE_FEATURE_NAMES,
    PAIR_SUBSET_STABILITY_FEATURE_NAME,
)

CONFIDENCE_FEATURE_DIRECTIONS: dict[str, ConfidenceDirection] = {
    "top1_top2_margin": "higher",
    "top1_top_k_margin": "higher",
    "top1_softmax_concentration": "higher",
    "normalized_softmax_entropy": "lower",
    PAIR_SUBSET_STABILITY_FEATURE_NAME: "higher",
}


@dataclass(frozen=True)
class ConfidenceObservation:
    """One labelled validation ranking and its frozen confidence features."""

    source_result: str
    source_result_sha256: str
    split_role: str
    instance_index: int
    instance_id: str
    true_final_round_subkey: int
    true_rank: int
    target_in_top_k: bool
    features: RankingConfidenceFeatures

    def to_dict(self) -> dict[str, object]:
        """Serialize the observation without exposing any private key data."""

        return {
            "source_result": self.source_result,
            "source_result_sha256": self.source_result_sha256,
            "split_role": self.split_role,
            "instance_index": self.instance_index,
            "instance_id": self.instance_id,
            "true_final_round_subkey": self.true_final_round_subkey,
            "true_rank": self.true_rank,
            "target_in_top_k": self.target_in_top_k,
            "features": self.features.to_dict(),
        }


@dataclass(frozen=True)
class RiskCoveragePoint:
    """Empirical error at one accepted prefix of a confidence ordering."""

    feature_name: str
    direction: ConfidenceDirection
    accepted_count: int
    total_count: int
    coverage: float
    error_count: int
    risk: float
    accepted_instance_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Serialize one risk--coverage point."""

        return {
            "feature_name": self.feature_name,
            "direction": self.direction,
            "accepted_count": self.accepted_count,
            "total_count": self.total_count,
            "coverage": self.coverage,
            "error_count": self.error_count,
            "risk": self.risk,
            "accepted_instance_ids": list(self.accepted_instance_ids),
        }


def _feature_direction(feature_name: str) -> ConfidenceDirection:
    try:
        return CONFIDENCE_FEATURE_DIRECTIONS[feature_name]
    except KeyError as error:
        allowed = ", ".join(CONFIDENCE_FEATURE_NAMES)
        raise ValueError(
            f"unknown confidence feature {feature_name!r}; expected one of {allowed}"
        ) from error


def _feature_value(
    observation: ConfidenceObservation,
    feature_name: str,
) -> float:
    value = getattr(observation.features, feature_name, None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"confidence feature {feature_name!r} is not numeric")
    value_float = float(value)
    if not math.isfinite(value_float):
        raise ValueError(f"confidence feature {feature_name!r} is not finite")
    return value_float


def risk_coverage_curve(
    observations: list[ConfidenceObservation] | tuple[ConfidenceObservation, ...],
    *,
    feature_name: str,
) -> tuple[RiskCoveragePoint, ...]:
    """Return descriptive accepted-prefix risk for one confidence feature.

    Higher margins and concentration are treated as more confident. Entropy
    is treated as more confident when it is smaller. Ties are resolved by
    instance id and source path, so the curve is reproducible. The curve has
    one point for every non-empty accepted prefix and does not fit a threshold.
    """

    direction = _feature_direction(feature_name)
    if not observations:
        raise ValueError("at least one confidence observation is required")
    indexed = list(enumerate(observations))
    for _, observation in indexed:
        if not isinstance(observation, ConfidenceObservation):
            raise TypeError("observations must contain ConfidenceObservation values")
        _feature_value(observation, feature_name)
    reverse = direction == "higher"
    indexed.sort(
        key=lambda item: (
            -_feature_value(item[1], feature_name) if reverse
            else _feature_value(item[1], feature_name),
            item[1].instance_id,
            item[1].source_result,
            item[0],
        )
    )

    accepted: list[ConfidenceObservation] = []
    points: list[RiskCoveragePoint] = []
    total_count = len(indexed)
    for observation_index, observation in indexed:
        del observation_index
        accepted.append(observation)
        error_count = sum(not item.target_in_top_k for item in accepted)
        accepted_count = len(accepted)
        points.append(
            RiskCoveragePoint(
                feature_name=feature_name,
                direction=direction,
                accepted_count=accepted_count,
                total_count=total_count,
                coverage=accepted_count / total_count,
                error_count=error_count,
                risk=error_count / accepted_count,
                accepted_instance_ids=tuple(
                    item.instance_id for item in accepted
                ),
            )
        )
    return tuple(points)


def all_risk_coverage_curves(
    observations: list[ConfidenceObservation] | tuple[ConfidenceObservation, ...],
    *,
    feature_names: Sequence[str] = CONFIDENCE_FEATURE_NAMES,
) -> dict[str, tuple[RiskCoveragePoint, ...]]:
    """Return curves for a pre-declared set of Phase 6 feature candidates."""

    return {
        feature_name: risk_coverage_curve(observations, feature_name=feature_name)
        for feature_name in feature_names
    }
