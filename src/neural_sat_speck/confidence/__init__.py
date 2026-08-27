"""Post-hoc confidence features for neural candidate rankings."""

from .features import (
    RankingConfidenceFeatures,
    candidate_is_in_top_k,
    extract_ranking_confidence,
    extract_serialized_ranking_confidence,
)
from .metrics import (
    CONFIDENCE_FEATURE_DIRECTIONS,
    CONFIDENCE_FEATURE_NAMES,
    EXTENDED_CONFIDENCE_FEATURE_NAMES,
    PAIR_SUBSET_STABILITY_FEATURE_NAME,
    ConfidenceObservation,
    RiskCoveragePoint,
    all_risk_coverage_curves,
    risk_coverage_curve,
)

__all__ = [
    "RankingConfidenceFeatures",
    "candidate_is_in_top_k",
    "extract_ranking_confidence",
    "extract_serialized_ranking_confidence",
    "CONFIDENCE_FEATURE_DIRECTIONS",
    "CONFIDENCE_FEATURE_NAMES",
    "EXTENDED_CONFIDENCE_FEATURE_NAMES",
    "PAIR_SUBSET_STABILITY_FEATURE_NAME",
    "ConfidenceObservation",
    "RiskCoveragePoint",
    "all_risk_coverage_curves",
    "risk_coverage_curve",
]
