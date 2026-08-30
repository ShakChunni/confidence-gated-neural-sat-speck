"""Deterministic Phase 6C confidence calibration and selective metrics.

Phase 6C has one deliberately narrow confidence feature: the separation of
the first and eighth candidates in the complete pair-aware ranking, scaled by
the standard deviation of all candidate scores.  This module contains the
small, dependency-free Platt calibrator and the metrics used to freeze a
selective threshold.  It does not access a true key; labels are accepted only
by the calibration/evaluation caller.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .features import extract_serialized_ranking_confidence

PHASE6C_FEATURE_NAME = "top1_top8_margin_over_score_std"
PHASE6C_TOP_K = 8
SCORE_STD_EPSILON = 1e-12
PLATT_L2_LAMBDA = 1.0
PLATT_CALIBRATOR_SCHEMA = "neural_sat_speck.phase6c_platt_calibrator.v1"


class CalibrationFitError(ValueError):
    """Raised when the frozen two-parameter calibrator cannot converge."""


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    normalised = float(value)
    if not math.isfinite(normalised):
        raise ValueError(f"{name} must be finite")
    return normalised


def _validated_features(values: Sequence[float]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("features must be a numeric sequence")
    result = tuple(_finite_float(value, "feature") for value in values)
    if not result:
        raise ValueError("at least one calibration feature is required")
    return result


def _validated_targets(values: Sequence[bool], expected_count: int) -> tuple[bool, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("targets must be a boolean sequence")
    result = tuple(values)
    if len(result) != expected_count:
        raise ValueError("features and targets must have equal length")
    if any(not isinstance(value, bool) for value in result):
        raise TypeError("targets must contain booleans")
    if not any(result) or all(result):
        raise CalibrationFitError(
            "Platt calibration requires both positive and negative targets"
        )
    return result


def _sigmoid(value: float) -> float:
    """Evaluate a logistic sigmoid without overflow."""

    if value >= 0.0:
        exponent = math.exp(-value) if value < 745.0 else 0.0
        return 1.0 / (1.0 + exponent)
    exponent = math.exp(value) if value > -745.0 else 0.0
    return exponent / (1.0 + exponent)


def _logistic_loss(
    intercept: float,
    slope: float,
    features: Sequence[float],
    targets: Sequence[bool],
    l2_lambda: float,
) -> float:
    total = 0.0
    for feature, target in zip(features, targets, strict=True):
        eta = intercept + slope * feature
        # log(1 + exp(eta)) - y*eta, evaluated stably.
        total += max(0.0, eta) + math.log1p(math.exp(-abs(eta)))
        if target:
            total -= eta
    return total + 0.5 * l2_lambda * slope * slope


@dataclass(frozen=True)
class CalibrationObservation:
    """One labelled validation ranking used by the frozen calibrator."""

    instance_id: str
    feature: float
    target_top8: bool

    def __post_init__(self) -> None:
        if not isinstance(self.instance_id, str) or not self.instance_id:
            raise ValueError("instance_id must be non-empty text")
        _finite_float(self.feature, "feature")
        if not isinstance(self.target_top8, bool):
            raise TypeError("target_top8 must be a boolean")

    def to_dict(self) -> dict[str, object]:
        """Serialize a calibration row without adding ranking inputs."""

        return {
            "instance_id": self.instance_id,
            "feature": float(self.feature),
            "target_top8": self.target_top8,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CalibrationObservation:
        """Reload a serialized calibration row."""

        if not isinstance(payload, Mapping):
            raise TypeError("calibration observation must be an object")
        return cls(
            instance_id=payload.get("instance_id"),
            feature=payload.get("feature"),
            target_top8=payload.get("target_top8"),
        )


def top8_score_separation(ranking_document: Mapping[str, Any]) -> float:
    """Return the frozen ``(score_1 - score_8) / max(std, 1e-12)`` feature.

    ``ranking_document`` must contain a complete finite ranking.  The
    candidate order controls which scores are positions one and eight; the
    true key is never read.  The runner separately enforces the complete
    65,536-candidate domain required by the prospective protocol.
    """

    features = extract_serialized_ranking_confidence(
        ranking_document,
        top_k=PHASE6C_TOP_K,
    )
    denominator = max(float(features.score_std), SCORE_STD_EPSILON)
    value = float(features.top1_top_k_margin) / denominator
    if not math.isfinite(value):
        raise ValueError("Phase 6C score-separation feature is not finite")
    return value


@dataclass(frozen=True)
class PlattCalibrator:
    """A fitted one-feature logistic calibrator with deterministic metadata."""

    feature_mean: float
    feature_std: float
    intercept: float
    slope: float
    l2_lambda: float
    fit_count: int
    positive_count: int
    iterations: int
    gradient_tolerance: float
    step_tolerance: float

    def __post_init__(self) -> None:
        for name, value in (
            ("feature_mean", self.feature_mean),
            ("feature_std", self.feature_std),
            ("intercept", self.intercept),
            ("slope", self.slope),
            ("l2_lambda", self.l2_lambda),
            ("gradient_tolerance", self.gradient_tolerance),
            ("step_tolerance", self.step_tolerance),
        ):
            _finite_float(value, name)
        if self.feature_std <= SCORE_STD_EPSILON:
            raise ValueError("feature_std must be greater than the calibration epsilon")
        if self.l2_lambda < 0.0:
            raise ValueError("l2_lambda must be non-negative")
        if self.fit_count <= 0 or self.positive_count <= 0:
            raise ValueError("calibrator counts must be positive")
        if self.positive_count >= self.fit_count:
            raise ValueError("calibrator must record both target classes")
        if self.iterations <= 0:
            raise ValueError("calibrator iterations must be positive")
        if self.gradient_tolerance <= 0.0 or self.step_tolerance <= 0.0:
            raise ValueError("calibrator tolerances must be positive")

    def predict(self, feature: float) -> float:
        """Return the frozen estimated probability for one feature value."""

        value = _finite_float(feature, "feature")
        standardised = (value - self.feature_mean) / self.feature_std
        probability = _sigmoid(self.intercept + self.slope * standardised)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise RuntimeError("calibrator produced an invalid probability")
        return probability

    def predict_many(self, features: Sequence[float]) -> tuple[float, ...]:
        """Return probabilities in input order."""

        return tuple(self.predict(value) for value in features)

    def to_dict(self) -> dict[str, object]:
        """Serialize the fitted calibrator and its numerical contract."""

        return {
            "schema": PLATT_CALIBRATOR_SCHEMA,
            "feature_name": PHASE6C_FEATURE_NAME,
            "standardization": {
                "mean": self.feature_mean,
                "std": self.feature_std,
            },
            "intercept": self.intercept,
            "slope": self.slope,
            "l2_lambda": self.l2_lambda,
            "fit_count": self.fit_count,
            "positive_count": self.positive_count,
            "iterations": self.iterations,
            "gradient_tolerance": self.gradient_tolerance,
            "step_tolerance": self.step_tolerance,
            "arithmetic": "float64",
            "intercept_penalty": "none",
            "slope_penalty": "l2",
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PlattCalibrator:
        """Reload a calibrator only when its frozen schema is intact."""

        if not isinstance(payload, Mapping):
            raise TypeError("calibrator payload must be an object")
        if payload.get("schema") != PLATT_CALIBRATOR_SCHEMA:
            raise ValueError("calibrator schema is not recognised")
        if payload.get("feature_name") != PHASE6C_FEATURE_NAME:
            raise ValueError("calibrator feature is not the frozen Phase 6C feature")
        standardization = payload.get("standardization")
        if not isinstance(standardization, Mapping):
            raise ValueError("calibrator standardization is missing")
        if payload.get("arithmetic") != "float64":
            raise ValueError("calibrator arithmetic is not float64")
        if payload.get("intercept_penalty") != "none":
            raise ValueError("calibrator intercept penalty is not frozen")
        if payload.get("slope_penalty") != "l2":
            raise ValueError("calibrator slope penalty is not frozen")
        return cls(
            feature_mean=standardization.get("mean"),
            feature_std=standardization.get("std"),
            intercept=payload.get("intercept"),
            slope=payload.get("slope"),
            l2_lambda=payload.get("l2_lambda"),
            fit_count=payload.get("fit_count"),
            positive_count=payload.get("positive_count"),
            iterations=payload.get("iterations"),
            gradient_tolerance=payload.get("gradient_tolerance"),
            step_tolerance=payload.get("step_tolerance"),
        )


def fit_platt_calibrator(
    features: Sequence[float],
    targets: Sequence[bool],
    *,
    l2_lambda: float = PLATT_L2_LAMBDA,
    max_iterations: int = 100,
    gradient_tolerance: float = 1e-10,
    step_tolerance: float = 1e-12,
) -> PlattCalibrator:
    """Fit the deterministic two-parameter float64 Newton calibrator.

    The intercept is unpenalized and the slope has an L2 penalty.  Newton
    updates use a deterministic backtracking line search; a failed line
    search or an exhausted iteration budget raises :class:`CalibrationFitError`.
    """

    raw_features = _validated_features(features)
    raw_targets = _validated_targets(targets, len(raw_features))
    penalty = _finite_float(l2_lambda, "l2_lambda")
    if penalty < 0.0:
        raise ValueError("l2_lambda must be non-negative")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be an integer")
    if max_iterations <= 0:
        raise CalibrationFitError("calibrator did not converge within max_iterations")
    grad_tol = _finite_float(gradient_tolerance, "gradient_tolerance")
    step_tol = _finite_float(step_tolerance, "step_tolerance")
    if grad_tol <= 0.0 or step_tol <= 0.0:
        raise ValueError("calibration tolerances must be positive")

    feature_mean = math.fsum(raw_features) / len(raw_features)
    centered = tuple(value - feature_mean for value in raw_features)
    feature_variance = math.fsum(value * value for value in centered) / len(centered)
    feature_std = math.sqrt(max(0.0, feature_variance))
    if feature_std <= SCORE_STD_EPSILON:
        raise CalibrationFitError("calibration features have zero variance")
    standardised = tuple(value / feature_std for value in centered)
    positive_count = sum(raw_targets)
    prevalence = positive_count / len(raw_targets)
    intercept = math.log(prevalence / (1.0 - prevalence))
    slope = 0.0
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        probabilities = tuple(
            _sigmoid(intercept + slope * feature) for feature in standardised
        )
        weights = tuple(probability * (1.0 - probability) for probability in probabilities)
        gradient_intercept = math.fsum(
            probability - target
            for probability, target in zip(probabilities, raw_targets, strict=True)
        )
        gradient_slope = math.fsum(
            (probability - target) * feature
            for probability, target, feature in zip(
                probabilities,
                raw_targets,
                standardised,
                strict=True,
            )
        ) + penalty * slope
        gradient_norm = max(abs(gradient_intercept), abs(gradient_slope))
        if gradient_norm <= grad_tol:
            break
        h00 = math.fsum(weights)
        h01 = math.fsum(
            weight * feature
            for weight, feature in zip(weights, standardised, strict=True)
        )
        h11 = math.fsum(
            weight * feature * feature
            for weight, feature in zip(weights, standardised, strict=True)
        ) + penalty
        determinant = h00 * h11 - h01 * h01
        if not math.isfinite(determinant) or determinant <= 0.0:
            raise CalibrationFitError("calibrator Hessian is not positive definite")
        delta_intercept = (
            -gradient_intercept * h11 + gradient_slope * h01
        ) / determinant
        delta_slope = (
            gradient_intercept * h01 - gradient_slope * h00
        ) / determinant
        if not math.isfinite(delta_intercept) or not math.isfinite(delta_slope):
            raise CalibrationFitError("calibrator Newton step is not finite")
        old_loss = _logistic_loss(
            intercept,
            slope,
            standardised,
            raw_targets,
            penalty,
        )
        accepted = False
        step = 1.0
        while step >= 2.0**-52:
            candidate_intercept = intercept + step * delta_intercept
            candidate_slope = slope + step * delta_slope
            candidate_loss = _logistic_loss(
                candidate_intercept,
                candidate_slope,
                standardised,
                raw_targets,
                penalty,
            )
            if math.isfinite(candidate_loss) and candidate_loss <= old_loss:
                intercept = candidate_intercept
                slope = candidate_slope
                accepted = True
                break
            step *= 0.5
        if not accepted:
            raise CalibrationFitError("calibrator backtracking line search failed")
        if max(abs(step * delta_intercept), abs(step * delta_slope)) <= step_tol:
            # A tiny update with a non-tiny gradient is a genuine failure, not
            # a reason to silently loosen the frozen tolerances.
            raise CalibrationFitError("calibrator Newton step stagnated")
    else:
        raise CalibrationFitError("calibrator did not converge within max_iterations")

    return PlattCalibrator(
        feature_mean=feature_mean,
        feature_std=feature_std,
        intercept=intercept,
        slope=slope,
        l2_lambda=penalty,
        fit_count=len(raw_features),
        positive_count=positive_count,
        iterations=iterations,
        gradient_tolerance=grad_tol,
        step_tolerance=step_tol,
    )


@dataclass(frozen=True)
class ThresholdSelection:
    """The frozen selective threshold decision, including failures."""

    feasible: bool
    threshold: float | None
    accepted_count: int
    total_count: int
    coverage: float
    empirical_risk: float | None
    risk_limit: float
    minimum_accepted: int
    candidate_threshold_count: int
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.total_count < 0 or self.accepted_count < 0:
            raise ValueError("threshold counts must be non-negative")
        if self.accepted_count > self.total_count:
            raise ValueError("accepted_count cannot exceed total_count")
        if not 0.0 <= self.coverage <= 1.0:
            raise ValueError("threshold coverage must be in [0, 1]")
        _finite_float(self.risk_limit, "risk_limit")
        if not 0.0 <= self.risk_limit <= 1.0:
            raise ValueError("risk_limit must be in [0, 1]")
        if self.minimum_accepted < 0:
            raise ValueError("minimum_accepted must be non-negative")
        if self.candidate_threshold_count < 0:
            raise ValueError("candidate_threshold_count must be non-negative")
        if self.threshold is not None:
            _finite_float(self.threshold, "threshold")
        if self.empirical_risk is not None:
            risk = _finite_float(self.empirical_risk, "empirical_risk")
            if not 0.0 <= risk <= 1.0:
                raise ValueError("empirical_risk must be in [0, 1]")
        if self.feasible != (self.threshold is not None):
            raise ValueError("feasible must agree with threshold presence")

    def to_dict(self) -> dict[str, object]:
        """Serialize the threshold result."""

        return {
            "feasible": self.feasible,
            "threshold": self.threshold,
            "accepted_count": self.accepted_count,
            "total_count": self.total_count,
            "coverage": self.coverage,
            "empirical_risk": self.empirical_risk,
            "risk_limit": self.risk_limit,
            "minimum_accepted": self.minimum_accepted,
            "candidate_threshold_count": self.candidate_threshold_count,
            "reason": self.reason,
        }


def _validated_probabilities_targets(
    probabilities: Sequence[float],
    targets: Sequence[bool],
) -> tuple[tuple[float, ...], tuple[bool, ...]]:
    if isinstance(probabilities, (str, bytes)) or isinstance(targets, (str, bytes)):
        raise TypeError("probabilities and targets must be sequences")
    values = tuple(_finite_float(value, "probability") for value in probabilities)
    labels = tuple(targets)
    if not values:
        raise ValueError("at least one probability is required")
    if len(values) != len(labels):
        raise ValueError("probabilities and targets must have equal length")
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("probabilities must be in [0, 1]")
    if any(not isinstance(value, bool) for value in labels):
        raise TypeError("targets must contain booleans")
    return values, labels


def select_threshold(
    probabilities: Sequence[float],
    targets: Sequence[bool],
    *,
    risk_limit: float = 0.10,
    minimum_accepted: int = 16,
) -> ThresholdSelection:
    """Select the maximum-coverage unique probability threshold.

    Feasibility is evaluated on all accepted rows with ``p >= threshold``.
    Ties are resolved by lower empirical risk and then higher threshold.  If
    no threshold meets both frozen constraints, the returned result records a
    selection failure and contains no invented threshold.
    """

    values, labels = _validated_probabilities_targets(probabilities, targets)
    limit = _finite_float(risk_limit, "risk_limit")
    if not 0.0 <= limit <= 1.0:
        raise ValueError("risk_limit must be in [0, 1]")
    if isinstance(minimum_accepted, bool) or not isinstance(minimum_accepted, int):
        raise TypeError("minimum_accepted must be an integer")
    if minimum_accepted < 0:
        raise ValueError("minimum_accepted must be non-negative")
    thresholds = tuple(sorted(set(values), reverse=True))
    feasible: list[tuple[int, float, float]] = []
    for threshold in thresholds:
        accepted = [
            target for probability, target in zip(values, labels, strict=True)
            if probability >= threshold
        ]
        accepted_count = len(accepted)
        if accepted_count == 0:
            continue
        risk = sum(not target for target in accepted) / accepted_count
        if accepted_count >= minimum_accepted and risk <= limit:
            feasible.append((accepted_count, risk, threshold))
    if not feasible:
        return ThresholdSelection(
            feasible=False,
            threshold=None,
            accepted_count=0,
            total_count=len(values),
            coverage=0.0,
            empirical_risk=None,
            risk_limit=limit,
            minimum_accepted=minimum_accepted,
            candidate_threshold_count=len(thresholds),
            reason="no_unique_probability_threshold_meets_risk_and_count_constraints",
        )
    accepted_count, risk, threshold = min(
        feasible,
        key=lambda item: (-item[0], item[1], -item[2]),
    )
    return ThresholdSelection(
        feasible=True,
        threshold=threshold,
        accepted_count=accepted_count,
        total_count=len(values),
        coverage=accepted_count / len(values),
        empirical_risk=risk,
        risk_limit=limit,
        minimum_accepted=minimum_accepted,
        candidate_threshold_count=len(thresholds),
    )


@dataclass(frozen=True)
class RiskCoveragePoint:
    """Empirical risk at one unique probability threshold."""

    threshold: float
    accepted_count: int
    total_count: int
    coverage: float
    error_count: int
    risk: float
    accepted_instance_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Serialize one risk-coverage point."""

        return {
            "threshold": self.threshold,
            "accepted_count": self.accepted_count,
            "total_count": self.total_count,
            "coverage": self.coverage,
            "error_count": self.error_count,
            "risk": self.risk,
            "accepted_instance_ids": list(self.accepted_instance_ids),
        }


def risk_coverage_curve(
    probabilities: Sequence[float],
    targets: Sequence[bool],
    *,
    instance_ids: Sequence[str] | None = None,
) -> tuple[RiskCoveragePoint, ...]:
    """Return deterministic risk-coverage points for unique thresholds."""

    values, labels = _validated_probabilities_targets(probabilities, targets)
    if instance_ids is None:
        ids = tuple(str(index) for index in range(len(values)))
    else:
        ids = tuple(instance_ids)
        if len(ids) != len(values) or any(not isinstance(value, str) for value in ids):
            raise ValueError("instance_ids must match probabilities and contain text")
    points: list[RiskCoveragePoint] = []
    for threshold in sorted(set(values), reverse=True):
        accepted_indices = [
            index
            for index, probability in enumerate(values)
            if probability >= threshold
        ]
        accepted_indices.sort(key=lambda index: (ids[index], index))
        error_count = sum(not labels[index] for index in accepted_indices)
        accepted_count = len(accepted_indices)
        points.append(
            RiskCoveragePoint(
                threshold=threshold,
                accepted_count=accepted_count,
                total_count=len(values),
                coverage=accepted_count / len(values),
                error_count=error_count,
                risk=error_count / accepted_count,
                accepted_instance_ids=tuple(ids[index] for index in accepted_indices),
            )
        )
    return tuple(points)


@dataclass(frozen=True)
class CalibrationBin:
    """One of the five equal-width [0, 1] calibration bins."""

    index: int
    lower: float
    upper: float
    accepted_count: int
    mean_probability: float | None
    empirical_rate: float | None
    absolute_gap: float | None

    def to_dict(self) -> dict[str, object]:
        """Serialize one calibration bin."""

        return {
            "index": self.index,
            "interval": {
                "lower": self.lower,
                "upper": self.upper,
                "left_closed": True,
                "right_closed": self.index == 4,
            },
            "accepted_count": self.accepted_count,
            "mean_probability": self.mean_probability,
            "empirical_rate": self.empirical_rate,
            "absolute_gap": self.absolute_gap,
        }


def equal_width_ece(
    probabilities: Sequence[float],
    targets: Sequence[bool],
) -> tuple[float, tuple[CalibrationBin, ...]]:
    """Compute five-bin equal-width ECE with explicit endpoint handling."""

    values, labels = _validated_probabilities_targets(probabilities, targets)
    bins: list[CalibrationBin] = []
    weighted_gap = 0.0
    for index in range(5):
        lower = index / 5.0
        upper = (index + 1) / 5.0
        indices = [
            row
            for row, probability in enumerate(values)
            if (probability >= lower)
            and (probability < upper if index < 4 else probability <= upper)
        ]
        count = len(indices)
        if count:
            mean_probability = math.fsum(values[row] for row in indices) / count
            empirical_rate = sum(labels[row] for row in indices) / count
            absolute_gap = abs(mean_probability - empirical_rate)
            weighted_gap += count / len(values) * absolute_gap
        else:
            mean_probability = None
            empirical_rate = None
            absolute_gap = None
        bins.append(
            CalibrationBin(
                index=index,
                lower=lower,
                upper=upper,
                accepted_count=count,
                mean_probability=mean_probability,
                empirical_rate=empirical_rate,
                absolute_gap=absolute_gap,
            )
        )

    return weighted_gap, tuple(bins)


def roc_auc_with_ties(
    probabilities: Sequence[float],
    targets: Sequence[bool],
) -> float:
    """Compute ROC-AUC using 0.5 credit for every exact score tie."""

    values, labels = _validated_probabilities_targets(probabilities, targets)
    positives = [value for value, target in zip(values, labels, strict=True) if target]
    negatives = [value for value, target in zip(values, labels, strict=True) if not target]
    if not positives or not negatives:
        raise ValueError("ROC-AUC requires both target classes")
    wins = math.fsum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def brier_scores(
    probabilities: Sequence[float],
    targets: Sequence[bool],
) -> tuple[float, float, float]:
    """Return Brier score, prevalence, and constant-prevalence reference."""

    values, labels = _validated_probabilities_targets(probabilities, targets)
    prevalence = sum(labels) / len(labels)
    score = math.fsum(
        (probability - target) ** 2
        for probability, target in zip(values, labels, strict=True)
    ) / len(values)
    reference = prevalence * (1.0 - prevalence)
    return score, prevalence, reference


def bootstrap_mean_interval(
    differences: Sequence[float],
    *,
    replicates: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a deterministic percentile interval for a paired mean.

    This helper is intentionally small and uses a local ``random.Random`` so
    the confidence report cannot perturb any attack or model random state.
    """

    values = _validated_features(differences)
    if isinstance(replicates, bool) or not isinstance(replicates, int):
        raise TypeError("replicates must be an integer")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    level = _finite_float(confidence, "confidence")
    if not 0.0 < level < 1.0:
        raise ValueError("confidence must be between zero and one")
    generator = random.Random(seed)
    means = sorted(
        math.fsum(values[generator.randrange(len(values))] for _ in values)
        / len(values)
        for _ in range(replicates)
    )
    alpha = (1.0 - level) / 2.0
    lower_index = max(0, min(replicates - 1, math.floor(alpha * replicates)))
    upper_index = max(
        0,
        min(replicates - 1, math.ceil((1.0 - alpha) * replicates) - 1),
    )
    return means[lower_index], means[upper_index]


def calibration_report(
    probabilities: Sequence[float],
    targets: Sequence[bool],
    *,
    instance_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    """Build the frozen calibration, ECE, AUC, and risk-coverage report."""

    brier, prevalence, reference = brier_scores(probabilities, targets)
    ece, bins = equal_width_ece(probabilities, targets)
    return {
        "feature_name": PHASE6C_FEATURE_NAME,
        "target": "true_pair_aware_rank <= 8",
        "sample_count": len(tuple(probabilities)),
        "positive_count": sum(targets),
        "prevalence": prevalence,
        "brier_score": brier,
        "constant_prevalence_brier_reference": reference,
        "roc_auc": roc_auc_with_ties(probabilities, targets),
        "ece": ece,
        "ece_bins": [item.to_dict() for item in bins],
        "risk_coverage": [
            item.to_dict()
            for item in risk_coverage_curve(
                probabilities,
                targets,
                instance_ids=instance_ids,
            )
        ],
    }


__all__ = [
    "CalibrationBin",
    "CalibrationFitError",
    "CalibrationObservation",
    "PHASE6C_FEATURE_NAME",
    "PHASE6C_TOP_K",
    "PLATT_CALIBRATOR_SCHEMA",
    "PLATT_L2_LAMBDA",
    "PlattCalibrator",
    "RiskCoveragePoint",
    "SCORE_STD_EPSILON",
    "ThresholdSelection",
    "brier_scores",
    "bootstrap_mean_interval",
    "calibration_report",
    "equal_width_ece",
    "fit_platt_calibrator",
    "roc_auc_with_ties",
    "risk_coverage_curve",
    "select_threshold",
    "top8_score_separation",
]
