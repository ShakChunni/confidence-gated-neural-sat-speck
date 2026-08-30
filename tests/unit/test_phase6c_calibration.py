"""Tests for the frozen Phase 6C confidence calibration contract."""

from __future__ import annotations

import json

import pytest

from neural_sat_speck.confidence.calibration import (
    CalibrationFitError,
    PlattCalibrator,
    bootstrap_mean_interval,
    calibration_report,
    equal_width_ece,
    fit_platt_calibrator,
    roc_auc_with_ties,
    select_threshold,
    top8_score_separation,
)


def _small_ranking(scores: tuple[float, ...]) -> dict[str, object]:
    return {
        "ranking": {
            "candidate_values": list(range(len(scores))),
            "candidate_scores": list(scores),
            "candidate_order": list(reversed(range(len(scores)))),
        }
    }


def test_phase6c_feature_uses_ordered_top8_and_all_score_std() -> None:
    scores = tuple(float(value) for value in range(8))
    feature = top8_score_separation(_small_ranking(scores))
    expected_std = (5.25) ** 0.5
    assert feature == pytest.approx(7.0 / expected_std)


def test_phase6c_feature_has_zero_score_std_guard() -> None:
    assert top8_score_separation(_small_ranking((0.0,) * 8)) == 0.0


def test_platt_fit_is_deterministic_bounded_and_serializable() -> None:
    features = (-2.0, -1.0, 0.0, 1.0, 2.0)
    targets = (False, False, False, True, True)
    first = fit_platt_calibrator(features, targets)
    second = fit_platt_calibrator(features, targets)

    assert first == second
    probabilities = first.predict_many(features)
    assert all(0.0 <= value <= 1.0 for value in probabilities)
    assert probabilities == tuple(sorted(probabilities))
    restored = PlattCalibrator.from_dict(json.loads(json.dumps(first.to_dict())))
    assert restored == first
    assert restored.predict_many(features) == probabilities


def test_platt_fit_rejects_nonconvergence_and_single_class() -> None:
    with pytest.raises(CalibrationFitError):
        fit_platt_calibrator((0.0, 1.0), (False, True), max_iterations=0)
    with pytest.raises(CalibrationFitError):
        fit_platt_calibrator((0.0, 1.0), (True, True))
    with pytest.raises(CalibrationFitError):
        fit_platt_calibrator((0.0, 0.0), (False, True))


def test_threshold_selection_is_unique_deterministic_and_fails_closed() -> None:
    selected = select_threshold(
        (0.95, 0.85, 0.75, 0.65),
        (True, True, False, False),
        risk_limit=0.10,
        minimum_accepted=2,
    )
    assert selected.feasible
    assert selected.threshold == 0.85
    assert selected.accepted_count == 2
    assert selected.coverage == 0.5
    assert selected.empirical_risk == 0.0

    tied = select_threshold(
        (0.9, 0.9, 0.8, 0.8),
        (True, False, True, False),
        risk_limit=0.5,
        minimum_accepted=2,
    )
    assert tied.feasible
    assert tied.threshold == 0.8
    assert tied.accepted_count == 4

    failed = select_threshold(
        tuple(1.0 - index / 20.0 for index in range(20)),
        (True,) * 14 + (False,) * 6,
        risk_limit=0.10,
        minimum_accepted=16,
    )
    assert not failed.feasible
    assert failed.threshold is None
    assert failed.reason is not None


def test_ece_boundaries_and_auc_tie_credit_are_explicit() -> None:
    probabilities = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    targets = (False, False, True, True, False, True)
    ece, bins = equal_width_ece(probabilities, targets)
    assert ece >= 0.0
    assert [item.accepted_count for item in bins] == [1, 1, 1, 1, 2]
    assert bins[0].lower == 0.0
    assert bins[4].upper == 1.0
    assert roc_auc_with_ties((0.5, 0.5, 0.2, 0.8), (True, False, False, True)) == pytest.approx(
        0.875
    )


def test_calibration_report_contains_prevalence_brier_and_risk_coverage() -> None:
    report = calibration_report(
        (0.9, 0.8, 0.2, 0.1),
        (True, True, False, False),
        instance_ids=("a", "b", "c", "d"),
    )
    assert report["prevalence"] == 0.5
    assert report["constant_prevalence_brier_reference"] == 0.25
    assert len(report["ece_bins"]) == 5
    assert len(report["risk_coverage"]) == 4


def test_bootstrap_mean_interval_is_seed_deterministic() -> None:
    first = bootstrap_mean_interval((1.0, 2.0, 3.0), replicates=100, seed=97101)
    second = bootstrap_mean_interval((1.0, 2.0, 3.0), replicates=100, seed=97101)
    assert first == second
