"""Tests for the Phase 6 post-hoc confidence boundary."""

from dataclasses import dataclass

import pytest

from neural_sat_speck.confidence import (
    CONFIDENCE_FEATURE_NAMES,
    EXTENDED_CONFIDENCE_FEATURE_NAMES,
    PAIR_SUBSET_STABILITY_FEATURE_NAME,
    ConfidenceObservation,
    all_risk_coverage_curves,
    candidate_is_in_top_k,
    extract_ranking_confidence,
    extract_serialized_ranking_confidence,
    risk_coverage_curve,
)


@dataclass(frozen=True)
class _Ranking:
    candidate_values: tuple[int, ...]
    candidate_scores: tuple[float, ...]
    candidate_order: tuple[int, ...]


def _ranking() -> _Ranking:
    return _Ranking(
        candidate_values=(0, 1, 2, 3),
        candidate_scores=(0.0, 1.0, 2.0, 3.0),
        candidate_order=(3, 2, 1, 0),
    )


def _observation(
    instance_id: str,
    order: tuple[int, ...],
    scores: tuple[float, ...],
    target: bool,
) -> ConfidenceObservation:
    features = extract_ranking_confidence(
        _Ranking((0, 1, 2, 3), scores, order),
        top_k=2,
    )
    return ConfidenceObservation(
        source_result=f"results/{instance_id}.json",
        source_result_sha256="0" * 64,
        split_role="validation",
        instance_index=0,
        instance_id=instance_id,
        true_final_round_subkey=0,
        true_rank=1 if target else 3,
        target_in_top_k=target,
        features=features,
    )


def test_features_use_ordered_scores_and_have_expected_values() -> None:
    features = extract_ranking_confidence(_ranking(), top_k=2)

    assert features.candidate_count == 4
    assert features.top1_candidate == 3
    assert features.top2_candidate == 2
    assert features.top_k_candidates == (3, 2)
    assert features.top1_score == 3.0
    assert features.top2_score == 2.0
    assert features.top1_top2_margin == 1.0
    assert features.top1_top_k_margin == 1.0
    assert features.score_mean == 1.5
    assert features.score_std == pytest.approx(1.1180339887)
    assert 0.0 < features.top1_softmax_concentration < 1.0
    assert 0.0 <= features.normalized_softmax_entropy <= 1.0


def test_serialized_features_and_target_membership_are_deterministic() -> None:
    document = {
        "ranking": {
            "candidate_values": [0, 1, 2, 3],
            "candidate_scores": [0.0, 1.0, 2.0, 3.0],
            "candidate_order": [3, 2, 1, 0],
        }
    }
    features = extract_serialized_ranking_confidence(document, top_k=2)

    assert candidate_is_in_top_k(features, 3)
    assert candidate_is_in_top_k(features, 2)
    assert not candidate_is_in_top_k(features, 1)
    assert features.to_dict() == extract_serialized_ranking_confidence(
        document,
        top_k=2,
    ).to_dict()


def test_pair_subset_stability_is_exposed_when_present() -> None:
    document = {
        "ranking": {
            "candidate_values": [0, 1, 2, 3],
            "candidate_scores": [0.0, 1.0, 2.0, 3.0],
            "candidate_order": [3, 2, 1, 0],
            "pair_subset_stability": {"full_top_k_jaccard": 0.75},
        }
    }
    features = extract_serialized_ranking_confidence(document, top_k=2)

    assert features.pair_subset_top_k_jaccard == 0.75
    assert features.to_dict()[PAIR_SUBSET_STABILITY_FEATURE_NAME] == 0.75


def test_tied_scores_keep_the_supplied_deterministic_order() -> None:
    ranking = _Ranking(
        candidate_values=(0, 1, 2, 3),
        candidate_scores=(0.0, 0.0, 0.0, 0.0),
        candidate_order=(2, 0, 3, 1),
    )
    features = extract_ranking_confidence(ranking, top_k=2)

    assert features.top_k_candidates == (2, 0)
    assert features.top1_softmax_concentration == pytest.approx(0.25)
    assert features.normalized_softmax_entropy == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("candidate_values", "candidate_scores", "candidate_order"),
    [
        ((0, 1), (0.0,), (0, 1)),
        ((0, 1), (0.0, 1.0), (0, 0)),
        ((0, 1), (0.0, 1.0), (0, 2)),
        ((True, 1), (0.0, 1.0), (True, 1)),
        ((0, 1), (float("nan"), 1.0), (0, 1)),
    ],
)
def test_malformed_rankings_are_rejected(
    candidate_values: tuple[object, ...],
    candidate_scores: tuple[float, ...],
    candidate_order: tuple[object, ...],
) -> None:
    ranking = _Ranking(
        candidate_values,  # type: ignore[arg-type]
        candidate_scores,
        candidate_order,  # type: ignore[arg-type]
    )
    with pytest.raises((TypeError, ValueError)):
        extract_ranking_confidence(ranking, top_k=2)


@pytest.mark.parametrize("top_k", [0, 1, 5, True, 2.0])
def test_invalid_top_k_is_rejected(top_k: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        extract_ranking_confidence(_ranking(), top_k=top_k)  # type: ignore[arg-type]


def test_risk_coverage_is_validation_only_and_deterministic() -> None:
    observations = (
        _observation("a", (0, 1, 2, 3), (4.0, 3.0, 2.0, 1.0), True),
        _observation("b", (1, 0, 2, 3), (3.0, 5.0, 2.0, 1.0), False),
        _observation("c", (2, 0, 1, 3), (0.0, 1.0, 6.0, -1.0), True),
    )
    curve = risk_coverage_curve(observations, feature_name="top1_top2_margin")

    assert [point.accepted_instance_ids for point in curve] == [
        ("c",),
        ("c", "b"),
        ("c", "b", "a"),
    ]
    assert [point.risk for point in curve] == [0.0, 0.5, pytest.approx(1 / 3)]
    assert curve[-1].coverage == 1.0
    assert curve[-1].error_count == 1

    curves = all_risk_coverage_curves(observations)
    assert tuple(curves) == CONFIDENCE_FEATURE_NAMES
    assert all(len(points) == 3 for points in curves.values())


def test_extended_risk_coverage_requires_stability_data() -> None:
    observations = (
        _observation("a", (0, 1, 2, 3), (4.0, 3.0, 2.0, 1.0), True),
    )
    with pytest.raises(ValueError):
        all_risk_coverage_curves(
            observations,
            feature_names=EXTENDED_CONFIDENCE_FEATURE_NAMES,
        )


def test_risk_coverage_rejects_unknown_or_empty_inputs() -> None:
    observation = _observation("a", (0, 1, 2, 3), (4.0, 3.0, 2.0, 1.0), True)
    with pytest.raises(ValueError):
        risk_coverage_curve((observation,), feature_name="unknown")
    with pytest.raises(ValueError):
        risk_coverage_curve((), feature_name="top1_top2_margin")
