"""Unit tests for screened multi-fidelity pair-group subkey ranking."""

import pytest

from neural_sat_speck.attacks.screened_group_ranking import (
    rank_pair_group_subkeys_screened,
)
from neural_sat_speck.data.attack import AttackInstanceConfig, generate_attack_instances
from neural_sat_speck.neural.group_model import build_pair_group_model
from neural_sat_speck.neural.model import GohrResNet


def _instance():
    return generate_attack_instances(
        AttackInstanceConfig(
            instance_count=1,
            pair_count=64,
            rounds=6,
            seed=64001,
            split_role="validation",
        ),
        generated_at="2026-08-27T00:00:00+08:00",
    ).instances[0]


def _model():
    baseline = GohrResNet()
    model = build_pair_group_model(baseline, variant="b1_mean", seed=63001)
    model.eval()
    return model


def test_screened_ranking_invariants() -> None:
    instance = _instance()
    model = _model()
    candidates = tuple(range(100))

    ranking = rank_pair_group_subkeys_screened(
        instance,
        model,
        screen_pairs=16,
        screen_top_k=20,
        candidate_values=candidates,
        candidate_batch_size=32,
        device="cpu",
    )

    assert len(ranking.candidate_values) == 100
    assert len(ranking.candidate_scores) == 100
    assert len(ranking.candidate_order) == 100
    assert set(ranking.candidate_order) == set(candidates)
    assert ranking.pair_count == 64
    assert ranking.group_size == 64
    assert ranking.total_seconds > 0.0


def test_screened_ranking_validations() -> None:
    instance = _instance()
    model = _model()

    with pytest.raises(ValueError, match="screen_pairs must be 8, 16, or 32"):
        rank_pair_group_subkeys_screened(instance, model, screen_pairs=10)

    with pytest.raises(ValueError, match="screen_top_k must be between 1 and 65536"):
        rank_pair_group_subkeys_screened(instance, model, screen_top_k=0)
