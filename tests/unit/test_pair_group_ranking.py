"""Tests for complete-domain pair-aware ranking mechanics."""

import torch

from neural_sat_speck.attacks.group_ranking import (
    rank_pair_group_subkeys,
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


def test_pair_group_candidate_batching_and_mode_restore() -> None:
    instance = _instance()
    model = _model()
    model.train()
    candidates = tuple(instance.final_round_subkey ^ offset for offset in range(5))
    first = rank_pair_group_subkeys(
        instance,
        model,
        candidate_values=candidates,
        candidate_batch_size=1,
        device="cpu",
    )
    second = rank_pair_group_subkeys(
        instance,
        model,
        candidate_values=candidates,
        candidate_batch_size=3,
        device="cpu",
    )
    assert torch.allclose(
        torch.tensor(first.candidate_scores),
        torch.tensor(second.candidate_scores),
        atol=1e-6,
        rtol=0.0,
    )
    assert first.candidate_order == second.candidate_order
    assert first.peak_feature_tensor_bytes == 1 * 64 * 4 * 16 * 4
    assert model.training


def test_pair_group_ties_use_ascending_candidate_value() -> None:
    instance = _instance()
    model = _model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    ranking = rank_pair_group_subkeys(
        instance,
        model,
        candidate_values=(9, 2, 7),
        candidate_batch_size=2,
    )
    assert ranking.candidate_values == (2, 7, 9)
    assert ranking.candidate_order == (2, 7, 9)
    assert ranking.rank_interval(7) == (1, 3)
    assert ranking.rank(7) == 2
