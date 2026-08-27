"""Tests for the Phase 4 final-round subkey ranking boundary."""

from dataclasses import replace

import pytest
import torch
from torch import nn

from neural_sat_speck.attacks.ranking import (
    ALL_FINAL_ROUND_SUBKEYS,
    evaluate_subkey_ranking,
    rank_final_round_subkeys,
)
from neural_sat_speck.data.attack import (
    AttackInstance,
    AttackInstanceConfig,
    generate_attack_instances,
)
from neural_sat_speck.data.generator import encode_ciphertext_pair
from neural_sat_speck.neural.dataset import features_to_tensor
from neural_sat_speck.speck.cipher import encrypt_block
from neural_sat_speck.speck.partial import partial_decrypt_final_round


class _PairFeatureDistanceModel(nn.Module):
    """Deterministic test model that prefers a supplied state pair exactly."""

    def __init__(self, target_features: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("target_features", target_features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pair_count = self.target_features.shape[0]
        if features.shape[0] % pair_count:
            raise AssertionError("test features are not candidate-major")
        repeated_targets = self.target_features.repeat(
            features.shape[0] // pair_count,
            1,
            1,
        )
        return -((features - repeated_targets) ** 2).sum(dim=(1, 2))


class _ConstantModel(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.zeros(features.shape[0], device=features.device)


class _WrongShapeModel(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.zeros(max(0, features.shape[0] - 1), device=features.device)


def _instance() -> AttackInstance:
    generated = generate_attack_instances(
        AttackInstanceConfig(
            instance_count=1,
            pair_count=2,
            rounds=4,
            seed=61001,
            split_role="test",
        ),
        generated_at="2026-08-27T00:00:00+00:00",
    )
    return generated.instances[0]


def _target_features(instance: AttackInstance) -> torch.Tensor:
    target_rows = []
    for pair in instance.pairs:
        state0 = partial_decrypt_final_round(
            pair.ciphertext0,
            instance.final_round_subkey,
        )
        state1 = partial_decrypt_final_round(
            pair.ciphertext1,
            instance.final_round_subkey,
        )
        encoded = encode_ciphertext_pair(state0, state1)
        target_rows.append(features_to_tensor([encoded])[0])
    return torch.stack(target_rows)


def test_true_candidate_maps_to_the_expected_pre_final_state() -> None:
    instance = _instance()
    for pair in instance.pairs:
        assert partial_decrypt_final_round(
            pair.ciphertext0,
            instance.final_round_subkey,
        ) == encrypt_block(pair.plaintext0, instance.master_key, rounds=3)
        assert partial_decrypt_final_round(
            pair.ciphertext1,
            instance.final_round_subkey,
        ) == encrypt_block(pair.plaintext1, instance.master_key, rounds=3)


def test_correct_candidate_ranks_first_and_model_mode_is_restored() -> None:
    instance = _instance()
    model = _PairFeatureDistanceModel(_target_features(instance))
    model.train()
    true_subkey = instance.final_round_subkey
    candidates = (true_subkey ^ 1, true_subkey, true_subkey ^ 2)

    ranking = rank_final_round_subkeys(
        instance,
        model,
        candidate_values=candidates,
        candidate_batch_size=3,
        device="cpu",
    )

    assert ranking.candidate_order[0] == true_subkey
    assert ranking.rank(true_subkey) == 1
    assert ranking.rank_interval(true_subkey) == (1, 1)
    assert ranking.top_k(1) == (true_subkey,)
    assert ranking.peak_feature_tensor_bytes == 3 * 2 * 4 * 16 * 4
    assert model.training


def test_candidate_batching_produces_identical_scores_and_order() -> None:
    instance = _instance()
    candidates = tuple(
        instance.final_round_subkey ^ offset for offset in (0, 1, 2, 3, 4)
    )
    first = rank_final_round_subkeys(
        instance,
        _PairFeatureDistanceModel(_target_features(instance)),
        candidate_values=candidates,
        candidate_batch_size=1,
    )
    second = rank_final_round_subkeys(
        instance,
        _PairFeatureDistanceModel(_target_features(instance)),
        candidate_values=candidates,
        candidate_batch_size=3,
    )

    assert first.candidate_values == second.candidate_values
    assert first.candidate_scores == second.candidate_scores
    assert first.candidate_order == second.candidate_order


def test_score_ties_use_ascending_candidate_value() -> None:
    instance = _instance()
    ranking = rank_final_round_subkeys(
        instance,
        _ConstantModel(),
        candidate_values=(9, 2, 7),
        candidate_batch_size=2,
    )

    assert ranking.candidate_values == (2, 7, 9)
    assert ranking.candidate_order == (2, 7, 9)
    assert ranking.rank_interval(7) == (1, 3)
    assert ranking.rank(7) == 2


@pytest.mark.parametrize(
    "candidate_values",
    [(), (1, 1), (-1,), (1 << 16,), (True,)],
)
def test_invalid_candidate_sets_are_rejected(candidate_values: tuple[object, ...]) -> None:
    instance = _instance()
    with pytest.raises((TypeError, ValueError)):
        rank_final_round_subkeys(
            instance,
            _ConstantModel(),
            candidate_values=candidate_values,  # type: ignore[arg-type]
        )


def test_empty_instance_is_rejected() -> None:
    instance = _instance()
    empty_instance = replace(instance, pairs=())
    with pytest.raises(ValueError, match="at least one pair"):
        rank_final_round_subkeys(empty_instance, _ConstantModel())


def test_model_must_return_one_logit_per_row() -> None:
    with pytest.raises(ValueError, match="one scalar logit"):
        rank_final_round_subkeys(
            _instance(),
            _WrongShapeModel(),
            candidate_values=(0,),
        )


def test_default_candidate_set_is_the_complete_16_bit_domain() -> None:
    assert len(ALL_FINAL_ROUND_SUBKEYS) == 1 << 16
    assert ALL_FINAL_ROUND_SUBKEYS[0] == 0
    assert ALL_FINAL_ROUND_SUBKEYS[-1] == 0xFFFF


def test_evaluation_reports_missing_true_candidate_without_guessing() -> None:
    instance = _instance()
    ranking = rank_final_round_subkeys(
        instance,
        _ConstantModel(),
        candidate_values=(0, 1, 2),
    )

    evaluation = evaluate_subkey_ranking(
        ranking,
        instance.final_round_subkey,
        top_k_values=(0, 1, 3),
    )
    assert evaluation["true_subkey_in_candidate_set"] is False
    assert evaluation["true_rank"] is None
    assert evaluation["top_k_membership"] == {"0": False, "1": False, "3": False}
