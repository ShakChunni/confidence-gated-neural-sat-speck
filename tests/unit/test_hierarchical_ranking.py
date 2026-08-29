"""Focused tests for the fixed-budget H1 hierarchical ranker."""

from __future__ import annotations

import inspect
import json
from dataclasses import fields

import pytest
import torch
from torch import nn

from neural_sat_speck.attacks.group_ranking import PairGroupSubkeyRanking
from neural_sat_speck.attacks.hierarchical_ranking import (
    HIERARCHICAL_METHOD_ID,
    HIERARCHICAL_WINDOW_SIZE,
    HierarchicalSubkeyRanking,
    rank_hierarchical_subkeys,
)
from neural_sat_speck.attacks.ranking import (
    ALL_FINAL_ROUND_SUBKEYS,
    rank_final_round_subkeys,
)
from neural_sat_speck.data.attack import (
    AttackInstance,
    AttackInstanceConfig,
    generate_attack_instances,
)

_B1_WINDOW = (15, 3, 11, 7, 0, 12, 4, 8, 14, 2, 10, 6, 1, 13, 5, 9)


class _ConstantModel(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.zeros(features.shape[0], device=features.device)


class _FeatureSumModel(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features.sum(dim=(1, 2))


class _NegativeFeatureSumModel(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return -features.sum(dim=(1, 2))


class _CountingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rows_seen = 0
        self.calls = 0

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.rows_seen += int(features.shape[0])
        return torch.zeros(features.shape[0], device=features.device)


@pytest.fixture(scope="module")
def instance() -> AttackInstance:
    return generate_attack_instances(
        AttackInstanceConfig(
            instance_count=1,
            pair_count=64,
            rounds=6,
            seed=67101,
            split_role="validation",
        ),
        generated_at="2026-08-28T00:00:00+08:00",
    ).instances[0]


def _b1_ranking() -> PairGroupSubkeyRanking:
    tail = tuple(value for value in ALL_FINAL_ROUND_SUBKEYS if value not in _B1_WINDOW)
    order = _B1_WINDOW + tail
    return PairGroupSubkeyRanking(
        candidate_values=ALL_FINAL_ROUND_SUBKEYS,
        candidate_scores=tuple(float(-value) for value in ALL_FINAL_ROUND_SUBKEYS),
        candidate_order=order,
        pair_count=64,
        group_size=64,
        candidate_batch_size=256,
        score_aggregation="group_logit",
        partial_decryption_seconds=1.0,
        inference_seconds=2.0,
        sorting_seconds=3.0,
        total_seconds=7.0,
        peak_feature_tensor_bytes=1234,
    )


def test_h1_preserves_b1_window_set_and_tail(instance: AttackInstance) -> None:
    b1 = _b1_ranking()
    result = rank_hierarchical_subkeys(instance, b1, _ConstantModel(), device="cpu")

    assert result.method_id == HIERARCHICAL_METHOD_ID
    assert set(result.reranked_top16) == set(b1.candidate_order[:16])
    assert result.b1_top16_input_order == b1.candidate_order[:16]
    assert result.candidate_order[16:] == b1.candidate_order[16:]
    assert result.b1_tail == b1.candidate_order[16:]
    assert result.candidate_order[:16] == result.reranked_top16
    assert len(result.candidate_order) == 1 << 16
    assert len(set(result.candidate_order)) == 1 << 16
    assert set(result.candidate_order) == set(ALL_FINAL_ROUND_SUBKEYS)


def test_h1_b0_ties_use_ascending_candidate_value(instance: AttackInstance) -> None:
    result = rank_hierarchical_subkeys(
        instance,
        _b1_ranking(),
        _ConstantModel(),
        device="cpu",
    )

    assert result.reranked_top16 == tuple(sorted(_B1_WINDOW))
    assert result.b0_window_candidate_values == tuple(sorted(_B1_WINDOW))
    assert result.b0_window_scores == (0.0,) * HIERARCHICAL_WINDOW_SIZE


def test_h1_invokes_b0_only_for_the_sixteen_candidate_window(
    instance: AttackInstance,
) -> None:
    model = _CountingModel()
    result = rank_hierarchical_subkeys(
        instance,
        _b1_ranking(),
        model,
        b0_candidate_batch_size=1,
        device="cpu",
    )

    assert model.calls == HIERARCHICAL_WINDOW_SIZE
    assert model.rows_seen == HIERARCHICAL_WINDOW_SIZE * len(instance.pairs)
    assert len(result.b0_scores) == HIERARCHICAL_WINDOW_SIZE


def test_h1_uses_existing_b0_sum_logits_sorting(instance: AttackInstance) -> None:
    b1 = _b1_ranking()
    expected = rank_final_round_subkeys(
        instance,
        _FeatureSumModel(),
        candidate_values=b1.candidate_order[:HIERARCHICAL_WINDOW_SIZE],
        candidate_batch_size=HIERARCHICAL_WINDOW_SIZE,
        device="cpu",
    )
    result = rank_hierarchical_subkeys(
        instance,
        b1,
        _FeatureSumModel(),
        device="cpu",
    )

    assert result.reranked_top16 == expected.candidate_order
    assert result.b0_window_candidate_values == expected.candidate_values
    assert result.b0_window_scores == expected.candidate_scores


def test_h1_scalar_and_batched_window_scores_match(instance: AttackInstance) -> None:
    b1 = _b1_ranking()
    scalar = rank_hierarchical_subkeys(
        instance,
        b1,
        _FeatureSumModel(),
        b0_candidate_batch_size=1,
        device="cpu",
    )
    batched = rank_hierarchical_subkeys(
        instance,
        b1,
        _FeatureSumModel(),
        b0_candidate_batch_size=HIERARCHICAL_WINDOW_SIZE,
        device="cpu",
    )

    assert scalar.b0_window_candidate_values == batched.b0_window_candidate_values
    assert scalar.b0_window_scores == batched.b0_window_scores
    assert scalar.candidate_order == batched.candidate_order


def test_h1_rerun_is_deterministic(instance: AttackInstance) -> None:
    first = rank_hierarchical_subkeys(
        instance,
        _b1_ranking(),
        _FeatureSumModel(),
        device="cpu",
    )
    second = rank_hierarchical_subkeys(
        instance,
        _b1_ranking(),
        _FeatureSumModel(),
        device="cpu",
    )

    assert first.candidate_order == second.candidate_order
    assert first.b0_window_scores == second.b0_window_scores


def test_h1_serialization_round_trip(instance: AttackInstance) -> None:
    result = rank_hierarchical_subkeys(
        instance,
        _b1_ranking(),
        _NegativeFeatureSumModel(),
        device="cpu",
        b1_checkpoint_sha256="b1-hash",
        b0_checkpoint_sha256="b0-hash",
        b1_model_id="speck32_64_b1_mean_g64",
    )
    payload = json.loads(json.dumps(result.to_dict()))
    restored = HierarchicalSubkeyRanking.from_dict(payload)

    assert restored == result
    assert restored.b1_checkpoint_sha256 == "b1-hash"
    assert restored.b0_checkpoint_sha256 == "b0-hash"


def test_h1_deserialization_rejects_order_inconsistent_with_b0_scores(
    instance: AttackInstance,
) -> None:
    result = rank_hierarchical_subkeys(
        instance,
        _b1_ranking(),
        _FeatureSumModel(),
        device="cpu",
    )
    payload = result.to_dict()
    payload["reranked_top16_order"][0:2] = reversed(
        payload["reranked_top16_order"][0:2]
    )
    payload["candidate_order"][0:2] = payload["reranked_top16_order"][0:2]

    with pytest.raises(ValueError, match="recorded B0 scores"):
        HierarchicalSubkeyRanking.from_dict(payload)


@pytest.mark.parametrize("field", ("tie_rule", "candidate_domain"))
def test_h1_deserialization_rejects_corrupt_protocol_metadata(
    instance: AttackInstance,
    field: str,
) -> None:
    result = rank_hierarchical_subkeys(
        instance,
        _b1_ranking(),
        _ConstantModel(),
        device="cpu",
    )
    payload = result.to_dict()
    payload[field] = None

    with pytest.raises(ValueError, match="tie rule|complete domain"):
        HierarchicalSubkeyRanking.from_dict(payload)


def test_h1_timing_fields_are_finite_and_non_negative(
    instance: AttackInstance,
) -> None:
    result = rank_hierarchical_subkeys(
        instance,
        _b1_ranking(),
        _ConstantModel(),
        device="cpu",
    )
    timing_names = {
        field.name
        for field in fields(HierarchicalSubkeyRanking)
        if field.name.endswith("seconds")
    }

    assert timing_names
    for name in timing_names:
        value = getattr(result, name)
        assert isinstance(value, float)
        assert torch.isfinite(torch.tensor(value))
        assert value >= 0.0
    assert result.total_seconds >= result.b1_ranking_seconds
    assert result.total_seconds >= result.b0_window_seconds


def test_h1_ordering_api_does_not_accept_or_serialize_true_key(
    instance: AttackInstance,
) -> None:
    assert "true_subkey" not in inspect.signature(rank_hierarchical_subkeys).parameters
    result = rank_hierarchical_subkeys(
        instance,
        _b1_ranking(),
        _ConstantModel(),
        device="cpu",
    )
    assert "true_final_round_subkey" not in result.to_dict()


def test_h1_rejects_incomplete_or_duplicate_b1_order(
    instance: AttackInstance,
) -> None:
    malformed = _b1_ranking()
    object.__setattr__(malformed, "candidate_order", (0,) + malformed.candidate_order[:-1])
    with pytest.raises(ValueError, match="complete candidate permutation"):
        rank_hierarchical_subkeys(instance, malformed, _ConstantModel())


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"window_size": 8}, "window_size"),
        ({"b0_candidate_batch_size": 0}, "b0_candidate_batch_size"),
    ],
)
def test_h1_rejects_unfrozen_budget_and_invalid_batch_size(
    instance: AttackInstance,
    kwargs: dict[str, int],
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        rank_hierarchical_subkeys(instance, _b1_ranking(), _ConstantModel(), **kwargs)


def test_h1_rejects_wrong_instance_pair_count(instance: AttackInstance) -> None:
    short_instance = AttackInstance(
        instance_id=instance.instance_id,
        master_key=instance.master_key,
        final_round_subkey=instance.final_round_subkey,
        pairs=instance.pairs[:-1],
    )
    with pytest.raises(ValueError, match="exactly 64 pairs"):
        rank_hierarchical_subkeys(short_instance, _b1_ranking(), _ConstantModel())


def test_h1_rejects_wrong_ranking_type(instance: AttackInstance) -> None:
    with pytest.raises(TypeError, match="PairGroupSubkeyRanking"):
        rank_hierarchical_subkeys(instance, object(), _ConstantModel())  # type: ignore[arg-type]


def test_h1_rejects_non_model_b0_input(instance: AttackInstance) -> None:
    with pytest.raises(TypeError, match="torch.nn.Module"):
        rank_hierarchical_subkeys(instance, _b1_ranking(), object())  # type: ignore[arg-type]
