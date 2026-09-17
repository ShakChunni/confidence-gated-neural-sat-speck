"""Unit tests for the hierarchical multi-round-key attack.

The end-to-end tests inject test-double rankers instead of running a frozen
checkpoint. A complete-domain ranking with the frozen B1 checkpoint costs
several minutes per instance on this machine, which is far outside a unit-test
budget, so the pipeline is exercised with two small ``PairGroupRanker``
subclasses that satisfy the same ranking interface:

* :class:`_NearestCandidateStubRanker` is an oracle double. It scores each
  candidate group by its distance to the feature row of one chosen round key, so
  the test can place the true round key anywhere in the complete-domain order
  (rank one, or second behind a deliberately promoted wrong candidate). Only the
  test-owned object receives the true round key; the implementation under test
  never does, and ``test_search_does_not_read_declared_ground_truth`` proves
  that the search ignores the instance's declared key material.
* :class:`_MeanFeatureStubRanker` is truth-free and its complete-domain order is
  arbitrary, which is what the pruning tests need: the true final round key is
  ranked outside the beam, so a ranking failure is observed deterministically.

Round counts are kept at five so the exact backend solves in milliseconds; the
seven-round residual-depth boundary itself is covered by
``tests/unit/test_multi_round_recovery.py``. The frozen B1 checkpoint is still
loaded and accepted by the stage interface in
``test_frozen_b1_checkpoint_satisfies_the_stage_interface``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from neural_sat_speck.attacks.hierarchical_multi_round import (
    HIERARCHICAL_ATTACK_STRATEGY,
    MAX_CANDIDATE_BATCH_SIZE,
    HierarchicalAttackResult,
    HierarchicalStage,
    StageDiagnostic,
    run_hierarchical_multi_round_attack,
)
from neural_sat_speck.attacks.multi_round_instance import partial_decrypt_instance
from neural_sat_speck.attacks.multi_round_recovery import (
    STATUS_EXHAUSTED_WITHOUT_UNRESOLVED,
)
from neural_sat_speck.attacks.recovery import (
    independent_verification_pairs,
    known_pairs_for_attack_instance,
)
from neural_sat_speck.data.attack import (
    AttackInstance,
    AttackInstanceConfig,
    generate_attack_instances,
)
from neural_sat_speck.neural.group_model import PairGroupRanker
from neural_sat_speck.neural.group_train import load_pair_group_checkpoint
from neural_sat_speck.neural.model import GohrResNet
from neural_sat_speck.neural.numpy_inference import (
    NumpyInferenceModule,
    NumpyRankerInference,
    vectorized_final_round_features,
)
from neural_sat_speck.speck.key_schedule import expand_key

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
B1_CHECKPOINT = REPOSITORY_ROOT / "models" / "speck32_64_phase6a_b1_mean_g64.pt"
B1_CHECKPOINT_SHA256 = (
    "a93d29a76a521b6f3fd0f1dff943df9f74de3857169b9d735c9aa478817954d5"
)
B1_MODEL_ID = "speck32_64_b1_mean_g64"

ROUNDS = 5
SEED = 4001
KNOWN_PAIRS = 4
VERIFICATION_PAIRS = 2
VERIFICATION_SEED = 999999
BONUS_SCORE = 1.0e6


class _NearestCandidateStubRanker(PairGroupRanker):
    """Oracle test double: distance in feature space to one chosen round key.

    ``rank_pair_group_subkeys`` passes a ranker only the partially decrypted
    features of each candidate, so the double precomputes the feature row of a
    chosen round key and scores every candidate by its negative squared distance
    to that row. An optional second row can be promoted by a large bonus, which
    places a deliberately wrong candidate ahead of the chosen one.
    """

    def __init__(
        self,
        pivot_features: np.ndarray,
        bonus_features: np.ndarray | None = None,
    ) -> None:
        super().__init__(variant="b1_mean")
        self._pivot = torch.from_numpy(
            np.asarray(pivot_features, dtype=np.float32)
        ).reshape(1, -1)
        self._bonus = (
            None
            if bonus_features is None
            else torch.from_numpy(np.asarray(bonus_features, dtype=np.float32)).reshape(
                1, -1
            )
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return one distance score per candidate group."""

        rows = features.reshape(features.shape[0], -1)
        scores = -((rows - self._pivot) ** 2).sum(dim=1)
        if self._bonus is None:
            return scores
        promoted = (rows == self._bonus).all(dim=1)
        return torch.where(promoted, scores + BONUS_SCORE, scores)


class _MeanFeatureStubRanker(PairGroupRanker):
    """Truth-free test double whose complete-domain order is arbitrary."""

    def __init__(self) -> None:
        super().__init__(variant="b1_mean")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return the mean feature activation of each candidate group."""

        return features.mean(dim=(1, 2, 3))


def _build_instance(
    rounds: int = ROUNDS,
    pair_count: int = 64,
    seed: int = SEED,
) -> AttackInstance:
    """Build one ephemeral attack instance for the pipeline tests."""

    return generate_attack_instances(
        AttackInstanceConfig(
            instance_count=1,
            pair_count=pair_count,
            rounds=rounds,
            seed=seed,
            split_role="train",
        ),
        generated_at="2026-09-10T00:00:00+08:00",
    ).instances[0]


def _nearest_stub(
    instance: AttackInstance,
    pivot_value: int,
    bonus_value: int | None = None,
) -> _NearestCandidateStubRanker:
    """Build an oracle double that ranks ``pivot_value`` first."""

    pivot = vectorized_final_round_features(instance.pairs, [pivot_value])
    if bonus_value is None:
        return _NearestCandidateStubRanker(pivot)
    bonus = vectorized_final_round_features(instance.pairs, [bonus_value])
    return _NearestCandidateStubRanker(pivot, bonus)


def _run(
    instance: AttackInstance,
    stages: list[HierarchicalStage],
    known_pairs: tuple,
    verification_pairs: tuple,
) -> HierarchicalAttackResult:
    """Run the attack with the test's solver-timeout envelope."""

    return run_hierarchical_multi_round_attack(
        instance,
        rounds=ROUNDS,
        stages=stages,
        known_pairs=known_pairs,
        verification_pairs=verification_pairs,
        solver_timeout=30.0,
    )


@pytest.fixture(scope="module")
def instance() -> AttackInstance:
    return _build_instance()


@pytest.fixture(scope="module")
def known_pairs(instance: AttackInstance) -> tuple:
    return known_pairs_for_attack_instance(instance)[:KNOWN_PAIRS]


@pytest.fixture(scope="module")
def verification_pairs(instance: AttackInstance) -> tuple:
    return independent_verification_pairs(
        instance.master_key,
        rounds=ROUNDS,
        pair_count=VERIFICATION_PAIRS,
        seed=VERIFICATION_SEED,
    )


@pytest.fixture(scope="module")
def two_stage_runs(
    instance: AttackInstance,
    known_pairs: tuple,
    verification_pairs: tuple,
) -> tuple[HierarchicalAttackResult, HierarchicalAttackResult]:
    """Two identical two-stage runs that refute a wrong pair before the true one.

    Stage zero keeps the true final round key. Stage one ranks a deliberately
    promoted wrong round key first and the true one second, so the tuple order
    is ``(wrong, true)`` and the run must test exactly two tuples.
    """

    schedule = expand_key(instance.master_key, rounds=ROUNDS)
    true_final, true_second = schedule[-1], schedule[-2]
    wrong_second = (true_second + 1) & 0xFFFF
    descended = partial_decrypt_instance(
        instance,
        true_final,
        expected_next_subkey=true_second,
        label=f"{instance.instance_id}-descended",
    )
    stages = [
        HierarchicalStage(
            round_index=ROUNDS - 1,
            model=_nearest_stub(instance, true_final),
            beam_width=1,
        ),
        HierarchicalStage(
            round_index=ROUNDS - 2,
            model=_nearest_stub(descended, true_second, wrong_second),
            beam_width=3,
        ),
    ]
    return tuple(
        _run(instance, stages, known_pairs, verification_pairs) for _ in range(2)
    )


@pytest.fixture(scope="module")
def pruned_result(
    instance: AttackInstance,
    known_pairs: tuple,
    verification_pairs: tuple,
) -> HierarchicalAttackResult:
    """Two-stage run whose stage-zero beam holds only a wrong final round key."""

    stages = [
        HierarchicalStage(
            round_index=ROUNDS - 1,
            model=_MeanFeatureStubRanker(),
            beam_width=1,
        ),
        HierarchicalStage(
            round_index=ROUNDS - 2,
            model=_MeanFeatureStubRanker(),
            beam_width=3,
        ),
    ]
    return _run(instance, stages, known_pairs, verification_pairs)


def test_hierarchical_stage_rejects_invalid_arguments() -> None:
    """Stage fields are strictly typed and the model must be a group ranker."""

    model = _MeanFeatureStubRanker()
    plain_model = GohrResNet()
    plain_model.eval()
    non_group_numpy = NumpyInferenceModule(NumpyRankerInference(plain_model))
    assert non_group_numpy.is_group_model is False

    assert HierarchicalStage(round_index=4, model=model, beam_width=7).beam_width == 7
    with pytest.raises(TypeError):
        HierarchicalStage(round_index=True, model=model, beam_width=1)
    with pytest.raises(TypeError):
        HierarchicalStage(round_index=4.0, model=model, beam_width=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        HierarchicalStage(round_index=-1, model=model, beam_width=1)
    with pytest.raises(ValueError):
        HierarchicalStage(round_index=4, model=model, beam_width=0)
    with pytest.raises(TypeError):
        HierarchicalStage(round_index=4, model=model, beam_width=True)
    with pytest.raises(TypeError):
        HierarchicalStage(round_index=4, model=nn.Linear(1, 1), beam_width=1)
    with pytest.raises(TypeError):
        HierarchicalStage(round_index=4, model=plain_model, beam_width=1)
    with pytest.raises(TypeError):
        HierarchicalStage(round_index=4, model=non_group_numpy, beam_width=1)


def test_attack_rejects_invalid_arguments(
    instance: AttackInstance,
    known_pairs: tuple,
    verification_pairs: tuple,
) -> None:
    stage = HierarchicalStage(
        round_index=ROUNDS - 1,
        model=_MeanFeatureStubRanker(),
        beam_width=1,
    )
    small_instance = _build_instance(pair_count=8)

    with pytest.raises(TypeError):
        _run("instance", [stage], known_pairs, verification_pairs)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        _run(small_instance, [stage], known_pairs, verification_pairs)
    with pytest.raises(TypeError):
        _run(instance, [object()], known_pairs, verification_pairs)  # type: ignore[list-item]
    with pytest.raises(ValueError):
        _run(instance, [], known_pairs, verification_pairs)
    with pytest.raises(ValueError):
        _run(
            instance,
            [HierarchicalStage(round_index=ROUNDS - 2, model=stage.model, beam_width=1)],
            known_pairs,
            verification_pairs,
        )
    with pytest.raises(ValueError):
        _run(
            instance,
            [
                HierarchicalStage(round_index=ROUNDS - 1, model=stage.model, beam_width=1),
                HierarchicalStage(round_index=ROUNDS - 1, model=stage.model, beam_width=1),
            ],
            known_pairs,
            verification_pairs,
        )
    with pytest.raises(ValueError):
        _run(
            instance,
            [
                HierarchicalStage(
                    round_index=round_index,
                    model=stage.model,
                    beam_width=1,
                )
                for round_index in (ROUNDS - 1, ROUNDS - 3)
            ],
            known_pairs,
            verification_pairs,
        )
    for invalid_rounds in (1, ROUNDS - 1):
        with pytest.raises(ValueError):
            run_hierarchical_multi_round_attack(
                instance,
                rounds=invalid_rounds,
                stages=[stage],
                known_pairs=known_pairs,
                verification_pairs=verification_pairs,
            )
    for invalid_batch_size in (0, -1, MAX_CANDIDATE_BATCH_SIZE + 1):
        with pytest.raises(ValueError):
            run_hierarchical_multi_round_attack(
                instance,
                rounds=ROUNDS,
                stages=[stage],
                known_pairs=known_pairs,
                verification_pairs=verification_pairs,
                candidate_batch_size=invalid_batch_size,
            )
    with pytest.raises(TypeError):
        run_hierarchical_multi_round_attack(
            instance,
            rounds=ROUNDS,
            stages=[stage],
            known_pairs=known_pairs,
            verification_pairs=verification_pairs,
            candidate_batch_size=True,
        )
    for invalid_timeout in (0.0, -1.0):
        with pytest.raises(ValueError):
            run_hierarchical_multi_round_attack(
                instance,
                rounds=ROUNDS,
                stages=[stage],
                known_pairs=known_pairs,
                verification_pairs=verification_pairs,
                solver_timeout=invalid_timeout,
            )
    with pytest.raises(TypeError):
        run_hierarchical_multi_round_attack(
            instance,
            rounds=ROUNDS,
            stages=[stage],
            known_pairs=known_pairs,
            verification_pairs=verification_pairs,
            solver_timeout="30.0",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError):
        run_hierarchical_multi_round_attack(
            instance,
            rounds=ROUNDS,
            stages=[stage],
            known_pairs=known_pairs,
            verification_pairs=verification_pairs,
            device=123,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        run_hierarchical_multi_round_attack(
            instance,
            rounds=ROUNDS,
            stages=[stage],
            known_pairs=(),
            verification_pairs=verification_pairs,
        )
    with pytest.raises(ValueError):
        run_hierarchical_multi_round_attack(
            instance,
            rounds=ROUNDS,
            stages=[stage],
            known_pairs=known_pairs,
            verification_pairs=(),
        )


def test_single_stage_attack_recovers_five_round_instance_key(
    instance: AttackInstance,
    known_pairs: tuple,
    verification_pairs: tuple,
) -> None:
    """A beam holding the true final round key recovers the master key exactly."""

    schedule = expand_key(instance.master_key, rounds=ROUNDS)
    result = _run(
        instance,
        [
            HierarchicalStage(
                round_index=ROUNDS - 1,
                model=_nearest_stub(instance, schedule[-1]),
                beam_width=1,
            )
        ],
        known_pairs,
        verification_pairs,
    )

    assert result.strategy == HIERARCHICAL_ATTACK_STRATEGY
    assert result.success is True
    assert result.status == "verified_recovery"
    assert result.failure_cause is None
    assert tuple(result.recovered_master_key) == tuple(instance.master_key)
    assert result.candidates_attempted == 1
    assert result.solver_seconds >= 0.0
    assert result.total_seconds >= result.ranking_seconds
    assert len(result.stages) == 1
    stage = result.stages[0]
    assert isinstance(stage, StageDiagnostic)
    assert stage.stage_index == 0
    assert stage.round_index == ROUNDS - 1
    assert stage.candidates_ranked == 1 << 16
    assert stage.beam_width == 1
    assert stage.top_candidate == schedule[-1]
    assert stage.true_round_key_rank == 1
    assert stage.ranking_seconds > 0.0

    payload = result.to_dict()
    assert payload["stages"][0]["true_round_key_rank"] == 1
    assert payload["recovered_master_key"] == list(instance.master_key)
    assert json.loads(json.dumps(payload)) == payload


def test_two_stage_attack_refutes_a_wrong_pair_before_verifying(
    instance: AttackInstance,
    two_stage_runs: tuple[HierarchicalAttackResult, HierarchicalAttackResult],
) -> None:
    """Tuple order is honoured across stages: the wrong pair is tested first."""

    schedule = expand_key(instance.master_key, rounds=ROUNDS)
    wrong_second = (schedule[-2] + 1) & 0xFFFF
    result = two_stage_runs[0]

    assert result.success is True
    assert result.failure_cause is None
    assert tuple(result.recovered_master_key) == tuple(instance.master_key)
    assert result.candidates_attempted == 2
    assert [stage.round_index for stage in result.stages] == [ROUNDS - 1, ROUNDS - 2]
    assert result.stages[0].top_candidate == schedule[-1]
    assert result.stages[1].top_candidate == wrong_second
    assert result.stages[1].true_round_key_rank == 2
    assert result.stages[1].candidates_ranked == 1 << 16


def test_identical_runs_produce_identical_search_and_order(
    two_stage_runs: tuple[HierarchicalAttackResult, HierarchicalAttackResult],
) -> None:
    """Two identical runs agree on every search-side quantity and on the order."""

    def projection(result: HierarchicalAttackResult) -> tuple:
        return (
            result.success,
            result.status,
            result.failure_cause,
            result.candidates_attempted,
            tuple(result.recovered_master_key or ()),
            tuple(
                (
                    stage.stage_index,
                    stage.round_index,
                    stage.candidates_ranked,
                    stage.beam_width,
                    stage.top_candidate,
                    stage.true_round_key_rank,
                )
                for stage in result.stages
            ),
        )

    first, second = two_stage_runs

    assert projection(first) == projection(second)
    assert first.candidates_attempted == 2


def test_beam_width_bounds_the_number_of_tested_tuples(
    pruned_result: HierarchicalAttackResult,
) -> None:
    """Beam pruning limits the exact tests to the beam product, not the domain."""

    assert pruned_result.stages[0].true_round_key_rank > 1
    assert pruned_result.candidates_attempted == 3
    assert pruned_result.candidates_attempted == (
        pruned_result.stages[0].beam_width * pruned_result.stages[1].beam_width
    )


def test_pruned_first_stage_candidate_reports_ranking_failure(
    pruned_result: HierarchicalAttackResult,
) -> None:
    """A beam that never held the true final round key is a ranking failure."""

    assert pruned_result.success is False
    assert pruned_result.recovered_master_key is None
    assert pruned_result.failure_cause == "ranking"
    assert pruned_result.status == STATUS_EXHAUSTED_WITHOUT_UNRESOLVED
    assert pruned_result.stages[0].true_round_key_rank is not None
    assert pruned_result.stages[0].true_round_key_rank > (
        pruned_result.stages[0].beam_width
    )
    # The true prefix never reached stage one, so its rank is unresolvable.
    assert pruned_result.stages[1].true_round_key_rank is None


def test_pruning_inside_the_final_stage_reports_ranking_failure(
    instance: AttackInstance,
    known_pairs: tuple,
    verification_pairs: tuple,
) -> None:
    """A final-stage prune keeps its true rank diagnostic and reports ranking."""

    schedule = expand_key(instance.master_key, rounds=ROUNDS)
    true_final, true_second = schedule[-1], schedule[-2]
    wrong_second = (true_second + 1) & 0xFFFF
    descended = partial_decrypt_instance(
        instance,
        true_final,
        expected_next_subkey=true_second,
        label=f"{instance.instance_id}-descended",
    )
    result = _run(
        instance,
        [
            HierarchicalStage(
                round_index=ROUNDS - 1,
                model=_nearest_stub(instance, true_final),
                beam_width=1,
            ),
            HierarchicalStage(
                round_index=ROUNDS - 2,
                model=_nearest_stub(descended, true_second, wrong_second),
                beam_width=1,
            ),
        ],
        known_pairs,
        verification_pairs,
    )

    assert result.success is False
    assert result.failure_cause == "ranking"
    assert result.candidates_attempted == 1
    assert result.stages[0].true_round_key_rank == 1
    assert result.stages[1].true_round_key_rank == 2


def test_search_does_not_read_declared_ground_truth(
    instance: AttackInstance,
    known_pairs: tuple,
    verification_pairs: tuple,
) -> None:
    """Replacing the instance's declared keys leaves the search bit-identical.

    The pairs are left untouched, so the lying instance only changes the key
    material the diagnostics may read. Every search-side quantity must stay the
    same, while the post-search truth ranks must follow the declared schedule.
    """

    lying_instance = AttackInstance(
        instance_id=f"{instance.instance_id}-lying",
        master_key=tuple(
            (word + 1) & 0xFFFF for word in instance.master_key
        ),  # type: ignore[arg-type]
        final_round_subkey=(instance.final_round_subkey + 1) & 0xFFFF,
        pairs=instance.pairs,
    )
    honest_schedule = expand_key(instance.master_key, rounds=ROUNDS)
    lying_schedule = expand_key(lying_instance.master_key, rounds=ROUNDS)
    assert lying_schedule[-1] != honest_schedule[-1]

    stages = [
        HierarchicalStage(
            round_index=ROUNDS - 1,
            model=_MeanFeatureStubRanker(),
            beam_width=1,
        ),
        HierarchicalStage(
            round_index=ROUNDS - 2,
            model=_MeanFeatureStubRanker(),
            beam_width=3,
        ),
    ]
    honest = _run(instance, stages, known_pairs, verification_pairs)
    lying = _run(lying_instance, stages, known_pairs, verification_pairs)

    honest_search = [
        (stage.candidates_ranked, stage.beam_width, stage.top_candidate)
        for stage in honest.stages
    ]
    lying_search = [
        (stage.candidates_ranked, stage.beam_width, stage.top_candidate)
        for stage in lying.stages
    ]
    assert honest_search == lying_search
    assert honest.candidates_attempted == lying.candidates_attempted
    assert honest.status == lying.status
    assert honest.failure_cause == lying.failure_cause
    assert honest.stages[0].true_round_key_rank != lying.stages[0].true_round_key_rank


def test_frozen_b1_checkpoint_satisfies_the_stage_interface() -> None:
    """The frozen five-round B1 checkpoint is accepted as a stage ranker."""

    digest = hashlib.sha256(B1_CHECKPOINT.read_bytes()).hexdigest()
    assert digest == B1_CHECKPOINT_SHA256
    model, metadata = load_pair_group_checkpoint(B1_CHECKPOINT, device="cpu")
    assert isinstance(model, PairGroupRanker)
    assert model.model_id == B1_MODEL_ID
    assert metadata["model_id"] == B1_MODEL_ID

    assert HierarchicalStage(round_index=5, model=model, beam_width=16).beam_width == 16
    wrapped = NumpyInferenceModule(NumpyRankerInference(model))
    assert HierarchicalStage(round_index=5, model=wrapped, beam_width=16).beam_width == 16


def test_screened_hierarchical_stage_validation_and_execution() -> None:
    """HierarchicalStage validates screened parameters and executes screened ranking."""
    model = PairGroupRanker(variant="b1_mean")

    # Validation checks
    with pytest.raises(ValueError, match="screen_pairs must be 8, 16, or 32"):
        HierarchicalStage(
            round_index=5,
            model=model,
            beam_width=16,
            screened=True,
            screen_pairs=10,
        )

    with pytest.raises(ValueError, match="screen_top_k must be between 1 and 65536"):
        HierarchicalStage(
            round_index=5,
            model=model,
            beam_width=16,
            screened=True,
            screen_top_k=0,
        )

    stage = HierarchicalStage(
        round_index=5,
        model=model,
        beam_width=16,
        screened=True,
        screen_pairs=16,
        screen_top_k=512,
    )
    assert stage.screened is True
    assert stage.screen_pairs == 16
    assert stage.screen_top_k == 512


def test_stage_cache_and_excluded_paths_cascade(
    instance: AttackInstance,
    known_pairs: tuple,
    verification_pairs: tuple,
) -> None:
    """Cascade reusing stage_cache and excluded_paths avoids duplicate work."""

    schedule = expand_key(instance.master_key, rounds=ROUNDS)
    true_final, true_second = schedule[-1], schedule[-2]
    wrong_second = (true_second + 1) & 0xFFFF
    descended = partial_decrypt_instance(
        instance,
        true_final,
        expected_next_subkey=true_second,
        label=f"{instance.instance_id}-descended",
    )
    stage0 = HierarchicalStage(
        round_index=ROUNDS - 1,
        model=_nearest_stub(instance, true_final),
        beam_width=1,
    )
    stage1_narrow = HierarchicalStage(
        round_index=ROUNDS - 2,
        model=_nearest_stub(descended, true_second, wrong_second),
        beam_width=1,
    )
    stage1_wide = HierarchicalStage(
        round_index=ROUNDS - 2,
        model=_nearest_stub(descended, true_second, wrong_second),
        beam_width=2,
    )

    stage_cache: dict = {}
    res_narrow = run_hierarchical_multi_round_attack(
        instance,
        rounds=ROUNDS,
        stages=[stage0, stage1_narrow],
        known_pairs=known_pairs,
        verification_pairs=verification_pairs,
        solver_timeout=30.0,
        stage_cache=stage_cache,
    )
    assert res_narrow.success is False
    assert res_narrow.candidates_attempted == 1
    assert len(res_narrow.attempted_paths) == 1
    assert () in stage_cache
    assert (true_final,) in stage_cache

    res_wide = run_hierarchical_multi_round_attack(
        instance,
        rounds=ROUNDS,
        stages=[stage0, stage1_wide],
        known_pairs=known_pairs,
        verification_pairs=verification_pairs,
        solver_timeout=30.0,
        stage_cache=stage_cache,
        excluded_paths=set(res_narrow.attempted_paths),
    )
    assert res_wide.success is True
    assert res_wide.candidates_attempted == 1
    assert tuple(res_wide.recovered_master_key) == tuple(instance.master_key)


