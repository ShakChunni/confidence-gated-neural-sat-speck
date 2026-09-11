"""Hierarchical multi-round-key attack with truth-free neural beam ranking.

The exact backend completes a SPECK32/64 instance only while
``rounds - (assumed trailing round keys) <= 5`` (measured; see
``docs/PHASE11_SCALING_STRATEGY.md``). A seven-round attack therefore has to
rank the last **two** round keys: rank the final round key, partially decrypt
the instance one round under each surviving candidate, rank the round key below
it on the resulting shorter instance, and only then verify candidate tuples with
the exact model.

:mod:`neural_sat_speck.attacks.multi_round_instance` provides the one-round
descent and :mod:`neural_sat_speck.attacks.multi_round_recovery` the ordered
exact testing; this module is the beam search that connects them. It is
deliberately separate from :mod:`neural_sat_speck.attacks.hierarchical_ranking`,
which is the frozen Phase 6 H1 composition of two rankers on a *single* round
key and is not used here.

Leakage rule (critical)
-----------------------
The ground-truth round keys must never influence ranking, beam pruning, or
candidate-tuple ordering. The search in :func:`run_hierarchical_multi_round_attack`
therefore reads no key material at all: it calls
:func:`neural_sat_speck.attacks.group_ranking.rank_pair_group_subkeys`, which
never consults the instance's stored ``final_round_subkey``, and it descends
with an explicit placeholder subkey that is never read back. Ground truth is
read exclusively by :func:`_diagnose_stages`, which runs strictly after the
search has finished and only fills diagnostic fields; nothing it computes is
fed back into the search, the beams, or the tuple order.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from neural_sat_speck.attacks.group_ranking import rank_pair_group_subkeys
from neural_sat_speck.attacks.screened_group_ranking import (
    rank_pair_group_subkeys_screened,
)
from neural_sat_speck.attacks.multi_round_instance import partial_decrypt_instance
from neural_sat_speck.attacks.multi_round_recovery import (
    run_multi_round_key_recovery,
)
from neural_sat_speck.attacks.recovery import MasterKey
from neural_sat_speck.data.attack import AttackInstance
from neural_sat_speck.neural.group_model import PairGroupRanker
from neural_sat_speck.neural.numpy_inference import NumpyInferenceModule
from neural_sat_speck.sat.isolation import IsolatedSolverFactory
from neural_sat_speck.sat.model import KnownPair
from neural_sat_speck.speck.key_schedule import expand_key
from neural_sat_speck.speck.primitives import WORD_MASK

HIERARCHICAL_ATTACK_STRATEGY = "hierarchical_multi_round_beam"
HIERARCHICAL_SOLVER_NAME = "hierarchical_multi_round_exact"
MAX_CANDIDATE_BATCH_SIZE = 1024
DESCENT_SUBKEY_PLACEHOLDER = 0
FAILURE_CAUSE_RANKING = "ranking"
FAILURE_CAUSE_SAT_BACKEND = "sat_backend"
STATUS_NO_RANKED_CANDIDATES = "no_ranked_candidates"


def _validate_integer(value: object, name: str, *, minimum: int) -> int:
    """Validate a bounded integer without accepting booleans."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _validate_seconds(value: object, name: str) -> float:
    """Validate a finite, non-negative timing measurement."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    normalised = float(value)
    if not math.isfinite(normalised) or normalised < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return normalised


def _validate_word(value: object, name: str) -> int:
    """Validate a 16-bit word."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 0 <= value <= WORD_MASK:
        raise ValueError(f"{name} must be a 16-bit word")
    return value


def _validate_stage_model(model: object, name: str) -> None:
    """Reject anything the pair-group ranking entry point cannot score."""

    if not isinstance(model, (PairGroupRanker, NumpyInferenceModule)):
        raise TypeError(f"{name} must be a PairGroupRanker or NumpyInferenceModule")
    if isinstance(model, NumpyInferenceModule) and not model.is_group_model:
        raise TypeError(f"{name} NumPy model must wrap a PairGroupRanker")


def _validate_timeout(timeout: object) -> float | None:
    """Validate the optional per-candidate solver budget."""

    if timeout is None:
        return None
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise TypeError("solver_timeout must be a number or None")
    if timeout <= 0:
        raise ValueError("solver_timeout must be positive")
    return float(timeout)


def _validate_device(device: object) -> None:
    """Validate the optional ranking device without resolving it."""

    if device is None:
        return
    if not isinstance(device, (str, torch.device)):
        raise TypeError("device must be a string, torch.device, or None")


def _validate_materialised_pairs(pairs: object, name: str) -> tuple[KnownPair, ...]:
    """Materialise a non-empty sequence of SAT constraint pairs."""

    if isinstance(pairs, (str, bytes)) or not isinstance(pairs, Sequence):
        raise TypeError(f"{name} must be a sequence of plaintext/ciphertext pairs")
    materialised = tuple(pairs)
    if not materialised:
        raise ValueError(f"{name} must contain at least one pair")
    return materialised


def _validate_master_key(master_key: object) -> MasterKey:
    """Validate a decoded master key as four 16-bit words."""

    if isinstance(master_key, (str, bytes)) or not isinstance(master_key, Sequence):
        raise TypeError("recovered_master_key must be a sequence of four words")
    words = tuple(master_key)
    if len(words) != 4:
        raise ValueError("recovered_master_key must contain four words")
    return (
        _validate_word(words[0], "recovered_master_key[0]"),
        _validate_word(words[1], "recovered_master_key[1]"),
        _validate_word(words[2], "recovered_master_key[2]"),
        _validate_word(words[3], "recovered_master_key[3]"),
    )


@dataclass(frozen=True)
class HierarchicalStage:
    """One complete-domain ranking stage of the hierarchical attack.

    ``round_index`` is the trailing round-key index this stage targets,
    ``model`` the pair-group ranker that scores it, and ``beam_width`` the
    number of top-ranked candidates kept from each complete-domain ranking.
    ``model`` is typed ``Any`` because the stage accepts either a
    :class:`~neural_sat_speck.neural.group_model.PairGroupRanker` or a
    group-model :class:`~neural_sat_speck.neural.numpy_inference.NumpyInferenceModule`;
    the concrete type is enforced at construction.
    """

    round_index: int
    model: Any
    beam_width: int
    screened: bool = False
    screen_pairs: int = 16
    screen_top_k: int = 512

    def __post_init__(self) -> None:
        """Validate the stage's index, ranker interface, and beam width."""

        _validate_integer(self.round_index, "stages round_index", minimum=0)
        _validate_integer(self.beam_width, "stages beam_width", minimum=1)
        _validate_stage_model(self.model, "stages model")
        if self.screened:
            if self.screen_pairs not in (8, 16, 32):
                raise ValueError(f"screen_pairs must be 8, 16, or 32, got {self.screen_pairs}")
            if self.screen_top_k <= 0 or self.screen_top_k > 65536:
                raise ValueError(f"screen_top_k must be between 1 and 65536, got {self.screen_top_k}")
            if not isinstance(self.model, PairGroupRanker):
                raise TypeError("screened stage requires a PairGroupRanker model")


@dataclass(frozen=True)
class StageDiagnostic:
    """Per-stage record of the completed search and its post-hoc truth ranks.

    ``candidates_ranked`` counts every candidate scored at the stage, i.e. the
    complete domain once per surviving parent prefix. ``top_candidate`` is the
    surviving candidate with the smallest cumulative stage-rank sum, so for
    stage zero it is the head of the single complete-domain ranking.
    ``ranking_seconds`` covers that stage's rankings plus the one-round descent
    of the candidates it kept. ``true_round_key_rank`` is the one-based rank of
    the true round key for ``round_index`` inside the complete-domain ranking
    computed for the true prefix; it is ``None`` when no ground truth is
    available or when the true prefix had already been pruned, and it is only
    ever read after the search has finished.
    """

    stage_index: int
    round_index: int
    candidates_ranked: int
    beam_width: int
    top_candidate: int
    ranking_seconds: float
    true_round_key_rank: int | None

    def __post_init__(self) -> None:
        """Enforce the diagnostic's structural and range invariants."""

        _validate_integer(self.stage_index, "stage_index", minimum=0)
        _validate_integer(self.round_index, "round_index", minimum=0)
        _validate_integer(self.candidates_ranked, "candidates_ranked", minimum=1)
        _validate_integer(self.beam_width, "beam_width", minimum=1)
        _validate_word(self.top_candidate, "top_candidate")
        _validate_seconds(self.ranking_seconds, "ranking_seconds")
        if self.true_round_key_rank is not None:
            _validate_integer(
                self.true_round_key_rank,
                "true_round_key_rank",
                minimum=1,
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the stage record, including its post-search truth rank."""

        return {
            "stage_index": self.stage_index,
            "round_index": self.round_index,
            "candidates_ranked": self.candidates_ranked,
            "beam_width": self.beam_width,
            "top_candidate": self.top_candidate,
            "ranking_seconds": self.ranking_seconds,
            "true_round_key_rank": self.true_round_key_rank,
        }


@dataclass(frozen=True)
class HierarchicalAttackResult:
    """Complete outcome of one hierarchical multi-round-key attack.

    ``ranking_seconds`` is the sum of the stage records. ``total_seconds`` is
    the wall clock of the whole call and therefore also covers the descent,
    tuple assembly, exact verification, and the post-search diagnostic pass, so
    it may exceed the sum of the recorded stage totals.
    """

    strategy: str
    success: bool
    status: str
    stages: tuple[StageDiagnostic, ...]
    candidates_attempted: int
    recovered_master_key: MasterKey | None
    ranking_seconds: float
    solver_seconds: float
    verification_seconds: float
    total_seconds: float
    failure_cause: str | None
    attempted_paths: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        """Enforce the result's accounting and outcome invariants."""

        if self.strategy != HIERARCHICAL_ATTACK_STRATEGY:
            raise ValueError(f"strategy must be {HIERARCHICAL_ATTACK_STRATEGY!r}")
        if not isinstance(self.success, bool):
            raise TypeError("success must be a boolean")
        if not isinstance(self.status, str) or not self.status:
            raise ValueError("status must be a non-empty string")
        if not isinstance(self.stages, tuple) or not self.stages:
            raise ValueError("stages must be a non-empty tuple of diagnostics")
        for stage in self.stages:
            if not isinstance(stage, StageDiagnostic):
                raise TypeError("stages must contain StageDiagnostic values")
        _validate_integer(self.candidates_attempted, "candidates_attempted", minimum=0)
        if self.recovered_master_key is not None:
            _validate_master_key(self.recovered_master_key)
        _validate_seconds(self.ranking_seconds, "ranking_seconds")
        _validate_seconds(self.solver_seconds, "solver_seconds")
        _validate_seconds(self.verification_seconds, "verification_seconds")
        _validate_seconds(self.total_seconds, "total_seconds")
        if not isinstance(self.attempted_paths, tuple):
            raise TypeError("attempted_paths must be a tuple of path tuples")
        if self.failure_cause not in (
            None,
            FAILURE_CAUSE_RANKING,
            FAILURE_CAUSE_SAT_BACKEND,
        ):
            raise ValueError(
                "failure_cause must be 'ranking', 'sat_backend', or None"
            )
        if self.success != (self.failure_cause is None):
            raise ValueError("failure_cause is None exactly when success is True")
        if self.success != (self.recovered_master_key is not None):
            raise ValueError(
                "recovered_master_key is set exactly when success is True"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the attack outcome, its stage records, and its timings."""

        return {
            "strategy": self.strategy,
            "success": self.success,
            "status": self.status,
            "failure_cause": self.failure_cause,
            "stages": [stage.to_dict() for stage in self.stages],
            "candidates_attempted": self.candidates_attempted,
            "attempted_paths": [list(path) for path in self.attempted_paths],
            "recovered_master_key": (
                list(self.recovered_master_key)
                if self.recovered_master_key is not None
                else None
            ),
            "timing_seconds": {
                "ranking": self.ranking_seconds,
                "solver": self.solver_seconds,
                "verification": self.verification_seconds,
                "total": self.total_seconds,
            },
        }


def _validate_attack_inputs(
    instance: AttackInstance,
    rounds: int,
    stages: Sequence[HierarchicalStage],
    solver_timeout: float | None,
    candidate_batch_size: int,
    device: torch.device | str | None,
) -> tuple[HierarchicalStage, ...]:
    """Validate every argument and return the materialised stage tuple."""

    if not isinstance(instance, AttackInstance):
        raise TypeError("instance must be an AttackInstance")
    if len(instance.pairs) != 64:
        raise ValueError(
            "hierarchical multi-round ranking requires an instance with 64 pairs"
        )
    _validate_integer(rounds, "rounds", minimum=2)
    if isinstance(stages, (str, bytes)) or not isinstance(stages, Sequence):
        raise TypeError("stages must be a sequence of HierarchicalStage values")
    materialised = tuple(stages)
    if not materialised:
        raise ValueError("stages must contain at least one ranking stage")
    for stage in materialised:
        if not isinstance(stage, HierarchicalStage):
            raise TypeError("stages must contain HierarchicalStage values")
    if len(materialised) >= rounds:
        raise ValueError(
            "the exact backend needs at least one unassumed round; "
            "len(stages) must be smaller than rounds"
        )
    if materialised[0].round_index != rounds - 1:
        raise ValueError("the first stage must target the final round key")
    for previous, current in zip(materialised, materialised[1:]):
        if current.round_index != previous.round_index - 1:
            raise ValueError(
                "stage round indices must descend by exactly one round"
            )
    _validate_timeout(solver_timeout)
    batch_size = _validate_integer(
        candidate_batch_size,
        "candidate_batch_size",
        minimum=1,
    )
    if batch_size > MAX_CANDIDATE_BATCH_SIZE:
        raise ValueError(
            "candidate_batch_size must not exceed "
            f"{MAX_CANDIDATE_BATCH_SIZE} (AGENTS.md section 16.1)"
        )
    _validate_device(device)
    return materialised


def _true_round_schedule(
    instance: AttackInstance,
    rounds: int,
) -> tuple[int, ...] | None:
    """Expand the instance master key for post-search diagnostics only.

    This is the only place in the module that touches ground truth. It must be
    called after the search has finished, and its return value must never reach
    any ranking, pruning, or ordering decision. ``None`` means the caller did
    not supply usable ground truth, in which case the diagnostic truth ranks are
    reported as unavailable.
    """

    try:
        return tuple(expand_key(instance.master_key, rounds=rounds))
    except (TypeError, ValueError):
        return None


def _diagnose_stages(
    *,
    instance: AttackInstance,
    rounds: int,
    stages: tuple[HierarchicalStage, ...],
    retained_orders: list[dict[tuple[int, ...], tuple[int, ...]]],
    candidates_ranked: list[int],
    top_candidates: list[int],
    stage_ranking_seconds: list[float],
    ordered_paths: list[tuple[tuple[int, ...], int]],
    success: bool,
) -> tuple[tuple[StageDiagnostic, ...], str | None]:
    """Build the stage diagnostics and classify a failure, after the search.

    Every quantity computed here is post hoc: the true round-key ranks come from
    :func:`_true_round_schedule`, and ``failure_cause`` is ``"ranking"`` only
    when ground truth shows that the true round-key tuple was never assembled
    because it fell outside the beams. Any other failure is attributed to the
    exact backend, including a solver timeout that stopped the run before the
    true tuple was reached.
    """

    schedule = _true_round_schedule(instance, rounds)
    diagnostics: list[StageDiagnostic] = []
    true_prefix: list[int] = []
    truth_available = schedule is not None

    for stage_index, stage in enumerate(stages):
        true_rank: int | None = None
        if truth_available and 0 <= stage.round_index < len(schedule):
            true_prefix.append(schedule[stage.round_index])
            order = retained_orders[stage_index].get(tuple(true_prefix[:-1]))
            if order is not None:
                true_rank = order.index(true_prefix[-1]) + 1
        else:
            truth_available = False
        diagnostics.append(
            StageDiagnostic(
                stage_index=stage_index,
                round_index=stage.round_index,
                candidates_ranked=candidates_ranked[stage_index],
                beam_width=stage.beam_width,
                top_candidate=top_candidates[stage_index],
                ranking_seconds=stage_ranking_seconds[stage_index],
                true_round_key_rank=true_rank,
            )
        )

    if success:
        return tuple(diagnostics), None
    true_path = tuple(true_prefix) if truth_available else None
    assembled = true_path is not None and any(
        prefix == true_path for prefix, _ in ordered_paths
    )
    if true_path is not None and not assembled:
        return tuple(diagnostics), FAILURE_CAUSE_RANKING
    return tuple(diagnostics), FAILURE_CAUSE_SAT_BACKEND


def run_hierarchical_multi_round_attack(
    instance: AttackInstance,
    *,
    rounds: int,
    stages: Sequence[HierarchicalStage],
    known_pairs: Sequence[KnownPair],
    verification_pairs: Sequence[KnownPair],
    solver_timeout: float | None = None,
    candidate_batch_size: int = 1024,
    device: torch.device | str | None = None,
    stage_cache: dict[tuple[int, ...], Any] | None = None,
    excluded_paths: set[tuple[int, ...]] | None = None,
) -> HierarchicalAttackResult:
    """Attack ``rounds`` rounds by ranking its trailing round keys in a beam.

    Stage zero ranks the complete candidate domain for the final round key on
    ``instance``. Each surviving candidate descends one round through
    :func:`~neural_sat_speck.attacks.multi_round_instance.partial_decrypt_instance`,
    and the next stage ranks the round key below it on the descended instance.
    Only the final stage's survivors become ``{round_index: value}`` tuples,
    which are ordered deterministically by the sum of their stage ranks (ties by
    tuple value) and tested in that order through
    :func:`~neural_sat_speck.attacks.multi_round_recovery.run_multi_round_key_recovery`,
    stopping at the first independently verified recovery.

    Ranking is delegated to
    :func:`~neural_sat_speck.attacks.group_ranking.rank_pair_group_subkeys`,
    which performs the one-round partial decryption of the stage's own instance
    internally; the descent applied between stages is therefore applied exactly
    once per assumed round key. See the module docstring for the leakage rule.
    """

    stage_tuple = _validate_attack_inputs(
        instance,
        rounds,
        stages,
        solver_timeout,
        candidate_batch_size,
        device,
    )
    known = _validate_materialised_pairs(known_pairs, "known_pairs")
    verification = _validate_materialised_pairs(
        verification_pairs,
        "verification_pairs",
    )
    timeout = _validate_timeout(solver_timeout)
    started = time.perf_counter()
    ranking_seconds = 0.0

    # Search state: (assumed values in stage order, cumulative rank sum, the
    # instance this stage ranks on). No entry carries key material.
    frontier: list[tuple[tuple[int, ...], int, AttackInstance]] = [
        ((), 0, instance)
    ]
    retained_orders: list[dict[tuple[int, ...], tuple[int, ...]]] = []
    stage_candidates_ranked: list[int] = []
    stage_top_candidates: list[int] = []
    stage_ranking_seconds: list[float] = []
    ordered_paths: list[tuple[tuple[int, ...], int]] = []

    for stage_index, stage in enumerate(stage_tuple):
        ranking_seconds_before_stage = ranking_seconds
        orders: dict[tuple[int, ...], tuple[int, ...]] = {}
        survivors: list[tuple[tuple[int, ...], int, int, AttackInstance]] = []
        candidates_ranked = 0

        for prefix, rank_sum, current in frontier:
            if stage_cache is not None and prefix in stage_cache:
                ranking = stage_cache[prefix]
            elif stage.screened:
                ranking = rank_pair_group_subkeys_screened(
                    current,
                    stage.model,
                    screen_pairs=stage.screen_pairs,
                    screen_top_k=stage.screen_top_k,
                    candidate_batch_size=candidate_batch_size,
                    device=device,
                )
                if stage_cache is not None:
                    stage_cache[prefix] = ranking
            else:
                ranking = rank_pair_group_subkeys(
                    current,
                    stage.model,
                    candidate_batch_size=candidate_batch_size,
                    device=device,
                )
                if stage_cache is not None:
                    stage_cache[prefix] = ranking
            ranking_seconds += float(ranking.total_seconds)
            candidates_ranked += len(ranking.candidate_values)
            orders[prefix] = ranking.candidate_order
            for rank, value in enumerate(
                ranking.candidate_order[: stage.beam_width],
                start=1,
            ):
                survivors.append((prefix + (value,), rank_sum + rank, value, current))

        stage_candidates_ranked.append(candidates_ranked)
        stage_top_candidates.append(
            min(survivors, key=lambda entry: (entry[1], entry[0]))[2]
        )
        retained_orders.append(orders)

        if stage_index + 1 < len(stage_tuple):
            # Descend one round for the next stage. The descended instance's
            # stored ``final_round_subkey`` is a placeholder: the true next
            # round key is ground truth and must not enter the search. Ranking
            # never reads that field, and no diagnostic in this module reads it.
            descent_started = time.perf_counter()
            frontier = [
                (
                    prefix,
                    rank_sum,
                    partial_decrypt_instance(
                        current,
                        value,
                        expected_next_subkey=DESCENT_SUBKEY_PLACEHOLDER,
                        label=f"{current.instance_id}-r{stage.round_index - 1}",
                    ),
                )
                for prefix, rank_sum, value, current in survivors
            ]
            ranking_seconds += time.perf_counter() - descent_started
        else:
            ordered_paths = sorted(
                ((prefix, rank_sum) for prefix, rank_sum, _, _ in survivors),
                key=lambda entry: (entry[1], entry[0]),
            )
        stage_ranking_seconds.append(ranking_seconds - ranking_seconds_before_stage)

    # Exact verification: one fresh isolated model, candidates in tuple order.
    solver_seconds = 0.0
    verification_seconds = 0.0
    candidates_attempted = 0
    recovered: MasterKey | None = None
    attempted_tuples: tuple[tuple[int, ...], ...] = ()
    if ordered_paths:
        effective_paths = (
            [entry for entry in ordered_paths if entry[0] not in excluded_paths]
            if excluded_paths
            else ordered_paths
        )
        if effective_paths:
            round_indices = tuple(stage.round_index for stage in stage_tuple)
            candidate_round_keys = [
                dict(zip(round_indices, prefix, strict=True))
                for prefix, _ in effective_paths
            ]
            factory = IsolatedSolverFactory.from_pairs(
                known,
                rounds=rounds,
                instance_id=instance.instance_id,
            )
            model = factory.build_isolated_model(HIERARCHICAL_SOLVER_NAME)
            recovery = run_multi_round_key_recovery(
                model,
                candidate_round_keys,
                known,
                verification,
                rounds=rounds,
                solver_timeout=timeout,
                strategy=HIERARCHICAL_ATTACK_STRATEGY,
            )
            success = recovery.success
            status = recovery.status
            candidates_attempted = recovery.candidates_attempted
            recovered = recovery.recovered_master_key
            solver_seconds = float(recovery.solver_seconds)
            verification_seconds = float(recovery.verification_seconds)
            attempted_tuples = tuple(
                prefix for prefix, _ in effective_paths[:candidates_attempted]
            )
        else:
            success = False
            status = STATUS_NO_RANKED_CANDIDATES
    else:
        success = False
        status = STATUS_NO_RANKED_CANDIDATES

    # Post-search diagnostics: the only ground-truth read in this module.
    stage_diagnostics, failure_cause = _diagnose_stages(
        instance=instance,
        rounds=rounds,
        stages=stage_tuple,
        retained_orders=retained_orders,
        candidates_ranked=stage_candidates_ranked,
        top_candidates=stage_top_candidates,
        stage_ranking_seconds=stage_ranking_seconds,
        ordered_paths=ordered_paths,
        success=success,
    )

    return HierarchicalAttackResult(
        strategy=HIERARCHICAL_ATTACK_STRATEGY,
        success=success,
        status=status,
        stages=stage_diagnostics,
        candidates_attempted=candidates_attempted,
        recovered_master_key=recovered,
        ranking_seconds=ranking_seconds,
        solver_seconds=solver_seconds,
        verification_seconds=verification_seconds,
        total_seconds=time.perf_counter() - started,
        failure_cause=failure_cause,
        attempted_paths=attempted_tuples,
    )


__all__ = [
    "DESCENT_SUBKEY_PLACEHOLDER",
    "FAILURE_CAUSE_RANKING",
    "FAILURE_CAUSE_SAT_BACKEND",
    "HIERARCHICAL_ATTACK_STRATEGY",
    "HIERARCHICAL_SOLVER_NAME",
    "MAX_CANDIDATE_BATCH_SIZE",
    "STATUS_NO_RANKED_CANDIDATES",
    "HierarchicalAttackResult",
    "HierarchicalStage",
    "StageDiagnostic",
    "run_hierarchical_multi_round_attack",
]
