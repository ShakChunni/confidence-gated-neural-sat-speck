"""Exact recovery conditioned on a run of trailing round keys.

Phase 11A supplied only the final round key and the exact backend still failed at
seven rounds. The residual-depth benchmark established that the solver completes
an instance while ``rounds - (assumed trailing round keys) <= 5``, so a
seven-round attack needs two consecutive trailing round keys and an eight-round
attack needs three.

Like :mod:`neural_sat_speck.attacks.recovery`, this module contains no neural
decision logic: a caller supplies an ordered sequence of multi-round-key
candidates, and each candidate is tested against one exact
:class:`SpeckSatModel`. Solver timeouts are never collapsed into UNSAT, and a
recovery is only reported when independent fresh-pair verification succeeds.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from neural_sat_speck.attacks.recovery import (
    MasterKey,
    verify_master_key,
)
from neural_sat_speck.sat.model import KnownPair, SpeckSatModel
from neural_sat_speck.sat.multi_round import assumptions_for_round_keys

STATUS_GLOBAL_TIMEOUT = "global_timeout"
STATUS_EXHAUSTED_WITHOUT_UNRESOLVED = "candidates_exhausted_without_unresolved_attempts"
STATUS_EXHAUSTED_CANDIDATES = "candidates_exhausted_without_unresolved_attempts"
STATUS_UNSUCCESSFUL_WITH_UNRESOLVED_TIMEOUTS = "unsuccessful_with_unresolved_timeouts"
STATUS_VERIFIED_RECOVERY = "verified_recovery"
STATUS_SOLVER_TIMEOUT = "solver_timeout"
STATUS_CANDIDATE_TIMEOUT = "candidate_timeout"
STATUS_PROVEN_UNSAT = "proven_unsat"
STATUS_VERIFICATION_FAILURE = "verification_failure"
STATUS_DECODING_FAILURE = "decoding_failure"
STATUS_SOLVER_EXCEPTION = "solver_exception"
STATUS_SOLVER_UNKNOWN = "solver_unknown"
STATUS_UNSAT_OR_TIMEOUT = "unsat_or_timeout"


@dataclass(frozen=True)
class MultiRoundAttempt:
    """One solver call made while testing a multi-round-key candidate."""

    candidate_position: int
    round_keys: tuple[tuple[int, int], ...]
    solver_status: str
    satisfiable: bool
    timed_out: bool | None
    solver_seconds: float
    verification_seconds: float
    recovered_master_key: MasterKey | None
    known_pairs_verified: bool | None
    independent_pairs_verified: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_position": self.candidate_position,
            "round_keys": [list(entry) for entry in self.round_keys],
            "solver_status": self.solver_status,
            "satisfiable": self.satisfiable,
            "timed_out": self.timed_out,
            "solver_seconds": self.solver_seconds,
            "verification_seconds": self.verification_seconds,
            "recovered_master_key": (
                list(self.recovered_master_key)
                if self.recovered_master_key is not None
                else None
            ),
            "known_pairs_verified": self.known_pairs_verified,
            "independent_pairs_verified": self.independent_pairs_verified,
        }


@dataclass(frozen=True)
class MultiRoundRecoveryResult:
    """Complete result of one multi-round-key ordered-candidate recovery run."""

    strategy: str
    status: str
    success: bool
    rounds: int
    assumed_round_keys: int
    residual_rounds: int
    candidates_attempted: int
    solver_calls: int
    attempts: tuple[MultiRoundAttempt, ...]
    recovered_master_key: MasterKey | None
    solver_seconds: float
    verification_seconds: float
    total_seconds: float
    rebuilding_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "status": self.status,
            "success": self.success,
            "rounds": self.rounds,
            "assumed_round_keys": self.assumed_round_keys,
            "residual_rounds": self.residual_rounds,
            "candidates_attempted": self.candidates_attempted,
            "solver_calls": self.solver_calls,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "recovered_master_key": (
                list(self.recovered_master_key)
                if self.recovered_master_key is not None
                else None
            ),
            "timing_seconds": {
                "solver": self.solver_seconds,
                "rebuilding": self.rebuilding_seconds,
                "verification": self.verification_seconds,
                "total": self.total_seconds,
            },
        }


def _materialise_candidates(
    candidate_round_keys: Sequence[Mapping[int, int]],
    name: str,
) -> tuple[Mapping[int, int], ...]:
    try:
        materialised = tuple(candidate_round_keys)
    except TypeError as error:
        raise TypeError(
            f"{name} must be a sequence of round-key mappings"
        ) from error
    if not materialised:
        raise ValueError(f"{name} must contain at least one candidate")
    for candidate in materialised:
        if not isinstance(candidate, Mapping):
            raise TypeError(
                f"{name} entries must be mappings of round index to round key value"
            )
    return materialised


def _materialise_pairs(
    pairs: Sequence[KnownPair],
    name: str,
) -> tuple[KnownPair, ...]:
    try:
        materialised = tuple(pairs)
    except TypeError as error:
        raise TypeError(
            f"{name} must be a sequence of plaintext/ciphertext pairs"
        ) from error
    if not materialised:
        raise ValueError(f"{name} must contain at least one pair")
    return materialised


def _validate_timeout(time_limit: float | None, name: str = "solver_timeout") -> float | None:
    if time_limit is not None:
        if isinstance(time_limit, bool) or not isinstance(time_limit, (int, float)):
            raise TypeError(f"{name} must be a number or None")
        if time_limit < 0:
            raise ValueError(f"{name} must be non-negative")
    return float(time_limit) if time_limit is not None else None


def run_multi_round_key_recovery(
    model: SpeckSatModel,
    candidate_round_keys: Sequence[Mapping[int, int]],
    known_pairs: Sequence[KnownPair],
    verification_pairs: Sequence[KnownPair],
    *,
    rounds: int,
    solver_timeout: float | None = None,
    global_timeout: float | None = None,
    strategy: str = "multi_round_key",
    solver_factory: Any | None = None,
) -> MultiRoundRecoveryResult:
    """Test ordered multi-round-key candidates against one exact model.

    Every candidate is a ``{round_index: value}`` mapping that is turned into
    temporary solver assumptions. The first candidate whose SAT solution decodes
    to a master key that passes both the known-pair and the independent
    fresh-pair checks is returned as the verified recovery.

    A candidate solver timeout leaves the solver in an unresolved state. To prevent
    state contamination, the model is freshly rebuilt via ``solver_factory`` (or a
    new ``SpeckSatModel``), charging rebuild time to solver_seconds, and search
    continues unless the global deadline has expired.
    """

    if not isinstance(model, SpeckSatModel):
        raise TypeError("model must be a SpeckSatModel")
    candidates = _materialise_candidates(candidate_round_keys, "candidate_round_keys")
    known = _materialise_pairs(known_pairs, "known_pairs")
    verification = _materialise_pairs(verification_pairs, "verification_pairs")
    if isinstance(rounds, bool) or not isinstance(rounds, int):
        raise TypeError("rounds must be an integer")
    timeout = _validate_timeout(solver_timeout, "solver_timeout")
    g_timeout = _validate_timeout(global_timeout, "global_timeout")

    assumed = len(candidates[0])
    for candidate in candidates:
        if len(candidate) != assumed:
            raise ValueError(
                "every candidate must assume the same number of round keys"
            )

    started = perf_counter()
    if g_timeout is not None and g_timeout <= 0.0:
        return MultiRoundRecoveryResult(
            strategy=strategy,
            status=STATUS_GLOBAL_TIMEOUT,
            success=False,
            rounds=rounds,
            assumed_round_keys=assumed,
            residual_rounds=rounds - assumed,
            candidates_attempted=0,
            solver_calls=0,
            attempts=(),
            recovered_master_key=None,
            solver_seconds=0.0,
            verification_seconds=0.0,
            total_seconds=0.0,
            rebuilding_seconds=0.0,
        )

    attempts: list[MultiRoundAttempt] = []
    recovered: MasterKey | None = None
    status = STATUS_EXHAUSTED_WITHOUT_UNRESOLVED
    success = False
    solver_seconds = 0.0
    rebuilding_seconds = 0.0
    verification_seconds = 0.0
    global_deadline = (started + g_timeout) if g_timeout is not None else None

    current_model = model
    any_solver_timed_out = False

    for position, candidate in enumerate(candidates):
        now = perf_counter()
        if global_deadline is not None and now >= global_deadline:
            status = STATUS_GLOBAL_TIMEOUT
            break

        eff_timeout = timeout
        if global_deadline is not None:
            remaining = global_deadline - now
            if remaining <= 0.0:
                status = STATUS_GLOBAL_TIMEOUT
                break
            eff_timeout = min(timeout, remaining) if timeout is not None else remaining

        assumptions = assumptions_for_round_keys(current_model, candidate)
        solver_started = perf_counter()
        result = current_model.solve(assumptions, time_limit=eff_timeout)
        attempt_solver_seconds = perf_counter() - solver_started
        solver_seconds += attempt_solver_seconds

        attempt_recovered: MasterKey | None = None
        known_verified: bool | None = None
        independent_verified: bool | None = None
        attempt_verification_seconds = 0.0

        if result.satisfiable and result.solution is not None:
            attempt_recovered = current_model.master_key_from_solution(result.solution)
            verification_started = perf_counter()
            verification_result = verify_master_key(
                attempt_recovered,
                known,
                verification,
                rounds,
            )
            attempt_verification_seconds = perf_counter() - verification_started
            verification_seconds += attempt_verification_seconds
            known_verified = verification_result.known_pairs_verified
            independent_verified = verification_result.independent_pairs_verified

        attempts.append(
            MultiRoundAttempt(
                candidate_position=position,
                round_keys=tuple(sorted(candidate.items())),
                solver_status=result.status,
                satisfiable=result.satisfiable,
                timed_out=result.timed_out,
                solver_seconds=attempt_solver_seconds,
                verification_seconds=attempt_verification_seconds,
                recovered_master_key=attempt_recovered,
                known_pairs_verified=known_verified,
                independent_pairs_verified=independent_verified,
            )
        )

        if result.satisfiable and result.solution is not None:
            if verification_result.verified:
                recovered = attempt_recovered
                status = STATUS_VERIFIED_RECOVERY
                success = True
                break
            continue
        if result.status in ("unsat_or_timeout", "solver_timeout") or result.timed_out:
            any_solver_timed_out = True
            # Purge potentially corrupted solver state by rebuilding a fresh isolated model
            rebuild_start = perf_counter()
            if solver_factory is not None and hasattr(solver_factory, "rebuild_isolated_model"):
                current_model = solver_factory.rebuild_isolated_model(strategy)
            else:
                current_model = SpeckSatModel.from_known_pairs(known, rounds=rounds)
            rebuild_seconds = perf_counter() - rebuild_start
            rebuilding_seconds += rebuild_seconds
            solver_seconds += rebuild_seconds

            if global_deadline is not None and perf_counter() >= global_deadline:
                status = STATUS_GLOBAL_TIMEOUT
                break
            continue
        if result.status == "unsat":
            continue
        status = STATUS_UNSAT_OR_TIMEOUT
        break

    if not success and status not in (STATUS_GLOBAL_TIMEOUT, STATUS_UNSAT_OR_TIMEOUT):
        if any_solver_timed_out:
            status = STATUS_UNSUCCESSFUL_WITH_UNRESOLVED_TIMEOUTS
        else:
            status = STATUS_EXHAUSTED_WITHOUT_UNRESOLVED

    return MultiRoundRecoveryResult(
        strategy=strategy,
        status=status,
        success=success,
        rounds=rounds,
        assumed_round_keys=assumed,
        residual_rounds=rounds - assumed,
        candidates_attempted=len(attempts),
        solver_calls=len(attempts),
        attempts=tuple(attempts),
        recovered_master_key=recovered,
        solver_seconds=solver_seconds,
        rebuilding_seconds=rebuilding_seconds,
        verification_seconds=verification_seconds,
        total_seconds=perf_counter() - started,
    )


def random_multi_round_candidates(
    count: int,
    round_indices: Sequence[int],
    *,
    seed: int,
) -> tuple[dict[int, int], ...]:
    """Generate reproducible distinct random round-key candidate tuples.

    Samples uniformly from the Cartesian product of 16-bit words across the
    specified round indices.
    """
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("count must be an integer")
    if count <= 0:
        raise ValueError("count must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if isinstance(round_indices, (str, bytes)) or not isinstance(round_indices, Sequence):
        raise TypeError("round_indices must be a sequence of round indices")
    indices = tuple(round_indices)
    if not indices:
        raise ValueError("round_indices must not be empty")
    for r in indices:
        if isinstance(r, bool) or not isinstance(r, int) or r < 0:
            raise ValueError("round_indices entries must be non-negative integers")
    if len(set(indices)) != len(indices):
        raise ValueError("round_indices must not contain duplicate rounds")
    max_possible = (1 << 16) ** len(indices)
    if count > max_possible:
        raise ValueError(f"count {count} exceeds total domain {max_possible}")

    rng = random.Random(seed)
    seen: set[tuple[int, ...]] = set()
    candidates: list[dict[int, int]] = []
    while len(candidates) < count:
        cand_tuple = tuple(rng.getrandbits(16) for _ in indices)
        if cand_tuple in seen:
            continue
        seen.add(cand_tuple)
        candidates.append(dict(zip(indices, cand_tuple, strict=True)))
    return tuple(candidates)


def run_random_multi_round_recovery(
    model: SpeckSatModel,
    known_pairs: Sequence[KnownPair],
    verification_pairs: Sequence[KnownPair],
    *,
    rounds: int,
    round_indices: Sequence[int] = (6, 5),
    candidate_budget: int = 1024,
    seed: int = 0,
    solver_timeout: float | None = None,
    global_timeout: float | None = None,
    strategy: str = "random_multi_round_sat",
    solver_factory: Any | None = None,
) -> MultiRoundRecoveryResult:
    """Evaluate exact recovery under random multi-round-key candidates."""
    candidates = random_multi_round_candidates(
        count=candidate_budget,
        round_indices=round_indices,
        seed=seed,
    )
    return run_multi_round_key_recovery(
        model=model,
        candidate_round_keys=candidates,
        known_pairs=known_pairs,
        verification_pairs=verification_pairs,
        rounds=rounds,
        solver_timeout=solver_timeout,
        global_timeout=global_timeout,
        strategy=strategy,
        solver_factory=solver_factory,
    )


def run_multi_round_sat_only_recovery(
    model: SpeckSatModel,
    known_pairs: Sequence[KnownPair],
    verification_pairs: Sequence[KnownPair],
    *,
    rounds: int,
    solver_timeout: float | None = None,
    global_timeout: float | None = None,
    strategy: str = "sat_only",
) -> MultiRoundRecoveryResult:
    """Solve the exact model once without round-key assumptions."""
    if not isinstance(model, SpeckSatModel):
        raise TypeError("model must be a SpeckSatModel")
    known = _materialise_pairs(known_pairs, "known_pairs")
    verification = _materialise_pairs(verification_pairs, "verification_pairs")
    if isinstance(rounds, bool) or not isinstance(rounds, int):
        raise TypeError("rounds must be an integer")
    timeout = _validate_timeout(solver_timeout, "solver_timeout")
    g_timeout = _validate_timeout(global_timeout, "global_timeout")

    started = perf_counter()
    if g_timeout is not None and g_timeout <= 0.0:
        return MultiRoundRecoveryResult(
            strategy=strategy,
            status=STATUS_GLOBAL_TIMEOUT,
            success=False,
            rounds=rounds,
            assumed_round_keys=0,
            residual_rounds=rounds,
            candidates_attempted=0,
            solver_calls=0,
            attempts=(),
            recovered_master_key=None,
            solver_seconds=0.0,
            rebuilding_seconds=0.0,
            verification_seconds=0.0,
            total_seconds=0.0,
        )

    eff_timeout = timeout
    if g_timeout is not None:
        eff_timeout = min(timeout, g_timeout) if timeout is not None else g_timeout

    solver_started = perf_counter()
    result = model.solve(time_limit=eff_timeout)
    solver_seconds = perf_counter() - solver_started

    recovered: MasterKey | None = None
    verification_seconds = 0.0
    known_verified: bool | None = None
    independent_verified: bool | None = None
    status = STATUS_PROVEN_UNSAT
    success = False

    if result.status in ("unsat_or_timeout", "solver_timeout") or result.timed_out:
        if g_timeout is not None and (perf_counter() - started) >= g_timeout:
            status = STATUS_GLOBAL_TIMEOUT
        else:
            status = STATUS_SOLVER_TIMEOUT
    elif result.satisfiable and result.solution is not None:
        recovered_candidate = model.master_key_from_solution(result.solution)
        v_started = perf_counter()
        verification_result = verify_master_key(
            recovered_candidate,
            known,
            verification,
            rounds,
        )
        verification_seconds = perf_counter() - v_started
        known_verified = verification_result.known_pairs_verified
        independent_verified = verification_result.independent_pairs_verified
        if verification_result.verified:
            recovered = recovered_candidate
            status = STATUS_VERIFIED_RECOVERY
            success = True
        else:
            status = STATUS_VERIFICATION_FAILURE
    elif not result.satisfiable:
        status = STATUS_PROVEN_UNSAT

    attempt = MultiRoundAttempt(
        candidate_position=0,
        round_keys=(),
        solver_status=result.status,
        satisfiable=result.satisfiable,
        timed_out=result.timed_out,
        solver_seconds=solver_seconds,
        verification_seconds=verification_seconds,
        recovered_master_key=recovered if success else (model.master_key_from_solution(result.solution) if result.satisfiable and result.solution is not None else None),
        known_pairs_verified=known_verified,
        independent_pairs_verified=independent_verified,
    )

    return MultiRoundRecoveryResult(
        strategy=strategy,
        status=status,
        success=success,
        rounds=rounds,
        assumed_round_keys=0,
        residual_rounds=rounds,
        candidates_attempted=0,
        solver_calls=1,
        attempts=(attempt,),
        recovered_master_key=recovered,
        solver_seconds=solver_seconds,
        rebuilding_seconds=0.0,
        verification_seconds=verification_seconds,
        total_seconds=perf_counter() - started,
    )


__all__ = [
    "MultiRoundAttempt",
    "MultiRoundRecoveryResult",
    "STATUS_CANDIDATE_TIMEOUT",
    "STATUS_DECODING_FAILURE",
    "STATUS_EXHAUSTED_CANDIDATES",
    "STATUS_EXHAUSTED_WITHOUT_UNRESOLVED",
    "STATUS_GLOBAL_TIMEOUT",
    "STATUS_PROVEN_UNSAT",
    "STATUS_SOLVER_EXCEPTION",
    "STATUS_SOLVER_TIMEOUT",
    "STATUS_SOLVER_UNKNOWN",
    "STATUS_UNSAT_OR_TIMEOUT",
    "STATUS_UNSUCCESSFUL_WITH_UNRESOLVED_TIMEOUTS",
    "STATUS_VERIFICATION_FAILURE",
    "STATUS_VERIFIED_RECOVERY",
    "random_multi_round_candidates",
    "run_multi_round_key_recovery",
    "run_multi_round_sat_only_recovery",
    "run_random_multi_round_recovery",
]
