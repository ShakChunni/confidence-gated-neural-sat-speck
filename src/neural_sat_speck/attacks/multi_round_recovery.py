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


def _validate_timeout(time_limit: float | None) -> float | None:
    if time_limit is not None:
        if isinstance(time_limit, bool) or not isinstance(time_limit, (int, float)):
            raise TypeError("solver_timeout must be a number or None")
        if time_limit <= 0:
            raise ValueError("solver_timeout must be positive")
    return time_limit


def run_multi_round_key_recovery(
    model: SpeckSatModel,
    candidate_round_keys: Sequence[Mapping[int, int]],
    known_pairs: Sequence[KnownPair],
    verification_pairs: Sequence[KnownPair],
    *,
    rounds: int,
    solver_timeout: float | None = None,
    strategy: str = "multi_round_key",
) -> MultiRoundRecoveryResult:
    """Test ordered multi-round-key candidates against one exact model.

    Every candidate is a ``{round_index: value}`` mapping that is turned into
    temporary solver assumptions. The first candidate whose SAT solution decodes
    to a master key that passes both the known-pair and the independent
    fresh-pair checks is returned as the verified recovery. A solver timeout
    stops the run and is reported as ``solver_timeout``; it is never treated as
    proof that the candidate is wrong.
    """

    if not isinstance(model, SpeckSatModel):
        raise TypeError("model must be a SpeckSatModel")
    candidates = _materialise_candidates(candidate_round_keys, "candidate_round_keys")
    known = _materialise_pairs(known_pairs, "known_pairs")
    verification = _materialise_pairs(verification_pairs, "verification_pairs")
    if isinstance(rounds, bool) or not isinstance(rounds, int):
        raise TypeError("rounds must be an integer")
    timeout = _validate_timeout(solver_timeout)

    assumed = len(candidates[0])
    for candidate in candidates:
        if len(candidate) != assumed:
            raise ValueError(
                "every candidate must assume the same number of round keys"
            )

    attempts: list[MultiRoundAttempt] = []
    recovered: MasterKey | None = None
    status = "exhausted_candidates"
    success = False
    solver_seconds = 0.0
    verification_seconds = 0.0
    started = perf_counter()

    for position, candidate in enumerate(candidates):
        assumptions = assumptions_for_round_keys(model, candidate)
        solver_started = perf_counter()
        result = model.solve(assumptions, time_limit=timeout)
        attempt_solver_seconds = perf_counter() - solver_started
        solver_seconds += attempt_solver_seconds

        attempt_recovered: MasterKey | None = None
        known_verified: bool | None = None
        independent_verified: bool | None = None
        attempt_verification_seconds = 0.0

        if result.satisfiable and result.solution is not None:
            attempt_recovered = model.master_key_from_solution(result.solution)
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
                status = "verified_recovery"
                success = True
                break
            continue
        if result.status == "solver_timeout" or result.timed_out:
            status = "solver_timeout"
            break
        if result.status == "unsat":
            continue
        status = "unsat_or_timeout"
        break

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
        verification_seconds=verification_seconds,
        total_seconds=perf_counter() - started,
    )
