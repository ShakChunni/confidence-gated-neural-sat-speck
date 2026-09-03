"""Exact Phase 5 candidate testing and independently verified recovery.

The recovery boundary deliberately contains no neural decision logic. A caller
provides an ordered sequence of complete 16-bit final-round subkey candidates;
this module tests that order against one exact :class:`SpeckSatModel` and
records enough information to compare methods fairly.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any, TypeAlias

from neural_sat_speck.data.attack import AttackInstance
from neural_sat_speck.sat.model import KnownPair, SpeckSatModel
from neural_sat_speck.speck.cipher import Block, encrypt_block
from neural_sat_speck.speck.key_schedule import expand_key
from neural_sat_speck.speck.primitives import WORD_MASK

MasterKey: TypeAlias = tuple[int, int, int, int]


def _validate_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _validate_word(value: object, name: str) -> int:
    word = _validate_int(value, name)
    if not 0 <= word <= WORD_MASK:
        raise ValueError(f"{name} must be a 16-bit word")
    return word


def _normalise_master_key(master_key: Sequence[int]) -> MasterKey:
    try:
        words = tuple(master_key)
    except TypeError as error:
        raise TypeError("master_key must be a sequence of four words") from error
    if len(words) != 4:
        raise ValueError("master_key must contain four words")
    return tuple(
        _validate_word(word, f"master_key[{index}]")
        for index, word in enumerate(words)
    )  # type: ignore[return-value]


def _validate_positive(value: object, name: str) -> int:
    integer = _validate_int(value, name)
    if integer <= 0:
        raise ValueError(f"{name} must be positive")
    return integer


def _materialise_pairs(
    pairs: Sequence[KnownPair],
    name: str,
) -> tuple[KnownPair, ...]:
    try:
        materialised = tuple(pairs)
    except TypeError as error:
        raise TypeError(f"{name} must be a sequence of plaintext/ciphertext pairs") from error
    if not materialised:
        raise ValueError(f"{name} must contain at least one pair")
    return materialised


def known_pairs_for_attack_instance(
    instance: AttackInstance,
    *,
    pair_count: int | None = None,
) -> tuple[KnownPair, ...]:
    """Expand related attack pairs into the known pairs consumed by SAT.

    One attack pair contains two plaintext/ciphertext observations. The exact
    SAT model constrains both observations independently, while the relation
    between the two plaintexts remains available to the neural ranking stage.
    ``pair_count`` is an explicit pilot control; omitting it uses every pair.
    """

    if not isinstance(instance, AttackInstance):
        raise TypeError("instance must be an AttackInstance")
    if pair_count is None:
        selected_pairs = instance.pairs
    else:
        count = _validate_positive(pair_count, "pair_count")
        if count > len(instance.pairs):
            raise ValueError("pair_count exceeds the attack instance pair count")
        selected_pairs = instance.pairs[:count]
    if not selected_pairs:
        raise ValueError("attack instance must contain at least one pair")
    return tuple(
        known_pair
        for pair in selected_pairs
        for known_pair in (
            (pair.plaintext0, pair.ciphertext0),
            (pair.plaintext1, pair.ciphertext1),
        )
    )


def _random_block(rng: random.Random) -> Block:
    return rng.randrange(1 << 16), rng.randrange(1 << 16)


def independent_verification_pairs(
    master_key: Sequence[int],
    rounds: int,
    pair_count: int,
    seed: int,
) -> tuple[KnownPair, ...]:
    """Generate fresh plaintext/ciphertext pairs for post-SAT verification.

    The key is used only as the oracle for constructing the ciphertexts. The
    generated plaintexts are independent of the candidate order and the SAT
    assumptions. A dedicated seed must be recorded by the caller.
    """

    key = _normalise_master_key(master_key)
    round_count = _validate_positive(rounds, "rounds")
    count = _validate_positive(pair_count, "pair_count")
    _validate_int(seed, "seed")
    # expand_key performs the upper-bound check and keeps this validation in
    # the same source of truth as the cipher implementation.
    expand_key(key, round_count)
    rng = random.Random(seed)
    return tuple(
        (plaintext, encrypt_block(plaintext, key, round_count))
        for plaintext in (_random_block(rng) for _ in range(count))
    )


@dataclass(frozen=True)
class VerificationResult:
    """Separate known-pair and fresh-pair checks for one decoded key."""

    known_pairs_verified: bool
    independent_pairs_verified: bool

    @property
    def verified(self) -> bool:
        """Return true only when both checks pass."""

        return self.known_pairs_verified and self.independent_pairs_verified

    def to_dict(self) -> dict[str, bool]:
        return {
            "known_pairs_verified": self.known_pairs_verified,
            "independent_pairs_verified": self.independent_pairs_verified,
            "verified": self.verified,
        }


def verify_master_key(
    master_key: Sequence[int],
    known_pairs: Sequence[KnownPair],
    verification_pairs: Sequence[KnownPair],
    rounds: int,
) -> VerificationResult:
    """Verify a candidate key against supplied and fresh observations."""

    key = _normalise_master_key(master_key)
    round_count = _validate_positive(rounds, "rounds")
    known = _materialise_pairs(known_pairs, "known_pairs")
    independent = _materialise_pairs(verification_pairs, "verification_pairs")
    expand_key(key, round_count)

    def matches(pairs: Sequence[KnownPair]) -> bool:
        return all(
            encrypt_block(plaintext, key, round_count) == ciphertext
            for plaintext, ciphertext in pairs
        )

    return VerificationResult(
        known_pairs_verified=matches(known),
        independent_pairs_verified=matches(independent),
    )


def random_candidate_order(
    seed: int,
    candidate_values: Sequence[int] = range(1 << 16),
) -> tuple[int, ...]:
    """Return a reproducible shuffled order of 16-bit candidate values."""

    _validate_int(seed, "seed")
    values = tuple(candidate_values)
    if not values:
        raise ValueError("candidate_values must not be empty")
    for value in values:
        _validate_word(value, "candidate value")
    if len(set(values)) != len(values):
        raise ValueError("candidate_values must not contain duplicates")
    shuffled = list(values)
    random.Random(seed).shuffle(shuffled)
    return tuple(shuffled)


@dataclass(frozen=True)
class CandidateAttempt:
    """One solver call made while testing a candidate order."""

    candidate_position: int
    candidate: int
    solution_number: int
    solver_status: str
    outcome: str
    satisfiable: bool
    timed_out: bool | None
    solver_seconds: float
    verification_seconds: float
    recovered_master_key: MasterKey | None
    final_round_subkey_matches: bool | None
    known_pairs_verified: bool | None
    independent_pairs_verified: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_position": self.candidate_position,
            "candidate": self.candidate,
            "solution_number": self.solution_number,
            "solver_status": self.solver_status,
            "outcome": self.outcome,
            "satisfiable": self.satisfiable,
            "timed_out": self.timed_out,
            "solver_seconds": self.solver_seconds,
            "verification_seconds": self.verification_seconds,
            "recovered_master_key": (
                list(self.recovered_master_key)
                if self.recovered_master_key is not None
                else None
            ),
            "final_round_subkey_matches": self.final_round_subkey_matches,
            "known_pairs_verified": self.known_pairs_verified,
            "independent_pairs_verified": self.independent_pairs_verified,
        }


@dataclass(frozen=True)
class RecoveryResult:
    """Complete result of one SAT-only or ordered-candidate recovery run."""

    strategy: str
    status: str
    success: bool
    candidate_budget: int | None
    candidates_attempted: int
    solver_calls: int
    attempts: tuple[CandidateAttempt, ...]
    recovered_master_key: MasterKey | None
    decoded_master_key: MasterKey | None
    winning_candidate: int | None
    winning_candidate_position: int | None
    solver_seconds: float
    verification_seconds: float
    total_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "status": self.status,
            "success": self.success,
            "candidate_budget": self.candidate_budget,
            "candidates_attempted": self.candidates_attempted,
            "solver_calls": self.solver_calls,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "recovered_master_key": (
                list(self.recovered_master_key)
                if self.recovered_master_key is not None
                else None
            ),
            "decoded_master_key": (
                list(self.decoded_master_key)
                if self.decoded_master_key is not None
                else None
            ),
            "winning_candidate": self.winning_candidate,
            "winning_candidate_position": self.winning_candidate_position,
            "timing_seconds": {
                "solver": self.solver_seconds,
                "verification": self.verification_seconds,
                "total": self.total_seconds,
            },
        }


def _validate_model(model: SpeckSatModel) -> SpeckSatModel:
    if not isinstance(model, SpeckSatModel):
        raise TypeError("model must be a SpeckSatModel")
    return model


def _validate_timeout(time_limit: float | None) -> float | None:
    if time_limit is not None:
        if isinstance(time_limit, bool) or not isinstance(time_limit, (int, float)):
            raise TypeError("solver_timeout must be a number or None")
        if time_limit <= 0:
            raise ValueError("solver_timeout must be positive")
    return time_limit


def _make_result(
    *,
    strategy: str,
    status: str,
    success: bool,
    candidate_budget: int | None,
    candidates_attempted: int,
    attempts: Sequence[CandidateAttempt],
    recovered_master_key: MasterKey | None,
    decoded_master_key: MasterKey | None = None,
    winning_candidate: int | None,
    winning_candidate_position: int | None,
    solver_seconds: float,
    verification_seconds: float,
    started: float,
    solver_calls: int | None = None,
) -> RecoveryResult:
    return RecoveryResult(
        strategy=strategy,
        status=status,
        success=success,
        candidate_budget=candidate_budget,
        candidates_attempted=candidates_attempted,
        attempts=tuple(attempts),
        recovered_master_key=recovered_master_key,
        winning_candidate=winning_candidate,
        winning_candidate_position=winning_candidate_position,
        solver_seconds=solver_seconds,
        verification_seconds=verification_seconds,
        total_seconds=perf_counter() - started,
        decoded_master_key=decoded_master_key,
        solver_calls=(len(attempts) if solver_calls is None else solver_calls),
    )


def run_sat_only_recovery(
    model: SpeckSatModel,
    known_pairs: Sequence[KnownPair],
    verification_pairs: Sequence[KnownPair],
    *,
    solver_timeout: float | None = None,
    strategy: str = "sat_only",
) -> RecoveryResult:
    """Solve the exact model once and independently verify its decoded key."""

    _validate_model(model)
    if not isinstance(strategy, str) or not strategy:
        raise ValueError("strategy must be a non-empty string")
    known = _materialise_pairs(known_pairs, "known_pairs")
    independent = _materialise_pairs(verification_pairs, "verification_pairs")
    timeout = _validate_timeout(solver_timeout)
    started = perf_counter()
    result = model.solve(time_limit=timeout)
    solver_seconds = result.elapsed_seconds
    if result.status in ("unsat_or_timeout", "solver_timeout"):
        return _make_result(
            strategy=strategy,
            status="solver_timeout",
            success=False,
            candidate_budget=None,
            candidates_attempted=0,
            attempts=(),
            recovered_master_key=None,
            winning_candidate=None,
            winning_candidate_position=None,
            solver_seconds=solver_seconds,
            verification_seconds=0.0,
            started=started,
            solver_calls=1,
        )
    if not result.satisfiable:
        return _make_result(
            strategy=strategy,
            status="unsat",
            success=False,
            candidate_budget=None,
            candidates_attempted=0,
            attempts=(),
            recovered_master_key=None,
            winning_candidate=None,
            winning_candidate_position=None,
            solver_seconds=solver_seconds,
            verification_seconds=0.0,
            started=started,
            solver_calls=1,
        )
    if result.solution is None:
        raise RuntimeError("CryptoMiniSat returned SAT without a solution model")

    recovered_key = model.master_key_from_solution(result.solution)
    final_subkey = expand_key(recovered_key, model.rounds)[-1]
    verification_started = perf_counter()
    verification = verify_master_key(
        recovered_key,
        known,
        independent,
        model.rounds,
    )
    verification_seconds = perf_counter() - verification_started
    success = verification.verified
    return _make_result(
        strategy=strategy,
        status="recovered" if success else "sat_unverified",
        success=success,
        candidate_budget=None,
        candidates_attempted=0,
        attempts=(),
        recovered_master_key=recovered_key if success else None,
        decoded_master_key=recovered_key,
        winning_candidate=final_subkey if success else None,
        winning_candidate_position=None,
        solver_seconds=solver_seconds,
        verification_seconds=verification_seconds,
        started=started,
        solver_calls=1,
    )


def run_candidate_recovery(
    model: SpeckSatModel,
    candidate_order: Sequence[int],
    known_pairs: Sequence[KnownPair],
    verification_pairs: Sequence[KnownPair],
    *,
    candidate_budget: int,
    solver_timeout: float | None = None,
    max_solutions_per_candidate: int = 1,
    strategy: str = "candidate_order",
) -> RecoveryResult:
    """Test an ordered candidate prefix with temporary SAT assumptions.

    A time-limited false solve result is returned as ``solver_timeout`` and
    stops the run. It is never counted as an UNSAT candidate. If a SAT model
    fails independent verification, an optional blocking clause can enumerate
    another master-key model under the same final-subkey assumption.
    """

    _validate_model(model)
    if not isinstance(strategy, str) or not strategy:
        raise ValueError("strategy must be a non-empty string")
    known = _materialise_pairs(known_pairs, "known_pairs")
    independent = _materialise_pairs(verification_pairs, "verification_pairs")
    timeout = _validate_timeout(solver_timeout)
    budget = _validate_positive(candidate_budget, "candidate_budget")
    max_solutions = _validate_positive(
        max_solutions_per_candidate,
        "max_solutions_per_candidate",
    )

    order = tuple(candidate_order)
    if not order:
        raise ValueError("candidate_order must not be empty")
    for candidate in order:
        _validate_word(candidate, "candidate")
    if len(set(order)) != len(order):
        raise ValueError("candidate_order must not contain duplicates")
    if budget > len(order):
        raise ValueError("candidate_budget cannot exceed candidate_order length")

    started = perf_counter()
    attempts: list[CandidateAttempt] = []
    solver_seconds = 0.0
    verification_seconds = 0.0

    for candidate_position, candidate in enumerate(order[:budget], start=1):
        for solution_number in range(1, max_solutions + 1):
            result = model.solve_final_round_subkey(
                candidate,
                time_limit=timeout,
            )
            solver_seconds += result.elapsed_seconds
            if result.status in ("unsat_or_timeout", "solver_timeout"):
                attempts.append(
                    CandidateAttempt(
                        candidate_position=candidate_position,
                        candidate=candidate,
                        solution_number=solution_number,
                        solver_status=result.status,
                        outcome="inconclusive",
                        satisfiable=False,
                        timed_out=result.timed_out,
                        solver_seconds=result.elapsed_seconds,
                        verification_seconds=0.0,
                        recovered_master_key=None,
                        final_round_subkey_matches=None,
                        known_pairs_verified=None,
                        independent_pairs_verified=None,
                    )
                )
                return _make_result(
                    strategy=strategy,
                    status="solver_timeout",
                    success=False,
                    candidate_budget=budget,
                    candidates_attempted=candidate_position,
                    attempts=attempts,
                    recovered_master_key=None,
                    winning_candidate=None,
                    winning_candidate_position=None,
                    solver_seconds=solver_seconds,
                    verification_seconds=verification_seconds,
                    started=started,
                )

            if not result.satisfiable:
                attempts.append(
                    CandidateAttempt(
                        candidate_position=candidate_position,
                        candidate=candidate,
                        solution_number=solution_number,
                        solver_status=result.status,
                        outcome="unsat",
                        satisfiable=False,
                        timed_out=result.timed_out,
                        solver_seconds=result.elapsed_seconds,
                        verification_seconds=0.0,
                        recovered_master_key=None,
                        final_round_subkey_matches=None,
                        known_pairs_verified=None,
                        independent_pairs_verified=None,
                    )
                )
                break

            if result.solution is None:
                raise RuntimeError(
                    "CryptoMiniSat returned SAT without a solution model"
                )
            recovered_key = model.master_key_from_solution(result.solution)
            final_round_subkey_matches = (
                expand_key(recovered_key, model.rounds)[-1] == candidate
            )
            verification_started = perf_counter()
            verification = verify_master_key(
                recovered_key,
                known,
                independent,
                model.rounds,
            )
            elapsed_verification = perf_counter() - verification_started
            verification_seconds += elapsed_verification
            verified = final_round_subkey_matches and verification.verified
            attempts.append(
                CandidateAttempt(
                    candidate_position=candidate_position,
                    candidate=candidate,
                    solution_number=solution_number,
                    solver_status=result.status,
                    outcome="recovered" if verified else "sat_unverified",
                    satisfiable=True,
                    timed_out=result.timed_out,
                    solver_seconds=result.elapsed_seconds,
                    verification_seconds=elapsed_verification,
                    recovered_master_key=recovered_key,
                    final_round_subkey_matches=final_round_subkey_matches,
                    known_pairs_verified=verification.known_pairs_verified,
                    independent_pairs_verified=verification.independent_pairs_verified,
                )
            )
            if verified:
                return _make_result(
                    strategy=strategy,
                    status="recovered",
                    success=True,
                    candidate_budget=budget,
                    candidates_attempted=candidate_position,
                    attempts=attempts,
                    recovered_master_key=recovered_key,
                    winning_candidate=candidate,
                    winning_candidate_position=candidate_position,
                    solver_seconds=solver_seconds,
                    verification_seconds=verification_seconds,
                    started=started,
                )
            if solution_number < max_solutions:
                model.add_blocking_clause(result.solution)

    return _make_result(
        strategy=strategy,
        status="budget_exhausted",
        success=False,
        candidate_budget=budget,
        candidates_attempted=budget,
        attempts=attempts,
        recovered_master_key=None,
        winning_candidate=None,
        winning_candidate_position=None,
        solver_seconds=solver_seconds,
        verification_seconds=verification_seconds,
        started=started,
    )
