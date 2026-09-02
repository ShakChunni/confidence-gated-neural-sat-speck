"""Verification-Triggered Cascaded Fallback (VTC-B1-B0).

VTC uses exact cryptographic verification failure as an empirical routing
signal to invoke a complementary secondary ranker (B0) lazily only when
the primary ranker (B1) fails within its top-16 candidate budget.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from neural_sat_speck.attacks.group_ranking import (
    PairGroupSubkeyRanking,
    evaluate_pair_group_subkey_ranking,
    rank_pair_group_subkeys,
)
from neural_sat_speck.attacks.ranking import (
    SubkeyRanking,
    evaluate_subkey_ranking,
    rank_final_round_subkeys,
)
from neural_sat_speck.attacks.recovery import (
    CandidateAttempt,
    KnownPair,
    MasterKey,
    _materialise_pairs,
    _validate_positive,
    _validate_timeout,
    verify_master_key,
)
from neural_sat_speck.data.attack import AttackInstance
from neural_sat_speck.neural.numpy_inference import NumpyInferenceModule
from neural_sat_speck.sat.model import SpeckSatModel
from neural_sat_speck.speck.key_schedule import expand_key

VTC_PRIMARY_BUDGET: int = 16
VTC_FALLBACK_ALLOWANCE: int = 16
VTC_CUMULATIVE_MAX_BUDGET: int = 32


@dataclass(frozen=True)
class VtcAttackResult:
    """Result of a Verification-Triggered Cascaded Fallback attack run."""

    strategy: str
    status: str
    success: bool
    b0_invoked: bool
    winning_stage: str | None
    winning_candidate: int | None
    winning_candidate_position: int | None
    recovered_master_key: MasterKey | None
    decoded_master_key: MasterKey | None
    candidates_attempted: int
    b1_candidates_attempted: int
    b0_candidates_attempted: int
    b1_candidates_sequence: tuple[int, ...]
    b0_candidates_sequence: tuple[int, ...]
    cumulative_candidates_sequence: tuple[int, ...]
    attempts: tuple[CandidateAttempt, ...]
    t_rank_b1: float
    t_rank_b0: float
    t_rank_total: float
    t_sat: float
    t_ver: float
    t_routing: float
    t_online: float
    b1_true_rank: int | None
    b0_true_rank: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "status": self.status,
            "success": self.success,
            "b0_invoked": self.b0_invoked,
            "winning_stage": self.winning_stage,
            "winning_candidate": self.winning_candidate,
            "winning_candidate_position": self.winning_candidate_position,
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
            "candidates_attempted": self.candidates_attempted,
            "b1_candidates_attempted": self.b1_candidates_attempted,
            "b0_candidates_attempted": self.b0_candidates_attempted,
            "b1_candidates_sequence": list(self.b1_candidates_sequence),
            "b0_candidates_sequence": list(self.b0_candidates_sequence),
            "cumulative_candidates_sequence": list(self.cumulative_candidates_sequence),
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "timing_seconds": {
                "t_rank_b1": self.t_rank_b1,
                "t_rank_b0": self.t_rank_b0,
                "t_rank_total": self.t_rank_total,
                "t_sat": self.t_sat,
                "t_ver": self.t_ver,
                "t_routing": self.t_routing,
                "t_online": self.t_online,
            },
            "rank_metrics": {
                "b1_true_rank": self.b1_true_rank,
                "b0_true_rank": self.b0_true_rank,
            },
        }


def run_vtc_attack(
    instance: AttackInstance,
    b1_infer: NumpyInferenceModule,
    b0_infer: NumpyInferenceModule,
    known_pairs: Sequence[KnownPair],
    verification_pairs: Sequence[KnownPair],
    *,
    rounds: int = 6,
    k1_budget: int = VTC_PRIMARY_BUDGET,
    b0_allowance: int = VTC_FALLBACK_ALLOWANCE,
    kmax_budget: int = VTC_CUMULATIVE_MAX_BUDGET,
    solver_timeout: float | None = 10.0,
    candidate_batch_size: int = 4096,
    precomputed_b1_ranking: PairGroupSubkeyRanking | None = None,
    precomputed_b0_ranking: SubkeyRanking | None = None,
    strategy: str = "vtc_b1_b0_sat",
) -> VtcAttackResult:
    """Execute Verification-Triggered Cascaded Fallback with independent SAT solving."""
    known = _materialise_pairs(known_pairs, "known_pairs")
    independent = _materialise_pairs(verification_pairs, "verification_pairs")
    timeout = _validate_timeout(solver_timeout)
    k1 = _validate_positive(k1_budget, "k1_budget")
    b0_max = _validate_positive(b0_allowance, "b0_allowance")
    kmax = _validate_positive(kmax_budget, "kmax_budget")

    # Step 1: Construct a brand-new independent SAT solver instance for VTC
    sat_model = SpeckSatModel.from_known_pairs(known, rounds=rounds)

    # Step 2: Compute or acquire the Primary B1 complete-domain ranking
    if precomputed_b1_ranking is not None:
        b1_ranking = precomputed_b1_ranking
        t_rank_b1 = b1_ranking.total_seconds
    else:
        t0_b1 = perf_counter()
        b1_ranking = rank_pair_group_subkeys(
            instance,
            b1_infer,
            candidate_batch_size=candidate_batch_size,
        )
        t_rank_b1 = b1_ranking.total_seconds or (perf_counter() - t0_b1)

    b1_order = tuple(b1_ranking.candidate_order)
    b1_eval = evaluate_pair_group_subkey_ranking(
        b1_ranking,
        instance.final_round_subkey,
    )
    b1_true_rank = b1_eval["true_rank"]

    # Step 3: Test Primary B1 Top-16 Candidates
    attempts: list[CandidateAttempt] = []
    b1_seq: list[int] = []
    b0_seq: list[int] = []
    cumulative_seq: list[int] = []

    solver_seconds = 0.0
    verification_seconds = 0.0
    routing_seconds = 0.0

    recovered_master_key: MasterKey | None = None
    decoded_master_key: MasterKey | None = None
    winning_candidate: int | None = None
    winning_candidate_position: int | None = None
    winning_stage: str | None = None
    success = False
    status = "budget_exhausted"

    b1_prefix = b1_order[:k1]
    for candidate_position, candidate in enumerate(b1_prefix, start=1):
        b1_seq.append(candidate)
        cumulative_seq.append(candidate)

        sub_res = sat_model.solve_final_round_subkey(candidate, time_limit=timeout)
        solver_seconds += sub_res.elapsed_seconds

        if sub_res.status == "unsat_or_timeout":
            attempts.append(
                CandidateAttempt(
                    candidate_position=candidate_position,
                    candidate=candidate,
                    solution_number=1,
                    solver_status="solver_timeout",
                    outcome="timeout",
                    satisfiable=False,
                    timed_out=True,
                    solver_seconds=sub_res.elapsed_seconds,
                    verification_seconds=0.0,
                    recovered_master_key=None,
                    final_round_subkey_matches=None,
                    known_pairs_verified=None,
                    independent_pairs_verified=None,
                )
            )
            status = "solver_timeout"
            break

        if not sub_res.satisfiable:
            attempts.append(
                CandidateAttempt(
                    candidate_position=candidate_position,
                    candidate=candidate,
                    solution_number=1,
                    solver_status="unsat",
                    outcome="unsat",
                    satisfiable=False,
                    timed_out=False,
                    solver_seconds=sub_res.elapsed_seconds,
                    verification_seconds=0.0,
                    recovered_master_key=None,
                    final_round_subkey_matches=None,
                    known_pairs_verified=None,
                    independent_pairs_verified=None,
                )
            )
            continue

        if sub_res.solution is None:
            raise RuntimeError("CryptoMiniSat returned SAT without a model")

        rec_key = sat_model.master_key_from_solution(sub_res.solution)
        decoded_master_key = rec_key
        final_subkey = expand_key(rec_key, rounds)[-1]
        subkey_matches = final_subkey == candidate

        t_ver_start = perf_counter()
        verif = verify_master_key(rec_key, known, independent, rounds)
        t_ver_elapsed = perf_counter() - t_ver_start
        verification_seconds += t_ver_elapsed

        attempts.append(
            CandidateAttempt(
                candidate_position=candidate_position,
                candidate=candidate,
                solution_number=1,
                solver_status="sat",
                outcome="verified" if verif.verified else "unverified",
                satisfiable=True,
                timed_out=False,
                solver_seconds=sub_res.elapsed_seconds,
                verification_seconds=t_ver_elapsed,
                recovered_master_key=rec_key if verif.verified else None,
                final_round_subkey_matches=subkey_matches,
                known_pairs_verified=verif.known_pairs_verified,
                independent_pairs_verified=verif.independent_pairs_verified,
            )
        )

        if verif.verified:
            success = True
            status = "recovered"
            winning_candidate = candidate
            winning_candidate_position = candidate_position
            winning_stage = "primary_b1"
            recovered_master_key = rec_key
            break

    # Step 4: Early Termination Check
    if success or status == "solver_timeout":
        b0_invoked = False
        t_rank_b0 = 0.0
        t_rank_total = t_rank_b1
        t_online = t_rank_total + solver_seconds + verification_seconds + routing_seconds

        return VtcAttackResult(
            strategy=strategy,
            status=status,
            success=success,
            b0_invoked=b0_invoked,
            winning_stage=winning_stage,
            winning_candidate=winning_candidate,
            winning_candidate_position=winning_candidate_position,
            recovered_master_key=recovered_master_key,
            decoded_master_key=decoded_master_key,
            candidates_attempted=len(attempts),
            b1_candidates_attempted=len(b1_seq),
            b0_candidates_attempted=0,
            b1_candidates_sequence=tuple(b1_seq),
            b0_candidates_sequence=(),
            cumulative_candidates_sequence=tuple(cumulative_seq),
            attempts=tuple(attempts),
            t_rank_b1=t_rank_b1,
            t_rank_b0=t_rank_b0,
            t_rank_total=t_rank_total,
            t_sat=solver_seconds,
            t_ver=verification_seconds,
            t_routing=routing_seconds,
            t_online=t_online,
            b1_true_rank=b1_true_rank,
            b0_true_rank=None,
        )

    # Step 5: Verification-Triggered Escalation to Lazy B0
    t_route_start = perf_counter()
    b0_invoked = True

    if precomputed_b0_ranking is not None:
        b0_ranking = precomputed_b0_ranking
        t_rank_b0 = b0_ranking.total_seconds
    else:
        t0_b0 = perf_counter()
        b0_ranking = rank_final_round_subkeys(
            instance,
            b0_infer,
            candidate_batch_size=candidate_batch_size,
        )
        t_rank_b0 = b0_ranking.total_seconds or (perf_counter() - t0_b0)

    b0_order = tuple(b0_ranking.candidate_order)
    b0_eval = evaluate_subkey_ranking(b0_ranking, instance.final_round_subkey)
    b0_true_rank = b0_eval["true_rank"]

    # Filter B0 candidates: exclude all candidates already tested in B1 top-16
    seen_in_b1 = set(b1_prefix)
    b0_unique_candidates = [c for c in b0_order if c not in seen_in_b1][:b0_max]
    routing_seconds += perf_counter() - t_route_start

    # Step 6: Test B0 Fallback Candidates (up to b0_max new unique, cumulative <= kmax)
    for b0_idx, candidate in enumerate(b0_unique_candidates, start=1):
        if len(cumulative_seq) >= kmax:
            break

        candidate_position = len(attempts) + 1
        b0_seq.append(candidate)
        cumulative_seq.append(candidate)

        sub_res = sat_model.solve_final_round_subkey(candidate, time_limit=timeout)
        solver_seconds += sub_res.elapsed_seconds

        if sub_res.status == "unsat_or_timeout":
            attempts.append(
                CandidateAttempt(
                    candidate_position=candidate_position,
                    candidate=candidate,
                    solution_number=1,
                    solver_status="solver_timeout",
                    outcome="timeout",
                    satisfiable=False,
                    timed_out=True,
                    solver_seconds=sub_res.elapsed_seconds,
                    verification_seconds=0.0,
                    recovered_master_key=None,
                    final_round_subkey_matches=None,
                    known_pairs_verified=None,
                    independent_pairs_verified=None,
                )
            )
            status = "solver_timeout"
            break

        if not sub_res.satisfiable:
            attempts.append(
                CandidateAttempt(
                    candidate_position=candidate_position,
                    candidate=candidate,
                    solution_number=1,
                    solver_status="unsat",
                    outcome="unsat",
                    satisfiable=False,
                    timed_out=False,
                    solver_seconds=sub_res.elapsed_seconds,
                    verification_seconds=0.0,
                    recovered_master_key=None,
                    final_round_subkey_matches=None,
                    known_pairs_verified=None,
                    independent_pairs_verified=None,
                )
            )
            continue

        if sub_res.solution is None:
            raise RuntimeError("CryptoMiniSat returned SAT without a model")

        rec_key = sat_model.master_key_from_solution(sub_res.solution)
        decoded_master_key = rec_key
        final_subkey = expand_key(rec_key, rounds)[-1]
        subkey_matches = final_subkey == candidate

        t_ver_start = perf_counter()
        verif = verify_master_key(rec_key, known, independent, rounds)
        t_ver_elapsed = perf_counter() - t_ver_start
        verification_seconds += t_ver_elapsed

        attempts.append(
            CandidateAttempt(
                candidate_position=candidate_position,
                candidate=candidate,
                solution_number=1,
                solver_status="sat",
                outcome="verified" if verif.verified else "unverified",
                satisfiable=True,
                timed_out=False,
                solver_seconds=sub_res.elapsed_seconds,
                verification_seconds=t_ver_elapsed,
                recovered_master_key=rec_key if verif.verified else None,
                final_round_subkey_matches=subkey_matches,
                known_pairs_verified=verif.known_pairs_verified,
                independent_pairs_verified=verif.independent_pairs_verified,
            )
        )

        if verif.verified:
            success = True
            status = "recovered"
            winning_candidate = candidate
            winning_candidate_position = candidate_position
            winning_stage = "fallback_b0"
            recovered_master_key = rec_key
            break

    t_rank_total = t_rank_b1 + t_rank_b0
    t_online = t_rank_total + solver_seconds + verification_seconds + routing_seconds

    return VtcAttackResult(
        strategy=strategy,
        status=status,
        success=success,
        b0_invoked=b0_invoked,
        winning_stage=winning_stage,
        winning_candidate=winning_candidate,
        winning_candidate_position=winning_candidate_position,
        recovered_master_key=recovered_master_key,
        decoded_master_key=decoded_master_key,
        candidates_attempted=len(attempts),
        b1_candidates_attempted=len(b1_seq),
        b0_candidates_attempted=len(b0_seq),
        b1_candidates_sequence=tuple(b1_seq),
        b0_candidates_sequence=tuple(b0_seq),
        cumulative_candidates_sequence=tuple(cumulative_seq),
        attempts=tuple(attempts),
        t_rank_b1=t_rank_b1,
        t_rank_b0=t_rank_b0,
        t_rank_total=t_rank_total,
        t_sat=solver_seconds,
        t_ver=verification_seconds,
        t_routing=routing_seconds,
        t_online=t_online,
        b1_true_rank=b1_true_rank,
        b0_true_rank=b0_true_rank,
    )


__all__ = [
    "VTC_CUMULATIVE_MAX_BUDGET",
    "VTC_FALLBACK_ALLOWANCE",
    "VTC_PRIMARY_BUDGET",
    "VtcAttackResult",
    "run_vtc_attack",
]
