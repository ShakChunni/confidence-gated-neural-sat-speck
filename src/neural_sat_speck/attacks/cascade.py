"""Adaptive cascade state machine for reliability-gated complementary rescue (G1 + R1).

This module implements the execution state machine specified in
docs/FALLBACK_CONFIDENCE_REDESIGN.md, coupling neural guidance with exact SAT
solving and master-key verification.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Final

from neural_sat_speck.attacks.r1 import R1RescueRerankerProtocol, R1ShortlistBuilder
from neural_sat_speck.attacks.recovery import (
    CandidateAttempt,
    RecoveryResult,
    independent_verification_pairs,
    run_candidate_recovery,
)
from neural_sat_speck.confidence.stability import G1FeatureVector, ReliabilityEstimator
from neural_sat_speck.data.attack import AttackInstance
from neural_sat_speck.data.seed_policy import SeedRole, validate_seed_for_role
from neural_sat_speck.sat.isolation import IsolatedSolverFactory

DEFAULT_RELIABILITY_THRESHOLD: Final[float] = 0.50
DEFAULT_PRIMARY_BUDGET: Final[int] = 16
DEFAULT_SHORTLIST_SIZE: Final[int] = 64
DEFAULT_SOLVER_TIMEOUT: Final[float] = 10.0


@dataclass
class CascadeExecutionResult:
    """Detailed record of an adaptive cascade recovery attempt."""

    instance_id: str
    success: bool
    status: str
    route_selected: str  # 'b1_direct' or 'r1_rescue'
    reliability_score: float
    escalated_to_b0: bool
    recovered_master_key: tuple[int, int, int, int] | None
    winning_candidate: int | None
    total_candidates_attempted: int
    attempts: tuple[CandidateAttempt, ...] = field(default_factory=tuple)

    # Component timings (seconds)
    t_stage1_b1_rank: float = 0.0
    t_reliability_gate: float = 0.0
    t_r1_rerank: float = 0.0
    t_sat: float = 0.0
    t_verification: float = 0.0
    t_lazy_b0_rank: float = 0.0
    t_total_online: float = 0.0


def run_adaptive_cascade_recovery(
    instance: AttackInstance,
    b1_candidate_order: Sequence[int],
    t_b1_ranking: float,
    g1_features: G1FeatureVector,
    reliability_estimator: ReliabilityEstimator,
    r1_reranker: R1RescueRerankerProtocol,
    b0_ranking_callback: Callable[[], tuple[Sequence[int], float]],
    reliability_threshold: float = DEFAULT_RELIABILITY_THRESHOLD,
    primary_budget: int = DEFAULT_PRIMARY_BUDGET,
    shortlist_size: int = DEFAULT_SHORTLIST_SIZE,
    solver_timeout: float = DEFAULT_SOLVER_TIMEOUT,
    verification_pair_count: int = 2,
    verification_seed: int = 999999,
    seed_role: SeedRole | str = SeedRole.DEMO,
) -> CascadeExecutionResult:
    """Execute the full adaptive cascade state machine on an attack instance.

    Guarantees
    ----------
    1. Fresh solver isolation per stage via IsolatedSolverFactory.
    2. Lazy B0 ranking is never computed unless B1/R1 SAT verification fails.
    3. Seed governance policy is strictly enforced.
    4. Exact wall-clock tracking for all components.
    """
    # Enforce seed validation
    validate_seed_for_role(verification_seed, seed_role)

    known_pairs = [
        (p.plaintext0, p.ciphertext0)
        for p in instance.pairs
    ]

    verif_pairs = independent_verification_pairs(
        instance.master_key,
        rounds=len(instance.pairs[0].plaintext0) * 3,  # 6 rounds
        pair_count=verification_pair_count,
        seed=verification_seed,
    )

    factory = IsolatedSolverFactory.from_pairs(
        known_pairs,
        rounds=6,
        instance_id=instance.instance_id,
    )

    t_gate_start = time.perf_counter()
    rel_score = reliability_estimator.predict_reliability(g1_features)
    t_gate = time.perf_counter() - t_gate_start

    t_rerank = 0.0
    t_sat = 0.0
    t_ver = 0.0
    t_lazy_b0 = 0.0
    total_attempts: list[CandidateAttempt] = []

    # Step 1: Reliability Gating
    if rel_score >= reliability_threshold:
        route = "b1_direct"
        candidates_to_test = tuple(b1_candidate_order[:primary_budget])
    else:
        route = "r1_rescue"
        shortlist_builder = R1ShortlistBuilder(shortlist_size=shortlist_size)
        shortlist = shortlist_builder.build_shortlist(b1_candidate_order)

        t_rerank_start = time.perf_counter()
        reranked = r1_reranker.rerank_shortlist(shortlist, known_pairs)
        t_rerank = time.perf_counter() - t_rerank_start
        candidates_to_test = tuple(reranked[:primary_budget])

    # Step 2: Primary SAT Solving under isolated model
    sat_model = factory.build_isolated_model(f"cascade_stage1_{route}")
    rec_res: RecoveryResult = run_candidate_recovery(
        sat_model,
        candidates_to_test,
        known_pairs,
        verif_pairs,
        candidate_budget=len(candidates_to_test),
        solver_timeout=solver_timeout,
        strategy=f"cascade_{route}",
    )

    t_sat += rec_res.solver_seconds
    t_ver += rec_res.verification_seconds
    total_attempts.extend(rec_res.attempts)

    escalated = False

    # Step 3: Success or Verification-Triggered Safety Escalation
    if rec_res.success:
        final_success = True
        final_status = "success"
        recovered_key = rec_res.recovered_master_key
        winning_cand = rec_res.winning_candidate
    else:
        # Escalate to Lazy B0 Fallback
        escalated = True
        t_b0_start = time.perf_counter()
        b0_order, t_b0_calc = b0_ranking_callback()
        t_lazy_b0 = time.perf_counter() - t_b0_start

        # Fresh isolated solver for escalated B0 run
        b0_sat_model = factory.build_isolated_model("cascade_stage2_lazy_b0")
        b0_candidates = tuple(b0_order[:primary_budget])

        rec_b0: RecoveryResult = run_candidate_recovery(
            b0_sat_model,
            b0_candidates,
            known_pairs,
            verif_pairs,
            candidate_budget=len(b0_candidates),
            solver_timeout=solver_timeout,
            strategy="cascade_lazy_b0",
        )

        t_sat += rec_b0.solver_seconds
        t_ver += rec_b0.verification_seconds
        total_attempts.extend(rec_b0.attempts)

        final_success = rec_b0.success
        final_status = rec_b0.status
        recovered_key = rec_b0.recovered_master_key
        winning_cand = rec_b0.winning_candidate

    t_total = (
        t_b1_ranking
        + t_gate
        + t_rerank
        + t_sat
        + t_ver
        + t_lazy_b0
    )

    return CascadeExecutionResult(
        instance_id=instance.instance_id,
        success=final_success,
        status=final_status,
        route_selected=route,
        reliability_score=rel_score,
        escalated_to_b0=escalated,
        recovered_master_key=recovered_key,
        winning_candidate=winning_cand,
        total_candidates_attempted=len(total_attempts),
        attempts=tuple(total_attempts),
        t_stage1_b1_rank=t_b1_ranking,
        t_reliability_gate=t_gate,
        t_r1_rerank=t_rerank,
        t_sat=t_sat,
        t_verification=t_ver,
        t_lazy_b0_rank=t_lazy_b0,
        t_total_online=t_total,
    )
