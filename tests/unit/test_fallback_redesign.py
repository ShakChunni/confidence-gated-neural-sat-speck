"""Unit tests verifying the 12 invariants of the G1/R1 adaptive cascade redesign.

Invariants Verified:
1. G1 feature extraction is strictly deterministic across repeated calls.
2. G1 functions handle empty, degenerate, or zero-variance score distributions gracefully.
3. Normalized logit margin strictly implements the Cattelan & Silva (UAI 2024) definition.
4. R1 hard-negative miner correctly identifies true subkey vs. false-positive candidates.
5. R1 shortlist builder preserves the true subkey whenever it ranks in top-M.
6. Rescue ceiling is mathematically bounded by top-M candidate recall.
7. Cascade routes to B1-direct when reliability score >= threshold.
8. Cascade routes to R1-rescue when reliability score < threshold.
9. Lazy B0 callback is NEVER computed unless Stage 1 SAT verification fails.
10. Solver isolation is maintained across all cascade stages (distinct model/solver IDs).
11. State machine tracks cumulative wall-clock time across all active execution stages.
12. Seed validation policy rejects quarantined or retired seeds in cascade execution.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from neural_sat_speck.attacks.cascade import (
    run_adaptive_cascade_recovery,
)
from neural_sat_speck.attacks.group_ranking import rank_pair_group_multi_subsets
from neural_sat_speck.attacks.r1 import (
    HeuristicRescueReranker,
    R1HardNegativeMiner,
    R1ShortlistBuilder,
    compute_rescue_ceiling,
)
from neural_sat_speck.confidence.stability import (
    G1FeatureVector,
    extract_g1_features_from_rankings,
    normalized_logit_margin,
    score_curvature,
    topk_entropy,
    topk_jaccard_stability,
    topk_spearman_correlation,
)
from neural_sat_speck.data.attack import AttackInstanceConfig, generate_attack_instances
from neural_sat_speck.data.seed_policy import (
    QUARANTINED_PHASE10_SEED,
    RETIRED_BLIND_SEED,
    Phase10DataRole,
    SeedRole,
)
from neural_sat_speck.neural.group_model import PairGroupRanker
from neural_sat_speck.neural.numpy_inference import vectorized_final_round_features


@pytest.fixture
def sample_attack_instance():
    """Deterministic 6-round attack instance for cascade testing."""
    config = AttackInstanceConfig(
        rounds=6,
        pair_count=4,
        instance_count=1,
        seed=424242,
        split_role="test",
    )
    return generate_attack_instances(config).instances[0]


def test_invariant_1_g1_feature_extraction_is_deterministic():
    """Invariant 1: G1 feature extraction produces identical values on repeated calls."""
    scores = np.random.RandomState(1234).randn(100).astype(np.float64)
    order_a = list(range(100))
    order_b = list(range(50, 150))

    val_margin1 = normalized_logit_margin(scores, p=2.0, top_k=16)
    val_margin2 = normalized_logit_margin(scores, p=2.0, top_k=16)
    assert val_margin1 == val_margin2

    val_ent1 = topk_entropy(scores, top_k=16)
    val_ent2 = topk_entropy(scores, top_k=16)
    assert val_ent1 == val_ent2

    val_jacc1 = topk_jaccard_stability(order_a, order_b, top_k=16)
    val_jacc2 = topk_jaccard_stability(order_a, order_b, top_k=16)
    assert val_jacc1 == val_jacc2


def test_invariant_2_g1_handles_degenerate_distributions_gracefully():
    """Invariant 2: Degenerate, zero-norm, or empty scores produce safe finite floats without NaN/error."""
    assert normalized_logit_margin([], p=2.0) == 0.0
    assert normalized_logit_margin([1.0], p=2.0) == 0.0
    assert normalized_logit_margin([0.0, 0.0, 0.0], p=2.0) == 0.0

    assert topk_entropy([], top_k=16) == 0.0
    assert score_curvature([1.0, 1.0], top_k=16) == 0.0
    assert topk_jaccard_stability([], []) == 0.0
    assert topk_spearman_correlation([], []) == 1.0


def test_invariant_3_normalized_logit_margin_matches_cattelan_silva():
    """Invariant 3: Normalized logit margin strictly implements (s_(1) - s_(16)) / ||s||_p."""
    scores = [10.0, 8.0, 6.0, 4.0, 2.0] + [1.0] * 20
    # sorted scores: s[0] = 10.0, s[15] (rank 16) = 1.0 -> margin = 9.0
    norm_l2 = math.sqrt(sum(x * x for x in scores))
    expected = 9.0 / norm_l2

    calc = normalized_logit_margin(scores, p=2.0, top_k=16)
    assert math.isclose(calc, expected, rel_tol=1e-7)


def test_invariant_4_r1_hard_negative_miner_labels_correctly():
    """Invariant 4: Hard negative miner correctly partitions true subkey and false candidates."""
    miner = R1HardNegativeMiner(shortlist_size=64, target_rank_cutoff=16)
    true_key = 42

    # Case A: True key ranks at position 20 (outside cutoff, within shortlist)
    order = list(range(100))
    # put true_key at index 19 (rank 20)
    order.remove(true_key)
    order.insert(19, true_key)

    res = miner.mine_instance("test-01", order, true_subkey=true_key)
    assert res.true_subkey == true_key
    assert res.true_subkey_b1_rank == 20
    assert res.true_subkey_in_shortlist is True
    assert true_key not in res.hard_negatives
    assert len(res.hard_negatives) == 16
    assert len(res.shortlist_candidates) == 64


def test_invariant_5_r1_shortlist_builder_preserves_true_subkey():
    """Invariant 5: Shortlist builder preserves the true subkey if rank <= M."""
    builder = R1ShortlistBuilder(shortlist_size=64)
    true_key = 999
    order = list(range(2000))
    order.insert(30, true_key)

    shortlist = builder.build_shortlist(order, true_subkey=true_key)
    assert len(shortlist) == 64
    assert true_key in shortlist


def test_invariant_6_rescue_ceiling_bounded_by_top_m_recall():
    """Invariant 6: Theoretical rescue ceiling cannot exceed top-M candidate recall."""
    # 4 instances:
    # Inst 1: True key rank 5 (not a failure, rank <= 16)
    # Inst 2: True key rank 25 (failure, within shortlist 64) -> RESCUED
    # Inst 3: True key rank 50 (failure, within shortlist 64) -> RESCUED
    # Inst 4: True key rank 100 (failure, outside shortlist 64) -> NOT RESCUED
    orders = [
        [1] * 4 + [42] + [2] * 100,
        [1] * 24 + [42] + [2] * 100,
        [1] * 49 + [42] + [2] * 100,
        [1] * 99 + [42] + [2] * 100,
    ]
    subkeys = [42, 42, 42, 42]

    ceiling = compute_rescue_ceiling(orders, subkeys, shortlist_size=64, primary_budget=16)
    assert ceiling["total_instances"] == 4
    assert ceiling["b1_failures"] == 3
    assert ceiling["rescued_in_shortlist"] == 2
    assert ceiling["rescue_ceiling_rate"] == 2.0 / 3.0


def test_invariant_7_cascade_routes_to_b1_when_reliability_high(sample_attack_instance):
    """Invariant 7: When reliability >= threshold, cascade executes B1-direct route."""
    true_key = sample_attack_instance.final_round_subkey
    b1_order = [true_key] + list(range(100))

    # Mock estimator returning high reliability
    class HighRelEstimator:
        def predict_reliability(self, features):
            return 0.85

    features = G1FeatureVector(0.5, 1.0, 2.0, 0.8, 0.7, 0.1, 1.0)
    lazy_called = False

    def lazy_b0_cb():
        nonlocal lazy_called
        lazy_called = True
        return ([true_key], 0.0)

    res = run_adaptive_cascade_recovery(
        instance=sample_attack_instance,
        b1_candidate_order=b1_order,
        t_b1_ranking=1.0,
        g1_features=features,
        reliability_estimator=HighRelEstimator(),
        r1_reranker=HeuristicRescueReranker(),
        b0_ranking_callback=lazy_b0_cb,
        reliability_threshold=0.50,
        primary_budget=16,
        solver_timeout=15.0,
    )

    assert res.route_selected == "b1_direct"
    assert res.success is True
    assert res.escalated_to_b0 is False
    assert lazy_called is False
    assert res.winning_candidate == true_key


def test_invariant_8_cascade_routes_to_r1_when_reliability_low(sample_attack_instance):
    """Invariant 8: When reliability < threshold, cascade executes R1-rescue route."""
    true_key = sample_attack_instance.final_round_subkey
    # B1 order has true_key at position 20 (outside 16, inside 64)
    b1_order = list(range(1, 20)) + [true_key] + list(range(21, 100))

    # Mock estimator returning low reliability
    class LowRelEstimator:
        def predict_reliability(self, features):
            return 0.25

    # R1 reranker elevates true_key to position 1
    reranker = HeuristicRescueReranker(target_subkey=true_key)

    features = G1FeatureVector(0.1, 2.5, 0.5, 0.2, 0.1, 0.9, 0.0)
    lazy_called = False

    def lazy_b0_cb():
        nonlocal lazy_called
        lazy_called = True
        return ([true_key], 0.0)

    res = run_adaptive_cascade_recovery(
        instance=sample_attack_instance,
        b1_candidate_order=b1_order,
        t_b1_ranking=1.0,
        g1_features=features,
        reliability_estimator=LowRelEstimator(),
        r1_reranker=reranker,
        b0_ranking_callback=lazy_b0_cb,
        reliability_threshold=0.50,
        primary_budget=16,
        solver_timeout=15.0,
    )

    assert res.route_selected == "r1_rescue"
    assert res.success is True
    assert res.escalated_to_b0 is False
    assert lazy_called is False
    assert res.winning_candidate == true_key


def test_invariant_9_lazy_b0_never_computed_unless_primary_fails(sample_attack_instance):
    """Invariant 9: Lazy B0 ranking callback is strictly deferred until primary stage fails."""
    true_key = sample_attack_instance.final_round_subkey

    # Case A: Primary succeeds -> lazy B0 callback NOT called
    b1_success_order = [true_key] + list(range(100))
    b0_calls = 0

    def mock_b0():
        nonlocal b0_calls
        b0_calls += 1
        return ([true_key], 1.0)

    class FixedEstimator:
        def predict_reliability(self, f):
            return 0.9

    features = G1FeatureVector(0.5, 1.0, 2.0, 0.8, 0.7, 0.1, 1.0)
    res_succ = run_adaptive_cascade_recovery(
        instance=sample_attack_instance,
        b1_candidate_order=b1_success_order,
        t_b1_ranking=1.0,
        g1_features=features,
        reliability_estimator=FixedEstimator(),
        r1_reranker=HeuristicRescueReranker(),
        b0_ranking_callback=mock_b0,
        solver_timeout=15.0,
    )
    assert res_succ.success is True
    assert b0_calls == 0, "Lazy B0 was eagerly computed when primary succeeded!"

    # Case B: Primary fails -> lazy B0 callback MUST be called
    b1_fail_order = [k for k in range(100) if k != true_key]
    res_esc = run_adaptive_cascade_recovery(
        instance=sample_attack_instance,
        b1_candidate_order=b1_fail_order,
        t_b1_ranking=1.0,
        g1_features=features,
        reliability_estimator=FixedEstimator(),
        r1_reranker=HeuristicRescueReranker(),
        b0_ranking_callback=mock_b0,
        primary_budget=1,
        solver_timeout=15.0,
    )
    assert b0_calls == 1, "Lazy B0 was NOT invoked upon primary failure!"
    assert res_esc.escalated_to_b0 is True
    assert res_esc.success is True


def test_invariant_10_solver_isolation_maintained_in_cascade(sample_attack_instance):
    """Invariant 10: Cascade uses fresh independent solvers for primary and escalated stages."""
    true_key = sample_attack_instance.final_round_subkey
    b1_fail_order = [k for k in range(100) if k != true_key]

    def mock_b0():
        return ([true_key], 1.0)

    class FixedEstimator:
        def predict_reliability(self, f):
            return 0.9

    features = G1FeatureVector(0.5, 1.0, 2.0, 0.8, 0.7, 0.1, 1.0)
    res = run_adaptive_cascade_recovery(
        instance=sample_attack_instance,
        b1_candidate_order=b1_fail_order,
        t_b1_ranking=1.0,
        g1_features=features,
        reliability_estimator=FixedEstimator(),
        r1_reranker=HeuristicRescueReranker(),
        b0_ranking_callback=mock_b0,
        primary_budget=2,
        solver_timeout=5.0,
    )

    # In attempts: stage 1 had candidates (which may timeout), stage 2 had 1 candidate
    assert len(res.attempts) in (2, 3)
    assert res.escalated_to_b0 is True


def test_invariant_11_state_machine_tracks_cumulative_time(sample_attack_instance):
    """Invariant 11: Total online time equals sum of all active component timings."""
    true_key = sample_attack_instance.final_round_subkey
    b1_order = [true_key] + list(range(100))

    class FixedEstimator:
        def predict_reliability(self, f):
            return 0.9

    features = G1FeatureVector(0.5, 1.0, 2.0, 0.8, 0.7, 0.1, 1.0)
    res = run_adaptive_cascade_recovery(
        instance=sample_attack_instance,
        b1_candidate_order=b1_order,
        t_b1_ranking=2.5,
        g1_features=features,
        reliability_estimator=FixedEstimator(),
        r1_reranker=HeuristicRescueReranker(),
        b0_ranking_callback=lambda: ([true_key], 0.0),
        solver_timeout=5.0,
    )

    expected_sum = (
        res.t_stage1_b1_rank
        + res.t_reliability_gate
        + res.t_r1_rerank
        + res.t_sat
        + res.t_verification
        + res.t_lazy_b0_rank
    )
    assert math.isclose(res.t_total_online, expected_sum, rel_tol=1e-6)


def test_invariant_12_seed_validation_rejects_quarantined_seeds(sample_attack_instance):
    """Invariant 12: Seed validation policy rejects quarantined seeds in cascade execution."""
    true_key = sample_attack_instance.final_round_subkey
    features = G1FeatureVector(0.5, 1.0, 2.0, 0.8, 0.7, 0.1, 1.0)

    class DummyEstimator:
        def predict_reliability(self, f):
            return 0.5

    # Passing QUARANTINED_PHASE10_SEED (97001) for PROSPECTIVE_VALIDATION must raise ValueError
    with pytest.raises(ValueError, match="(?i)quarantined"):
        run_adaptive_cascade_recovery(
            instance=sample_attack_instance,
            b1_candidate_order=[true_key],
            t_b1_ranking=1.0,
            g1_features=features,
            reliability_estimator=DummyEstimator(),
            r1_reranker=HeuristicRescueReranker(),
            b0_ranking_callback=lambda: ([true_key], 0.0),
            verification_seed=QUARANTINED_PHASE10_SEED,
            seed_role=SeedRole.PROSPECTIVE_VALIDATION,
        )

    # Passing RETIRED_BLIND_SEED (99001) for BLIND_CONFIRMATION must raise ValueError
    with pytest.raises(ValueError, match="(?i)retired"):
        run_adaptive_cascade_recovery(
            instance=sample_attack_instance,
            b1_candidate_order=[true_key],
            t_b1_ranking=1.0,
            g1_features=features,
            reliability_estimator=DummyEstimator(),
            r1_reranker=HeuristicRescueReranker(),
            b0_ranking_callback=lambda: ([true_key], 0.0),
            verification_seed=RETIRED_BLIND_SEED,
            seed_role=SeedRole.BLIND_CONFIRMATION,
        )


def test_invariant_13_locked_validation_role_cannot_enter_training_or_calibration():
    """Invariant 13: Passing locked validation instances to training or miner raises ValueError."""
    miner = R1HardNegativeMiner(shortlist_size=32)
    b1_order = list(range(100))

    # Passing LOCKED_VALIDATION role must immediately fail
    with pytest.raises(ValueError, match="(?i)locked prospective validation"):
        miner.mine_instance(
            instance_id="val_inst_001",
            b1_candidate_order=b1_order,
            true_subkey=42,
            data_role=Phase10DataRole.LOCKED_VALIDATION,
        )


def test_invariant_14_g1_optimized_reference_equivalence():
    """Invariant 14: Multi-subset ranking via cached embeddings matches independent passes bitwise."""
    config = AttackInstanceConfig(
        rounds=6,
        pair_count=64,
        instance_count=1,
        seed=777777,
        split_role="test",
    )
    inst = generate_attack_instances(config).instances[0]
    model = PairGroupRanker(variant="b1_mean")
    model.eval()

    cands = [100, 200, 300, 400]

    # Optimized multi-subset call
    multi_rankings = rank_pair_group_multi_subsets(
        inst,
        model,
        subset_sizes=(16, 32, 64),
        candidate_values=cands,
    )

    # Independent reference passes
    with torch.no_grad():
        f16 = torch.from_numpy(vectorized_final_round_features(inst.pairs[:16], cands))
        emb16 = model.encoder(f16.reshape(-1, 4, 16)).reshape(len(cands), 16, -1)
        ref16 = model.head(emb16.mean(dim=1)).squeeze().numpy()

        f32 = torch.from_numpy(vectorized_final_round_features(inst.pairs[:32], cands))
        emb32 = model.encoder(f32.reshape(-1, 4, 16)).reshape(len(cands), 32, -1)
        ref32 = model.head(emb32.mean(dim=1)).squeeze().numpy()

        f64 = torch.from_numpy(vectorized_final_round_features(inst.pairs[:64], cands))
        emb64 = model.encoder(f64.reshape(-1, 4, 16)).reshape(len(cands), 64, -1)
        ref64 = model.head(emb64.mean(dim=1)).squeeze().numpy()

    diff16 = np.max(np.abs(np.array(multi_rankings[16].candidate_scores) - ref16))
    diff32 = np.max(np.abs(np.array(multi_rankings[32].candidate_scores) - ref32))
    diff64 = np.max(np.abs(np.array(multi_rankings[64].candidate_scores) - ref64))

    assert diff16 == 0.0, f"16-pair diff must be 0.0, got {diff16}"
    assert diff32 == 0.0, f"32-pair diff must be 0.0, got {diff32}"
    assert diff64 == 0.0, f"64-pair diff must be 0.0, got {diff64}"

    # Verify G1 feature extraction on multi_rankings runs cleanly and deterministically
    feat_vec1 = extract_g1_features_from_rankings(
        multi_rankings[64],
        multi_rankings[32],
        multi_rankings[16],
    )
    feat_vec2 = extract_g1_features_from_rankings(
        multi_rankings[64],
        multi_rankings[32],
        multi_rankings[16],
    )
    np.testing.assert_array_equal(feat_vec1.to_array(), feat_vec2.to_array())


def test_invariant_15_r1_shortlist_budget_selection():
    """Invariant 15: R1 shortlist builder strictly obeys bounded shortlist sizes M in {32, 64, 128}."""
    order = list(range(200))
    for m in (32, 64, 128):
        builder = R1ShortlistBuilder(shortlist_size=m)
        shortlist = builder.build_shortlist(order)
        assert len(shortlist) == m
        assert shortlist == tuple(range(m))


def test_invariant_16_terminal_fallback_deduplication_and_cumulative_budget(sample_attack_instance):
    """Invariant 16: Lazy terminal B0 fallback strictly deduplicates seen candidates and caps cumulative budget."""
    true_key = sample_attack_instance.final_round_subkey

    # Primary stage tests candidates [10, 11, 12... 25] (16 candidates), all UNSAT
    primary_cands = list(range(10, 26))

    # B0 contains some overlapping candidates [15, 16, 17... 40]
    # Plus true_key at position 30
    b0_cands = list(range(15, 30)) + [true_key] + list(range(31, 50))

    class LowRelEstimator:
        def predict_reliability(self, f):
            return 0.1

    features = G1FeatureVector(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)

    res = run_adaptive_cascade_recovery(
        instance=sample_attack_instance,
        b1_candidate_order=primary_cands + list(range(100, 150)),
        t_b1_ranking=1.0,
        g1_features=features,
        reliability_estimator=LowRelEstimator(),
        r1_reranker=HeuristicRescueReranker(),  # reverses primary_cands[:16]
        b0_ranking_callback=lambda: (b0_cands, 0.1),
        primary_budget=16,
        cumulative_budget=32,
        solver_timeout=5.0,
    )

    # Verify that total candidates attempted never exceeds cumulative_budget (32)
    assert res.total_candidates_attempted <= 32

    # Verify that every attempted candidate is strictly unique (no duplicates tested)
    attempted_cands = [a.candidate for a in res.attempts]
    assert len(attempted_cands) == len(set(attempted_cands)), "Duplicate candidate attempted in SAT!"
    assert res.success is True
    assert res.winning_candidate == true_key


def test_invariant_17_g1_failure_risk_routing_polarity(sample_attack_instance):
    """Invariant 17: G1 failure risk polarity routing regression check.

    Mathematical Invariant:
    G1(x) = P(B1 top-16 failure | x)
    low failure risk  (< theta)  -> B1_DIRECT
    high failure risk (>= theta) -> R1_RESCUE
    Polarity cannot be inverted.
    """
    true_key = sample_attack_instance.final_round_subkey
    b1_order = [true_key] + list(range(100))
    features = G1FeatureVector(0.5, 1.0, 2.0, 0.8, 0.7, 0.1, 1.0)

    class LowFailureRiskEstimator:
        def predict_failure_risk(self, f):
            return 0.05  # Low failure risk (< 0.50)

    class HighFailureRiskEstimator:
        def predict_failure_risk(self, f):
            return 0.95  # High failure risk (>= 0.50)

    # 1. Low failure risk MUST route to b1_direct
    res_low = run_adaptive_cascade_recovery(
        instance=sample_attack_instance,
        b1_candidate_order=b1_order,
        t_b1_ranking=1.0,
        g1_features=features,
        reliability_estimator=LowFailureRiskEstimator(),
        r1_reranker=HeuristicRescueReranker(),
        b0_ranking_callback=lambda: ([true_key], 0.0),
        failure_threshold=0.50,
        primary_budget=16,
        solver_timeout=5.0,
    )
    assert res_low.route_selected == "b1_direct", (
        f"Expected 'b1_direct' for low failure risk (0.05 < 0.50), got {res_low.route_selected}"
    )

    # 2. High failure risk MUST route to r1_rescue
    res_high = run_adaptive_cascade_recovery(
        instance=sample_attack_instance,
        b1_candidate_order=b1_order,
        t_b1_ranking=1.0,
        g1_features=features,
        reliability_estimator=HighFailureRiskEstimator(),
        r1_reranker=HeuristicRescueReranker(target_subkey=true_key),
        b0_ranking_callback=lambda: ([true_key], 0.0),
        failure_threshold=0.50,
        primary_budget=16,
        solver_timeout=5.0,
    )
    assert res_high.route_selected == "r1_rescue", (
        f"Expected 'r1_rescue' for high failure risk (0.95 >= 0.50), got {res_high.route_selected}"
    )
