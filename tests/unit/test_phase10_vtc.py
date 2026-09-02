"""Unit tests for Phase 10 Verification-Triggered Cascaded Fallback (VTC-B1-B0)."""

from unittest.mock import MagicMock

import pytest

from neural_sat_speck.attacks.group_ranking import PairGroupSubkeyRanking
from neural_sat_speck.attacks.ranking import SubkeyRanking
from neural_sat_speck.attacks.recovery import (
    independent_verification_pairs,
    known_pairs_for_attack_instance,
)
from neural_sat_speck.attacks.vtc import (
    run_vtc_attack,
)
from neural_sat_speck.data.attack import AttackInstance, AttackPair
from neural_sat_speck.sat.model import SpeckSatModel
from neural_sat_speck.speck.cipher import encrypt_block
from neural_sat_speck.speck.key_schedule import expand_key


def _make_dummy_b1_ranking(order: list[int], total_seconds: float = 78.5) -> PairGroupSubkeyRanking:
    domain = tuple(range(65536))
    score_map = {cand: float(65536 - rank) for rank, cand in enumerate(order)}
    scores = tuple(score_map[c] for c in domain)
    return PairGroupSubkeyRanking(
        candidate_values=domain,
        candidate_scores=scores,
        candidate_order=tuple(order),
        pair_count=64,
        group_size=64,
        candidate_batch_size=4096,
        score_aggregation="group_logit",
        partial_decryption_seconds=1.0,
        inference_seconds=total_seconds - 1.1,
        sorting_seconds=0.1,
        total_seconds=total_seconds,
        peak_feature_tensor_bytes=1000,
    )


def _make_dummy_b0_ranking(order: list[int], total_seconds: float = 79.2) -> SubkeyRanking:
    domain = tuple(range(65536))
    score_map = {cand: float(65536 - rank) for rank, cand in enumerate(order)}
    scores = tuple(score_map[c] for c in domain)
    return SubkeyRanking(
        candidate_values=domain,
        candidate_scores=scores,
        candidate_order=tuple(order),
        pair_count=64,
        candidate_batch_size=4096,
        score_aggregation="sum_logits",
        partial_decryption_seconds=1.0,
        inference_seconds=total_seconds - 1.1,
        sorting_seconds=0.1,
        total_seconds=total_seconds,
        peak_feature_tensor_bytes=1000,
        pair_subset_stability=None,
    )


@pytest.fixture
def sample_attack_instance() -> AttackInstance:
    master_key = (0x1234, 0x5678, 0x9ABC, 0xDEF0)
    rounds = 6
    round_keys = expand_key(master_key, rounds)
    final_subkey = round_keys[-1]

    pairs = []
    for i in range(64):
        p0 = (0x1000 + i, 0x2000 + i)
        p1 = (p0[0] ^ 0x0040, p0[1])
        c0 = encrypt_block(p0, master_key, rounds)
        c1 = encrypt_block(p1, master_key, rounds)
        pairs.append(AttackPair(plaintext0=p0, ciphertext0=c0, plaintext1=p1, ciphertext1=c1))

    return AttackInstance(
        instance_id="test-fixture-0000",
        master_key=master_key,
        final_round_subkey=final_subkey,
        pairs=tuple(pairs),
    )


def test_solver_isolation_invariant(sample_attack_instance: AttackInstance) -> None:
    """Verify that every compared method receives an independent solver instance."""
    all_pairs = known_pairs_for_attack_instance(sample_attack_instance)
    known = all_pairs[:4]

    m_b0 = SpeckSatModel.from_known_pairs(known, 6)
    m_b1 = SpeckSatModel.from_known_pairs(known, 6)
    m_eager = SpeckSatModel.from_known_pairs(known, 6)
    m_vtc = SpeckSatModel.from_known_pairs(known, 6)

    models = [m_b0, m_b1, m_eager, m_vtc]
    model_ids = {id(m) for m in models}
    solver_ids = {id(m.solver) for m in models}

    assert len(model_ids) == 4, "All 4 method models must be distinct objects"
    assert len(solver_ids) == 4, "All 4 underlying CryptoMiniSat solvers must be distinct instances"


def test_vtc_early_termination_on_b1_success(sample_attack_instance: AttackInstance) -> None:
    """Verify that when B1 top-16 contains the true subkey, B0 is never computed."""
    true_key = sample_attack_instance.final_round_subkey

    all_keys = [c for c in range(65536) if c != true_key]
    b1_order = all_keys[:2] + [true_key] + all_keys[2:]
    b1_ranking = _make_dummy_b1_ranking(b1_order, total_seconds=78.5)

    dummy_infer = MagicMock()
    all_pairs = known_pairs_for_attack_instance(sample_attack_instance)
    known = all_pairs[:4]
    ver = independent_verification_pairs(sample_attack_instance.master_key, 6, 2, 99999)

    res = run_vtc_attack(
        sample_attack_instance,
        b1_infer=dummy_infer,
        b0_infer=dummy_infer,
        known_pairs=known,
        verification_pairs=ver,
        rounds=6,
        precomputed_b1_ranking=b1_ranking,
        precomputed_b0_ranking=None,
    )

    assert res.success is True
    assert res.status == "recovered"
    assert res.winning_stage == "primary_b1"
    assert res.winning_candidate == true_key
    assert res.winning_candidate_position == 3
    assert res.b0_invoked is False
    assert res.t_rank_b0 == 0.0
    assert res.t_rank_total == 78.5
    assert len(res.attempts) == 3
    assert res.b0_candidates_attempted == 0
    assert res.b1_candidates_attempted == 3


def test_vtc_escalation_on_b1_exhaustion(sample_attack_instance: AttackInstance) -> None:
    """Verify that when B1 top-16 fails, B0 is lazily computed and rescues the key."""
    true_key = sample_attack_instance.final_round_subkey

    all_keys = [c for c in range(65536) if c != true_key]
    b1_order = all_keys[:24] + [true_key] + all_keys[24:]
    b1_ranking = _make_dummy_b1_ranking(b1_order, total_seconds=78.5)

    b0_order = all_keys[:1] + [true_key] + all_keys[1:]
    b0_ranking = _make_dummy_b0_ranking(b0_order, total_seconds=79.2)

    dummy_infer = MagicMock()
    all_pairs = known_pairs_for_attack_instance(sample_attack_instance)
    known = all_pairs[:4]
    ver = independent_verification_pairs(sample_attack_instance.master_key, 6, 2, 99999)

    res = run_vtc_attack(
        sample_attack_instance,
        b1_infer=dummy_infer,
        b0_infer=dummy_infer,
        known_pairs=known,
        verification_pairs=ver,
        rounds=6,
        precomputed_b1_ranking=b1_ranking,
        precomputed_b0_ranking=b0_ranking,
    )

    assert res.success is True
    assert res.status == "recovered"
    assert res.winning_stage == "fallback_b0"
    assert res.winning_candidate == true_key
    # all_keys[0] was in B1 top-16, so it is deduplicated; true_key is tested as attempt 17 (16 from B1 + 1 new from B0)
    assert res.winning_candidate_position == 17
    assert res.b0_invoked is True
    assert res.t_rank_b0 == 79.2
    assert res.t_rank_total == 78.5 + 79.2
    assert res.b1_candidates_attempted == 16
    assert res.b0_candidates_attempted == 1
    assert res.candidates_attempted == 17


def test_vtc_candidate_deduplication(sample_attack_instance: AttackInstance) -> None:
    """Verify that candidates already attempted in B1 top-16 are skipped in B0."""
    true_key = sample_attack_instance.final_round_subkey

    all_keys = [c for c in range(65536) if c != true_key]
    # Overlap: first 5 keys in B1 are also at start of B0
    overlap_keys = all_keys[:5]
    other_keys = all_keys[5:]

    b1_order = overlap_keys + other_keys + [true_key]
    b1_ranking = _make_dummy_b1_ranking(b1_order, total_seconds=80.0)

    # B0 starts with overlap_keys, then true_key, then remaining
    b0_order = overlap_keys + [true_key] + other_keys
    b0_ranking = _make_dummy_b0_ranking(b0_order, total_seconds=80.0)

    dummy_infer = MagicMock()
    all_pairs = known_pairs_for_attack_instance(sample_attack_instance)
    known = all_pairs[:4]
    ver = independent_verification_pairs(sample_attack_instance.master_key, 6, 2, 99999)

    res = run_vtc_attack(
        sample_attack_instance,
        b1_infer=dummy_infer,
        b0_infer=dummy_infer,
        known_pairs=known,
        verification_pairs=ver,
        rounds=6,
        precomputed_b1_ranking=b1_ranking,
        precomputed_b0_ranking=b0_ranking,
    )

    assert res.success is True
    assert res.b0_candidates_sequence[0] == true_key
    assert res.b0_candidates_attempted == 1
    assert res.candidates_attempted == 17  # 16 from B1 + 1 from B0


def test_vtc_budget_capping(sample_attack_instance: AttackInstance) -> None:
    """Verify that VTC terminates when cumulative attempts reach kmax = 32."""
    all_keys = list(range(65536))
    b1_order = all_keys
    b1_ranking = _make_dummy_b1_ranking(b1_order, total_seconds=80.0)

    b0_order = all_keys[::-1]
    b0_ranking = _make_dummy_b0_ranking(b0_order, total_seconds=80.0)

    dummy_infer = MagicMock()
    all_pairs = known_pairs_for_attack_instance(sample_attack_instance)
    known = all_pairs[:4]
    ver = independent_verification_pairs(sample_attack_instance.master_key, 6, 2, 99999)

    res = run_vtc_attack(
        sample_attack_instance,
        b1_infer=dummy_infer,
        b0_infer=dummy_infer,
        known_pairs=known,
        verification_pairs=ver,
        rounds=6,
        precomputed_b1_ranking=b1_ranking,
        precomputed_b0_ranking=b0_ranking,
    )

    assert res.success is False
    assert res.status == "budget_exhausted"
    assert res.b0_invoked is True
    assert res.candidates_attempted == 32
    assert res.b1_candidates_attempted == 16
    assert res.b0_candidates_attempted == 16
    assert len(res.cumulative_candidates_sequence) == 32
    assert len(set(res.cumulative_candidates_sequence)) == 32


def test_vtc_serialization_roundtrip(sample_attack_instance: AttackInstance) -> None:
    """Verify dictionary serialization contains all required Phase 10 fields."""
    true_key = sample_attack_instance.final_round_subkey
    all_keys = [c for c in range(65536) if c != true_key]
    b1_order = [true_key] + all_keys
    b1_ranking = _make_dummy_b1_ranking(b1_order, total_seconds=80.0)

    dummy_infer = MagicMock()
    all_pairs = known_pairs_for_attack_instance(sample_attack_instance)
    known = all_pairs[:4]
    ver = independent_verification_pairs(sample_attack_instance.master_key, 6, 2, 99999)

    res = run_vtc_attack(
        sample_attack_instance,
        b1_infer=dummy_infer,
        b0_infer=dummy_infer,
        known_pairs=known,
        verification_pairs=ver,
        rounds=6,
        precomputed_b1_ranking=b1_ranking,
    )

    d = res.to_dict()
    assert d["strategy"] == "vtc_b1_b0_sat"
    assert d["success"] is True
    assert d["b0_invoked"] is False
    assert "timing_seconds" in d
    assert "rank_metrics" in d
    assert d["timing_seconds"]["t_rank_b1"] == 80.0
    assert d["timing_seconds"]["t_rank_b0"] == 0.0
