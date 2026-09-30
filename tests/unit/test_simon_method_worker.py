"""Unit and adversarial tests for the isolated SIMON method worker harness."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

import neural_sat_speck.simon.method_worker as method_worker_module
from neural_sat_speck.simon.attack_instance import (
    SimonAttackInstanceConfig,
    generate_simon_attack_instances,
)
from neural_sat_speck.simon.b1_data import save_simon_checkpoint
from neural_sat_speck.simon.faithful_b1_model import SimonPairGroupRanker
from neural_sat_speck.simon.key_schedule import expand_key
from neural_sat_speck.simon.method_worker import (
    EIGHT_COMPARATORS,
    generate_faithful_b1_hybrid_queue,
    generate_random_tuples,
    run_method_worker,
)
from neural_sat_speck.simon.sat_model import SatSolveResult


def test_corrupt_progress_journal_is_flagged_and_partial_evidence_retained(tmp_path):
    journal = tmp_path / "progress.jsonl"
    journal.write_text(
        json.dumps({"event": "queue", "total_queue_count": 1,
                    "full_queue": [[1, 2]], "queue_sample": [[1, 2]]}) + "\n"
        + json.dumps({"event": "attempt_start", "candidate_k8": 1,
                      "candidate_k9": 2, "started_at_monotonic": 100.1}) + "\n"
        + '{"event":"attempt"\n', encoding="utf-8"
    )
    rec = method_worker_module._read_progress_file(
        journal, parent_elapsed=1.0, stopped_at_monotonic=101.0,
    )
    assert rec.journal_error is not None
    assert "line 3" in rec.journal_error
    assert rec.in_flight_attempt is not None
    assert rec.candidates_remaining_count == 0


def test_progress_fsync_failure_is_not_silenced(tmp_path, monkeypatch):
    def fail_fsync(_fd):
        raise OSError("forced journal write failure")

    monkeypatch.setattr(method_worker_module.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="forced journal write failure"):
        method_worker_module._log_progress_event(
            str(tmp_path / "progress.jsonl"), {"event": "init", "method": "test"}
        )


def test_missing_progress_journal_is_flagged(tmp_path):
    rec = method_worker_module._read_progress_file(tmp_path / "missing.jsonl")
    assert rec.journal_error is not None
    assert "missing" in rec.journal_error


def _synthetic_checkpoint_metadata() -> dict[str, object]:
    """Schema fixture only; it does not claim real SIMON training occurred."""
    return {
        "training_status": "complete",
        "training_manifest_sha256": "1" * 64,
        "train_dataset_checksum": "2" * 64,
        "validation_dataset_checksum": "3" * 64,
        "selected_epoch": 1,
    }


@pytest.fixture(scope="module")
def demo_instance():
    cfg = SimonAttackInstanceConfig(
        instance_count=2,
        pair_count=32,
        rounds=10,
        seed=999999,
        split_role="demo",
    )
    return generate_simon_attack_instances(cfg)[1]


@pytest.fixture(scope="module")
def demo_instance_r9():
    cfg = SimonAttackInstanceConfig(
        instance_count=1,
        pair_count=32,
        rounds=9,
        seed=999999,
        split_role="demo",
    )
    return generate_simon_attack_instances(cfg)[0]


# 1. Oracle Control: Recovery and Verification
def test_oracle_control_recovers_and_verifies_on_2_pairs(demo_instance):
    round_keys = expand_key(demo_instance.master_key, 10)
    true_k8, true_k9 = round_keys[8], round_keys[9]

    outcome = run_method_worker(
        method="oracle_a2_n4",
        rounds=10,
        attack_pairs=tuple((p.ciphertext0, p.ciphertext1) for p in demo_instance.pairs),
        known_pairs=demo_instance.known_pairs,
        verification_pairs=demo_instance.verification_pairs,
        oracle_trailing_keys=(true_k8, true_k9),
        candidate_cutoff_seconds=60.0,
        global_deadline_seconds=90.0,
        use_parent_watchdog=False,
    )

    assert outcome.outcome == "verified_recovery"
    assert outcome.verified_on_fresh_pairs is True
    assert outcome.recovered_master_key == demo_instance.master_key
    assert outcome.candidates_tested_count == 1
    assert outcome.candidates_remaining_count == 0
    assert len(outcome.candidate_attempts) == 1
    assert outcome.candidate_attempts[0].status == "sat"
    assert outcome.peak_rss_gib > 0.001


# 2. Compared Methods Isolation: Reject Oracle Keys and Mock Queues
@pytest.mark.parametrize("method", sorted(EIGHT_COMPARATORS))
def test_compared_methods_reject_oracle_and_mock_queues(demo_instance, method):
    attack_pairs = tuple((p.ciphertext0, p.ciphertext1) for p in demo_instance.pairs)

    # Rejects oracle_trailing_keys
    with pytest.raises(ValueError, match="cannot receive oracle trailing keys"):
        run_method_worker(
            method=method,
            rounds=10,
            attack_pairs=attack_pairs,
            known_pairs=demo_instance.known_pairs,
            verification_pairs=demo_instance.verification_pairs,
            oracle_trailing_keys=(0x1234, 0x5678),
            use_parent_watchdog=False,
        )

    # Rejects mock_candidate_queue
    with pytest.raises(ValueError, match="cannot receive mock candidate queue"):
        run_method_worker(
            method=method,
            rounds=10,
            attack_pairs=attack_pairs,
            known_pairs=demo_instance.known_pairs,
            verification_pairs=demo_instance.verification_pairs,
            mock_candidate_queue=[(0x1234, 0x5678)],
            use_parent_watchdog=False,
        )


# 3. Empty Queue Rejection
def test_empty_queue_worker_rejected(demo_instance):
    attack_pairs = tuple((p.ciphertext0, p.ciphertext1) for p in demo_instance.pairs)

    with pytest.raises(ValueError, match="cannot be empty|empty candidate queue"):
        run_method_worker(
            method="test_mock_queue",
            rounds=10,
            attack_pairs=attack_pairs,
            known_pairs=demo_instance.known_pairs,
            verification_pairs=demo_instance.verification_pairs,
            mock_candidate_queue=[],
            use_parent_watchdog=False,
        )


# 4. Reversed Key Order Fails (Chronological Order Invariant)
def test_reversed_key_order_fails_or_unsat(demo_instance_r9):
    round_keys = expand_key(demo_instance_r9.master_key, 9)
    true_k7, true_k8 = round_keys[7], round_keys[8]
    assert true_k7 != true_k8, "True trailing keys must be distinct for reversal test"

    attack_pairs = tuple((p.ciphertext0, p.ciphertext1) for p in demo_instance_r9.pairs)

    # Chronological (k7, k8) succeeds
    chronological_outcome = run_method_worker(
        method="oracle_a2_n4",
        rounds=9,
        attack_pairs=attack_pairs,
        known_pairs=demo_instance_r9.known_pairs,
        verification_pairs=demo_instance_r9.verification_pairs,
        oracle_trailing_keys=(true_k7, true_k8),
        candidate_cutoff_seconds=10.0,
        global_deadline_seconds=30.0,
        use_parent_watchdog=False,
    )
    assert chronological_outcome.outcome == "verified_recovery"

    # Reversed (k8, k7) yields UNSAT (residual 7 rounds proves UNSAT in <0.05s)
    reversed_outcome = run_method_worker(
        method="test_mock_queue",
        rounds=9,
        attack_pairs=attack_pairs,
        known_pairs=demo_instance_r9.known_pairs,
        verification_pairs=demo_instance_r9.verification_pairs,
        mock_candidate_queue=[(true_k8, true_k7)],
        candidate_cutoff_seconds=10.0,
        global_deadline_seconds=30.0,
        use_parent_watchdog=False,
    )
    assert reversed_outcome.outcome == "queue_exhausted"
    assert reversed_outcome.verified_on_fresh_pairs is False
    assert len(reversed_outcome.candidate_attempts) == 1
    assert reversed_outcome.candidate_attempts[0].status == "unsat"


# 5. Queue Exhaustion on Proven UNSAT Wrong Tuples
def test_queue_exhaustion_on_proven_unsat_wrong_tuples(demo_instance_r9):
    wrong_tuples = [(0x1234, 0x5678), (0xAAAA, 0x5555)]

    outcome = run_method_worker(
        method="test_mock_queue",
        rounds=9,
        attack_pairs=tuple((p.ciphertext0, p.ciphertext1) for p in demo_instance_r9.pairs),
        known_pairs=demo_instance_r9.known_pairs,
        verification_pairs=demo_instance_r9.verification_pairs,
        mock_candidate_queue=wrong_tuples,
        candidate_cutoff_seconds=10.0,
        global_deadline_seconds=30.0,
        use_parent_watchdog=False,
    )

    assert outcome.outcome == "queue_exhausted"
    assert outcome.verified_on_fresh_pairs is False
    assert outcome.candidates_tested_count == 2
    assert outcome.candidates_remaining_count == 0
    for attempt in outcome.candidate_attempts:
        assert attempt.status == "unsat"
        assert attempt.rebuild_occurred is False


# 6. Rebuild Policy Triggered on Candidate Timeout
def test_rebuild_after_timeout_policy_triggered(demo_instance):
    mock_tuples = [(0x1234, 0x5678), (0x50A8, 0x0A0A)]

    outcome = run_method_worker(
        method="test_mock_queue",
        rounds=10,
        attack_pairs=tuple((p.ciphertext0, p.ciphertext1) for p in demo_instance.pairs),
        known_pairs=demo_instance.known_pairs,
        verification_pairs=demo_instance.verification_pairs,
        mock_candidate_queue=mock_tuples,
        candidate_cutoff_seconds=0.0001,
        global_deadline_seconds=30.0,
        use_parent_watchdog=False,
    )

    assert len(outcome.candidate_attempts) >= 1
    first_att = outcome.candidate_attempts[0]
    assert first_att.status == "solver_timeout"
    assert first_att.rebuild_occurred is True


# 7. Parent Process Watchdog Termination on Deliberately Slow Ranker
def test_parent_watchdog_terminates_slow_ranker(demo_instance):
    outcome = run_method_worker(
        method="test_slow_ranker",
        rounds=10,
        attack_pairs=tuple((p.ciphertext0, p.ciphertext1) for p in demo_instance.pairs),
        known_pairs=demo_instance.known_pairs,
        verification_pairs=demo_instance.verification_pairs,
        global_deadline_seconds=0.5,
        use_parent_watchdog=True,
    )

    assert outcome.outcome == "global_deadline"
    assert outcome.complete_online_wall_seconds < 2.5
    assert outcome.verified_on_fresh_pairs is False


# 8. Parent Process Watchdog Termination on Deliberately Slow Solver
def test_parent_watchdog_terminates_slow_solver(demo_instance):
    outcome = run_method_worker(
        method="test_slow_solver",
        rounds=10,
        attack_pairs=tuple((p.ciphertext0, p.ciphertext1) for p in demo_instance.pairs),
        known_pairs=demo_instance.known_pairs,
        verification_pairs=demo_instance.verification_pairs,
        global_deadline_seconds=0.5,
        use_parent_watchdog=True,
    )

    assert outcome.outcome == "global_deadline"
    assert outcome.complete_online_wall_seconds < 2.5
    assert outcome.verified_on_fresh_pairs is False


# 9. Unknown Solver Status Handling and Rebuild
def test_unknown_solver_status_triggers_rebuild(demo_instance):
    mock_tuples = [(0x1234, 0x5678), (0x2345, 0x6789)]

    call_count = 0

    def dummy_unknown_solver(model, assumptions, time_limit):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return SatSolveResult(False, None, 0.01, False, "unsat_or_timeout")
        else:
            return SatSolveResult(False, None, 0.01, False, "custom_anomaly")

    outcome = run_method_worker(
        method="test_mock_queue",
        rounds=10,
        attack_pairs=tuple((p.ciphertext0, p.ciphertext1) for p in demo_instance.pairs),
        known_pairs=demo_instance.known_pairs,
        verification_pairs=demo_instance.verification_pairs,
        mock_candidate_queue=mock_tuples,
        solver_stub=dummy_unknown_solver,
        candidate_cutoff_seconds=10.0,
        global_deadline_seconds=30.0,
        use_parent_watchdog=False,
    )

    assert len(outcome.candidate_attempts) == 2
    att1 = outcome.candidate_attempts[0]
    assert att1.status == "unsat_or_timeout"
    assert att1.rebuild_occurred is True

    att2 = outcome.candidate_attempts[1]
    assert att2.status == "unknown_custom_anomaly"
    assert att2.rebuild_occurred is True


# 10. Verification Failure Handling (SAT solution that fails fresh pairs)
def test_verification_failure_after_sat(demo_instance):
    mock_tuples = [(0x1234, 0x5678), (0x2345, 0x6789)]

    # Stub that returns a satisfying assignment whose master key is wrong (e.g., all zeros)
    # which will fail independent verification
    call_count = 0

    def bogus_sat_solver(model, assumptions, time_limit):
        nonlocal call_count
        call_count += 1
        # Build solution mapping all variables to False (yielding master key 0x0000000000000000)
        bogus_sol = [False] * (model.variable_count + 1)
        return SatSolveResult(True, bogus_sol, 0.05, False, "sat")

    outcome = run_method_worker(
        method="test_mock_queue",
        rounds=10,
        attack_pairs=tuple((p.ciphertext0, p.ciphertext1) for p in demo_instance.pairs),
        known_pairs=demo_instance.known_pairs,
        verification_pairs=demo_instance.verification_pairs,
        mock_candidate_queue=mock_tuples,
        solver_stub=bogus_sat_solver,
        candidate_cutoff_seconds=10.0,
        global_deadline_seconds=30.0,
        use_parent_watchdog=False,
    )

    assert outcome.outcome == "verification_failure"
    assert outcome.verified_on_fresh_pairs is False
    assert outcome.candidates_tested_count == 2
    for att in outcome.candidate_attempts:
        assert att.status == "verification_failure"
        assert att.rebuild_occurred is True


# 11. Zero-Memory Placeholder Invariant (peak RSS must be non-zero)
def test_peak_rss_measured_never_zero(demo_instance):
    outcome = run_method_worker(
        method="sat_only_n4",
        rounds=10,
        attack_pairs=tuple((p.ciphertext0, p.ciphertext1) for p in demo_instance.pairs),
        known_pairs=demo_instance.known_pairs,
        verification_pairs=demo_instance.verification_pairs,
        global_deadline_seconds=0.2,
        use_parent_watchdog=True,
    )

    assert outcome.peak_rss_gib > 0.0
    assert isinstance(outcome.peak_rss_gib, float)


# 12. Dynamic Calculation of Numerical Differences Verdict
def test_numerical_verdict_not_hardcoded():
    # Verify that dynamic computation detects score differences and order changes
    scores_a = np.array([-10.5, -12.3, -15.0], dtype=np.float64)
    scores_b = np.array([-10.5, -12.3, -15.0], dtype=np.float64)
    cands = np.array([1, 2, 3], dtype=np.uint16)

    # Identical scores -> no changes
    order_a = np.lexsort((cands, -scores_a))
    order_b = np.lexsort((cands, -scores_b))
    mismatches = int(np.count_nonzero(cands[order_a] != cands[order_b]))
    max_diff = float(np.max(np.abs(scores_a - scores_b)))
    flag = bool(mismatches > 0 or max_diff > 1e-12)
    assert flag is False

    # Perturbed scores -> detects differences dynamically
    scores_perturbed = np.array([-10.5, -10.4, -15.0], dtype=np.float64)
    order_p = np.lexsort((cands, -scores_perturbed))
    mismatches_p = int(np.count_nonzero(cands[order_a] != cands[order_p]))
    max_diff_p = float(np.max(np.abs(scores_a - scores_perturbed)))
    flag_p = bool(mismatches_p > 0 or max_diff_p > 1e-12)
    assert flag_p is True
    assert mismatches_p > 0


# 13. Random Controls Deterministic and Distinct
def test_random_controls_generate_valid_queues():
    q128 = generate_random_tuples(128, seed=42)
    assert len(q128) == 128
    assert len(set(q128)) == 128
    for k8, k9 in q128:
        assert 0 <= k8 <= 0xFFFF
        assert 0 <= k9 <= 0xFFFF

    q1024 = generate_random_tuples(1024, seed=42)
    assert len(q1024) == 1024
    assert len(set(q1024)) == 1024
    for k8, k9 in q1024:
        assert 0 <= k8 <= 0xFFFF
        assert 0 <= k9 <= 0xFFFF

    # Deterministic replay
    assert generate_random_tuples(128, seed=42) == q128


# 14. Blocker 2: Production Comparators Reject Watchdog Bypass and Stubs
@pytest.mark.parametrize("method", sorted(EIGHT_COMPARATORS))
def test_production_comparators_reject_watchdog_bypass(demo_instance, method):
    attack_pairs = tuple((p.ciphertext0, p.ciphertext1) for p in demo_instance.pairs)
    with pytest.raises(ValueError, match="cannot bypass parent watchdog"):
        run_method_worker(
            method=method,
            rounds=10,
            attack_pairs=attack_pairs,
            known_pairs=demo_instance.known_pairs,
            verification_pairs=demo_instance.verification_pairs,
            use_parent_watchdog=False,
        )


@pytest.mark.parametrize("method", sorted(EIGHT_COMPARATORS))
def test_production_comparators_reject_solver_stub(demo_instance, method):
    attack_pairs = tuple((p.ciphertext0, p.ciphertext1) for p in demo_instance.pairs)
    with pytest.raises(ValueError, match="cannot use solver_stub"):
        run_method_worker(
            method=method,
            rounds=10,
            attack_pairs=attack_pairs,
            known_pairs=demo_instance.known_pairs,
            verification_pairs=demo_instance.verification_pairs,
            solver_stub=lambda m, a, t: SatSolveResult(False, None, 0.0, True, "solver_timeout"),
            use_parent_watchdog=True,
        )


# 15. Blocker 2: Watchdog Interruption and Progress Persistence
def test_watchdog_interruption_during_ranking(demo_instance):
    outcome = run_method_worker(
        method="test_mock_queue",
        rounds=10,
        attack_pairs=tuple((p.ciphertext0, p.ciphertext1) for p in demo_instance.pairs),
        known_pairs=demo_instance.known_pairs,
        verification_pairs=demo_instance.verification_pairs,
        mock_candidate_queue=[(0x1234, 0x5678)],
        global_deadline_seconds=3.0,
        use_parent_watchdog=True,
        test_interruption_point="ranking",
    )
    assert outcome.outcome == "global_deadline"
    assert "interrupted during ranking" in (outcome.error_message or "")
    assert outcome.candidates_tested_count == 0
    assert outcome.candidates_remaining_count == 0
    assert len(outcome.candidate_attempts) == 0


def test_watchdog_interruption_during_solving_preserves_attempts(demo_instance):
    outcome = run_method_worker(
        method="test_slow_solver",
        rounds=10,
        attack_pairs=tuple((p.ciphertext0, p.ciphertext1) for p in demo_instance.pairs),
        known_pairs=demo_instance.known_pairs,
        verification_pairs=demo_instance.verification_pairs,
        candidate_cutoff_seconds=10.0,
        global_deadline_seconds=0.8,
        use_parent_watchdog=True,
        test_interruption_point="solving",
    )
    assert outcome.outcome == "global_deadline"
    assert "interrupted during solving" in (outcome.error_message or "")
    assert outcome.candidates_tested_count == 1
    assert outcome.candidates_remaining_count == 1
    assert len(outcome.candidate_attempts) == 1
    att = outcome.candidate_attempts[0]
    assert att.candidate_k8 == 0x1234
    assert att.candidate_k9 == 0x5678
    assert att.status == "unsat"
    assert att.rebuild_occurred is False
    assert att.solve_wall_seconds > 0.0


def test_watchdog_interruption_during_rebuilding_preserves_attempts(demo_instance):
    outcome = run_method_worker(
        method="test_mock_queue",
        rounds=10,
        attack_pairs=tuple((p.ciphertext0, p.ciphertext1) for p in demo_instance.pairs),
        known_pairs=demo_instance.known_pairs,
        verification_pairs=demo_instance.verification_pairs,
        mock_candidate_queue=[(0x1234, 0x5678), (0x2345, 0x6789)],
        candidate_cutoff_seconds=0.0001,
        global_deadline_seconds=0.8,
        use_parent_watchdog=True,
        test_interruption_point="rebuilding",
    )
    assert outcome.outcome == "global_deadline"
    assert "interrupted during rebuilding" in (outcome.error_message or "")
    assert outcome.candidates_tested_count == 1
    assert outcome.candidates_remaining_count == 1
    assert len(outcome.candidate_attempts) == 1
    att = outcome.candidate_attempts[0]
    assert att.candidate_k8 == 0x1234
    assert att.candidate_k9 == 0x5678
    assert att.status == "solver_timeout"
    assert att.rebuild_occurred is False  # watchdog killed worker before rebuild completed


def test_watchdog_interruption_during_verification(demo_instance):
    call_count = 0

    def quick_sat_solver(model, assumptions, time_limit):
        nonlocal call_count
        call_count += 1
        # Return all zero solution
        sol = [False] * (model.variable_count + 1)
        return SatSolveResult(True, sol, 0.01, False, "sat")

    # In worker subprocess, solver_stub cannot be passed via subprocess JSON, so use test_slow_verification
    outcome = run_method_worker(
        method="test_slow_verification",
        rounds=10,
        attack_pairs=tuple((p.ciphertext0, p.ciphertext1) for p in demo_instance.pairs),
        known_pairs=demo_instance.known_pairs,
        verification_pairs=demo_instance.verification_pairs,
        global_deadline_seconds=0.8,
        use_parent_watchdog=True,
        test_interruption_point="verification",
    )
    assert outcome.outcome == "global_deadline"
    assert "interrupted during verification" in (outcome.error_message or "")


# 16. Parent Process Watchdog and Process-Tree Termination on Hanging Ranking Child
def test_parent_watchdog_terminates_hanging_ranking_child(demo_instance):
    outcome = run_method_worker(
        method="test_hanging_ranking_child",
        rounds=10,
        attack_pairs=tuple((p.ciphertext0, p.ciphertext1) for p in demo_instance.pairs),
        known_pairs=demo_instance.known_pairs,
        verification_pairs=demo_instance.verification_pairs,
        global_deadline_seconds=3.0,
        use_parent_watchdog=True,
    )

    assert outcome.outcome == "global_deadline"
    assert "interrupted during ranking" in (outcome.error_message or "")
    assert outcome.candidates_tested_count == 0
    assert len(outcome.child_pids) > 0

    # Wait briefly for OS to complete signal dispatch and reaping
    time.sleep(0.1)

    for pid in outcome.child_pids:
        # Confirm process is terminated via kill(pid, 0) with a polling deadline
        deadline = time.monotonic() + 2.0
        alive = True
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
                alive = True
                time.sleep(0.05)
            except ProcessLookupError:
                alive = False
                break
            except PermissionError:
                alive = True
                break
        assert not alive, f"Descendant process {pid} is still alive after watchdog termination!"

        # On Linux with /proc, verify process entry is absent or at most a reaped zombie
        proc_status = Path(f"/proc/{pid}/status")
        if proc_status.exists():
            try:
                content = proc_status.read_text()
                assert "State:\tZ" in content or "zombie" in content.lower(), (
                    f"Descendant process {pid} still active in /proc: {content}"
                )
            except (ProcessLookupError, FileNotFoundError, PermissionError):
                pass

        # If ps is permitted in the environment, also verify it is not listed as active
        try:
            ps_res = subprocess.run(["ps", "-p", str(pid)], capture_output=True, text=True)
            if ps_res.returncode == 0:
                lines = [line.strip() for line in ps_res.stdout.strip().splitlines() if line.strip()]
                assert len(lines) <= 1, f"Child PID {pid} still visible in ps output: {ps_res.stdout}"
        except (PermissionError, FileNotFoundError):
            # In sandboxes restricting external ps execution, os.kill(pid, 0) raising
            # ProcessLookupError is the authoritative POSIX process-table check.
            pass


# 17. Durable Progress Journaling Captures In-Flight Attempt and Untested Count
def test_durable_progress_journaling_recovers_in_flight_candidate(demo_instance):
    outcome = run_method_worker(
        method="test_slow_solver",
        rounds=10,
        attack_pairs=tuple((p.ciphertext0, p.ciphertext1) for p in demo_instance.pairs),
        known_pairs=demo_instance.known_pairs,
        verification_pairs=demo_instance.verification_pairs,
        candidate_cutoff_seconds=10.0,
        global_deadline_seconds=0.8,
        use_parent_watchdog=True,
        test_interruption_point="solving",
    )

    assert outcome.outcome == "global_deadline"
    assert "interrupted during solving" in (outcome.error_message or "")

    # Queue preservation
    assert len(outcome.full_queue) == 3
    assert outcome.full_queue == ((0x1234, 0x5678), (0x2345, 0x6789), (0x3456, 0x789A))

    # Completed attempt 0: unsat
    assert outcome.candidates_tested_count == 1
    assert len(outcome.candidate_attempts) == 1
    att0 = outcome.candidate_attempts[0]
    assert att0.candidate_k8 == 0x1234
    assert att0.candidate_k9 == 0x5678
    assert att0.status == "unsat"
    assert att0.rebuild_occurred is False
    assert att0.solve_wall_seconds > 0.0

    # In-flight attempt 1: interrupted candidate
    assert outcome.in_flight_attempt is not None
    in_flight = outcome.in_flight_attempt
    assert in_flight.candidate_k8 == 0x2345
    assert in_flight.candidate_k9 == 0x6789
    assert in_flight.status == "in_flight"
    assert in_flight.solve_wall_seconds > 0.0
    assert in_flight.rebuild_occurred is False

    # Untested count: candidate 2 was never reached
    assert outcome.candidates_remaining_count == 1
    assert outcome.untested_remaining_count == 1

    # all_attempts convenience property includes completed + in_flight
    assert len(outcome.all_attempts) == 2
    assert outcome.all_attempts[0].status == "unsat"
    assert outcome.all_attempts[1].status == "in_flight"


# 18. Taxonomy: SAT-only solver_timeout vs proven_unsat vs global_deadline
def test_taxonomy_sat_only_solver_timeout_vs_proven_unsat_vs_global_deadline(demo_instance):
    # 1. solver_timeout when solver reaches soft cutoff before parent deadline
    def timeout_stub(model, assumptions, time_limit):
        return SatSolveResult(False, None, 0.01, True, "solver_timeout")

    outcome_timeout = run_method_worker(
        method="test_sat_only",
        rounds=10,
        attack_pairs=tuple((p.ciphertext0, p.ciphertext1) for p in demo_instance.pairs),
        known_pairs=demo_instance.known_pairs,
        verification_pairs=demo_instance.verification_pairs,
        solver_stub=timeout_stub,
        global_deadline_seconds=30.0,
        use_parent_watchdog=False,
    )
    assert outcome_timeout.outcome == "solver_timeout"
    assert outcome_timeout.verified_on_fresh_pairs is False
    assert outcome_timeout.solving_wall_seconds > 0.0

    # 2. proven_unsat when formula is definitively unsatisfiable
    def unsat_stub(model, assumptions, time_limit):
        return SatSolveResult(False, None, 0.01, False, "unsat")
    outcome_unsat = run_method_worker(
        method="test_sat_only",
        rounds=10,
        attack_pairs=tuple((p.ciphertext0, p.ciphertext1) for p in demo_instance.pairs),
        known_pairs=demo_instance.known_pairs,
        verification_pairs=demo_instance.verification_pairs,
        solver_stub=unsat_stub,
        global_deadline_seconds=30.0,
        use_parent_watchdog=False,
    )
    assert outcome_unsat.outcome == "proven_unsat"
    assert outcome_unsat.verified_on_fresh_pairs is False

    # 3. global_deadline when parent watchdog expires during solving
    outcome_deadline = run_method_worker(
        method="test_sat_only",
        rounds=10,
        attack_pairs=tuple((p.ciphertext0, p.ciphertext1) for p in demo_instance.pairs),
        known_pairs=demo_instance.known_pairs,
        verification_pairs=demo_instance.verification_pairs,
        global_deadline_seconds=3.0,
        use_parent_watchdog=True,
        test_interruption_point="solving",
    )
    assert outcome_deadline.outcome == "global_deadline"
    assert "interrupted during solving" in (outcome_deadline.error_message or "")
    assert outcome_deadline.verified_on_fresh_pairs is False


# 19. Soft Cutoff vs Hard Watchdog Semantics and Rebuild Triggering
def test_soft_cutoff_vs_hard_watchdog_semantics(demo_instance):
    received_time_limits: list[float] = []

    def timeout_recording_stub(model, assumptions, time_limit):
        received_time_limits.append(time_limit)
        return SatSolveResult(False, None, 0.01, True, "solver_timeout")

    outcome = run_method_worker(
        method="test_mock_queue",
        rounds=10,
        attack_pairs=tuple((p.ciphertext0, p.ciphertext1) for p in demo_instance.pairs),
        known_pairs=demo_instance.known_pairs,
        verification_pairs=demo_instance.verification_pairs,
        mock_candidate_queue=[(0x1234, 0x5678)],
        candidate_cutoff_seconds=45.0,
        candidate_cutoff_soft_seconds=2.5,
        global_deadline_seconds=30.0,
        use_parent_watchdog=False,
        solver_stub=timeout_recording_stub,
    )

    # Verify soft cutoff was passed down to solver
    assert len(received_time_limits) == 1
    assert received_time_limits[0] == 2.5

    # Verify solver_timeout triggered rebuild
    assert len(outcome.candidate_attempts) == 1
    att = outcome.candidate_attempts[0]
    assert att.status == "solver_timeout"
    assert att.rebuild_occurred is True

    # Verify subprocess worker preserves candidate_cutoff_soft_seconds and rebuilds
    outcome_sub = run_method_worker(
        method="test_slow_rebuild",
        rounds=10,
        attack_pairs=tuple((p.ciphertext0, p.ciphertext1) for p in demo_instance.pairs),
        known_pairs=demo_instance.known_pairs,
        verification_pairs=demo_instance.verification_pairs,
        candidate_cutoff_soft_seconds=0.05,
        global_deadline_seconds=5.0,
        use_parent_watchdog=True,
    )
    assert len(outcome_sub.candidate_attempts) >= 1
    assert outcome_sub.candidate_attempts[0].status == "solver_timeout"
    assert outcome_sub.candidate_attempts[0].rebuild_occurred is False
    # The watchdog interrupts the deliberate rebuild sleep before a new solver exists.


# 20. Faithful B1 Hybrid Queue Generation
def test_faithful_b1_hybrid_queue_generation(demo_instance):
    """Verify faithful_b1_hybrid queue generator produces 1,024 chronological tuples."""
    class MockDdtTable:
        rounds = 7

        def log_score_row(self, left: int) -> np.ndarray:
            return np.zeros(1 << 16, dtype=np.float64)

    attack_pairs = tuple((p.ciphertext0, p.ciphertext1) for p in demo_instance.pairs[:16])
    queue, diag = generate_faithful_b1_hybrid_queue(
        attack_pairs,
        b1_model=SimonPairGroupRanker(),
        table_7r=MockDdtTable(),
        screen_pairs=16,
        full_pairs=16,
        parent_count=8,
        children_per_parent=128,
        return_diagnostics=True,
    )
    assert len(queue) == 1024
    assert diag["candidate_tuples_produced"] == 1024
    for cand_k8, cand_k9 in queue[:10]:
        assert 0 <= cand_k8 <= 0xFFFF
        assert 0 <= cand_k9 <= 0xFFFF


def test_production_b1_rejects_missing_checkpoint(demo_instance):
    attack_pairs = tuple((p.ciphertext0, p.ciphertext1) for p in demo_instance.pairs)
    with pytest.raises(ValueError, match="requires a checkpoint path and SHA-256"):
        run_method_worker(
            method="faithful_b1_hybrid",
            rounds=10,
            attack_pairs=attack_pairs,
            known_pairs=demo_instance.known_pairs,
            verification_pairs=demo_instance.verification_pairs,
        )


def test_b1_checkpoint_format_and_hash_are_enforced(tmp_path, monkeypatch, demo_instance):
    checkpoint = tmp_path / "b1.pt"
    save_simon_checkpoint(SimonPairGroupRanker(), checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    observed = []

    def fake_generate(*, pairs, b1_model, **kwargs):
        observed.append((len(pairs), b1_model.training))
        return [(0x1234, 0x5678)]

    monkeypatch.setattr(method_worker_module, "generate_b1_hybrid_8parent_queue", fake_generate)
    attack_pairs = tuple((p.ciphertext0, p.ciphertext1) for p in demo_instance.pairs)
    with pytest.raises(ValueError, match="completed-training provenance"):
        generate_faithful_b1_hybrid_queue(
            attack_pairs,
            checkpoint_path=checkpoint,
            checkpoint_sha256=digest,
            table_7r=object(),
        )
    save_simon_checkpoint(
        SimonPairGroupRanker(), checkpoint, metadata=_synthetic_checkpoint_metadata()
    )
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert generate_faithful_b1_hybrid_queue(
        attack_pairs,
        checkpoint_path=checkpoint,
        checkpoint_sha256=digest,
        table_7r=object(),
    ) == [(0x1234, 0x5678)]
    assert observed == [(32, False)]
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        generate_faithful_b1_hybrid_queue(
            attack_pairs,
            checkpoint_path=checkpoint,
            checkpoint_sha256="0" * 64,
            table_7r=object(),
        )
    with pytest.raises(ValueError, match="requires a checkpoint"):
        generate_faithful_b1_hybrid_queue(attack_pairs, table_7r=object())


def test_b1_dispatch_reaches_ranking_under_parent_watchdog(tmp_path, demo_instance):
    checkpoint = tmp_path / "b1.pt"
    save_simon_checkpoint(
        SimonPairGroupRanker(), checkpoint, metadata=_synthetic_checkpoint_metadata()
    )
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    attack_pairs = tuple((p.ciphertext0, p.ciphertext1) for p in demo_instance.pairs)
    outcome = run_method_worker(
        method="faithful_b1_hybrid",
        rounds=10,
        attack_pairs=attack_pairs,
        known_pairs=demo_instance.known_pairs,
        verification_pairs=demo_instance.verification_pairs,
        b1_checkpoint_path=checkpoint,
        b1_checkpoint_sha256=digest,
        global_deadline_seconds=2.0,
    )
    assert outcome.outcome == "global_deadline"


def test_b1_direct_dispatch_passes_only_pinned_checkpoint(monkeypatch, demo_instance):
    observed = []

    def fake_b1_queue(pairs, *, checkpoint_path, checkpoint_sha256):
        observed.append((len(pairs), checkpoint_path, checkpoint_sha256))
        return []

    monkeypatch.setattr(method_worker_module, "generate_faithful_b1_hybrid_queue", fake_b1_queue)
    attack_pairs = tuple((p.ciphertext0, p.ciphertext1) for p in demo_instance.pairs)
    outcome = method_worker_module._run_method_worker_direct(
        method="faithful_b1_hybrid",
        rounds=10,
        attack_pairs=attack_pairs,
        known_pairs=demo_instance.known_pairs,
        verification_pairs=demo_instance.verification_pairs,
        b1_checkpoint_path="/tmp/pinned-b1.pt",
        b1_checkpoint_sha256="a" * 64,
    )
    assert observed == [(32, "/tmp/pinned-b1.pt", "a" * 64)]
    assert outcome.outcome == "exception"
    assert "empty candidate queue" in (outcome.error_message or "")


def test_compared_method_rejects_swapped_sat_ciphertexts(demo_instance):
    attack_pairs = tuple((p.ciphertext0, p.ciphertext1) for p in demo_instance.pairs)
    known_pairs = list(demo_instance.known_pairs)
    known_pairs[0], known_pairs[2] = known_pairs[2], known_pairs[0]
    with pytest.raises(ValueError, match="must match SAT observations in order"):
        run_method_worker(
            method="sat_only_n4",
            rounds=10,
            attack_pairs=attack_pairs,
            known_pairs=known_pairs,
            verification_pairs=demo_instance.verification_pairs,
        )


def test_compared_method_rejects_wrong_plaintext_difference(demo_instance):
    attack_pairs = tuple((p.ciphertext0, p.ciphertext1) for p in demo_instance.pairs)
    known_pairs = list(demo_instance.known_pairs)
    pt, ct = known_pairs[1]
    known_pairs[1] = ((pt[0], pt[1] ^ 1), ct)
    with pytest.raises(ValueError, match="wrong input difference"):
        run_method_worker(
            method="sat_only_n4",
            rounds=10,
            attack_pairs=attack_pairs,
            known_pairs=known_pairs,
            verification_pairs=demo_instance.verification_pairs,
        )


def test_random_control_rejects_seed_override(demo_instance):
    attack_pairs = tuple((p.ciphertext0, p.ciphertext1) for p in demo_instance.pairs)
    with pytest.raises(ValueError, match="manifest-derived seed"):
        run_method_worker(
            method="random_control_128",
            rounds=10,
            attack_pairs=attack_pairs,
            known_pairs=demo_instance.known_pairs,
            verification_pairs=demo_instance.verification_pairs,
            random_seed=123,
        )


def test_ranking_timeout_is_not_reported_as_exception(monkeypatch, demo_instance):
    def timed_out_ranker(*args, **kwargs):
        raise TimeoutError("test ranking timeout")

    monkeypatch.setattr(method_worker_module, "generate_hybrid_8parent_queue", timed_out_ranker)
    outcome = method_worker_module._run_method_worker_direct(
        method="neural_classical_hybrid_8parent",
        rounds=10,
        attack_pairs=tuple((p.ciphertext0, p.ciphertext1) for p in demo_instance.pairs),
        known_pairs=demo_instance.known_pairs,
        verification_pairs=demo_instance.verification_pairs,
    )
    assert outcome.outcome == "ranking_timeout"
    assert outcome.candidates_tested_count == 0


def test_timeout_only_queue_is_unresolved_not_exhausted(demo_instance):
    def timed_out_solver(model, assumptions, time_limit):
        return SatSolveResult(False, None, 0.01, True, "solver_timeout")

    outcome = run_method_worker(
        method="test_mock_queue",
        rounds=10,
        attack_pairs=tuple((p.ciphertext0, p.ciphertext1) for p in demo_instance.pairs),
        known_pairs=demo_instance.known_pairs,
        verification_pairs=demo_instance.verification_pairs,
        mock_candidate_queue=[(0x1234, 0x5678)],
        solver_stub=timed_out_solver,
        use_parent_watchdog=False,
    )
    assert outcome.outcome == "candidate_unresolved"
    assert outcome.candidate_attempts[0].status == "solver_timeout"


def test_full_pairs_used_telemetry_on_queue_method(demo_instance):
    attack_pairs = tuple((p.ciphertext0, p.ciphertext1) for p in demo_instance.pairs)
    outcome = run_method_worker(
        method="test_mock_queue",
        rounds=10,
        attack_pairs=attack_pairs,
        known_pairs=demo_instance.known_pairs,
        verification_pairs=demo_instance.verification_pairs,
        mock_candidate_queue=[(1, 2)],
        solver_stub=lambda *_args: SatSolveResult(False, None, 0.0, False, "unsat"),
        use_parent_watchdog=False,
    )
    assert outcome.outcome == "queue_exhausted"
    assert outcome.full_pairs_used == len(attack_pairs)


def test_interrupted_sat_only_telemetry_records_solving_time_and_pairs(demo_instance):
    attack_pairs = tuple((p.ciphertext0, p.ciphertext1) for p in demo_instance.pairs)
    # 1. Test in-flight solving time recovery during interruption
    outcome_interrupted = run_method_worker(
        method="test_sat_only",
        rounds=10,
        attack_pairs=attack_pairs,
        known_pairs=demo_instance.known_pairs,
        verification_pairs=demo_instance.verification_pairs,
        global_deadline_seconds=2.0,
        use_parent_watchdog=True,
        test_interruption_point="solving",
    )
    assert outcome_interrupted.outcome == "global_deadline"
    assert outcome_interrupted.full_pairs_used == 32
    assert outcome_interrupted.solving_wall_seconds > 0.0
    assert outcome_interrupted.solving_wall_seconds <= 2.5

    # Before a phase is recorded the selected input count remains unknown.
    outcome_n4 = run_method_worker(
        method="sat_only_n4",
        rounds=10,
        attack_pairs=attack_pairs,
        known_pairs=demo_instance.known_pairs,
        verification_pairs=demo_instance.verification_pairs,
        global_deadline_seconds=0.001,
        use_parent_watchdog=True,
    )
    assert outcome_n4.outcome == "global_deadline"
    assert outcome_n4.full_pairs_used is None


def test_unguided_phase_duration_excludes_parent_and_child_startup(tmp_path, monkeypatch):
    journal = tmp_path / "progress.jsonl"
    journal.write_text(json.dumps({
        "event": "phase", "phase": "solving", "solving_kind": "unguided",
        "started_at_monotonic": 109.0, "full_pairs_used": 64,
    }) + "\n", encoding="utf-8")
    monkeypatch.setattr(method_worker_module.time, "perf_counter", lambda: 112.0)
    # Parent began at 100; nine seconds of startup must not become SAT time.
    rec = method_worker_module._read_progress_file(journal, parent_elapsed=12.0)
    assert rec.journal_error is None
    assert rec.solving_wall_seconds == 3.0
    assert rec.full_pairs_used == 64


def test_completed_unguided_phase_is_not_counted_until_parent_exit(tmp_path, monkeypatch):
    journal = tmp_path / "progress.jsonl"
    journal.write_text("\n".join(json.dumps(ev) for ev in [
        {"event": "phase", "phase": "solving", "solving_kind": "unguided",
         "started_at_monotonic": 100.0, "full_pairs_used": 2},
        {"event": "completed", "outcome": "solver_timeout", "solving_wall_seconds": 4.0},
    ]) + "\n", encoding="utf-8")
    monkeypatch.setattr(method_worker_module.time, "perf_counter", lambda: 130.0)
    rec = method_worker_module._read_progress_file(journal, parent_elapsed=30.0)
    assert rec.journal_error is None
    assert rec.solving_wall_seconds == 4.0


def test_candidate_phase_does_not_use_unguided_recovery_path(tmp_path):
    journal = tmp_path / "progress.jsonl"
    journal.write_text("\n".join(json.dumps(ev) for ev in [
        {"event": "phase", "phase": "solving", "candidate_index": 0},
        {"event": "attempt", "candidate_k8": 1, "candidate_k9": 2,
         "solve_wall_seconds": 3.0, "status": "unsat", "rebuild_occurred": False,
         "current_solving_wall_seconds": 4.0},
    ]) + "\n", encoding="utf-8")
    rec = method_worker_module._read_progress_file(journal, parent_elapsed=60.0)
    assert rec.journal_error is None
    assert rec.solving_wall_seconds == 4.0


def test_interrupted_candidate_duration_excludes_startup_and_journal_read(tmp_path, monkeypatch):
    journal = tmp_path / "progress.jsonl"
    journal.write_text("\n".join(json.dumps(ev) for ev in [
        {"event": "attempt", "candidate_k8": 1, "candidate_k9": 2,
         "solve_wall_seconds": 2.0, "status": "unsat", "rebuild_occurred": False,
         "current_solving_wall_seconds": 3.0},
        {"event": "attempt_start", "candidate_k8": 3, "candidate_k9": 4,
         "started_at_monotonic": 109.0, "elapsed_so_far": 2.0},
    ]) + "\n", encoding="utf-8")
    monkeypatch.setattr(method_worker_module.time, "perf_counter", lambda: 150.0)
    rec = method_worker_module._read_progress_file(
        journal, parent_elapsed=12.0, stopped_at_monotonic=112.0,
    )
    assert rec.journal_error is None
    assert rec.in_flight_attempt.solve_wall_seconds == 3.0
    assert rec.solving_wall_seconds == 6.0


@pytest.mark.parametrize(
    "timestamp", [None, False, True, "99.0", -1.0, 0.0, float("nan"), float("inf"), 101.0, 10**400],
)
def test_invalid_unguided_phase_clock_is_flagged(tmp_path, monkeypatch, timestamp):
    journal = tmp_path / "progress.jsonl"
    journal.write_text(json.dumps({
        "event": "phase", "phase": "solving", "solving_kind": "unguided",
        "started_at_monotonic": timestamp,
    }) + "\n", encoding="utf-8")
    monkeypatch.setattr(method_worker_module.time, "perf_counter", lambda: 100.0)
    rec = method_worker_module._read_progress_file(journal, parent_elapsed=10.0)
    assert rec.journal_error is not None
    assert "Invalid in-flight solving phase" in rec.journal_error


def test_interrupted_unguided_verification_retains_completed_solve_and_journals_verification(tmp_path):
    journal = tmp_path / "progress.jsonl"
    journal.write_text("\n".join(json.dumps(ev) for ev in [
        {"event": "phase", "phase": "solving", "solving_kind": "unguided",
         "started_at_monotonic": 100.0, "full_pairs_used": 2},
        {"event": "solve_complete", "solving_wall_seconds": 4.5},
        {"event": "phase", "phase": "verification", "started_at_monotonic": 105.0},
    ]) + "\n", encoding="utf-8")
    rec = method_worker_module._read_progress_file(
        journal, parent_elapsed=10.0, stopped_at_monotonic=107.0,
    )
    assert rec.journal_error is None
    assert rec.last_phase == "verification"
    assert rec.solving_wall_seconds == 4.5
    assert rec.verification_wall_seconds == 2.0


def test_interrupted_guided_verification_retains_candidate_solve_and_journals_verification(tmp_path):
    journal = tmp_path / "progress.jsonl"
    journal.write_text("\n".join(json.dumps(ev) for ev in [
        {"event": "queue", "total_queue_count": 2, "full_queue": [[1, 2], [3, 4]], "queue_sample": [[1, 2]]},
        {"event": "attempt_start", "candidate_index": 0, "candidate_k8": 1, "candidate_k9": 2,
         "started_at_monotonic": 100.0},
        {"event": "phase", "phase": "solving", "candidate_index": 0, "candidate_k8": 1, "candidate_k9": 2,
         "started_at_monotonic": 100.0},
        {"event": "candidate_solve_complete", "candidate_index": 0, "candidate_k8": 1, "candidate_k9": 2,
         "solve_wall_seconds": 0.5, "status": "sat"},
        {"event": "phase", "phase": "verification", "candidate_index": 0, "candidate_k8": 1, "candidate_k9": 2,
         "started_at_monotonic": 101.0},
    ]) + "\n", encoding="utf-8")
    rec = method_worker_module._read_progress_file(
        journal, parent_elapsed=5.0, stopped_at_monotonic=104.0,
    )
    assert rec.journal_error is None
    assert rec.last_phase == "verification"
    assert rec.in_flight_attempt is not None
    assert rec.in_flight_attempt.solve_wall_seconds == 0.5
    assert rec.solving_wall_seconds == 0.5
    assert rec.verification_wall_seconds == 3.0


def test_interrupted_initial_formula_construction_journals_phase_and_duration(tmp_path):
    journal = tmp_path / "progress.jsonl"
    journal.write_text("\n".join(json.dumps(ev) for ev in [
        {"event": "queue", "total_queue_count": 2, "full_queue": [[1, 2], [3, 4]], "queue_sample": [[1, 2]]},
        {"event": "phase", "phase": "initial_formula_construction", "started_at_monotonic": 100.0},
    ]) + "\n", encoding="utf-8")
    rec = method_worker_module._read_progress_file(
        journal, parent_elapsed=4.0, stopped_at_monotonic=103.5,
    )
    assert rec.journal_error is None
    assert rec.last_phase == "initial_formula_construction"
    assert rec.solving_wall_seconds == 3.5
    assert rec.candidates_tested_count == 0
    assert rec.candidates_remaining_count == 2


def test_interrupted_rebuilding_journals_phase_and_duration(tmp_path):
    journal = tmp_path / "progress.jsonl"
    journal.write_text("\n".join(json.dumps(ev) for ev in [
        {"event": "queue", "total_queue_count": 2, "full_queue": [[1, 2], [3, 4]], "queue_sample": [[1, 2]]},
        {"event": "attempt_start", "candidate_index": 0, "candidate_k8": 1, "candidate_k9": 2, "started_at_monotonic": 100.0},
        {"event": "candidate_solve_complete", "candidate_index": 0, "candidate_k8": 1, "candidate_k9": 2, "solve_wall_seconds": 0.5, "status": "sat"},
        {"event": "attempt", "candidate_k8": 1, "candidate_k9": 2, "solve_wall_seconds": 0.5, "status": "verification_failure",
         "rebuild_occurred": False, "current_solving_wall_seconds": 0.5, "current_verification_wall_seconds": 0.1},
        {"event": "phase", "phase": "rebuilding", "candidate_index": 0, "candidate_k8": 1, "candidate_k9": 2, "started_at_monotonic": 101.0},
    ]) + "\n", encoding="utf-8")
    rec = method_worker_module._read_progress_file(
        journal, parent_elapsed=5.0, stopped_at_monotonic=103.0,
    )
    assert rec.journal_error is None
    assert rec.last_phase == "rebuilding"
    assert len(rec.attempts) == 1
    assert rec.attempts[0].rebuild_occurred is False
    assert rec.solving_wall_seconds == 2.5  # 0.5 solve + 2.0 rebuild
    assert rec.verification_wall_seconds == 0.1


def test_multiple_verification_failures_and_rebuilds_component_accounting(demo_instance):
    outcome = run_method_worker(
        method="test_multiple_verification_failures",
        rounds=10,
        attack_pairs=tuple((p.ciphertext0, p.ciphertext1) for p in demo_instance.pairs),
        known_pairs=demo_instance.known_pairs,
        verification_pairs=demo_instance.verification_pairs,
        mock_candidate_queue=[(0x1111, 0x2222), (0x3333, 0x4444), (0x5555, 0x6666)],
        use_parent_watchdog=False,
    )
    assert outcome.outcome == "verified_recovery"
    assert outcome.verified_on_fresh_pairs is True
    assert outcome.candidates_tested_count == 3
    assert len(outcome.candidate_attempts) == 3
    assert outcome.candidate_attempts[0].status == "verification_failure"
    assert outcome.candidate_attempts[0].rebuild_occurred is True
    assert outcome.candidate_attempts[1].status == "verification_failure"
    assert outcome.candidate_attempts[1].rebuild_occurred is True
    assert outcome.candidate_attempts[2].status == "sat"
    assert outcome.candidate_attempts[2].rebuild_occurred is False
    assert outcome.verification_wall_seconds > 0.0
    assert outcome.solving_wall_seconds > 0.0
    assert outcome.solving_wall_seconds + outcome.verification_wall_seconds <= outcome.complete_online_wall_seconds + 0.05
