"""Unit tests for the SIMON R=10 SAT-only 300s control runner.

Verifies:
- 64-pair opened demo instance cohort specification
- Formula clause and variable statistics for N=4 and N=128
- Worker solve behavior and status mapping
- Parent watchdog timeout handling
- CLI parser options and defaults
- Resume logic skipping already completed (instance, method) pairs
- Record schema completeness for all required fields
- Rigorous scientific notes in summary
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import scripts.benchmark_simon_r10_sat_only_300s_controls as runner
from neural_sat_speck.sat.model import SatSolveResult


def test_demo_instances_specification() -> None:
    """Verify demo instances are generated with 64 differential pairs (128 observations)."""
    instances = runner.generate_demo_instances()
    assert len(instances) == 8
    for inst in instances:
        assert inst.rounds == 10
        assert inst.split_role == "demo"
        assert len(inst.pairs) == 64
        assert len(inst.known_pairs) == 128
        assert len(inst.verification_pairs) == 2


def test_formula_statistics_exact_match() -> None:
    """Verify exact formula clause and variable counts for 4 and 128 observations."""
    instances = runner.generate_demo_instances()
    inst = instances[0]

    # 4 observations (2 pairs)
    m4 = runner.SimonSatModel.from_known_pairs(inst.known_pairs[:4], rounds=10)
    assert m4.variable_count == 1856
    assert m4.cnf_clause_count == 2176
    assert m4.xor_clause_count == 1024
    assert m4.cnf_clause_count + m4.xor_clause_count == 3200

    # 128 observations (64 pairs)
    m128 = runner.SimonSatModel.from_known_pairs(inst.known_pairs[:128], rounds=10)
    assert m128.variable_count == 45504
    assert m128.cnf_clause_count == 69632
    assert m128.xor_clause_count == 20864
    assert m128.cnf_clause_count + m128.xor_clause_count == 90496


def test_worker_classifies_300s_timeout_correctly(monkeypatch) -> None:
    """Verify worker passes 300.0s time limit and classifies solver timeout."""
    passed_time_limits = []

    class MockModel:
        variable_count = 1856
        cnf_clause_count = 2176
        xor_clause_count = 1024

        @classmethod
        def from_known_pairs(cls, pairs, rounds):
            assert len(pairs) == 4
            assert rounds == 10
            return cls()

        def solve(self, assumptions=(), time_limit=300.0):
            passed_time_limits.append(time_limit)
            return SatSolveResult(
                satisfiable=False,
                solution=None,
                elapsed_seconds=300.1,
                timed_out=True,
                status="solver_timeout",
            )

    monkeypatch.setattr(runner, "SimonSatModel", MockModel)
    inst = runner.generate_demo_instances()[0]
    payload = {
        "rounds": 10,
        "method": "sat_only_n4",
        "obs_count": 4,
        "time_limit": 300.0,
        "known_pairs": [[list(p[0]), list(p[1])] for p in inst.known_pairs[:4]],
        "verification_pairs": [
            [list(p[0]), list(p[1])] for p in inst.verification_pairs
        ],
    }
    result = runner.worker_solve(payload)
    assert passed_time_limits == [300.0]
    assert result["outcome"] == "solver_timeout"
    assert result["solver_status"] == "solver_timeout"
    assert result["recovered_master_key"] is None
    assert result["verified_on_fresh_pairs"] is False
    assert result["variables"] == 1856
    assert result["total_clauses"] == 3200
    assert result["peak_rss_gib"] > 0.0


def test_worker_classifies_verified_recovery(monkeypatch) -> None:
    """Verify worker handles successful solve, key extraction, and fresh verification."""
    inst = runner.generate_demo_instances()[0]

    class MockModel:
        variable_count = 1856
        cnf_clause_count = 2176
        xor_clause_count = 1024

        @classmethod
        def from_known_pairs(cls, pairs, rounds):
            return cls()

        def solve(self, assumptions=(), time_limit=300.0):
            return SatSolveResult(
                satisfiable=True,
                solution=(True,) * 1857,
                elapsed_seconds=12.5,
                timed_out=False,
                status="sat",
            )

        def master_key_from_solution(self, solution):
            return inst.master_key

    monkeypatch.setattr(runner, "SimonSatModel", MockModel)
    payload = {
        "rounds": 10,
        "method": "sat_only_n4",
        "obs_count": 4,
        "time_limit": 300.0,
        "known_pairs": [[list(p[0]), list(p[1])] for p in inst.known_pairs[:4]],
        "verification_pairs": [
            [list(p[0]), list(p[1])] for p in inst.verification_pairs
        ],
    }
    result = runner.worker_solve(payload)
    assert result["outcome"] == "verified_recovery"
    assert result["solver_status"] == "sat"
    assert result["recovered_master_key"] == list(inst.master_key)
    assert result["verified_on_fresh_pairs"] is True


def test_parent_watchdog_timeout(monkeypatch) -> None:
    """Verify parent watchdog timeout triggers parent_wall_timeout outcome."""
    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 305.0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    inst = runner.generate_demo_instances()[0]
    rec = runner.evaluate_control(
        inst, 0, "sat_only_n4", 4, time_limit=300.0, watchdog_seconds=305.0
    )
    assert rec["outcome"] == "parent_wall_timeout"
    assert rec["verified_recovery_within_cutoff"] is False
    assert "Subprocess timed out" in (rec["stderr"] or "")
    assert rec["solve_wall_seconds"] is None
    assert rec["peak_rss_gib"] is None
    assert rec["variables"] is None


def test_cli_parser_defaults_and_options() -> None:
    """Verify CLI argument defaults for time-limit, watchdog, resume, etc."""
    help_result = subprocess.run(
        [sys.executable, str(Path(runner.__file__).resolve()), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    source = help_result.stdout
    assert "--time-limit" in source
    assert "--watchdog-seconds" in source
    assert "--resume" in source
    assert "--max-instances" in source
    assert "--methods" in source


def test_mandatory_record_fields(monkeypatch) -> None:
    """Verify all 18 mandatory fields specified in Task 1 are present in evaluate_control record."""
    def fake_run(cmd, **kwargs):
        worker_output = {
            "outcome": "solver_timeout",
            "solver_status": "solver_timeout",
            "satisfiable": False,
            "timed_out": True,
            "verified_on_fresh_pairs": False,
            "recovered_master_key": None,
            "model_build_wall_seconds": 0.05,
            "solve_wall_seconds": 300.2,
            "verification_wall_seconds": 0.0,
            "worker_wall_seconds": 300.3,
            "variables": 1856,
            "cnf_clauses": 2176,
            "xor_clauses": 1024,
            "total_clauses": 3200,
            "peak_rss_gib": 0.12,
        }
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = json.dumps(worker_output)
        mock_proc.stderr = ""
        return mock_proc

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    inst = runner.generate_demo_instances()[0]
    rec = runner.evaluate_control(inst, 0, "sat_only_n4", 4)

    required_fields = [
        "instance_id",
        "instance_index",
        "master_key",
        "method",
        "observations",
        "outcome",
        "solver_status",
        "model_build_wall_seconds",
        "solve_wall_seconds",
        "verification_wall_seconds",
        "online_wall_seconds",
        "variables",
        "cnf_clauses",
        "xor_clauses",
        "total_clauses",
        "peak_rss_gib",
        "truth_match_posthoc",
        "verified_recovery_within_cutoff",
        "stderr",
    ]
    for field in required_fields:
        assert field in rec, f"Missing required field: {field}"

    assert rec["instance_id"] == inst.instance_id
    assert rec["instance_index"] == 0
    assert rec["master_key"] == list(inst.master_key)
    assert rec["method"] == "sat_only_n4"
    assert rec["observations"] == 4


def test_resume_logic_skips_completed_runs(tmp_path: Path, monkeypatch) -> None:
    """Verify resume logic loads partial results and skips already completed (key, method) pairs."""
    output_json = tmp_path / "test_resume.json"
    instance = runner.generate_demo_instances()[0]

    # Pre-populate artifact with instance 0, sat_only_n4 completed
    partial_record = {
        "instance_id": instance.instance_id,
        "instance_index": 0,
        "master_key": list(instance.master_key),
        "method": "sat_only_n4",
        "observations": 4,
        "differential_pairs": 2,
        "outcome": "solver_timeout",
        "solver_status": "solver_timeout",
        "model_build_wall_seconds": 0.01,
        "solve_wall_seconds": 300.1,
        "verification_wall_seconds": 0.0,
        "online_wall_seconds": 300.2,
        "variables": 1856,
        "cnf_clauses": 2176,
        "xor_clauses": 1024,
        "total_clauses": 3200,
        "peak_rss_gib": 0.10,
        "truth_match_posthoc": False,
        "verified_recovery_within_cutoff": False,
        "stderr": None,
    }
    initial_artifact = {
        "schema": "simon_r10_sat_only_300s_controls.v1",
        "seed": runner.CLI_DEMO_SEED,
        "rounds": runner.ROUNDS,
        "pair_count": runner.PAIR_COUNT,
        "instance_count": 1,
        "methods": ["sat_only_n4", "sat_only_n128"],
        "time_limit_seconds": runner.DEFAULT_TIME_LIMIT_SECONDS,
        "watchdog_seconds": runner.DEFAULT_WATCHDOG_SECONDS,
        "records": [partial_record],
        "started_at_utc": "2026-09-28T10:00:00Z",
        "status": "IN_PROGRESS",
    }
    output_json.write_text(json.dumps(initial_artifact), encoding="utf-8")
    runner._write_checksum_sidecar(output_json)

    evaluated_calls = []

    def mock_evaluate_control(inst, idx, method, obs_count, **kwargs):
        evaluated_calls.append((idx, method))
        return {
            "instance_id": inst.instance_id,
            "instance_index": idx,
            "master_key": list(inst.master_key),
            "method": method,
            "observations": obs_count,
            "differential_pairs": obs_count // 2,
            "outcome": "solver_timeout",
            "solver_status": "solver_timeout",
            "model_build_wall_seconds": 0.02,
            "solve_wall_seconds": 300.1,
            "verification_wall_seconds": 0.0,
            "online_wall_seconds": 300.2,
            "variables": 45504,
            "cnf_clauses": 69632,
            "xor_clauses": 20864,
            "total_clauses": 90496,
            "peak_rss_gib": 0.12,
            "truth_match_posthoc": False,
            "verified_recovery_within_cutoff": False,
            "stderr": None,
        }

    monkeypatch.setattr(runner, "evaluate_control", mock_evaluate_control)

    # Run for 1 instance with resume=True
    result = runner.run_benchmark(
        output_json,
        resume=True,
        max_instances=1,
    )

    # Only sat_only_n128 should have been evaluated; sat_only_n4 was skipped!
    assert evaluated_calls == [(0, "sat_only_n128")]
    assert len(result["records"]) == 2
    assert result["records"][0]["method"] == "sat_only_n4"
    assert result["records"][1]["method"] == "sat_only_n128"
    assert result["status"] == "COMPLETE"

    # Verify sidecar sha256
    sidecar = output_json.with_suffix(output_json.suffix + ".sha256")
    assert sidecar.exists()
    assert runner.sha256_file(output_json) in sidecar.read_text()


def test_resume_refuses_corrupted_or_mismatched_record(tmp_path: Path) -> None:
    output_json = tmp_path / "resume.json"
    instance = runner.generate_demo_instances()[0]
    artifact = {
        "schema": "simon_r10_sat_only_300s_controls.v1",
        "seed": runner.CLI_DEMO_SEED,
        "rounds": runner.ROUNDS,
        "pair_count": runner.PAIR_COUNT,
        "instance_count": 1,
        "methods": ["sat_only_n4", "sat_only_n128"],
        "time_limit_seconds": runner.DEFAULT_TIME_LIMIT_SECONDS,
        "watchdog_seconds": runner.DEFAULT_WATCHDOG_SECONDS,
        "records": [{
            "instance_index": 0,
            "instance_id": instance.instance_id,
            "master_key": list(instance.master_key),
            "method": "sat_only_n4",
            "observations": 128,
        }],
    }
    output_json.write_text(json.dumps(artifact), encoding="utf-8")
    runner._write_checksum_sidecar(output_json)
    with pytest.raises(ValueError, match="record cohort does not match"):
        runner.run_benchmark(output_json, resume=True, max_instances=1)
    output_json.write_text("corrupted", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 sidecar mismatch"):
        runner.run_benchmark(output_json, resume=True, max_instances=1)


def test_run_refuses_to_overwrite_existing_evidence(tmp_path: Path) -> None:
    output_json = tmp_path / "existing.json"
    output_json.write_text("protected record", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        runner.run_benchmark(output_json, max_instances=1)
    assert output_json.read_text(encoding="utf-8") == "protected record"


def test_scientific_notes_and_claims_in_summary() -> None:
    """Verify summary includes honest notes withdrawing 45s extrapolation and refraining from unmeasured claims."""
    dummy_records = [
        {
            "instance_index": 0,
            "method": "sat_only_n4",
            "verified_recovery_within_cutoff": False,
            "outcome": "solver_timeout",
            "solve_wall_seconds": 300.2,
            "peak_rss_gib": 0.08,
            "variables": 1856,
            "cnf_clauses": 2176,
            "xor_clauses": 1024,
            "total_clauses": 3200,
        },
        {
            "instance_index": 0,
            "method": "sat_only_n128",
            "verified_recovery_within_cutoff": False,
            "outcome": "solver_timeout",
            "solve_wall_seconds": 300.4,
            "peak_rss_gib": 0.11,
            "variables": 45504,
            "cnf_clauses": 69632,
            "xor_clauses": 20864,
            "total_clauses": 90496,
        },
    ]
    summary = runner.compute_summary(dummy_records)
    assert "scientific_notes" in summary
    notes = summary["scientific_notes"]

    # 1. 45s withdrawal
    assert "status_of_prior_45s_control" in notes
    assert "withdrawn" in notes["status_of_prior_45s_control"].lower()

    # 2. Honest instrumentation and no unmeasured zero swapping / Gauss claims
    assert "memory_and_solver_instrumentation" in notes
    assert "getrusage" in notes["memory_and_solver_instrumentation"]
    assert "zero swapping" in notes["memory_and_solver_instrumentation"]


def test_summary_treats_parent_kill_as_censored_and_uses_true_median() -> None:
    records = [
        {
            "method": "sat_only_n128",
            "outcome": "parent_wall_timeout",
            "verified_recovery_within_cutoff": False,
            "solve_wall_seconds": 0.0,  # legacy placeholder, not a measurement
            "peak_rss_gib": 0.0,
            "variables": 0,
        },
        {
            "method": "sat_only_n128",
            "outcome": "solver_timeout",
            "verified_recovery_within_cutoff": False,
            "solve_wall_seconds": 301.0,
            "peak_rss_gib": 0.2,
            "variables": 45504,
            "cnf_clauses": 69632,
            "xor_clauses": 20864,
            "total_clauses": 90496,
        },
        {
            "method": "sat_only_n128",
            "outcome": "solver_timeout",
            "verified_recovery_within_cutoff": False,
            "solve_wall_seconds": 303.0,
            "peak_rss_gib": 0.4,
            "variables": 45504,
            "cnf_clauses": 69632,
            "xor_clauses": 20864,
            "total_clauses": 90496,
        },
    ]
    summary = runner.compute_summary(records)["sat_only_n128"]
    assert summary["solve_time_observed_count"] == 2
    assert summary["solve_time_censored_count"] == 1
    assert summary["median_solve_seconds"] == 302.0
    assert summary["median_peak_rss_gib"] == pytest.approx(0.3)
    assert summary["formula"]["variables"] == 45504
