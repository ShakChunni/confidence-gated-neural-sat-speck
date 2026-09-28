"""Focused tests for SIMON operating points map, cohort identity, and solver isolation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from neural_sat_speck.data.seed_policy import CLI_DEMO_SEED
from neural_sat_speck.simon.attack_instance import (
    DEFAULT_SIMON_INPUT_DIFFERENCE,
    SimonAttackInstanceConfig,
    generate_simon_attack_instances,
)
from neural_sat_speck.simon.sat_model import SimonSatModel
from neural_sat_speck.simon.verification import verify_simon_master_key
from scripts.map_simon_operating_points import (
    INSTANCE_COUNT,
    get_demo_instances,
    run_map,
    worker_solve,
)


def test_cohort_identity_slicing_and_divergence() -> None:
    """Verify that N=4 is sliced from the 32-pair cohort and does NOT match pair_count=2."""
    for rounds in (9, 10):
        cohort_32 = get_demo_instances(rounds)
        assert len(cohort_32) == INSTANCE_COUNT
        assert all(len(inst.pairs) == 32 for inst in cohort_32)
        assert all(len(inst.known_pairs) == 64 for inst in cohort_32)
        assert all(inst.split_role == "demo" for inst in cohort_32)

        # Slicing the first 4 observations (2 pairs) preserves the master key
        for inst in cohort_32:
            sliced = inst.known_pairs[:4]
            assert len(sliced) == 4
            assert sliced == (
                (inst.pairs[0].plaintext0, inst.pairs[0].ciphertext0),
                (inst.pairs[0].plaintext1, inst.pairs[0].ciphertext1),
                (inst.pairs[1].plaintext0, inst.pairs[1].ciphertext0),
                (inst.pairs[1].plaintext1, inst.pairs[1].ciphertext1),
            )

        # Generating with pair_count=2 produces DIFFERENT master keys for keys 1..7
        cohort_2 = generate_simon_attack_instances(
            SimonAttackInstanceConfig(
                instance_count=INSTANCE_COUNT,
                pair_count=2,
                rounds=rounds,
                seed=CLI_DEMO_SEED,
                split_role="demo",
                verification_pair_count=2,
                input_difference=DEFAULT_SIMON_INPUT_DIFFERENCE,
            )
        )
        assert cohort_2[0].master_key == cohort_32[0].master_key
        # Due to RNG consumption in the inner loop, subsequent keys diverge
        mismatched = [
            cohort_2[i].master_key != cohort_32[i].master_key
            for i in range(1, INSTANCE_COUNT)
        ]
        assert any(mismatched), "Different pair_count values must not be assumed matched"


def test_deterministic_replay() -> None:
    """Ensure that get_demo_instances replays identically."""
    first = get_demo_instances(9)
    second = get_demo_instances(9)
    assert len(first) == len(second) == INSTANCE_COUNT
    for a, b in zip(first, second):
        assert a.master_key == b.master_key
        assert a.known_pairs == b.known_pairs
        assert a.verification_pairs == b.verification_pairs


def test_independent_solver_construction() -> None:
    """Ensure that each solve call builds an independent solver instance."""
    inst = get_demo_instances(9)[0]
    known = inst.known_pairs[:4]
    model1 = SimonSatModel.from_known_pairs(known, rounds=9)
    model2 = SimonSatModel.from_known_pairs(known, rounds=9)
    assert model1.solver is not model2.solver
    assert model1.variable_count == model2.variable_count == 1664
    assert model1.cnf_clause_count == model2.cnf_clause_count == 1984
    assert model1.xor_clause_count == model2.xor_clause_count == 896


def test_worker_solve_budget_exhausted() -> None:
    """Ensure that zero or negative time limit results in budget_exhausted."""
    inst = get_demo_instances(9)[0]
    payload = {
        "rounds": 9,
        "known_pairs": inst.known_pairs[:4],
        "verification_pairs": inst.verification_pairs,
        "assumed_subkeys": [],
        "time_limit": -1.0,
    }
    result = worker_solve(payload)
    assert result["outcome"] == "budget_exhausted"
    assert result["solver_status"] == "not_started"
    assert result["recovered_master_key"] is None


def test_worker_solve_timeout_classification() -> None:
    """Ensure that solver timeout is accurately classified."""
    inst = get_demo_instances(10)[0]
    payload = {
        "rounds": 10,
        "known_pairs": inst.known_pairs[:64],
        "verification_pairs": inst.verification_pairs,
        "assumed_subkeys": [],
        "time_limit": 0.2,  # Model build takes ~0.02s; remaining ~0.18s causes solver timeout
    }
    result = worker_solve(payload)
    assert result["outcome"] == "solver_timeout"
    assert result["recovered_master_key"] is None


def test_fresh_pair_verification_rejects_wrong_key() -> None:
    """Ensure that fresh-pair verification rejects incorrect master keys."""
    inst = get_demo_instances(9)[0]
    true_key = inst.master_key
    assert verify_simon_master_key(true_key, inst.known_pairs, inst.verification_pairs, 9).verified

    wrong_key = (true_key[0] ^ 1, true_key[1], true_key[2], true_key[3])
    assert not verify_simon_master_key(wrong_key, inst.known_pairs, inst.verification_pairs, 9).verified


def test_run_map_workers_guard(tmp_path: Path) -> None:
    """Ensure run_map rejects workers outside (1, 2)."""
    with pytest.raises(ValueError, match="Workers must be 1 or 2"):
        run_map(tmp_path / "out.json", workers=3)


def test_parent_deadline_kills_overrunning_worker() -> None:
    """Ensure parent watchdog terminates a worker that exceeds cutoff and classifies as parent_wall_timeout."""
    import sys

    from scripts.map_simon_operating_points import evaluate_cell_key

    # Cutoff of 0.1s; worker attempts to sleep for 1.0s
    rec = evaluate_cell_key(
        rounds=9,
        observations=4,
        assumed_keys=0,
        instance_index=0,
        cutoff_seconds=0.1,
        worker_cmd=[sys.executable, "-c", "import time; time.sleep(1.0)"],
    )
    assert rec["outcome"] == "parent_wall_timeout"
    assert rec["verified_recovery_within_cutoff"] is False
    assert rec["online_wall_seconds"] < 0.8, "Parent watchdog must terminate worker near deadline"
    assert rec["cutoff_overshoot_seconds"] >= 0.0


def test_worker_finishes_just_within_deadline() -> None:
    """Ensure a worker that completes within cutoff is correctly recorded as verified_recovery."""
    import sys

    from scripts.map_simon_operating_points import evaluate_cell_key

    inst = get_demo_instances(9)[0]
    worker_script = (
        "import json, sys; "
        f"print(json.dumps({{'outcome': 'verified_recovery', 'recovered_master_key': {list(inst.master_key)}}})); "
        "sys.exit(0)"
    )
    rec = evaluate_cell_key(
        rounds=9,
        observations=4,
        assumed_keys=2,
        instance_index=0,
        cutoff_seconds=2.0,
        worker_cmd=[sys.executable, "-c", worker_script],
    )
    assert rec["outcome"] == "verified_recovery"
    assert rec["verified_recovery_within_cutoff"] is True
    assert rec["truth_match_posthoc"] is True
    assert rec["online_wall_seconds"] < 2.0


def test_deadline_classifications_separate_outcomes() -> None:
    """Ensure parent separately classifies solver_timeout, verification_failure, and worker_exception."""
    import sys

    from scripts.map_simon_operating_points import evaluate_cell_key

    # 1. solver_timeout
    script_timeout = "import json; print(json.dumps({'outcome': 'solver_timeout'})); sys.exit(0)"
    rec = evaluate_cell_key(
        rounds=9, observations=4, assumed_keys=0, instance_index=0, cutoff_seconds=2.0,
        worker_cmd=[sys.executable, "-c", f"import sys; {script_timeout}"],
    )
    assert rec["outcome"] == "solver_timeout"
    assert rec["verified_recovery_within_cutoff"] is False

    # 2. verification_failure
    script_vf = "import json; print(json.dumps({'outcome': 'verification_failure', 'recovered_master_key': [1,2,3,4]})); sys.exit(0)"
    rec = evaluate_cell_key(
        rounds=9, observations=4, assumed_keys=0, instance_index=0, cutoff_seconds=2.0,
        worker_cmd=[sys.executable, "-c", f"import sys; {script_vf}"],
    )
    assert rec["outcome"] == "verification_failure"
    assert rec["verified_recovery_within_cutoff"] is False

    # 3. worker_exception (crash)
    rec = evaluate_cell_key(
        rounds=9, observations=4, assumed_keys=0, instance_index=0, cutoff_seconds=2.0,
        worker_cmd=[sys.executable, "-c", "import sys; sys.exit(1)"],
    )
    assert rec["outcome"] == "worker_exception"
    assert rec["verified_recovery_within_cutoff"] is False

    # 4. late_verified_recovery (finishes with verified_recovery but parent timer exceeds cutoff)
    # Simulate child sleeping 0.15s under a 0.05s parent cutoff where timeout expired
    # Or child finishes with verified_recovery when parent cutoff is tiny and child sleeps 0.02s
    # When child exits, if online_seconds > cutoff, it classifies as late_verified_recovery
    inst = get_demo_instances(9)[0]
    script_late = (
        "import json, time, sys; "
        "time.sleep(0.06); "
        f"print(json.dumps({{'outcome': 'verified_recovery', 'recovered_master_key': {list(inst.master_key)}}})); "
        "sys.exit(0)"
    )
    rec = evaluate_cell_key(
        rounds=9, observations=4, assumed_keys=2, instance_index=0, cutoff_seconds=0.04,
        worker_cmd=[sys.executable, "-c", script_late],
    )
    # The parent either killed it via timeoutExpired (parent_wall_timeout) or if it slipped past, late_verified_recovery
    assert rec["outcome"] in ("parent_wall_timeout", "late_verified_recovery")
    assert rec["verified_recovery_within_cutoff"] is False


def test_run_map_r10_serial_profile_cells(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure run_map with profile='r10_serial' evaluates only the four R=10 cells."""
    from scripts import map_simon_operating_points

    calls = []

    def fake_evaluate(rounds: int, obs: int, assumed: int, idx: int, cutoff: float, **kwargs: Any) -> dict[str, Any]:
        calls.append((rounds, obs, assumed, idx))
        return {
            "rounds": rounds,
            "observations": obs,
            "assumed_keys": assumed,
            "instance_index": idx,
            "instance_id": f"dummy_{idx}",
            "master_key": [1, 2, 3, 4],
            "cutoff_seconds": cutoff,
            "online_wall_seconds": 0.01,
            "cutoff_overshoot_seconds": 0.0,
            "outcome": "verified_recovery",
            "verified_recovery_within_cutoff": True,
            "truth_match_posthoc": True,
            "worker": {},
            "stderr": None,
        }

    monkeypatch.setattr(map_simon_operating_points, "evaluate_cell_key", fake_evaluate)
    monkeypatch.setattr(map_simon_operating_points, "INSTANCE_COUNT", 2)  # fast check with 2 instances

    out_file = tmp_path / "serial_conf.json"
    data = map_simon_operating_points.run_map(out_file, resume=False, workers=1, profile="r10_serial")

    assert data["profile"] == "r10_serial"
    assert data["status"] == "COMPLETE"
    assert len(data["records"]) == 8  # 4 cells * 2 instances
    cell_keys = {(r["rounds"], r["observations"], r["assumed_keys"]) for r in data["records"]}
    assert cell_keys == {
        (10, 4, 2),
        (10, 64, 2),
        (10, 4, 0),
        (10, 64, 0),
    }


def test_resume_rejects_changed_profile_or_worker_count(tmp_path: Path) -> None:
    """A resume must not blend records from different cell or concurrency plans."""
    from scripts import map_simon_operating_points as operating_map

    output = tmp_path / "map.json"
    original = {
        "schema": "simon_operating_points_map.v1",
        "profile": "r10_serial",
        "cells": [],
        "seed": CLI_DEMO_SEED,
        "workers": 1,
        "strict_parent_deadline": True,
        "solver_configuration": {},
        "cohort_master_keys": {},
        "environment": {"git_head": operating_map._git_head()},
        "records": [],
    }
    import json

    output.write_text(json.dumps(original), encoding="utf-8")
    with pytest.raises(ValueError, match="changed operating-point manifest"):
        run_map(output, resume=True, workers=1, profile="full")
    with pytest.raises(ValueError, match="changed operating-point manifest"):
        run_map(output, resume=True, workers=2, profile="r10_serial")
    with pytest.raises(ValueError, match="unknown operating-point profile"):
        run_map(output, profile="unknown")
