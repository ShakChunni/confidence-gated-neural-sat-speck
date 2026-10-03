"""Unit and adversarial tests for the cross-cipher residual-depth guarded runner."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from neural_sat_speck.experiments.depth_map_runner import (
    PER_CELL_CUTOFF_SECONDS,
    TOTAL_CELLS_COUNT,
    AtomicDepthJournal,
    build_counterbalanced_schedule,
    get_git_commit,
    load_dataset,
    prepare_cell_payload,
    run_cell_with_parent_watchdog,
    validate_manifest_for_execution,
)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "configs" / "protocol" / "cross_cipher_residual_depth_manifest.json"
DATASET_PATH = ROOT / "datasets" / "demo_cross_cipher_depth_r6_r11" / "dataset.json"


def test_manifest_validation_passes():
    """Verify production manifest validates successfully."""
    manifest = validate_manifest_for_execution(MANIFEST_PATH, repo_root=ROOT)
    assert manifest["execution_status"] == "authorized_execution"
    assert "D-043" in manifest["governing_decisions"]


def test_manifest_validation_rejects_tampering(tmp_path: Path):
    """Adversarial test: Reject unauthorized status, missing D-043, or hash mismatch."""
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        valid_manifest = json.load(f)

    # 1. Reject unapproved status
    bad_status = copy.deepcopy(valid_manifest)
    bad_status["execution_status"] = "proposed_draft"
    p1 = tmp_path / "manifest_bad_status.json"
    p1.write_text(json.dumps(bad_status), encoding="utf-8")
    with pytest.raises(ValueError, match="Manifest status is not 'authorized_execution'"):
        validate_manifest_for_execution(p1, repo_root=ROOT)

    # 2. Reject missing D-043
    no_d43 = copy.deepcopy(valid_manifest)
    no_d43["governing_decisions"] = ["D-024", "D-029"]
    p2 = tmp_path / "manifest_no_d43.json"
    p2.write_text(json.dumps(no_d43), encoding="utf-8")
    with pytest.raises(ValueError, match="Manifest lacks governing decision D-043"):
        validate_manifest_for_execution(p2, repo_root=ROOT)

    # 3. Reject dataset hash mismatch
    bad_hash = copy.deepcopy(valid_manifest)
    bad_hash["dataset"]["sha256"] = "0" * 64
    p3 = tmp_path / "manifest_bad_hash.json"
    p3.write_text(json.dumps(bad_hash), encoding="utf-8")
    with pytest.raises(ValueError, match="Dataset SHA-256 mismatch"):
        validate_manifest_for_execution(p3, repo_root=ROOT)


def test_schedule_properties():
    """Verify 192 cells, unique IDs, equal cipher/key coverage."""
    schedule = build_counterbalanced_schedule()
    assert len(schedule) == TOTAL_CELLS_COUNT == 192

    cell_ids = [c["cell_id"] for c in schedule]
    assert len(set(cell_ids)) == 192

    speck_cells = [c for c in schedule if c["cipher"] == "speck32_64"]
    simon_cells = [c for c in schedule if c["cipher"] == "simon32_64"]
    assert len(speck_cells) == 96
    assert len(simon_cells) == 96

    for key_idx in range(8):
        k_cells = [c for c in schedule if c["instance_index"] == key_idx]
        assert len(k_cells) == 24


def test_parent_watchdog_terminates_hanging_worker(tmp_path: Path):
    """Adversarial test: Parent watchdog sends SIGKILL to hanging process group at cutoff."""
    # Create worker script that hangs ignoring SIGTERM
    hanging_script = tmp_path / "hanging_worker.py"
    hanging_script.write_text(
        """
import time, signal
signal.signal(signal.SIGTERM, signal.SIG_IGN)
time.sleep(30.0)
""",
        encoding="utf-8",
    )

    payload = {
        "cell_id": "test_hanging_cell",
        "cipher": "speck32_64",
        "rounds": 6,
        "assumed_keys": 0,
        "nominal_residual_depth": 6,
        "observations": 4,
        "instance_index": 0,
        "master_key": [0, 0, 0, 0],
        "cutoff_seconds": 0.3,  # Short cutoff for test
        "expected_git_commit": "test",
    }

    cmd = [sys.executable, str(hanging_script)]
    res = run_cell_with_parent_watchdog(payload, repo_root=ROOT, worker_cmd=cmd)

    assert res["outcome"] == "parent_watchdog_timeout"
    assert res["parent_wall_seconds"] >= 0.3
    assert res["verified_recovery"] is False


def test_unsat_triggers_correctness_alarm(tmp_path: Path):
    """Adversarial test: Any UNSAT must raise a correctness alarm and halt dispatch."""
    unsat_script = tmp_path / "unsat_worker.py"
    unsat_script.write_text(
        """
import json, sys
res = {
    "cell_id": "test_unsat_cell",
    "outcome": "proven_unsat",
    "solve_status": "unsat",
    "receipt_file": "dummy"
}
sys.stdout.write(json.dumps(res) + "\\n")
""",
        encoding="utf-8",
    )

    payload = {
        "cell_id": "test_unsat_cell",
        "cipher": "speck32_64",
        "rounds": 6,
        "assumed_keys": 0,
        "nominal_residual_depth": 6,
        "observations": 4,
        "instance_index": 0,
        "master_key": [0, 0, 0, 0],
        "cutoff_seconds": 5.0,
        "expected_git_commit": "test",
    }

    cmd = [sys.executable, str(unsat_script)]
    with pytest.raises(RuntimeError, match="CORRECTNESS ALARM"):
        run_cell_with_parent_watchdog(payload, repo_root=ROOT, worker_cmd=cmd)


def test_atomic_journal_and_interrupted_cell(tmp_path: Path):
    """Test atomic journal persistence and crash recovery of started-without-finished cell."""
    journal_path = tmp_path / "test_journal.json"
    journal = AtomicDepthJournal(journal_path)

    # 1. Record start of cell 1
    journal.record_start("cell_1")
    # Record finish of cell 1
    journal.record_finish("cell_1", {"cell_id": "cell_1", "outcome": "verified_recovery"})
    assert journal.is_cell_completed("cell_1")

    # 2. Record start of cell 2 without finish (simulating crash)
    journal.record_start("cell_2")

    # 3. Reload journal from disk as a fresh instance
    reloaded_journal = AtomicDepthJournal(journal_path)
    assert reloaded_journal.is_cell_completed("cell_1")
    assert reloaded_journal.is_cell_completed("cell_2")
    assert reloaded_journal.data["cells"]["cell_2"]["outcome"] == "infrastructure_interrupted"

    # 4. Sidecar integrity
    sidecar_path = journal_path.with_name("test_journal.json.sha256")
    assert sidecar_path.is_file()

    # Corrupt sidecar
    sidecar_path.write_text("0" * 64 + "  test_journal.json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Journal SHA-256 sidecar mismatch"):
        AtomicDepthJournal(journal_path)


def test_single_cell_real_solve():
    """Positive end-to-end test: Run 1 real subcritical SPECK cell and verify correct recovery."""
    dataset = load_dataset(DATASET_PATH)
    schedule = build_counterbalanced_schedule()

    # Find first subcritical cell: SPECK (6, 1) O=4 k=0
    target_cell = next(
        c for c in schedule
        if c["cipher"] == "speck32_64" and c["rounds"] == 6 and c["assumed_keys"] == 1 and c["observations"] == 4 and c["instance_index"] == 0
    )

    git_commit = get_git_commit(ROOT)
    tmpdir = ROOT / "runtime" / "worker_tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)

    payload = prepare_cell_payload(target_cell, dataset, ROOT, tmpdir, git_commit)
    res = run_cell_with_parent_watchdog(payload, repo_root=ROOT)

    assert res["outcome"] == "verified_recovery"
    assert res["verified_recovery"] is True
    assert res["recovered_master_key"] == payload["master_key"]
    assert res["solve_wall_seconds"] < 5.0
    assert res["variables_count"] > 0


def test_simon_single_cell_real_solve():
    """Positive end-to-end test: Run 1 real subcritical SIMON cell and verify correct recovery."""
    dataset = load_dataset(DATASET_PATH)
    schedule = build_counterbalanced_schedule()

    # Find first subcritical SIMON cell: SIMON (9, 1) O=4 k=0
    target_cell = next(
        c for c in schedule
        if c["cipher"] == "simon32_64" and c["rounds"] == 9 and c["assumed_keys"] == 1 and c["observations"] == 4 and c["instance_index"] == 0
    )

    git_commit = get_git_commit(ROOT)
    tmpdir = ROOT / "runtime" / "worker_tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)

    payload = prepare_cell_payload(target_cell, dataset, ROOT, tmpdir, git_commit)
    res = run_cell_with_parent_watchdog(payload, repo_root=ROOT)

    assert res["outcome"] == "verified_recovery"
    assert res["verified_recovery"] is True
    assert res["recovered_master_key"] == payload["master_key"]
    assert res["solve_wall_seconds"] < 5.0
    assert res["variables_count"] > 0


def test_late_recovery_rejected(tmp_path: Path):
    """Adversarial test: Any recovery emitted after the cutoff must be classified as parent_watchdog_timeout."""
    late_script = tmp_path / "late_worker.py"
    late_script.write_text(
        """
import json, sys, time
time.sleep(0.4)
res = {
    "cell_id": "test_late_cell",
    "outcome": "verified_recovery",
    "solve_status": "sat",
    "receipt_file": "dummy"
}
sys.stdout.write(json.dumps(res) + "\\n")
""",
        encoding="utf-8",
    )

    payload = {
        "cell_id": "test_late_cell",
        "cipher": "speck32_64",
        "rounds": 6,
        "assumed_keys": 0,
        "nominal_residual_depth": 6,
        "observations": 4,
        "instance_index": 0,
        "master_key": [0, 0, 0, 0],
        "cutoff_seconds": 0.2,  # Cutoff shorter than sleep
        "expected_git_commit": "test",
    }

    cmd = [sys.executable, str(late_script)]
    res = run_cell_with_parent_watchdog(payload, repo_root=ROOT, worker_cmd=cmd)

    assert res["outcome"] == "parent_watchdog_timeout"
    assert res["verified_recovery"] is False

