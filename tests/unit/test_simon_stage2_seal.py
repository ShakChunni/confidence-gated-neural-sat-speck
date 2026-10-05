"""Fail-closed recovery tests for the no-dispatch Stage 2 sealer."""

from __future__ import annotations

import ast
import copy
import fcntl
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from neural_sat_speck.backend.guard import atomic_json, read_checked_json
from neural_sat_speck.followups import stage2_runner, stage2_seal
from neural_sat_speck.followups.stage2_control import (
    PROFILE,
    fixed_jobs,
    streaming_sha256,
)


@pytest.fixture
def case(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    source_root.mkdir()
    cohort_path = tmp_path / "opened_cohort.json"
    cohort_path.write_text('{"ciphertext_fixture":true}\n')
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "mode": "pilot",
        "source_commit": "a" * 40,
        "source_root": str(source_root),
        "profile": copy.deepcopy(PROFILE),
        "files": {
            "python": {"path": str(source_root / ".venv/bin/python")},
            "tf_python": {"path": str(source_root / ".venv-tf/bin/python")},
        },
    }
    atomic_json(manifest_path, manifest)
    monkeypatch.setattr(
        stage2_seal,
        "validate_manifest",
        lambda value, root: {"files": {"cohort": cohort_path}},
    )
    monkeypatch.setattr(stage2_seal, "public_case", lambda data, job: {
        "cipher": "simon32_64",
        "rounds": 10,
        "key_index": job["key_index"],
        "instance_id": f"public-{job['key_index']}",
        "ciphertext_pairs": [[[0, 0], [0, 0]]] * 64,
    })
    monkeypatch.setattr(stage2_seal, "live_groups", lambda token: set())

    ledger_path = tmp_path / "ledger.json"
    now = time.perf_counter()
    baseline = {
        "schema": "stage2_control_results_v1",
        "mode": "pilot",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": streaming_sha256(manifest_path),
        "profile": copy.deepcopy(PROFILE),
        "started_utc": datetime.now(UTC).isoformat(),
        "campaign_started_monotonic": now,
        "campaign_deadline_monotonic": now + PROFILE["campaign_limit_seconds"],
        "status": "ACTIVE",
        "starts": {},
        "records": {},
        "sleep_assertion": {"supported": False},
    }
    atomic_json(tmp_path / "campaign_start.json", baseline)
    stage2_runner.save_ledger(ledger_path, baseline)
    return {
        "root": source_root,
        "manifest_path": manifest_path,
        "ledger_path": ledger_path,
        "output_path": tmp_path / "sealed.json",
        "baseline": baseline,
        "manifest": manifest,
        "tmp_path": tmp_path,
    }


def _write_start(case, job_index=0):
    job = fixed_jobs()[job_index]
    token = f"{job_index + 1:032x}"
    directory = case["tmp_path"] / "jobs" / job["job_id"] / token
    directory.mkdir(parents=True)
    start = {
        "job": job,
        "invocation": token,
        "directory": str(directory),
        "source_commit": case["manifest"]["source_commit"],
        "manifest_sha256": streaming_sha256(case["manifest_path"]),
        "started_utc": datetime.now(UTC).isoformat(),
    }
    atomic_json(directory / "campaign_job_start.json", start)
    return job, start, directory


def _save_started_snapshot(case, job, start):
    ledger = copy.deepcopy(case["baseline"])
    ledger["starts"][job["job_id"]] = {
        "invocation": start["invocation"],
        "directory": start["directory"],
        "started_utc": start["started_utc"],
    }
    stage2_runner.save_ledger(case["ledger_path"], ledger)
    return ledger


def _seal(case):
    return stage2_seal.seal(
        case["manifest_path"],
        case["ledger_path"],
        case["output_path"],
        case["root"],
    )


def test_torn_latest_uses_highest_checked_immutable_history(case):
    job, start, _ = _write_start(case)
    ledger = _save_started_snapshot(case, job, start)
    case["ledger_path"].write_text("interrupted latest replacement")

    sealed = _seal(case)

    assert sealed["status"] == "STOPPED_ALARM"
    assert sealed["sealed_from"]["ledger_sha256"] is None
    assert sealed["sealed_from"]["history_sequence"] == 1
    assert sealed["sealed_from"]["history_sha256"] == streaming_sha256(
        case["tmp_path"] / "ledger_history/0001.json"
    )
    assert sealed["starts"][job["job_id"]] == ledger["starts"][job["job_id"]]
    assert set(sealed["records"]) == {item["job_id"] for item in fixed_jobs()[:3]}
    row = sealed["records"][job["job_id"]]
    assert row["dispatched"] is True
    assert row["outcome"] == "interrupted"
    assert row["calls"] == []
    assert row["load_wall_seconds"] is None
    assert row["worker_wall_seconds"] is None
    assert "guardian_receipt" not in row
    assert read_checked_json(case["tmp_path"] / "ledger_history/0001.json") == ledger


def test_torn_history_falls_back_and_orphan_start_is_not_dispatched(case):
    job, start, directory = _write_start(case)
    _save_started_snapshot(case, job, start)
    history_path = case["tmp_path"] / "ledger_history/0001.json"
    history_path.write_text("torn immutable history version")

    sealed = _seal(case)

    row = sealed["records"][job["job_id"]]
    assert sealed["sealed_from"]["history_sequence"] == 0
    assert row["dispatched"] is False
    assert row["calls"] == []
    assert row["load_wall_seconds"] is None
    assert row["worker_wall_seconds"] is None
    assert row["directory"] == str(directory)
    assert row["invocation"] == start["invocation"]
    assert row["outcome"] == "unavailable_after_coordinator_interrupted"
    assert (directory / "campaign_job_start.json").exists()


def test_checked_start_without_guardian_receipt_is_unavailable_and_keeps_forensics(case):
    job, start, directory = _write_start(case)
    _save_started_snapshot(case, job, start)

    sealed = _seal(case)

    row = sealed["records"][job["job_id"]]
    assert row["dispatched"] is True
    assert row["directory"] == str(directory)
    assert row["invocation"] == start["invocation"]
    assert row["calls"] == []
    assert row["load_wall_seconds"] is None
    assert row["worker_wall_seconds"] is None
    assert "guardian_receipt" not in row
    assert (directory / "campaign_job_start.json").exists()


def test_guardian_proven_no_worker_start_keeps_receipt_without_claiming_dispatch(case, monkeypatch):
    job, start, directory = _write_start(case)
    _save_started_snapshot(case, job, start)
    started = case["baseline"]["campaign_started_monotonic"] + 1.0
    deadline = case["baseline"]["campaign_deadline_monotonic"]
    config = {
        "invocation": start["invocation"],
        "directory": str(directory),
        "repo_root": case["manifest"]["source_root"],
        "worker_arguments": [
            case["manifest"]["files"]["tf_python"]["path"],
            str(case["root"] / "scripts/run_simon_matched_stage2_worker.py"),
            str(directory / "payload.json"),
        ],
        "receipt": str(directory / "guardian_receipt.json"),
        "job_id": job["job_id"],
        "coordinator_pid": os.getpid(),
        "job_started_monotonic": started,
        "campaign_deadline_monotonic": deadline,
        "load_seconds": 120.0,
        "call_seconds": 180.0,
        "expected_calls": 1,
    }
    payload = {"public_only": True}
    atomic_json(directory / "payload.json", payload)
    monkeypatch.setattr(stage2_seal, "public_payload", lambda *args: payload)
    atomic_json(directory / "guardian.json", config)
    config_hash = streaming_sha256(directory / "guardian.json")
    atomic_json(
        directory / "guardian_receipt.json",
        {
            "schema": "simon_stage2_guardian_receipt_v1",
            "invocation": start["invocation"],
            "job_id": job["job_id"],
            "config_sha256": config_hash,
            "accepted_events": [],
            "effective_deadline_monotonic": min(started + 120.0, deadline),
            "status": "campaign_timeout",
            "worker_pid": None,
            "process_group": None,
            "worker_not_started": True,
            "process_group_absent": True,
            "cleanup_completed_monotonic": started + 121.0,
        },
    )

    sealed = _seal(case)

    row = sealed["records"][job["job_id"]]
    assert row["dispatched"] is False
    assert row["outcome"] == "interrupted"
    assert row["directory"] == str(directory)
    assert row["invocation"] == start["invocation"]
    assert row["calls"] == []
    assert row["load_wall_seconds"] is None
    assert row["worker_wall_seconds"] is None
    assert row["guardian_receipt"]["worker_not_started"] is True
    assert row["accepted_events"] == []


def test_live_coordinator_lock_rejects_before_sealing(case):
    lock_path = case["ledger_path"].with_suffix(".lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with pytest.raises(BlockingIOError):
                _seal(case)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    assert not case["output_path"].exists()


def test_live_saved_invocation_group_rejects_sealer(case, monkeypatch):
    job, start, _ = _write_start(case)
    _save_started_snapshot(case, job, start)
    monkeypatch.setattr(stage2_seal, "live_groups", lambda token: {8765})

    with pytest.raises(RuntimeError, match="live guardian or worker"):
        _seal(case)
    assert not case["output_path"].exists()


def test_duplicate_invocation_starts_fail_instead_of_retrying(case):
    job = fixed_jobs()[0]
    for offset in (1, 2):
        token = f"{offset:032x}"
        directory = case["tmp_path"] / "jobs" / job["job_id"] / token
        directory.mkdir(parents=True)
        atomic_json(
            directory / "campaign_job_start.json",
            {
                "job": job,
                "invocation": token,
                "directory": str(directory),
                "source_commit": case["manifest"]["source_commit"],
                "manifest_sha256": streaming_sha256(case["manifest_path"]),
                "started_utc": datetime.now(UTC).isoformat(),
            },
        )

    with pytest.raises(ValueError, match="More than one immutable invocation"):
        _seal(case)
    assert not case["output_path"].exists()


def test_sorted_ledger_keys_keep_odd_key_stage2_prefix_in_fixed_job_order(case):
    jobs = fixed_jobs()
    starts = {}
    job_files = {}
    for index, job in enumerate(jobs[:6]):
        token = f"{index + 10:032x}"
        directory = case["tmp_path"] / "jobs" / job["job_id"] / token
        directory.mkdir(parents=True)
        started_utc = datetime.now(UTC).isoformat()
        starts[job["job_id"]] = {
            "invocation": token,
            "directory": str(directory),
            "started_utc": started_utc,
        }
        job_files[job["job_id"]] = {
            "job": job,
            "invocation": token,
            "directory": str(directory),
            "source_commit": case["manifest"]["source_commit"],
            "manifest_sha256": streaming_sha256(case["manifest_path"]),
            "started_utc": started_utc,
        }
    snapshot = {
        "starts": dict(sorted(starts.items())),
        "records": {job["job_id"]: {} for job in jobs[:5]},
    }

    merged, orphans = stage2_seal._merge_starts(
        case["tmp_path"], jobs, snapshot, job_files
    )

    assert not orphans
    assert set(merged) == {job["job_id"] for job in jobs[:6]}
    assert jobs[4]["job_id"].endswith("stage2_ddt7")
    assert jobs[5]["job_id"].endswith("stage2_vv8")


def test_sealer_never_dispatches_or_loads_ranking_code(case, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("sealer attempted worker dispatch")

    monkeypatch.setattr(stage2_runner, "run_job", forbidden)
    monkeypatch.setattr(stage2_runner.subprocess, "Popen", forbidden)

    sealed = _seal(case)

    assert sealed["status"] == "STOPPED_ALARM"
    source = Path(stage2_seal.__file__).read_text()
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any(
        token in name.lower()
        for name in imported
        for token in ("tensorflow", "ranking", "worker", "sat", "simon.cipher")
    )


def test_output_must_be_new_and_original_ledger_is_untouched(case, monkeypatch):
    original = case["ledger_path"].read_bytes()
    case["output_path"].write_text("preserve this existing artifact")
    monkeypatch.setattr(
        stage2_seal.fcntl,
        "flock",
        lambda *args: pytest.fail("existing output must be rejected before lock"),
    )

    with pytest.raises(FileExistsError, match="must be new"):
        _seal(case)
    assert case["ledger_path"].read_bytes() == original
    assert case["output_path"].read_text() == "preserve this existing artifact"
