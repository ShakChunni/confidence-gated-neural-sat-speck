"""Check rejected dispatch and persisted-call isolation independently of rankers."""

import copy
from pathlib import Path

import pytest

from neural_sat_speck.backend.guard import atomic_json
from neural_sat_speck.followups import stage2_runner as runner
from neural_sat_speck.followups.stage2_control import fixed_jobs, streaming_sha256


def test_invalid_manifest_reads_no_dataset_or_runtime(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    atomic_json(manifest_path, {"status": "PROPOSED"})
    observed = []
    monkeypatch.setattr(
        runner,
        "validate_manifest",
        lambda *args: (_ for _ in ()).throw(ValueError("unapproved")),
    )
    monkeypatch.setattr(runner, "public_case", lambda *args: observed.append("dataset"))
    monkeypatch.setattr(
        runner.subprocess, "Popen", lambda *args, **kwargs: observed.append("worker")
    )
    with pytest.raises(ValueError, match="unapproved"):
        runner.run(manifest_path, tmp_path / "ledger.json", tmp_path)
    assert observed == []
    assert not (tmp_path / "ledger.json").exists()


def test_existing_measurement_cannot_retry(tmp_path, monkeypatch):
    output = tmp_path / "ledger.json"
    output.write_text("preserved")
    monkeypatch.setattr(
        runner,
        "validate_manifest",
        lambda *args: pytest.fail("preflight after existing measurement"),
    )
    with pytest.raises(FileExistsError, match="No retry"):
        runner.run(tmp_path / "missing_manifest", output, tmp_path)
    assert output.read_text() == "preserved"


def test_orphan_sidecar_cannot_be_overwritten_as_a_new_measurement(tmp_path, monkeypatch):
    output = tmp_path / "ledger.json"
    sidecar = tmp_path / "ledger.json.sha256"
    sidecar.write_text("preserved torn measurement")
    monkeypatch.setattr(
        runner, "validate_manifest", lambda *args: pytest.fail("reused measurement")
    )
    with pytest.raises(FileExistsError, match="No retry"):
        runner.run(tmp_path / "manifest", output, tmp_path)
    assert sidecar.read_text() == "preserved torn measurement"


def test_torn_latest_ledger_preserves_immutable_checked_history(tmp_path):
    from neural_sat_speck.backend.guard import read_checked_json

    output = tmp_path / "ledger.json"
    first = {"starts": {}, "records": {}, "status": "ACTIVE"}
    runner.save_ledger(output, first)
    initial_bytes = (tmp_path / "ledger_history/0000.json").read_bytes()
    second = {"starts": {"one": {}}, "records": {}, "status": "ACTIVE"}
    runner.save_ledger(output, second)
    output.write_text("torn latest replacement")
    with pytest.raises(ValueError, match="Checksum"):
        read_checked_json(output)
    assert (tmp_path / "ledger_history/0000.json").read_bytes() == initial_bytes
    assert read_checked_json(tmp_path / "ledger_history/0000.json") == first
    assert read_checked_json(tmp_path / "ledger_history/0001.json") == second


def synthetic_files(directory, *, status="rank_timeout", returncode=-9):
    directory.mkdir()
    job = fixed_jobs()[1]
    identity = {"invocation": directory.name, "job_id": job["job_id"]}
    start = {
        **identity,
        "ordinal": 0,
        "scorer": "vv8",
        "parent": 123,
        "call_started_monotonic": 102.0,
    }
    result = {
        **start,
        "call_completed_monotonic": 103.0,
        "ordered_words": list(range(512)),
        "component_times_seconds": {"total": 1.0},
    }
    atomic_json(directory / "calls/0000_start.json", start)
    path = directory / "calls/0000_result.json"
    atomic_json(path, result)
    events = [
        {**identity, "event": "loaded", "completed_monotonic": 101.0},
        {
            **identity,
            "event": "call_complete",
            "ordinal": 0,
            "completed_monotonic": 104.0,
            "result": "calls/0000_result.json",
            "result_sha256": streaming_sha256(path),
        },
    ]
    receipt = {
        **identity,
        "process_group_absent": True,
        "accepted_events": events,
        "status": status,
        "worker_returncode": returncode,
        "cleanup_completed_monotonic": 285.0,
    }
    atomic_json(directory / "guardian_receipt.json", receipt)
    return job, receipt


def test_interrupted_parent_worker_keeps_only_accepted_complete_calls(tmp_path):
    directory = tmp_path / ("a" * 32)
    job, _ = synthetic_files(directory)
    # A valid-looking result without a guardian acceptance must not enter the row.
    atomic_json(
        directory / "calls/0001_result.json", {"ordered_words": list(range(512))}
    )
    row = runner.normalize_job(job, directory, 100.0, None)
    assert row["outcome"] == "rank_timeout"
    assert len(row["calls"]) == 1
    assert row["calls"][0]["phase_wall_seconds"] == 3.0
    assert row["load_wall_seconds"] == 1.0
    assert row["worker_wall_seconds"] == 185.0


def test_zero_exit_with_partial_calls_is_exception(tmp_path):
    directory = tmp_path / ("b" * 32)
    job, _ = synthetic_files(directory, status="worker_exit", returncode=0)
    assert (
        runner.normalize_job(job, directory, 100.0, None)["outcome"]
        == "worker_exception"
    )


@pytest.mark.parametrize(
    "field,value",
    [("job_id", "other"), ("invocation", "c" * 32), ("process_group_absent", False)],
)
def test_guardian_identity_or_cleanup_drift_fails(tmp_path, field, value):
    directory = tmp_path / ("d" * 32)
    job, receipt = synthetic_files(directory)
    receipt[field] = value
    atomic_json(directory / "guardian_receipt.json", receipt)
    with pytest.raises(ValueError):
        runner.normalize_job(job, directory, 100.0, None)


def test_checksums_do_not_allow_cross_invocation_result(tmp_path):
    directory = tmp_path / ("e" * 32)
    job, receipt = synthetic_files(directory)
    result_path = directory / "calls/0000_result.json"
    from neural_sat_speck.backend.guard import read_checked_json

    result = copy.deepcopy(read_checked_json(result_path))
    result["invocation"] = "f" * 32
    atomic_json(result_path, result)
    receipt["accepted_events"][1]["result_sha256"] = streaming_sha256(result_path)
    atomic_json(directory / "guardian_receipt.json", receipt)
    with pytest.raises(ValueError, match="identity"):
        runner.normalize_job(job, directory, 100.0, None)


def test_rank_payload_has_no_truth_dataset_path_or_solver_state():
    manifest = {
        "source_root": "/snapshot",
        "source_commit": "1" * 40,
        "files": {
            "source:src/test.py": {"path": "/snapshot/src/test.py", "sha256": "2" * 64},
            "cohort": {"path": "/truth/dataset.json", "sha256": "3" * 64},
        },
        "assets": {},
        "runtime": {},
    }
    case = {
        "cipher": "simon32_64",
        "rounds": 10,
        "key_index": 0,
        "instance_id": "demo0",
        "ciphertext_pairs": [[[0, 1], [2, 3]]] * 64,
    }
    payload = runner.public_payload(
        manifest, fixed_jobs()[0], case, [], "a" * 32, Path("/run")
    )
    assert "cohort" not in payload
    assert "/truth" not in str(payload)
    assert payload["public_case"] == case
    assert payload["source_files"] == {
        "src/test.py": manifest["files"]["source:src/test.py"]
    }


@pytest.mark.parametrize("drift", [None, "runtime", "asset", "depth", "source"])
def test_partial_rank_worker_requires_loaded_identity(tmp_path, monkeypatch, drift):
    from neural_sat_speck.followups import stage2_watchdog

    monkeypatch.setattr(stage2_watchdog, "live_groups", lambda token: [])
    directory = tmp_path / ("c" * 32)
    directory.mkdir()
    job = fixed_jobs()[1]
    identities = {name: {"size": 1} for name in ("vv9", "vv8", "ddt")}
    manifest = {
        "source_root": "/snapshot",
        "source_commit": "1" * 40,
        "jobs": [job],
        "files": {"tf_python": {"path": "/venv/bin/python"}},
        "runtime": {"tensorflow": {"packages": {"tensorflow": "2.16.2"}}},
        "assets": {
            name: {"identity": identity, "sha256": "2" * 64}
            for name, identity in identities.items()
        },
    }
    case = {"ciphertext_pairs": []}
    parents = list(range(8))
    atomic_json(
        directory / "payload.json",
        runner.public_payload(manifest, job, case, parents, directory.name, directory),
    )
    guardian = {
        "worker_arguments": [
            "/venv/bin/python",
            "/snapshot/scripts/run_simon_matched_stage2_worker.py",
            str(directory / "payload.json"),
        ],
        "invocation": directory.name,
        "job_id": job["job_id"],
        "campaign_deadline_monotonic": 1000.0,
        "load_seconds": 120.0,
        "call_seconds": 180.0,
        "expected_calls": 8,
    }
    atomic_json(directory / "guardian.json", guardian)
    atomic_json(
        directory / "process.json",
        {"invocation": directory.name, "pid": 12345, "process_group": 12345},
    )
    loaded = {
        "runtime": copy.deepcopy(manifest["runtime"]["tensorflow"]),
        "asset_identity": copy.deepcopy(identities),
        "loader_metadata": {"rounds": 8, "sha256": "2" * 64},
        "telemetry": {"git_commit": manifest["source_commit"]},
    }
    if drift == "runtime":
        loaded["runtime"]["packages"]["tensorflow"] = "other"
    elif drift == "asset":
        loaded["asset_identity"]["vv8"]["size"] = 2
    elif drift == "depth":
        loaded["loader_metadata"]["rounds"] = 7
    elif drift == "source":
        loaded["telemetry"]["git_commit"] = "3" * 40
    event_path = directory / "events/0000.json"
    atomic_json(event_path, loaded)
    event = {
        **loaded,
        "event_path": "events/0000.json",
        "event_sha256": streaming_sha256(event_path),
    }
    row = {
        **job,
        "directory": str(directory),
        "calls": [],
        "accepted_events": [event],
        "load_wall_seconds": 1.0,
        "outcome": "rank_timeout",
        "guardian_receipt": {
            "config_sha256": streaming_sha256(directory / "guardian.json"),
            "worker_pid": 12345,
            "process_group": 12345,
        },
    }
    if drift:
        with pytest.raises(ValueError, match="Partial worker"):
            runner.validate_execution_identity(manifest, row, case, parents, 1000.0)
    else:
        # There is deliberately no final worker_result.json after interruption.
        runner.validate_execution_identity(manifest, row, case, parents, 1000.0)
