from __future__ import annotations

import copy
import hashlib
import sys
import uuid
from pathlib import Path

import pytest

from neural_sat_speck.backend.guard import atomic_json
from neural_sat_speck.followups.stage2_receipts import (
    POWER_POLICY,
    RSS_LIMIT_BYTES,
    THREAD_ENV,
    validate_resource_receipts,
)


@pytest.fixture
def receipt_case(tmp_path: Path):
    source = Path(__file__).resolve().parents[2]
    temp = source / "runtime" / f"stage2_receipts_test_{uuid.uuid4().hex}"
    temp.mkdir(parents=True)
    directory = tmp_path / "invocation-0001"
    (directory / "events").mkdir(parents=True)
    (directory / "calls").mkdir()

    executable = str(Path(sys.executable).absolute())
    identity = {"device": 1, "inode": 2, "size": 3, "mtime_ns": 4}
    runtime = {"executable": executable, "python_version": "3.12.1"}
    manifest = {
        "source_root": str(source),
        "source_commit": "a" * 40,
        "files": {
            "python": {"path": executable, "sha256": "1" * 64},
            "tf_python": {"path": executable, "sha256": "2" * 64},
        },
        "runtime": {"python": runtime, "tensorflow": runtime},
        "assets": {
            name: {"identity": identity, "sha256": str(index) * 64}
            for index, name in enumerate(("vv9", "vv8", "ddt"), 1)
        },
    }
    invocation = directory.name
    job_id = "simon_r10_k00_stage1_vv9"
    telemetry = {
        "observed_at": "2026-10-05T05:00:00+00:00",
        "pid": 3003,
        "parent_pid": 2002,
        "thread_id": 501,
        "platform": "darwin",
        "qos_class": 33,
        "darwin_priority": 0,
        "darwin_bg_state": 0,
        "io_policy_state": 0,
        "thread_environment": {name: "8" if name == "TF_NUM_INTRAOP_THREADS" else "1" for name in THREAD_ENV},
        "tmpdir": str(temp),
        "tmpdir_safe": True,
        "git_commit": manifest["source_commit"],
        "python_executable": executable,
        "command_line": [
            str(source / "scripts/run_simon_matched_stage2_worker.py"),
            str(directory / "payload.json"),
        ],
        "sleep_assertions": {"supported": True},
        "is_certified": True,
        "rejection_reason": None,
    }
    loaded = {
        "schema": "stage2_rank_worker_event_v1",
        "event": "loaded",
        "event_index": 0,
        "invocation": invocation,
        "job_id": job_id,
        "scorer": "vv9",
        "completed_monotonic": 10.5,
        "load_wall_seconds": 0.5,
        "telemetry": telemetry,
        "runtime": runtime,
        "loader_metadata": {"kind": "published_vv_checkpoint", "rounds": 9},
        "asset_identity": {name: identity for name in ("vv9", "vv8", "ddt")},
    }
    event_path = directory / "events/0000.json"
    atomic_json(event_path, loaded)
    loaded_record = {
        **loaded,
        "event_path": "events/0000.json",
        "event_sha256": hashlib.sha256(event_path.read_bytes()).hexdigest(),
    }
    guardian = {
        "invocation": invocation,
        "job_id": job_id,
        "coordinator_pid": 1001,
        "job_started_monotonic": 10.0,
    }
    guardian_receipt = {
        "schema": "simon_stage2_guardian_receipt_v1",
        "invocation": invocation,
        "job_id": job_id,
        "watchdog_pid": 2002,
        "worker_pid": 3003,
        "process_group": 3003,
        "worker_not_started": False,
        "process_group_absent": True,
        "cleanup_completed_monotonic": 11.5,
    }
    process = {
        "pid": 3003,
        "process_group": 3003,
        "invocation": invocation,
        "job_id": job_id,
        "guardian_pid": 2002,
    }
    for name, value in (
        ("guardian.json", guardian),
        ("guardian_receipt.json", guardian_receipt),
        ("process.json", process),
    ):
        atomic_json(directory / name, value)
    policy = {
        "pid": 3003,
        "scope": "process_priority_and_background_policy",
        "native_thread_qos": "unavailable",
        "supported": True,
        "observations": {
            "nice": {"value": 0, "errno": 0},
            "background_policy": {"value": 0, "errno": 0},
        },
        "accepted": True,
    }
    power = {
        "utc": "2026-10-05T05:00:00+00:00",
        "platform": "darwin",
        "supported": True,
        "raw_assertions": "sleep assertion output",
        "raw_power": "Now drawing from 'AC Power'\n",
        "assertions_returncode": 0,
        "power_returncode": 0,
        "ac_power_observed": True,
        "sleep_assertion_observed": True,
        "accepted": True,
        "original_ac_policy_accepted": True,
        "required_policy": POWER_POLICY,
        "power_source": "AC Power",
    }
    samples = [
        {
            "monotonic": 10.25,
            "power": power,
            "memory": {"group_rss_bytes": 1024, "members_observed": 1, "accepted": True},
            "process_policy": policy,
        }
    ]
    atomic_json(directory / "environment_samples.json", samples)
    final = {
        "schema": "stage2_rank_worker_result_v1",
        "status": "complete",
        "telemetry": telemetry,
        "runtime": runtime,
        "loader_metadata": loaded["loader_metadata"],
    }
    atomic_json(directory / "worker_result.json", final)
    assertion = {
        "supported": True,
        "pid": 4004,
        "coordinator_pid": 1001,
        "arguments": ["/usr/bin/caffeinate", "-d", "-i", "-m", "-s", "-w", "1001"],
    }
    row = {
        "job_id": job_id,
        "scorer": "vv9",
        "outcome": "completed",
        "dispatched": True,
        "directory": str(directory),
        "invocation": invocation,
        "guardian_receipt": guardian_receipt,
        "accepted_events": [loaded_record],
        "sleep_assertion": assertion,
    }
    yield {"manifest": manifest, "row": row, "directory": directory, "loaded": loaded, "power": power, "policy": policy}
    temp.rmdir()


def test_completed_pilot_accepts_bound_rss_policy_and_ac_receipts(receipt_case):
    summary = validate_resource_receipts(
        receipt_case["manifest"], receipt_case["row"], require_worker_sample=True
    )
    assert summary["worker_samples"] == 1
    assert summary["accepted_worker_samples"] == 1
    assert summary["power_sources"] == ["AC Power"]
    assert summary["worker"]["pid"] == 3003


def test_battery_is_valid_under_explicit_author_policy(receipt_case):
    case = receipt_case
    samples_path = case["directory"] / "environment_samples.json"
    from neural_sat_speck.backend.guard import read_checked_json

    samples = read_checked_json(samples_path)
    power = {**case["power"], "ac_power_observed": False, "original_ac_policy_accepted": False, "power_source": "Battery Power", "raw_power": "Now drawing from 'Battery Power'\n"}
    samples[0]["power"] = power
    atomic_json(samples_path, samples)
    summary = validate_resource_receipts(case["manifest"], case["row"])
    assert summary["power_sources"] == ["Battery Power"]


def test_pilot_requires_observed_worker_member(receipt_case):
    case = receipt_case
    from neural_sat_speck.backend.guard import read_checked_json

    samples_path = case["directory"] / "environment_samples.json"
    sample = read_checked_json(samples_path)[0]
    sample["memory"] = {"group_rss_bytes": 0, "members_observed": 0, "accepted": True}
    sample["process_policy"]["observations"] = {
        "nice": {"value": -1, "errno": 3},
        "background_policy": {"value": -1, "errno": 3},
    }
    sample["process_policy"]["accepted"] = False
    atomic_json(samples_path, [sample])
    with pytest.raises(ValueError, match="accepted worker RSS"):
        validate_resource_receipts(case["manifest"], case["row"], require_worker_sample=True)


def test_rss_stop_accepts_only_a_real_over_limit_sample(receipt_case):
    case = receipt_case
    from neural_sat_speck.backend.guard import read_checked_json

    samples_path = case["directory"] / "environment_samples.json"
    sample = read_checked_json(samples_path)[0]
    sample["memory"] = {"group_rss_bytes": RSS_LIMIT_BYTES + 1, "members_observed": 1, "accepted": False}
    atomic_json(samples_path, [sample])
    case["row"]["outcome"] = "rss_stop"
    case["row"]["accepted_events"] = []
    (case["directory"] / "worker_result.json").unlink()
    (case["directory"] / "worker_result.json.sha256").unlink()
    summary = validate_resource_receipts(case["manifest"], case["row"], require_worker_sample=False)
    assert summary["rss_stop_observed"] is True
    case["row"]["outcome"] = "rank_timeout"
    with pytest.raises(ValueError, match="does not match rss_stop"):
        validate_resource_receipts(case["manifest"], case["row"], require_worker_sample=False)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("pid", 3999, "Telemetry PID"),
        ("parent_pid", 2999, "parent PID"),
        ("git_commit", "b" * 40, "source commit"),
        ("tmpdir_safe", False, "TMPDIR"),
        ("thread_environment", {name: "2" for name in THREAD_ENV}, "thread environment"),
        ("qos_class", 9, "QoS"),
    ],
)
def test_telemetry_must_match_observed_runtime_policy(receipt_case, field, value, message):
    case = receipt_case
    loaded = copy.deepcopy(case["loaded"])
    loaded["telemetry"][field] = value
    event_path = case["directory"] / "events/0000.json"
    atomic_json(event_path, loaded)
    case["row"]["accepted_events"][0] = {
        **loaded,
        "event_path": "events/0000.json",
        "event_sha256": hashlib.sha256(event_path.read_bytes()).hexdigest(),
    }
    final_path = case["directory"] / "worker_result.json"
    from neural_sat_speck.backend.guard import read_checked_json

    final = read_checked_json(final_path)
    final["telemetry"] = loaded["telemetry"]
    atomic_json(final_path, final)
    with pytest.raises(ValueError, match=message):
        validate_resource_receipts(case["manifest"], case["row"])


def test_process_policy_errno_is_not_hidden_by_accepted_flag(receipt_case):
    case = receipt_case
    from neural_sat_speck.backend.guard import read_checked_json

    samples_path = case["directory"] / "environment_samples.json"
    sample = read_checked_json(samples_path)[0]
    sample["process_policy"]["observations"]["nice"]["errno"] = 3
    atomic_json(samples_path, [sample])
    with pytest.raises(ValueError, match="contradicts observations"):
        validate_resource_receipts(case["manifest"], case["row"])


def test_sleep_assertion_owner_is_bound_to_guardian(receipt_case):
    case = receipt_case
    case["row"]["sleep_assertion"]["coordinator_pid"] = 1002
    case["row"]["sleep_assertion"]["arguments"][-1] = "1002"
    with pytest.raises(ValueError, match="not owned by coordinator"):
        validate_resource_receipts(case["manifest"], case["row"])


def test_corrupt_sample_sidecar_fails_closed(receipt_case):
    case = receipt_case
    sidecar = case["directory"] / "environment_samples.json.sha256"
    sidecar.write_text("0" * 64 + "  environment_samples.json\n")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        validate_resource_receipts(case["manifest"], case["row"])


def test_partial_power_only_sample_is_allowed_without_worker_observation(receipt_case):
    case = receipt_case
    from neural_sat_speck.backend.guard import read_checked_json

    samples_path = case["directory"] / "environment_samples.json"
    sample = read_checked_json(samples_path)[0]
    del sample["memory"]
    del sample["process_policy"]
    atomic_json(samples_path, [sample])
    case["row"]["outcome"] = "load_timeout"
    case["row"]["accepted_events"] = []
    (case["directory"] / "worker_result.json").unlink()
    (case["directory"] / "worker_result.json.sha256").unlink()
    summary = validate_resource_receipts(case["manifest"], case["row"], require_worker_sample=False)
    assert summary["worker_samples"] == 0
    with pytest.raises(ValueError, match="accepted worker RSS"):
        validate_resource_receipts(case["manifest"], case["row"], require_worker_sample=True)


def test_missing_samples_on_partial_row_remain_unobserved_not_zero_rss(receipt_case):
    case = receipt_case
    samples_path = case["directory"] / "environment_samples.json"
    samples_path.unlink()
    samples_path.with_name(samples_path.name + ".sha256").unlink()
    case["row"]["outcome"] = "rank_timeout"
    final_path = case["directory"] / "worker_result.json"
    final_path.unlink()
    final_path.with_name(final_path.name + ".sha256").unlink()
    summary = validate_resource_receipts(
        case["manifest"], case["row"], require_worker_sample=False
    )
    assert summary["monitoring_available"] is False
    assert summary["samples"] == 0
    assert summary["worker_samples"] == 0
    assert summary["worker"]["pid"] == 3003
    with pytest.raises(ValueError, match="Missing checked environment samples"):
        validate_resource_receipts(case["manifest"], case["row"], require_worker_sample=True)
