"""Regression tests for worker-exit sampling and its independent receipts.

All worker and rank artifacts below are synthetic.  These tests never launch a
SIMON worker, load a model, or evaluate a key.
"""

from __future__ import annotations

import errno
import hashlib
import os
import sys
import time
from pathlib import Path

import pytest

from neural_sat_speck.backend.guard import atomic_json, read_checked_json
from neural_sat_speck.followups import stage2_receipts
from neural_sat_speck.followups import stage2_runner as runner
from neural_sat_speck.followups.stage2_control import fixed_jobs

WORKER_PID = 3003
WATCHDOG_PID = 2002
COORDINATOR_PID = 1001


def policy(*, nice=(0, 0), background=(0, 0), accepted=None):
    observations = {
        "nice": {"value": nice[0], "errno": nice[1]},
        "background_policy": {"value": background[0], "errno": background[1]},
    }
    if accepted is None:
        accepted = all(item == {"value": 0, "errno": 0} for item in observations.values())
    return {
        "pid": WORKER_PID,
        "scope": "process_priority_and_background_policy",
        "native_thread_qos": "unavailable",
        "supported": True,
        "observations": observations,
        "accepted": accepted,
    }


def test_exit_classifier_accepts_only_esrch_and_successful_zero_queries():
    assert stage2_receipts.is_process_exit_policy(
        policy(nice=(-1, errno.ESRCH)), WORKER_PID
    )
    assert stage2_receipts.is_process_exit_policy(
        policy(nice=(-1, errno.ESRCH), background=(0, 0)), WORKER_PID
    )
    assert stage2_receipts.is_process_exit_policy(
        policy(nice=(-1, errno.ESRCH), background=(-1, errno.ESRCH)), WORKER_PID
    )
    assert not stage2_receipts.is_process_exit_policy(
        policy(nice=(1, 0), accepted=False), WORKER_PID
    )
    assert not stage2_receipts.is_process_exit_policy(
        policy(nice=(-1, errno.EIO), accepted=False), WORKER_PID
    )
    assert not stage2_receipts.is_process_exit_policy(
        policy(nice=(-1, errno.EPERM), accepted=False), WORKER_PID
    )
    unsupported = policy(nice=(-1, errno.ESRCH), accepted=False)
    unsupported["supported"] = False
    assert not stage2_receipts.is_process_exit_policy(unsupported, WORKER_PID)
    assert not stage2_receipts.is_process_exit_policy(
        policy(nice=(-1, errno.ESRCH)), WORKER_PID + 1
    )


def _write_worker_exit_artifacts(directory: Path, config: dict, *, returncode=0):
    """Write a synthetic, fully accepted eight-call guardian result."""
    job_id = config["job_id"]
    invocation = config["invocation"]
    job = next(item for item in fixed_jobs() if item["job_id"] == job_id)
    assert job["operation"] == "stage2"
    parents = list(range(100, 108))
    base = time.perf_counter()
    loaded_at = base
    events = [
        {
            "invocation": invocation,
            "job_id": job_id,
            "event": "loaded",
            "completed_monotonic": loaded_at,
            "load_wall_seconds": max(0.0, loaded_at - config["job_started_monotonic"]),
        }
    ]
    previous = loaded_at
    for ordinal, parent in enumerate(parents):
        started = previous + 0.0001
        completed = started + 0.0001
        event_completed = completed + 0.0001
        identity = {
            "invocation": invocation,
            "job_id": job_id,
            "scorer": job["scorer"],
            "ordinal": ordinal,
            "parent": parent,
            "call_started_monotonic": started,
        }
        atomic_json(directory / f"calls/{ordinal:04d}_start.json", identity)
        result_path = directory / f"calls/{ordinal:04d}_result.json"
        atomic_json(
            result_path,
            {
                **identity,
                "call_completed_monotonic": completed,
                "ordered_words": list(range(512)),
                "component_times_seconds": {"total": 0.0001},
            },
        )
        events.append(
            {
                "invocation": invocation,
                "job_id": job_id,
                "event": "call_complete",
                "ordinal": ordinal,
                "completed_monotonic": event_completed,
                "result": f"calls/{ordinal:04d}_result.json",
                "result_sha256": runner.streaming_sha256(result_path),
            }
        )
        previous = event_completed

    receipt = {
        "schema": "simon_stage2_guardian_receipt_v1",
        "status": "worker_exit",
        "worker_returncode": returncode,
        "invocation": invocation,
        "job_id": job_id,
        "watchdog_pid": WATCHDOG_PID,
        "worker_pid": WORKER_PID,
        "process_group": WORKER_PID,
        "worker_not_started": False,
        "process_group_absent": True,
        "config_sha256": runner.streaming_sha256(directory / "guardian.json"),
        "cleanup_completed_monotonic": time.perf_counter(),
        "accepted_events": events,
    }
    atomic_json(directory / "guardian_receipt.json", receipt)
    return receipt


def _run_job_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    process_policy=None,
    power_ok=True,
    rss_ok=True,
    returncode=0,
    receipt_mode="late",
    preexisting_receipt=False,
):
    job = fixed_jobs()[1]
    invocation = "a" * 32
    directory = tmp_path / invocation
    (directory / "calls").mkdir(parents=True)
    manifest = {
        "source_root": str(Path(__file__).resolve().parents[2]),
        "source_commit": "a" * 40,
        "files": {
            "python": {"path": sys.executable},
            "tf_python": {"path": sys.executable},
        },
        "assets": {},
        "runtime": {},
    }
    parents = list(range(100, 108))
    power = {"accepted": power_ok}
    memory = {
        "accepted": rss_ok,
        "group_rss_bytes": 1024 if rss_ok else stage2_receipts.RSS_LIMIT_BYTES + 1,
        "members_observed": 1,
    }
    sampled_policy = process_policy or policy(nice=(-1, errno.ESRCH), background=(0, 0))
    monkeypatch.setattr(runner, "power_sample", lambda: power)
    monkeypatch.setattr(runner, "memory_sample", lambda _group: memory)
    monkeypatch.setattr(runner, "native_policy_sample", lambda _pid: sampled_policy)
    monkeypatch.setattr(runner.time, "sleep", lambda _duration: None)

    class FakeGuardian:
        def __init__(self, arguments, **kwargs):
            assert kwargs["start_new_session"] is True
            self.directory = Path(arguments[-1]).parent
            self.config = read_checked_json(self.directory / "guardian.json")
            self.poll_count = 0
            self.terminated = False
            self.wrote_receipt = False
            atomic_json(
                self.directory / "process.json",
                {
                    "invocation": invocation,
                    "job_id": job["job_id"],
                    "pid": WORKER_PID,
                    "process_group": WORKER_PID,
                },
            )
            if preexisting_receipt:
                _write_worker_exit_artifacts(self.directory, self.config, returncode=returncode)
                self.wrote_receipt = True

        def poll(self):
            self.poll_count += 1
            return None if self.poll_count == 1 else 0

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            assert timeout == 10
            if receipt_mode == "missing":
                return returncode
            if receipt_mode == "corrupt":
                (self.directory / "guardian_receipt.json").write_text("{broken", encoding="utf-8")
                return returncode
            if not self.wrote_receipt:
                assert not (self.directory / "guardian_receipt.json").exists()
                assert not (self.directory / "monitor_complete.json").exists()
                _write_worker_exit_artifacts(
                    self.directory,
                    self.config,
                    returncode=(
                        -15 if self.terminated and returncode == 0 else returncode
                    ),
                )
                self.wrote_receipt = True
            return returncode

    fake_guardians = []

    def fake_popen(*arguments, **kwargs):
        guardian = FakeGuardian(*arguments, **kwargs)
        fake_guardians.append(guardian)
        return guardian

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    row = runner.run_job(
        manifest,
        job,
        {"ciphertext_pairs": []},
        parents,
        directory,
        time.perf_counter() + 3600,
        {},
    )
    return row, directory, fake_guardians[0]


def test_run_job_waits_for_late_guardian_receipt_after_esrch_sample(tmp_path, monkeypatch):
    row, directory, guardian = _run_job_case(tmp_path, monkeypatch)

    assert not guardian.terminated
    assert row["outcome"] == "completed"
    assert len(row["calls"]) == 8
    assert row["guardian_receipt"]["status"] == "worker_exit"
    samples = read_checked_json(directory / "environment_samples.json")
    assert samples[0]["process_policy"]["observations"]["nice"]["errno"] == errno.ESRCH
    monitor = read_checked_json(directory / "monitor_complete.json")
    assert monitor["schema"] == "stage2_monitor_complete_v1"
    assert monitor["invocation"] == directory.name
    assert monitor["job_id"] == row["job_id"]
    assert monitor["coordinator_pid"] == os.getpid()
    assert monitor["guardian_receipt_sha256"] == hashlib.sha256(
        (directory / "guardian_receipt.json").read_bytes()
    ).hexdigest()
    assert monitor["completed_monotonic"] >= row["guardian_receipt"]["cleanup_completed_monotonic"]


def test_nonzero_worker_exit_remains_worker_exception_with_accepted_calls(tmp_path, monkeypatch):
    row, _directory, guardian = _run_job_case(
        tmp_path,
        monkeypatch,
        returncode=7,
    )

    assert not guardian.terminated
    assert row["outcome"] == "worker_exception"
    assert len(row["calls"]) == 8


@pytest.mark.parametrize("receipt_mode", ["missing", "corrupt"])
def test_run_job_fails_closed_when_post_wait_receipt_is_missing_or_corrupt(
    tmp_path, monkeypatch, receipt_mode
):
    with pytest.raises((FileNotFoundError, ValueError)):
        _run_job_case(tmp_path, monkeypatch, receipt_mode=receipt_mode)


def test_real_priority_fault_is_not_excused_by_receipt_already_present(tmp_path, monkeypatch):
    real_fault = policy(nice=(1, 0), background=(0, 0), accepted=False)
    row, _directory, guardian = _run_job_case(
        tmp_path,
        monkeypatch,
        process_policy=real_fault,
        preexisting_receipt=True,
    )

    assert guardian.terminated
    assert row["outcome"] == "environment_alarm"


@pytest.mark.parametrize(
    ("power_ok", "rss_ok", "expected_outcome"),
    [(False, True, "environment_alarm"), (True, False, "rss_stop")],
)
def test_power_or_rss_alarm_overrides_esrch_exit_classification(
    tmp_path, monkeypatch, power_ok, rss_ok, expected_outcome
):
    row, directory, guardian = _run_job_case(
        tmp_path,
        monkeypatch,
        power_ok=power_ok,
        rss_ok=rss_ok,
    )

    assert guardian.terminated
    assert row["outcome"] == expected_outcome
    samples = read_checked_json(directory / "environment_samples.json")
    assert stage2_receipts.is_process_exit_policy(
        samples[0]["process_policy"], WORKER_PID
    )


def _resource_receipt_case(tmp_path: Path):
    source_root = Path(__file__).resolve().parents[2]
    invocation = "b" * 32
    job_id = "simon_r10_k00_stage2_vv8"
    directory = tmp_path / invocation
    directory.mkdir()
    guardian = {
        "invocation": invocation,
        "job_id": job_id,
        "coordinator_pid": COORDINATOR_PID,
        "job_started_monotonic": 10.0,
        "campaign_deadline_monotonic": 20.0,
    }
    atomic_json(directory / "guardian.json", guardian)
    receipt = {
        "schema": "simon_stage2_guardian_receipt_v1",
        "status": "worker_exit",
        "worker_returncode": 0,
        "invocation": invocation,
        "job_id": job_id,
        "watchdog_pid": WATCHDOG_PID,
        "worker_pid": WORKER_PID,
        "process_group": WORKER_PID,
        "worker_not_started": False,
        "process_group_absent": True,
        "config_sha256": runner.streaming_sha256(directory / "guardian.json"),
        "cleanup_completed_monotonic": 11.5,
        "accepted_events": [],
    }
    atomic_json(directory / "guardian_receipt.json", receipt)
    atomic_json(
        directory / "process.json",
        {
            "invocation": invocation,
            "job_id": job_id,
            "guardian_pid": WATCHDOG_PID,
            "pid": WORKER_PID,
            "process_group": WORKER_PID,
        },
    )
    receipt_path = directory / "guardian_receipt.json"
    monitor = {
        "schema": "stage2_monitor_complete_v1",
        "invocation": invocation,
        "job_id": job_id,
        "coordinator_pid": COORDINATOR_PID,
        "guardian_receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "completed_monotonic": 12.0,
    }
    atomic_json(directory / "monitor_complete.json", monitor)
    power = {
        "utc": "2026-10-06T00:00:00+00:00",
        "supported": True,
        "raw_assertions": "sleep assertion output",
        "raw_power": "Now drawing from 'AC Power'\n",
        "assertions_returncode": 0,
        "power_returncode": 0,
        "ac_power_observed": True,
        "sleep_assertion_observed": True,
        "accepted": True,
        "original_ac_policy_accepted": True,
        "required_policy": stage2_receipts.POWER_POLICY,
        "power_source": "AC Power",
    }
    live_policy = policy()
    exit_observation = policy(nice=(-1, errno.ESRCH), background=(0, 0))
    samples = [
        {
            "monotonic": 10.5,
            "power": power,
            "memory": {"group_rss_bytes": 1024, "members_observed": 1, "accepted": True},
            "process_policy": live_policy,
        },
        {
            "monotonic": 11.25,
            "power": power,
            "memory": {"group_rss_bytes": 0, "members_observed": 0, "accepted": True},
            "process_policy": exit_observation,
        },
    ]
    atomic_json(directory / "environment_samples.json", samples)
    row = {
        "job_id": job_id,
        "scorer": "vv8",
        "outcome": "worker_exception",
        "dispatched": True,
        "directory": str(directory),
        "invocation": invocation,
        "guardian_receipt": receipt,
        "accepted_events": [],
        "sleep_assertion": {
            "supported": True,
            "pid": 4004,
            "coordinator_pid": COORDINATOR_PID,
            "arguments": [
                "/usr/bin/caffeinate",
                "-d",
                "-i",
                "-m",
                "-s",
                "-w",
                str(COORDINATOR_PID),
            ],
        },
    }
    manifest = {"source_root": str(source_root), "source_commit": "a" * 40}
    return manifest, row, directory


def test_resource_validator_accepts_monitored_exit_sample_without_counting_it_live(tmp_path):
    manifest, row, _directory = _resource_receipt_case(tmp_path)

    summary = stage2_receipts.validate_resource_receipts(
        manifest, row, require_worker_sample=False
    )

    assert summary["worker_samples"] == 1
    assert summary["accepted_worker_samples"] == 1
    assert summary["exit_policy_observations"] == 1


def test_resource_validator_requires_checked_monitor_to_excuse_esrch(tmp_path):
    manifest, row, directory = _resource_receipt_case(tmp_path)
    (directory / "monitor_complete.json").unlink()
    (directory / "monitor_complete.json.sha256").unlink()

    with pytest.raises(ValueError, match="monitor|exit|terminal"):
        stage2_receipts.validate_resource_receipts(manifest, row)


@pytest.mark.parametrize(
    ("field", "value"),
    [("coordinator_pid", COORDINATOR_PID + 1), ("guardian_receipt_sha256", "0" * 64)],
)
def test_resource_validator_rejects_misbound_monitor_completion(tmp_path, field, value):
    manifest, row, directory = _resource_receipt_case(tmp_path)
    monitor_path = directory / "monitor_complete.json"
    monitor = read_checked_json(monitor_path)
    monitor[field] = value
    atomic_json(monitor_path, monitor)

    with pytest.raises(ValueError, match="monitor|exit|terminal"):
        stage2_receipts.validate_resource_receipts(manifest, row)


def test_resource_validator_rejects_live_policy_sample_after_guardian_cleanup(tmp_path):
    manifest, row, directory = _resource_receipt_case(tmp_path)
    samples_path = directory / "environment_samples.json"
    samples = read_checked_json(samples_path)
    samples = samples[:1]
    samples[0]["monotonic"] = 11.75
    atomic_json(samples_path, samples)

    with pytest.raises(ValueError, match="Live worker policy sample follows guardian cleanup"):
        stage2_receipts.validate_resource_receipts(manifest, row)


def test_resource_validator_rejects_live_sample_after_esrch_observation(tmp_path):
    manifest, row, directory = _resource_receipt_case(tmp_path)
    samples_path = directory / "environment_samples.json"
    samples = read_checked_json(samples_path)
    samples[0]["memory"] = {
        "group_rss_bytes": 0,
        "members_observed": 0,
        "accepted": True,
    }
    samples[0]["process_policy"] = policy(
        nice=(-1, errno.ESRCH), background=(0, 0)
    )
    samples[1]["memory"] = {
        "group_rss_bytes": 1024,
        "members_observed": 1,
        "accepted": True,
    }
    samples[1]["process_policy"] = policy()
    atomic_json(samples_path, samples)

    with pytest.raises(ValueError, match="Worker appeared live after an exit-policy observation"):
        stage2_receipts.validate_resource_receipts(
            manifest, row, require_worker_sample=False
        )


def test_resource_validator_rejects_monitor_at_or_after_campaign_deadline(tmp_path):
    manifest, row, directory = _resource_receipt_case(tmp_path)
    guardian_path = directory / "guardian.json"
    guardian = read_checked_json(guardian_path)
    guardian["campaign_deadline_monotonic"] = 12.0
    atomic_json(guardian_path, guardian)
    receipt_path = directory / "guardian_receipt.json"
    receipt = read_checked_json(receipt_path)
    receipt["config_sha256"] = runner.streaming_sha256(guardian_path)
    atomic_json(receipt_path, receipt)
    row["guardian_receipt"] = receipt
    monitor_path = directory / "monitor_complete.json"
    monitor = read_checked_json(monitor_path)
    monitor["guardian_receipt_sha256"] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    monitor["completed_monotonic"] = 12.0
    atomic_json(monitor_path, monitor)

    with pytest.raises(ValueError, match="campaign deadline|monitor|deadline"):
        stage2_receipts.validate_resource_receipts(
            manifest, row, require_worker_sample=False
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", None),
        ("worker_returncode", "0"),
        ("process_group_absent", False),
    ],
)
def test_resource_validator_requires_typed_terminal_worker_receipt(tmp_path, field, value):
    manifest, row, directory = _resource_receipt_case(tmp_path)
    receipt_path = directory / "guardian_receipt.json"
    receipt = read_checked_json(receipt_path)
    receipt[field] = value
    atomic_json(receipt_path, receipt)
    row["guardian_receipt"] = receipt
    monitor_path = directory / "monitor_complete.json"
    monitor = read_checked_json(monitor_path)
    monitor["guardian_receipt_sha256"] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    atomic_json(monitor_path, monitor)

    with pytest.raises(ValueError, match="terminal|process group|absent|worker policy"):
        stage2_receipts.validate_resource_receipts(manifest, row)
