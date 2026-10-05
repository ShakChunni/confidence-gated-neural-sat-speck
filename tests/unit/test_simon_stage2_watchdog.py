from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from neural_sat_speck.backend.guard import atomic_json, read_checked_json
from neural_sat_speck.followups import stage2_watchdog

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


FAKE_WORKER = textwrap.dedent(
    r'''
    import hashlib
    import json
    import os
    import subprocess
    import sys
    import threading
    import time
    from pathlib import Path
    from neural_sat_speck.backend.guard import atomic_json, canonical_bytes, sha256_file

    payload = json.loads(Path(sys.argv[1]).read_text())
    root = Path(payload["directory"])
    events = root / "events"
    calls = root / "calls"
    events.mkdir(exist_ok=True)
    calls.mkdir(exist_ok=True)
    mode = payload.get("mode", "success")

    if mode == "hang_load":
        Path(payload["marker"]).write_text("loading")
        time.sleep(60)
        raise SystemExit(0)
    if mode == "future_event":
        atomic_json(events / "0001.json", {
            "schema": "stage2_rank_worker_event_v1", "event_index": 1,
            "event": "call_complete", "invocation": payload["invocation"],
            "job_id": payload["job_id"], "ordinal": 0,
            "completed_monotonic": time.perf_counter(),
        })
        time.sleep(60)
        raise SystemExit(0)

    load_done = time.perf_counter()
    loaded_event = {
        "schema": "stage2_rank_worker_event_v1", "event_index": 0,
        "event": "loaded", "scorer": "vv9" if payload["expected_calls"] == 1 else "vv8",
        "invocation": payload["invocation"],
        "job_id": payload["job_id"], "completed_monotonic": load_done,
        "load_wall_seconds": 0.01, "model_sha256": "a" * 64,
    }
    if mode == "delayed_loaded_sidecar":
        loaded_path = events / "0000.json"
        loaded_raw = canonical_bytes(loaded_event)
        loaded_path.write_bytes(loaded_raw)
        def write_loaded_sidecar():
            time.sleep(0.02)
            (events / "0000.json.sha256").write_text(
                f"{hashlib.sha256(loaded_raw).hexdigest()}  0000.json\n"
            )
        threading.Thread(target=write_loaded_sidecar, daemon=True).start()
        time.sleep(0.03)
    else:
        atomic_json(events / "0000.json", loaded_event)
    if mode == "hang_load_after_event":
        time.sleep(60)
        raise SystemExit(0)

    previous_event_completed = load_done
    for ordinal in range(payload["expected_calls"]):
        call_started = time.perf_counter()
        if mode == "backdated_second_call" and ordinal == 1:
            call_started = previous_event_completed - 0.001
        scorer = "vv9" if payload["expected_calls"] == 1 else "vv8"
        parent = None if payload["expected_calls"] == 1 else ordinal
        atomic_json(calls / f"{ordinal:04d}_start.json", {
            "invocation": payload["invocation"], "job_id": payload["job_id"],
            "ordinal": ordinal, "scorer": scorer, "parent": parent,
            "call_started_monotonic": call_started,
        })
        if mode == "hang_call" and ordinal == 0:
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
            Path(payload["marker"]).write_text(str(child.pid))
            time.sleep(60)
            raise SystemExit(0)

        call_done = time.perf_counter()
        result = {
            "invocation": payload["invocation"], "job_id": payload["job_id"],
            "ordinal": ordinal, "scorer": scorer, "parent": parent,
            "ordered_words": list(range(512)),
            "component_times_seconds": {"transform": 0.01, "score": 0.02},
            "call_started_monotonic": call_started,
            "call_completed_monotonic": call_done,
        }
        if mode == "wrong_result_identity":
            result["job_id"] = "another-job"
        result_path = calls / f"{ordinal:04d}_result.json"
        atomic_json(result_path, result)
        completed = time.perf_counter()
        result_hash = sha256_file(result_path)
        if mode == "bad_result_hash":
            result_hash = "0" * 64
        event = {
            "schema": "stage2_rank_worker_event_v1", "event_index": ordinal + 1,
            "event": "call_complete", "invocation": payload["invocation"],
            "job_id": payload["job_id"], "ordinal": ordinal,
            "scorer": scorer, "parent": parent,
            "completed_monotonic": completed,
            "result": f"calls/{ordinal:04d}_result.json",
            "result_path": f"calls/{ordinal:04d}_result.json",
            "result_sha256": result_hash,
        }
        if mode == "wrong_event_identity":
            event["invocation"] = "f" * 32
        atomic_json(events / f"{ordinal + 1:04d}.json", event)
        previous_event_completed = completed
        if mode == "hang_finalize" and ordinal + 1 == payload["expected_calls"]:
            Path(payload["marker"]).write_text("finalizing")
            time.sleep(60)
            raise SystemExit(0)

    if mode == "slow_exit":
        Path(payload["marker"]).write_text("finalizing")
        time.sleep(60)
    '''
)


def _make_invocation(
    tmp_path: Path,
    *,
    mode: str = "success",
    expected_calls: int = 1,
    load_seconds: float = 1.0,
    call_seconds: float = 1.0,
    campaign_seconds: float = 10.0,
) -> tuple[Path, dict, Path]:
    invocation = f"{len(list(tmp_path.iterdir())) + 1:032x}"
    directory = tmp_path / invocation
    repo_root = tmp_path / "fake_repo"
    script = repo_root / "scripts" / "run_simon_matched_stage2_worker.py"
    directory.mkdir(parents=True)
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(FAKE_WORKER)
    payload_path = directory / "payload.json"
    marker = directory / "worker.marker"
    payload = {
        "directory": str(directory),
        "invocation": invocation,
        "job_id": "test-job-1",
        "expected_calls": expected_calls,
        "mode": mode,
        "marker": str(marker),
    }
    atomic_json(payload_path, payload)
    job_started = time.perf_counter()
    config = {
        "invocation": invocation,
        "directory": str(directory),
        "repo_root": str(repo_root),
        "coordinator_pid": os.getpid(),
        "worker_arguments": [sys.executable, str(script), str(payload_path)],
        "receipt": str(directory / "guardian_receipt.json"),
        "job_id": "test-job-1",
        "job_started_monotonic": job_started,
        "campaign_deadline_monotonic": job_started + campaign_seconds,
        "load_seconds": load_seconds,
        "call_seconds": call_seconds,
        "expected_calls": expected_calls,
    }
    config_path = directory / "guardian_config.json"
    atomic_json(config_path, config)
    return config_path, config, marker


def _run_guardian(config_path: Path, *, timeout: float = 8.0) -> subprocess.CompletedProcess:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(SRC), environment.get("PYTHONPATH", "")]
    )
    return subprocess.run(
        [sys.executable, "-m", "neural_sat_speck.followups.stage2_watchdog", str(config_path)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _receipt(config: dict) -> dict:
    return read_checked_json(Path(config["receipt"]))


def test_accepts_loaded_then_all_eight_calls_and_advances_deadline(tmp_path):
    config_path, config, _ = _make_invocation(
        tmp_path, expected_calls=8, load_seconds=2.0, call_seconds=1.0
    )
    completed = _run_guardian(config_path)
    assert completed.returncode == 0, completed.stderr
    receipt = _receipt(config)

    assert receipt["status"] == "worker_exit"
    assert receipt["returncode"] == 0
    assert receipt["process_group_absent"] is True
    accepted = receipt["accepted_events"]
    assert len(accepted) == 9
    assert [event["event"] for event in accepted] == ["loaded"] + [
        "call_complete"
    ] * 8
    assert [event.get("ordinal") for event in accepted[1:]] == list(range(8))
    assert all(len(event["event_sha256"]) == 64 for event in accepted)
    assert all(len(event["result_sha256"]) == 64 for event in accepted[1:])
    assert receipt["effective_deadline_monotonic"] == pytest.approx(
        accepted[-1]["completed_monotonic"] + config["call_seconds"]
    )


def test_waits_briefly_for_event_sidecar_without_rejecting_complete_event(tmp_path):
    config_path, config, _ = _make_invocation(
        tmp_path, mode="delayed_loaded_sidecar", load_seconds=1.0
    )
    completed = _run_guardian(config_path)
    assert completed.returncode == 0, completed.stderr
    receipt = _receipt(config)
    assert receipt["status"] == "worker_exit"
    assert [event["event"] for event in receipt["accepted_events"]] == [
        "loaded", "call_complete"
    ]


@pytest.mark.parametrize(
    ("mode", "expected_status", "accepted_count"),
    [
        ("hang_load", "load_timeout", 0),
        ("hang_call", "rank_timeout", 1),
        ("hang_finalize", "rank_timeout", 2),
    ],
)
def test_deadline_stops_hangs_and_retains_only_completed_calls(
    tmp_path, mode, expected_status, accepted_count
):
    config_path, config, _ = _make_invocation(
        tmp_path,
        mode=mode,
        load_seconds=0.45,
        call_seconds=0.45,
        campaign_seconds=4.0,
    )
    completed = _run_guardian(config_path)
    assert completed.returncode == 0, completed.stderr
    receipt = _receipt(config)

    assert receipt["status"] == expected_status
    assert len(receipt["accepted_events"]) == accepted_count
    assert receipt["process_group_absent"] is True
    if mode == "hang_finalize":
        last_call = receipt["accepted_events"][-1]
        assert receipt["effective_deadline_monotonic"] == pytest.approx(
            last_call["completed_monotonic"] + config["call_seconds"]
        )


@pytest.mark.parametrize(
    "mode",
    [
        "future_event",
        "bad_result_hash",
        "wrong_result_identity",
        "wrong_event_identity",
        "backdated_second_call",
    ],
)
def test_corrupt_or_nonsequential_journal_fails_closed(tmp_path, mode):
    config_path, config, _ = _make_invocation(
        tmp_path, mode=mode, expected_calls=8 if mode == "backdated_second_call" else 1,
        load_seconds=1.0, call_seconds=1.0,
    )
    completed = _run_guardian(config_path)
    assert completed.returncode != 0
    receipt = _receipt(config)
    assert receipt["status"] == "guardian_error"
    assert receipt["process_group_absent"] is True


def test_campaign_deadline_caps_loading_deadline(tmp_path):
    config_path, config, _ = _make_invocation(
        tmp_path,
        mode="hang_load",
        load_seconds=2.0,
        call_seconds=1.0,
        campaign_seconds=0.35,
    )
    completed = _run_guardian(config_path)
    assert completed.returncode == 0, completed.stderr
    receipt = _receipt(config)
    assert receipt["status"] == "campaign_timeout"
    assert receipt["effective_deadline_monotonic"] == config["campaign_deadline_monotonic"]
    assert receipt["process_group_absent"] is True


def test_configuration_drift_is_rejected_before_worker_dispatch(tmp_path):
    config_path, config, marker = _make_invocation(tmp_path)
    config["worker_arguments"][2] = str(tmp_path / "other" / "payload.json")
    atomic_json(config_path, config)
    completed = _run_guardian(config_path)
    assert completed.returncode != 0
    assert not marker.exists()
    assert not Path(config["receipt"]).exists()


def test_coordinator_loss_before_dispatch_writes_stop_receipt(tmp_path, monkeypatch):
    config_path, config, _ = _make_invocation(tmp_path)
    seen = iter((True, False))
    monkeypatch.setattr(stage2_watchdog, "_coordinator_is_parent", lambda _pid: next(seen))
    monkeypatch.setattr(stage2_watchdog.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(
        stage2_watchdog.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("worker dispatched after coordinator loss"),
    )

    receipt = stage2_watchdog.run(config_path)
    assert receipt["status"] == "coordinator_stop"
    assert receipt["worker_not_started"] is True
    assert receipt["process_group_absent"] is True
    assert receipt["accepted_events"] == []


def test_coordinator_loss_after_dispatch_stops_worker_group(tmp_path):
    config_path, config, marker = _make_invocation(
        tmp_path, mode="hang_call", load_seconds=2.0, call_seconds=5.0
    )
    coordinator = tmp_path / "coordinator.py"
    coordinator_script = textwrap.dedent(
        r'''
        import os, subprocess, sys, time
        from pathlib import Path
        from neural_sat_speck.backend.guard import atomic_json, read_checked_json
        config_path = Path(sys.argv[1])
        config = read_checked_json(config_path)
        config["coordinator_pid"] = os.getpid()
        atomic_json(config_path, config)
        child = subprocess.Popen([
            sys.executable, "-m", "neural_sat_speck.followups.stage2_watchdog", str(config_path)
        ])
        atomic_json(Path(config["directory"]) / "guardian.pid.json", {"pid": child.pid})
        marker = Path(sys.argv[2])
        deadline = time.perf_counter() + 4
        while time.perf_counter() < deadline and not marker.exists():
            time.sleep(0.01)
        os._exit(0)
        '''
    )
    coordinator.write_text(coordinator_script)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(SRC), environment.get("PYTHONPATH", "")]
    )
    parent = subprocess.Popen(
        [sys.executable, str(coordinator), str(config_path), str(marker)],
        cwd=ROOT,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    parent.wait(timeout=5)

    receipt_path = Path(config["receipt"])
    deadline = time.perf_counter() + 8
    while time.perf_counter() < deadline and not receipt_path.exists():
        time.sleep(0.02)
    assert receipt_path.exists()
    receipt = read_checked_json(receipt_path)
    assert receipt["status"] == "coordinator_stop"
    assert receipt["accepted_events"]
    assert receipt["process_group_absent"] is True


def test_live_groups_matches_only_exact_stage2_argv(tmp_path, monkeypatch):
    token = "a" * 32
    output = textwrap.dedent(
        f"""
        4321 /opt/python/bin/python /checkout/scripts/run_simon_matched_stage2_worker.py /data/{token}/payload.json
        5432 /opt/python/bin/python -m neural_sat_speck.followups.stage2_watchdog /data/{token}/guardian.json
        6543 /opt/python/bin/python /checkout/scripts/other_worker.py /data/{token}/payload.json
        7654 /opt/python/bin/python -m neural_sat_speck.followups.stage2_watchdog /data/other/guardian.json
        """
    )
    monkeypatch.setattr(
        stage2_watchdog.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, output, ""),
    )
    assert stage2_watchdog.live_groups(token) == {4321, 5432}
