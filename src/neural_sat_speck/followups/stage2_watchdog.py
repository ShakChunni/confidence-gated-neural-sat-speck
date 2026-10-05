"""Event-journal-aware process guardian for matched SIMON Stage 2 jobs.

The worker and guardian run in separate process groups.  The guardian observes
the worker without reaping it, accepts only a sequential checksummed journal,
and advances a bounded deadline after each persisted completion event.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from neural_sat_speck.backend.guard import (
    atomic_json,
    canonical_bytes,
    read_checked_json,
)

_INVOCATION = re.compile(r"[0-9a-f]{32}\Z")
_JOB_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_EVENT_FILE = re.compile(r"([0-9]{4})\.json\Z")
_SIDE_CAR_WAIT = 0.05
_POLL_SECONDS = 0.01


class GuardianError(RuntimeError):
    """A worker journal or guardian identity failed closed."""


class _PhaseDeadlineReached(Exception):
    """Validation/persistence crossed the currently active hard deadline."""


def _finite_number(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"Invalid {name}")
    return float(value)


def _validate_config(config: Any, config_path: Path) -> dict[str, Any]:
    required = {
        "invocation",
        "directory",
        "repo_root",
        "coordinator_pid",
        "worker_arguments",
        "receipt",
        "job_id",
        "job_started_monotonic",
        "campaign_deadline_monotonic",
        "load_seconds",
        "call_seconds",
        "expected_calls",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("Guardian configuration fields differ from the fixed schema")
    invocation = config["invocation"]
    if not isinstance(invocation, str) or _INVOCATION.fullmatch(invocation) is None:
        raise ValueError("Invalid invocation")
    if type(config["coordinator_pid"]) is not int or config["coordinator_pid"] <= 1:
        raise ValueError("Invalid coordinator PID")
    if not isinstance(config["job_id"], str) or _JOB_ID.fullmatch(config["job_id"]) is None:
        raise ValueError("Invalid job identity")
    if type(config["expected_calls"]) is not int or config["expected_calls"] not in (1, 8):
        raise ValueError("Expected call count must be one or eight")

    load_seconds = _finite_number(config["load_seconds"], "load limit")
    call_seconds = _finite_number(config["call_seconds"], "call limit")
    if not 0 < load_seconds <= 120 or not 0 < call_seconds <= 180:
        raise ValueError("Guardian limits exceed the fixed production caps")
    job_started = _finite_number(config["job_started_monotonic"], "job start")
    campaign_deadline = _finite_number(
        config["campaign_deadline_monotonic"], "campaign deadline"
    )
    if campaign_deadline <= job_started:
        raise ValueError("Campaign deadline must follow job start")
    if job_started > time.perf_counter() + 1e-6:
        raise ValueError("Recorded job start is in the future")

    directory = Path(config["directory"])
    repo_root = Path(config["repo_root"])
    receipt = Path(config["receipt"])
    if not all(path.is_absolute() for path in (directory, repo_root, receipt)):
        raise ValueError("Guardian paths must be absolute")
    directory = directory.absolute()
    repo_root = repo_root.absolute()
    if directory.name != invocation or not directory.is_dir():
        raise ValueError("Invocation directory identity mismatch")
    if not repo_root.is_dir():
        raise ValueError("Repository root is unavailable")
    if receipt.absolute() != directory / "guardian_receipt.json":
        raise ValueError("Guardian receipt must be inside its invocation directory")

    worker_arguments = config["worker_arguments"]
    if (
        not isinstance(worker_arguments, list)
        or len(worker_arguments) != 3
        or any(not isinstance(item, str) or not item for item in worker_arguments)
    ):
        raise ValueError("Worker argv must be executable, script and payload")
    executable, script, payload = worker_arguments
    if not Path(executable).is_absolute() or not Path(executable).is_file():
        raise ValueError("Worker interpreter must be an existing absolute path")
    expected_script = repo_root / "scripts" / "run_simon_matched_stage2_worker.py"
    expected_payload = directory / "payload.json"
    if Path(script).absolute() != expected_script or Path(payload).absolute() != expected_payload:
        raise ValueError("Worker argv is not bound to this invocation")
    if not expected_script.is_file() or not expected_payload.is_file():
        raise ValueError("Worker script or payload is unavailable")
    if Path(config_path).absolute().parent != directory:
        raise ValueError("Guardian config must be inside the invocation directory")
    if not hasattr(os, "waitid") or not hasattr(os, "WNOWAIT"):
        raise RuntimeError("Non-reaping child observation is required")

    return {
        **config,
        "directory": directory,
        "repo_root": repo_root,
        "receipt": receipt.absolute(),
        "worker_arguments": worker_arguments,
        "job_started_monotonic": job_started,
        "campaign_deadline_monotonic": campaign_deadline,
        "load_seconds": load_seconds,
        "call_seconds": call_seconds,
        "config_sha256": hashlib.sha256(canonical_bytes(config)).hexdigest(),
    }


def _read_checked_when_ready(path: Path, deadline: float) -> tuple[Any | None, str | None]:
    """Return (value, raw hash), waiting briefly only for a missing sidecar."""
    sidecar = path.with_name(path.name + ".sha256")
    if not path.exists():
        return None, None
    until = min(deadline, time.perf_counter() + _SIDE_CAR_WAIT)
    while not sidecar.exists() and time.perf_counter() < until:
        time.sleep(min(_POLL_SECONDS, max(0.0, until - time.perf_counter())))
    if not sidecar.exists():
        return None, None
    try:
        # Verify the checksum against the exact bytes that will be parsed.  The
        # shared helper is also called so duplicate keys/nonfinite values remain
        # subject to the repository's checked-JSON policy.
        checked = read_checked_json(path)
        raw = path.read_bytes()
        expected_sidecar = f"{hashlib.sha256(raw).hexdigest()}  {path.name}\n"
        if sidecar.read_text() != expected_sidecar:
            raise ValueError("Artifact changed while being read")
        value = json.loads(raw)
        if canonical_bytes(value) != raw or value != checked:
            raise ValueError("Artifact bytes and parsed value disagree")
    except (OSError, ValueError, TypeError) as error:
        raise GuardianError(f"Checked artifact failed: {path.name}: {error}") from error
    return value, hashlib.sha256(raw).hexdigest()


def _validate_completion(
    config: dict[str, Any], ordinal: int, event: Any, event_hash: str, deadline: float
) -> dict[str, Any]:
    directory = config["directory"]
    expected_event = ordinal + 1
    if not isinstance(event, dict):
        raise GuardianError("Call event must be a JSON object")
    if (
        event.get("schema") != "stage2_rank_worker_event_v1"
        or event.get("event") != "call_complete"
        or event.get("invocation") != config["invocation"]
        or event.get("job_id") != config["job_id"]
        or type(event.get("ordinal")) is not int
        or event["ordinal"] != ordinal
        or type(event.get("event_index")) is not int
        or event["event_index"] != expected_event
    ):
        raise GuardianError("Call event identity or sequence mismatch")
    completed = _finite_number(event.get("completed_monotonic"), "call completion time")
    if completed > time.perf_counter() + 1e-6:
        raise GuardianError("Call event completion is in the future")

    start_path = directory / "calls" / f"{ordinal:04d}_start.json"
    result_path = directory / "calls" / f"{ordinal:04d}_result.json"
    start, start_hash = _read_checked_when_ready(start_path, deadline)
    result, result_hash = _read_checked_when_ready(result_path, deadline)
    if time.perf_counter() >= deadline:
        raise _PhaseDeadlineReached
    if not isinstance(start, dict) or not isinstance(result, dict) or result_hash is None:
        raise GuardianError("Completion event lacks its checked start or result")

    required_identity = {"invocation": config["invocation"], "job_id": config["job_id"]}
    for record, name in ((start, "start"), (result, "result")):
        if any(record.get(key) != value for key, value in required_identity.items()):
            raise GuardianError(f"Call {name} identity mismatch")
        if type(record.get("ordinal")) is not int or record["ordinal"] != ordinal:
            raise GuardianError(f"Call {name} ordinal mismatch")
    expected_result = f"calls/{ordinal:04d}_result.json"
    if event.get("result") != expected_result:
        raise GuardianError("Completion event names an unexpected result path")
    if "result_path" in event and event["result_path"] != expected_result:
        raise GuardianError("Duplicate completion result paths disagree")
    if event.get("result_sha256") != result_hash:
        raise GuardianError("Completion event result hash mismatch")

    scorer = start.get("scorer")
    parent = start.get("parent")
    if scorer not in {"vv9", "vv8", "ddt7"}:
        raise GuardianError("Call start lacks scorer identity")
    if (config["expected_calls"] == 1 and (scorer != "vv9" or parent is not None)) or (
        config["expected_calls"] == 8 and (scorer == "vv9" or parent is None)
    ):
        raise GuardianError("Scorer or parent shape disagrees with the fixed call plan")
    if parent is not None and (type(parent) is not int or not 0 <= parent <= 0xFFFF):
        raise GuardianError("Call start has an invalid parent")
    start_time = _finite_number(start.get("call_started_monotonic"), "call start time")
    result_start = _finite_number(result.get("call_started_monotonic"), "result call start")
    result_completed = _finite_number(
        result.get("call_completed_monotonic"), "result call completion"
    )
    if (
        start_time != result_start
        or result_completed < start_time
        or completed < result_completed
    ):
        raise GuardianError("Call start/result/event times disagree")
    if result.get("scorer") != scorer or result.get("parent") != parent:
        raise GuardianError("Scorer or parent changed between start and result")
    if event.get("scorer") != scorer or event.get("parent") != parent:
        raise GuardianError("Scorer or parent changed between event and result")
    if scorer != config.get("_loaded_scorer"):
        raise GuardianError("Scorer changed after the loaded event")

    words = result.get("ordered_words")
    if (
        not isinstance(words, list)
        or len(words) != 512
        or any(type(word) is not int or not 0 <= word <= 0xFFFF for word in words)
        or len(set(words)) != 512
    ):
        raise GuardianError("Call result is not an ordered 512-word ranking")
    timings = result.get("component_times_seconds")
    if not isinstance(timings, dict) or not timings:
        raise GuardianError("Call result lacks component timing metadata")
    for name, seconds in timings.items():
        if not isinstance(name, str) or not name:
            raise GuardianError("Invalid component timing name")
        value = _finite_number(seconds, f"component timing {name}")
        if value < 0:
            raise GuardianError("Negative component timing")

    return {
        **event,
        "event_path": f"events/{expected_event:04d}.json",
        "event_sha256": event_hash,
        "start_path": f"calls/{ordinal:04d}_start.json",
        "start_sha256": start_hash,
        "result_path": f"calls/{ordinal:04d}_result.json",
        "result_sha256": result_hash,
        "call_started_monotonic": start_time,
        "call_completed_monotonic": result_completed,
        "completed_monotonic": completed,
        "scorer": scorer,
        "parent": parent,
    }


def _validate_loaded(config: dict[str, Any], event: Any, event_hash: str) -> dict[str, Any]:
    if not isinstance(event, dict) or any(
        event.get(key) != config[key] for key in ("invocation", "job_id")
    ):
        raise GuardianError("Loaded event identity mismatch")
    if (
        event.get("schema") != "stage2_rank_worker_event_v1"
        or event.get("event") != "loaded"
        or event.get("event_index") != 0
        or "ordinal" in event
    ):
        raise GuardianError("First journal event must describe model loading")
    scorer = event.get("scorer")
    if scorer not in {"vv9", "vv8", "ddt7"}:
        raise GuardianError("Loaded event lacks a fixed scorer identity")
    if (config["expected_calls"] == 1 and scorer != "vv9") or (
        config["expected_calls"] == 8 and scorer == "vv9"
    ):
        raise GuardianError("Loaded scorer disagrees with the fixed call count")
    completed = _finite_number(event.get("completed_monotonic"), "load completion time")
    load_wall = _finite_number(event.get("load_wall_seconds"), "load wall time")
    if load_wall < 0 or completed > time.perf_counter() + 1e-6:
        raise GuardianError("Invalid load event timing")
    if completed < config["job_started_monotonic"]:
        raise GuardianError("Load completed before the recorded job start")
    return {
        **event,
        "event_path": "events/0000.json",
        "event_sha256": event_hash,
        "completed_monotonic": completed,
        "scorer": scorer,
    }


def _assert_accepted_unchanged(config: dict[str, Any], accepted: list[dict[str, Any]]) -> None:
    for record in accepted:
        event_path = config["directory"] / record["event_path"]
        try:
            event = read_checked_json(event_path)
            raw_hash = hashlib.sha256(event_path.read_bytes()).hexdigest()
        except (OSError, ValueError) as error:
            raise GuardianError(f"Accepted event changed: {error}") from error
        if raw_hash != record["event_sha256"] or event != record["_source_event"]:
            raise GuardianError("Accepted event changed after acceptance")
        if "start_path" in record:
            start_path = config["directory"] / record["start_path"]
            try:
                read_checked_json(start_path)
                start_hash = hashlib.sha256(start_path.read_bytes()).hexdigest()
            except (OSError, ValueError) as error:
                raise GuardianError(f"Accepted start changed: {error}") from error
            if start_hash != record["start_sha256"]:
                raise GuardianError("Accepted start changed after acceptance")
        if "result_path" in record:
            result_path = config["directory"] / record["result_path"]
            try:
                _result = read_checked_json(result_path)
                result_hash = hashlib.sha256(result_path.read_bytes()).hexdigest()
            except (OSError, ValueError) as error:
                raise GuardianError(f"Accepted result changed: {error}") from error
            if result_hash != record["result_sha256"]:
                raise GuardianError("Accepted result changed after acceptance")


def _journal_sequence_error(config: dict[str, Any], next_index: int) -> str | None:
    event_dir = config["directory"] / "events"
    if not event_dir.is_dir():
        return None
    indices = []
    max_index = config["expected_calls"]
    for path in event_dir.iterdir():
        match = _EVENT_FILE.fullmatch(path.name)
        if match:
            index = int(match.group(1))
            if index > max_index:
                return "Journal contains an event beyond the configured call count"
            indices.append(index)
    present = set(indices)
    if any(index not in present for index in range(next_index, max(indices, default=-1) + 1)):
        return "Journal contains a skipped event"
    return None


def _snapshot_checked_journal(config: dict[str, Any]) -> dict[str, str]:
    """Hash complete journal artifacts visible at a coordinator-stop boundary."""
    directory = config["directory"]
    paths = [
        directory / "events" / f"{index:04d}.json"
        for index in range(config["expected_calls"] + 1)
    ]
    for ordinal in range(config["expected_calls"]):
        paths.extend(
            [
                directory / "calls" / f"{ordinal:04d}_start.json",
                directory / "calls" / f"{ordinal:04d}_result.json",
            ]
        )
    available: dict[str, str] = {}
    for path in paths:
        if not path.is_file() or not path.with_name(path.name + ".sha256").is_file():
            continue
        _value, digest = _read_checked_when_ready(path, time.perf_counter())
        if digest is not None:
            available[str(path.relative_to(directory))] = digest
    return available


def _group_absent(pgid: int, *, timeout: float = 2.0) -> bool:
    until = time.perf_counter() + timeout
    while time.perf_counter() < until:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        except OSError:
            return False
        time.sleep(0.01)
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    return False


def _coordinator_is_parent(coordinator_pid: int) -> bool:
    """Keep parent liveness tied to the live relationship, never a saved PID."""
    return os.getppid() == coordinator_pid


def run(config_path: Path) -> dict[str, Any]:
    """Run one worker under phase deadlines driven by accepted journal events."""
    if os.getppid() <= 1:
        raise RuntimeError("Guardian has no coordinator parent")
    raw_config = read_checked_json(config_path)
    config = _validate_config(raw_config, config_path)
    if not _coordinator_is_parent(config["coordinator_pid"]):
        raise RuntimeError("Guardian was not started by its recorded coordinator")

    stopped: int | None = None

    def stop_requested(signum: int, _frame: Any) -> None:
        nonlocal stopped
        stopped = signum

    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGINT, stop_requested)

    directory = config["directory"]
    accepted: list[dict[str, Any]] = []
    process: subprocess.Popen[bytes] | None = None
    current_phase = "loading"
    deadline = min(
        config["job_started_monotonic"] + config["load_seconds"],
        config["campaign_deadline_monotonic"],
    )
    status = "guardian_error"
    error_text: str | None = None
    returncode: int | None = None
    group_absent = True
    dispatched = False
    final_observed = time.perf_counter()
    stop_boundary: float | None = None
    stop_snapshot: dict[str, str] = {}

    def accept_event(expected_index: int, event: Any, event_hash: str | None) -> bool:
        nonlocal current_phase, deadline
        if time.perf_counter() >= deadline:
            raise _PhaseDeadlineReached
        if expected_index == 0:
            accepted_event = _validate_loaded(config, event, event_hash or "")
            if accepted_event["completed_monotonic"] >= deadline:
                raise GuardianError("Loaded event completed at or after its deadline")
            if time.perf_counter() >= deadline:
                raise _PhaseDeadlineReached
            accepted_event["_source_event"] = event
            config["_loaded_scorer"] = accepted_event["scorer"]
            accepted.append(accepted_event)
            current_phase = "ranking"
            deadline = min(
                accepted_event["completed_monotonic"] + config["call_seconds"],
                config["campaign_deadline_monotonic"],
            )
            return True
        ordinal = expected_index - 1
        if ordinal >= config["expected_calls"]:
            raise GuardianError("Worker wrote more calls than configured")
        accepted_event = _validate_completion(
            config, ordinal, event, event_hash or "", deadline
        )
        previous_completed = accepted[-1]["completed_monotonic"]
        if (
            accepted_event["call_started_monotonic"] < previous_completed
            or accepted_event["completed_monotonic"] < previous_completed
        ):
            raise GuardianError("Call began or completed before the preceding phase")
        if accepted_event["completed_monotonic"] >= deadline:
            raise GuardianError("Call completed at or after its deadline")
        if time.perf_counter() >= deadline:
            raise _PhaseDeadlineReached
        accepted_event["_source_event"] = event
        accepted.append(accepted_event)
        if ordinal + 1 < config["expected_calls"]:
            current_phase = "ranking"
            deadline = min(
                accepted_event["completed_monotonic"] + config["call_seconds"],
                config["campaign_deadline_monotonic"],
            )
        else:
            # Keep the last call's deadline during worker finalization.
            current_phase = "finalizing"
        return True

    def finish_receipt() -> dict[str, Any]:
        final = time.perf_counter()
        # Internal source snapshots are removed from the public event objects.
        public_events = [
            {key: value for key, value in event.items() if key != "_source_event"}
            for event in accepted
        ]
        receipt = {
            "schema": "simon_stage2_guardian_receipt_v1",
            "watchdog_pid": os.getpid(),
            "invocation": config["invocation"],
            "job_id": config["job_id"],
            "config_sha256": config["config_sha256"],
            "accepted_events": public_events,
            "current_phase": current_phase,
            "effective_deadline_monotonic": deadline,
            "deadline_monotonic": deadline,
            "status": status,
            "worker_pid": process.pid if process is not None else None,
            "process_group": process.pid if process is not None else None,
            "worker_returncode": returncode,
            "returncode": returncode,
            "worker_not_started": not dispatched,
            "process_group_absent": group_absent,
            "observed_monotonic": final_observed,
            "cleanup_completed_monotonic": final,
        }
        if stopped is not None:
            receipt["stop_signal"] = stopped
        if error_text is not None:
            receipt["error"] = error_text
        atomic_json(config["receipt"], receipt)
        return receipt

    try:
        if stopped is not None or not _coordinator_is_parent(config["coordinator_pid"]):
            status = "coordinator_stop"
            current_phase = "not_started"
            stop_snapshot = _snapshot_checked_journal(config)
            stop_boundary = time.perf_counter()
            final_observed = stop_boundary
        elif time.perf_counter() >= deadline:
            status = (
                "campaign_timeout"
                if deadline == config["campaign_deadline_monotonic"]
                else "load_timeout"
            )
            final_observed = time.perf_counter()
        else:
            stdout_path = directory / "worker_stdout.txt"
            stderr_path = directory / "worker_stderr.txt"
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                process = subprocess.Popen(
                    config["worker_arguments"],
                    cwd=config["repo_root"],
                    start_new_session=True,
                    stdout=stdout,
                    stderr=stderr,
                )
            dispatched = True
            atomic_json(
                directory / "process.json",
                {
                    "pid": process.pid,
                    "process_group": process.pid,
                    "invocation": config["invocation"],
                    "job_id": config["job_id"],
                    "guardian_pid": os.getpid(),
                },
            )

            while True:
                now = time.perf_counter()
                final_observed = now
                if stopped is not None or not _coordinator_is_parent(config["coordinator_pid"]):
                    status = "coordinator_stop"
                    stop_snapshot = _snapshot_checked_journal(config)
                    stop_boundary = time.perf_counter()
                    final_observed = stop_boundary
                    break
                # The observation deadline wins over a journal file already on disk:
                # event parsing and durable persistence are charged to that phase.
                if now >= deadline:
                    status = (
                        "campaign_timeout"
                        if deadline == config["campaign_deadline_monotonic"]
                        else ("load_timeout" if not accepted else "rank_timeout")
                    )
                    break

                expected_index = len(accepted)
                sequence_error = _journal_sequence_error(config, expected_index)
                if sequence_error:
                    raise GuardianError(sequence_error)
                event_path = directory / "events" / f"{expected_index:04d}.json"
                event, event_hash = _read_checked_when_ready(event_path, deadline)
                now = time.perf_counter()
                final_observed = now
                if now >= deadline:
                    status = (
                        "campaign_timeout"
                        if deadline == config["campaign_deadline_monotonic"]
                        else ("load_timeout" if not accepted else "rank_timeout")
                    )
                    break
                if event is not None:
                    try:
                        accept_event(expected_index, event, event_hash)
                    except _PhaseDeadlineReached:
                        status = (
                            "campaign_timeout"
                            if deadline == config["campaign_deadline_monotonic"]
                            else ("load_timeout" if not accepted else "rank_timeout")
                        )
                        break
                    continue

                try:
                    observed_exit = os.waitid(
                        os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT
                    )
                except ChildProcessError as error:
                    raise GuardianError("Worker was reaped outside the guardian") from error
                if observed_exit is not None:
                    status = "worker_exit"
                    break
                time.sleep(min(_POLL_SECONDS, max(0.0, deadline - time.perf_counter())))

    except GuardianError as error:
        status = "guardian_error"
        error_text = str(error)
    except Exception as error:
        status = "guardian_error"
        error_text = f"{type(error).__name__}: {error}"
    finally:
        final_observed = time.perf_counter()
        try:
            _assert_accepted_unchanged(config, accepted)
        except GuardianError as error:
            status = "guardian_error"
            error_text = str(error)
        if process is not None:
            group_signal_error: str | None = None
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as error:
                # Some kernels reject signalling an already-exited WNOWAIT leader.
                # Group absence below is the deciding evidence; a live group after
                # this error remains a hard cleanup failure.
                group_signal_error = f"{type(error).__name__}: {error}"
            try:
                returncode = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                status = "guardian_error"
                error_text = "Worker leader did not terminate after process-group stop"
            group_absent = _group_absent(process.pid)
            if not group_absent:
                status = "guardian_error"
                detail = f" after signal error {group_signal_error}" if group_signal_error else ""
                error_text = f"Worker process group absence was not proven{detail}"
        if status == "coordinator_stop" and dispatched and group_absent and stop_boundary is not None:
            # Once the worker group is gone, retain any fully persisted events
            # that were already complete when stop was observed. Do not wait for
            # sidecars or give a stopped phase any additional time.
            try:
                while len(accepted) <= config["expected_calls"]:
                    expected_index = len(accepted)
                    sequence_error = _journal_sequence_error(config, expected_index)
                    if sequence_error:
                        raise GuardianError(sequence_error)
                    event_path = directory / "events" / f"{expected_index:04d}.json"
                    sidecar = event_path.with_name(event_path.name + ".sha256")
                    relative_event = str(event_path.relative_to(directory))
                    if (
                        not event_path.exists()
                        or not sidecar.exists()
                        or relative_event not in stop_snapshot
                    ):
                        break
                    event, event_hash = _read_checked_when_ready(event_path, deadline)
                    if event is None:
                        break
                    if event_hash != stop_snapshot[relative_event]:
                        raise GuardianError("Stopped event differs from its stop-boundary snapshot")
                    completed = _finite_number(
                        event.get("completed_monotonic") if isinstance(event, dict) else None,
                        "stopped event completion time",
                    )
                    if completed > stop_boundary or completed >= deadline:
                        break
                    prior_deadline, prior_phase = deadline, current_phase
                    if expected_index > 0:
                        ordinal = expected_index - 1
                        start_relative = f"calls/{ordinal:04d}_start.json"
                        result_relative = f"calls/{ordinal:04d}_result.json"
                        if (
                            start_relative not in stop_snapshot
                            or result_relative not in stop_snapshot
                        ):
                            break
                    try:
                        accept_event(expected_index, event, event_hash)
                    except _PhaseDeadlineReached:
                        break
                    if expected_index > 0:
                        accepted_call = accepted[-1]
                        if (
                            stop_snapshot.get(accepted_call["start_path"])
                            != accepted_call["start_sha256"]
                            or stop_snapshot.get(accepted_call["result_path"])
                            != accepted_call["result_sha256"]
                        ):
                            accepted.pop()
                            deadline, current_phase = prior_deadline, prior_phase
                            raise GuardianError(
                                "Stopped call lacks checked start/result at the stop boundary"
                            )
            except GuardianError as error:
                status = "guardian_error"
                error_text = str(error)
        if time.perf_counter() >= config["campaign_deadline_monotonic"] and status == "worker_exit":
            status = "campaign_timeout"
        receipt = finish_receipt()

    if receipt["status"] == "guardian_error":
        raise RuntimeError(receipt.get("error", "Stage 2 guardian failed"))
    return receipt


def live_groups(token: str) -> set[int]:
    """Return exact live Stage 2 guardian/worker groups for one invocation."""
    if not isinstance(token, str) or _INVOCATION.fullmatch(token) is None:
        raise ValueError("Invalid invocation token")
    result = subprocess.run(
        ["/bin/ps", "-ax", "-o", "pgid=,command="],
        capture_output=True,
        text=True,
        timeout=1,
    )
    if result.returncode:
        raise RuntimeError("Cannot establish live Stage 2 process identity")
    groups: set[int] = set()
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) != 2 or token not in fields[1]:
            continue
        try:
            argv = shlex.split(fields[1])
        except ValueError:
            continue
        worker = (
            len(argv) == 3
            and Path(argv[1]).name == "run_simon_matched_stage2_worker.py"
            and Path(argv[1]).parent.name == "scripts"
            and Path(argv[2]).name == "payload.json"
            and Path(argv[2]).parent.name == token
            and re.search(r"(?:^|/)" + token + r"(?:/|$)", argv[2]) is not None
        )
        guardian = (
            len(argv) == 4
            and argv[1] == "-m"
            and argv[2] == "neural_sat_speck.followups.stage2_watchdog"
            and Path(argv[3]).suffix == ".json"
            and Path(argv[3]).parent.name == token
        )
        if worker or guardian:
            group = int(fields[0])
            if group <= 1 or group == os.getpgrp():
                raise RuntimeError("Unsafe Stage 2 process group")
            groups.add(group)
    return groups


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m neural_sat_speck.followups.stage2_watchdog CONFIG.json")
    receipt = run(Path(sys.argv[1]))
    if receipt["status"] == "guardian_error":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
