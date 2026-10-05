"""Independent validation of bounded SIMON Stage 2 resource receipts.

This module validates recorded coordinator samples and worker telemetry.  It
does not estimate memory between samples or claim an instantaneous peak.
"""

from __future__ import annotations

import errno
import hashlib
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from neural_sat_speck.backend.guard import read_checked_json

RSS_LIMIT_BYTES = 2_684_354_560
POWER_POLICY = "battery_allowed_author_instruction_20261005"
THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TF_NUM_INTEROP_THREADS",
    "TF_NUM_INTRAOP_THREADS",
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value: object) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _pid(value: object, label: str) -> int:
    _require(type(value) is int and value > 1, f"Invalid {label}")
    return value


def _checked_event(directory: Path, event: dict[str, Any]) -> dict[str, Any]:
    relative = event.get("event_path")
    digest = event.get("event_sha256")
    _require(
        isinstance(relative, str)
        and relative == "events/0000.json"
        and isinstance(digest, str)
        and _SHA256.fullmatch(digest) is not None,
        "Malformed accepted loaded-event reference",
    )
    path = directory / relative
    loaded = read_checked_json(path)
    actual_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    _require(actual_digest == digest, "Loaded-event hash differs from guardian receipt")
    public_record = {
        key: value
        for key, value in event.items()
        if key not in {"event_path", "event_sha256"}
    }
    _require(loaded == public_record, "Loaded event differs from accepted receipt")
    return loaded


def _validate_sleep_assertion(assertion: object) -> int:
    _require(isinstance(assertion, dict), "Missing owned sleep assertion")
    _require(assertion.get("supported") is True, "Sleep assertion is unsupported")
    owner = _pid(assertion.get("coordinator_pid"), "sleep assertion owner PID")
    _pid(assertion.get("pid"), "caffeinate PID")
    expected = ["/usr/bin/caffeinate", "-d", "-i", "-m", "-s", "-w", str(owner)]
    _require(assertion.get("arguments") == expected, "Sleep assertion is not owned by coordinator")
    return owner


def _validate_power(power: object) -> str:
    _require(isinstance(power, dict), "Resource sample lacks power receipt")
    source = power.get("power_source")
    _require(isinstance(source, str) and source in {"AC Power", "Battery Power"}, "Unsupported power source")
    _require(power.get("supported") is True, "Power receipt is unsupported")
    _require(power.get("accepted") is True, "Power receipt was not accepted")
    _require(power.get("required_policy") == POWER_POLICY, "Power policy differs from approval")
    _require(power.get("assertions_returncode") == 0, "Power assertions query failed")
    _require(power.get("power_returncode") == 0, "Power source query failed")
    _require(power.get("sleep_assertion_observed") is True, "Power sample did not observe sleep assertion")
    ac = source == "AC Power"
    _require(power.get("ac_power_observed") is ac, "Power source fields contradict")
    _require(power.get("original_ac_policy_accepted") is ac, "Original AC-policy receipt contradicts source")
    _require(isinstance(power.get("raw_assertions"), str), "Missing raw power assertions")
    _require(isinstance(power.get("raw_power"), str), "Missing raw power-source output")
    stamp = power.get("utc")
    _require(isinstance(stamp, str), "Missing power sample UTC timestamp")
    try:
        datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("Malformed power sample UTC timestamp") from error
    return source


def is_process_exit_policy(policy: object, worker_pid: int) -> bool:
    """Classify missing-process observations; this is not terminal exit proof."""
    if (
        not isinstance(policy, dict)
        or type(worker_pid) is not int
        or worker_pid <= 1
        or type(policy.get("pid")) is not int
        or policy.get("pid") != worker_pid
        or policy.get("supported") is not True
        or policy.get("scope") != "process_priority_and_background_policy"
        or policy.get("native_thread_qos") != "unavailable"
        or policy.get("accepted") is not False
    ):
        return False
    observations = policy.get("observations")
    if not isinstance(observations, dict) or set(observations) != {"nice", "background_policy"}:
        return False
    missing = False
    for observation in observations.values():
        if not isinstance(observation, dict):
            return False
        value, error = observation.get("value"), observation.get("errno")
        if type(value) is not int or type(error) is not int:
            return False
        if error == errno.ESRCH and value == -1:
            missing = True
        elif error != 0 or value != 0:
            return False
    return missing


def _validate_policy(
    policy: object, worker_pid: int, *, terminal_exit_proven: bool
) -> bool:
    _require(isinstance(policy, dict), "RSS sample lacks process-policy receipt")
    _require(policy.get("supported") is True, "Process-policy observation unsupported")
    _require(policy.get("scope") == "process_priority_and_background_policy", "Unexpected process-policy scope")
    _require(policy.get("native_thread_qos") == "unavailable", "Unexpected native thread QoS claim")
    _require(policy.get("pid") == worker_pid, "Process-policy PID differs from worker")
    observations = policy.get("observations")
    _require(isinstance(observations, dict) and set(observations) == {"nice", "background_policy"}, "Malformed process-policy observations")
    good = True
    for name, observation in observations.items():
        _require(isinstance(observation, dict), f"Malformed {name} process-policy observation")
        value, error = observation.get("value"), observation.get("errno")
        _require(type(value) is int and type(error) is int and error >= 0, f"Malformed {name} process-policy values")
        good = good and error == 0 and value == 0
    _require(policy.get("accepted") is good, "Process-policy acceptance contradicts observations")
    if not good:
        _require(
            terminal_exit_proven and is_process_exit_policy(policy, worker_pid),
            "Observed worker policy failed or lacks terminal exit proof",
        )
    return good


def _validate_tmpdir(telemetry: dict[str, Any], source_root: Path) -> str:
    raw = telemetry.get("tmpdir")
    _require(isinstance(raw, str) and Path(raw).is_absolute(), "Worker TMPDIR must be absolute")
    path = Path(raw).resolve(strict=True)
    _require(path.is_dir() and str(path) == raw, "Worker TMPDIR is not a resolved persistent directory")
    roots = ((source_root / "runtime").resolve(), (source_root / "results").resolve())
    inside = any(path == root or root in path.parents for root in roots)
    ephemeral = any(str(path) == item or str(path).startswith(item + "/") for item in ("/tmp", "/private/tmp", "/var/folders", "/private/var"))
    _require(inside and not ephemeral and telemetry.get("tmpdir_safe") is True, "Worker TMPDIR is not persistent project storage")
    return str(path)


def _same_interpreter(reported: object, expected: object, label: str) -> str:
    _require(isinstance(reported, str) and isinstance(expected, str), f"Missing {label} interpreter")
    actual_path, expected_path = Path(reported), Path(expected)
    _require(actual_path.is_absolute() and expected_path.is_absolute(), f"{label} interpreter must be absolute")
    _require(actual_path.parent.resolve() == expected_path.parent.resolve(), f"{label} interpreter environment directory differs")
    try:
        same = actual_path.samefile(expected_path)
    except OSError:
        same = False
    _require(same, f"{label} interpreter file identity differs")
    return str(actual_path)


def _validate_telemetry(
    telemetry: object,
    *,
    manifest: dict[str, Any],
    scorer: str,
    worker_pid: int,
    guardian_pid: int,
    directory: Path,
) -> dict[str, Any]:
    _require(isinstance(telemetry, dict), "Missing worker telemetry")
    source_root = Path(manifest["source_root"]).resolve(strict=True)
    _require(telemetry.get("git_commit") == manifest.get("source_commit"), "Worker source commit differs from manifest")
    _require(telemetry.get("pid") == worker_pid, "Telemetry PID differs from observed worker")
    _require(telemetry.get("parent_pid") == guardian_pid, "Telemetry parent PID differs from watchdog")
    _require(telemetry.get("platform") == "darwin", "Worker platform differs from supported runtime")
    _require(type(telemetry.get("thread_id")) is int and telemetry["thread_id"] > 0, "Invalid calling-thread identity")
    _require(telemetry.get("qos_class") in {33, 25, 21}, "Worker calling thread QoS is not accepted")
    for name in ("darwin_priority", "darwin_bg_state", "io_policy_state"):
        _require(type(telemetry.get(name)) is int and telemetry[name] == 0, f"Worker {name} is not standard")
    _require(telemetry.get("is_certified") is True and telemetry.get("rejection_reason") is None, "Worker telemetry certification failed")
    env = telemetry.get("thread_environment")
    _require(isinstance(env, dict) and all(env.get(name) == ("8" if name == "TF_NUM_INTRAOP_THREADS" else "1") for name in THREAD_ENV), "Worker numeric thread environment differs from frozen numeric/inference policy")
    _validate_tmpdir(telemetry, source_root)
    runtime_name = "python" if scorer == "ddt7" else "tensorflow"
    expected_runtime = manifest["runtime"][runtime_name]
    expected_file = manifest["files"]["python" if scorer == "ddt7" else "tf_python"]["path"]
    _same_interpreter(expected_runtime["executable"], expected_file, runtime_name)
    executable = _same_interpreter(telemetry.get("python_executable"), expected_file, runtime_name)
    _same_interpreter(executable, expected_runtime["executable"], runtime_name)
    argv = telemetry.get("command_line")
    _require(
        isinstance(argv, list)
        and len(argv) == 2
        and all(isinstance(item, str) for item in argv)
        and Path(argv[0]).resolve() == source_root / "scripts/run_simon_matched_stage2_worker.py"
        and Path(argv[1]).absolute() == directory / "payload.json",
        "Worker telemetry command line differs from pinned invocation",
    )
    observed_at = telemetry.get("observed_at")
    _require(isinstance(observed_at, str), "Missing worker telemetry timestamp")
    try:
        datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("Malformed worker telemetry timestamp") from error
    return {
        "pid": worker_pid,
        "parent_pid": guardian_pid,
        "python_executable": executable,
        "tmpdir": telemetry["tmpdir"],
        "qos_class": telemetry["qos_class"],
        "thread_id": telemetry["thread_id"],
    }


def validate_resource_receipts(
    manifest: dict[str, Any], row: dict[str, Any], require_worker_sample: bool = True
) -> dict[str, Any]:
    """Validate power, sampled RSS/process policy, and worker identity receipts.

    ``require_worker_sample`` requires at least one accepted sample containing
    an observed worker member.  For partial jobs, a genuine over-limit sample
    is accepted only when the recorded outcome is ``rss_stop``.  Missing worker
    members are not treated as zero-memory observations.
    """
    _require(isinstance(manifest, dict) and isinstance(row, dict), "Manifest and row must be objects")
    _require(type(require_worker_sample) is bool, "require_worker_sample must be boolean")
    assertion_owner = _validate_sleep_assertion(row.get("sleep_assertion"))
    raw_directory = row.get("directory")
    if not isinstance(raw_directory, str) or not raw_directory:
        _require(not require_worker_sample and row.get("dispatched") is False, "Dispatched row lacks invocation directory")
        return {"samples": 0, "monitoring_available": False, "worker_samples": 0, "power_sources": [], "worker": None}
    directory = Path(raw_directory)
    _require(directory.is_absolute() and directory.is_dir(), "Invocation directory is unavailable")
    invocation = row.get("invocation")
    job_id = row.get("job_id")
    _require(isinstance(invocation, str) and directory.name == invocation, "Invocation directory identity differs")
    _require(isinstance(job_id, str) and job_id, "Missing job identity")

    guardian = read_checked_json(directory / "guardian.json")
    receipt = read_checked_json(directory / "guardian_receipt.json")
    _require(row.get("guardian_receipt") == receipt, "Guardian receipt differs from ledger row")
    _require(guardian.get("invocation") == invocation and guardian.get("job_id") == job_id, "Guardian configuration identity differs")
    _require(guardian.get("coordinator_pid") == assertion_owner, "Sleep assertion is not owned by coordinator")
    _require(receipt.get("invocation") == invocation and receipt.get("job_id") == job_id, "Guardian receipt identity differs")
    _require(type(receipt.get("watchdog_pid")) is int and receipt["watchdog_pid"] > 1, "Invalid watchdog PID")
    _require(receipt.get("process_group_absent") is True, "Worker process group is not confirmed absent")

    samples_path = directory / "environment_samples.json"
    samples_sidecar = samples_path.with_name(samples_path.name + ".sha256")
    samples_present = samples_path.exists()
    if not samples_present:
        _require(not samples_sidecar.exists(), "Environment sample sidecar has no data file")
        _require(not require_worker_sample, "Missing checked environment samples")
        samples = []
    else:
        samples = read_checked_json(samples_path)
    _require(isinstance(samples, list), "Environment samples must be an array")
    process = None
    if receipt.get("worker_not_started") is not True:
        process = read_checked_json(directory / "process.json")
        worker_pid = _pid(process.get("pid"), "worker PID")
        _require(process.get("pid") == receipt.get("worker_pid"), "Worker PID differs from guardian receipt")
        _require(process.get("process_group") == worker_pid and receipt.get("process_group") == worker_pid, "Worker process-group identity differs")
        _require(process.get("guardian_pid") == receipt.get("watchdog_pid"), "Worker process record differs from watchdog")
        _require(process.get("invocation") == invocation and process.get("job_id") == job_id, "Worker process record identity differs")
    else:
        worker_pid = None

    started = guardian.get("job_started_monotonic")
    ended = receipt.get("cleanup_completed_monotonic")
    _require(_finite(started) and _finite(ended) and ended >= started, "Invalid coordinator sample interval")
    monitor_ended = ended
    monitor_proven = False
    monitor_path = directory / "monitor_complete.json"
    monitor_sidecar = monitor_path.with_name(monitor_path.name + ".sha256")
    if monitor_path.exists() or monitor_sidecar.exists():
        monitor = read_checked_json(monitor_path)
        _require(
            isinstance(monitor, dict)
            and set(monitor) == {"schema", "invocation", "job_id", "coordinator_pid", "guardian_receipt_sha256", "completed_monotonic"}
            and monitor.get("schema") == "stage2_monitor_complete_v1"
            and monitor.get("invocation") == invocation
            and monitor.get("job_id") == job_id
            and monitor.get("coordinator_pid") == assertion_owner
            and monitor.get("guardian_receipt_sha256") == hashlib.sha256((directory / "guardian_receipt.json").read_bytes()).hexdigest()
            and _finite(monitor.get("completed_monotonic"))
            and monitor["completed_monotonic"] >= ended,
            "Monitor completion receipt is not bound to guardian exit",
        )
        monitor_ended = monitor["completed_monotonic"]
        _require(
            _finite(guardian.get("campaign_deadline_monotonic"))
            and monitor_ended < guardian["campaign_deadline_monotonic"],
            "Monitor completion exceeds original campaign deadline",
        )
        monitor_proven = True
    if require_worker_sample:
        _require(monitor_proven, "Missing checked monitor completion receipt")
    terminal_exit_proven = (
        monitor_proven
        and receipt.get("schema") == "simon_stage2_guardian_receipt_v1"
        and receipt.get("status") in {"worker_exit", "load_timeout", "rank_timeout", "campaign_timeout", "coordinator_stop"}
        and type(receipt.get("worker_returncode")) is int
        and receipt.get("worker_not_started") is False
        and receipt.get("process_group_absent") is True
        and receipt.get("worker_pid") == worker_pid
    )
    last_clock = -math.inf
    accepted_worker_samples = 0
    observed_worker_samples = 0
    power_sources: list[str] = []
    rss_stopped = False
    exit_policy_observations = 0
    for index, sample in enumerate(samples):
        _require(isinstance(sample, dict), f"Malformed environment sample {index}")
        clock = sample.get("monotonic")
        _require(_finite(clock) and started <= clock <= monitor_ended and clock >= last_clock, f"Invalid environment sample clock {index}")
        last_clock = float(clock)
        power_sources.append(_validate_power(sample.get("power")))
        has_memory, has_policy = "memory" in sample, "process_policy" in sample
        _require(has_memory == has_policy, f"Incomplete worker sample {index}")
        if not has_memory:
            _require(clock <= ended, "Power-only sample follows guardian cleanup")
            continue
        _require(worker_pid is not None, "Resource sample exists without a dispatched worker")
        memory = sample["memory"]
        _require(isinstance(memory, dict), f"Malformed RSS sample {index}")
        rss, members, accepted = memory.get("group_rss_bytes"), memory.get("members_observed"), memory.get("accepted")
        _require(type(rss) is int and rss >= 0, f"Malformed RSS bytes in sample {index}")
        _require(type(members) is int and members >= 0, f"Malformed worker count in sample {index}")
        _require(type(accepted) is bool and accepted is (rss <= RSS_LIMIT_BYTES), f"RSS acceptance contradicts sample {index}")
        _require(members != 0 or rss == 0, f"RSS without observed members in sample {index}")
        policy_ok = _validate_policy(sample["process_policy"], worker_pid, terminal_exit_proven=terminal_exit_proven)
        if policy_ok:
            _require(exit_policy_observations == 0, "Worker appeared live after an exit-policy observation")
            _require(clock <= ended, "Live worker policy sample follows guardian cleanup")
        else:
            exit_policy_observations += 1
            if not accepted:
                _require(row.get("outcome") == "rss_stop", "Over-limit exit sample does not match rss_stop outcome")
                rss_stopped = True
            continue
        if members > 0:
            observed_worker_samples += 1
            if accepted:
                _require(policy_ok, f"Accepted RSS sample lacks accepted policy in sample {index}")
                accepted_worker_samples += 1
            else:
                _require(row.get("outcome") == "rss_stop", "Over-limit RSS sample does not match rss_stop outcome")
                rss_stopped = True

    _require(bool(samples) or not require_worker_sample, "No environment samples were recorded")
    if row.get("outcome") == "rss_stop":
        _require(rss_stopped, "rss_stop outcome lacks a genuine over-limit sample")
    if require_worker_sample:
        _require(accepted_worker_samples > 0, "No accepted worker RSS/process-policy sample was observed")

    worker_summary = None
    accepted_events = row.get("accepted_events", [])
    _require(isinstance(accepted_events, list), "Accepted events must be an array")
    loaded = None
    if accepted_events:
        _require(isinstance(accepted_events[0], dict), "Malformed loaded-event receipt")
        loaded = _checked_event(directory, accepted_events[0])
        _require(loaded.get("event") == "loaded" and loaded.get("invocation") == invocation and loaded.get("job_id") == job_id, "Loaded event identity differs")
        scorer = row.get("scorer")
        _require(scorer in {"vv9", "vv8", "ddt7"} and loaded.get("scorer") == scorer, "Loaded scorer differs from row")
        telemetry = loaded.get("telemetry")
        worker_summary = _validate_telemetry(
            telemetry,
            manifest=manifest,
            scorer=scorer,
            worker_pid=worker_pid,
            guardian_pid=receipt["watchdog_pid"],
            directory=directory,
        )
        runtime_name = "python" if scorer == "ddt7" else "tensorflow"
        _require(loaded.get("runtime") == manifest["runtime"][runtime_name], "Loaded runtime differs from pinned runtime")
        expected_assets = {name: record["identity"] for name, record in manifest["assets"].items()}
        _require(loaded.get("asset_identity") == expected_assets, "Loaded asset identities differ from manifest")
    elif require_worker_sample:
        raise ValueError("Completed pilot resource proof requires a loaded worker event")

    result_path = directory / "worker_result.json"
    if result_path.exists() or result_path.with_name(result_path.name + ".sha256").exists():
        final = read_checked_json(result_path)
        _require(loaded is not None, "Final worker result has no accepted loaded event")
        _require(final.get("telemetry") == loaded.get("telemetry"), "Final telemetry differs from loaded event")
        _require(final.get("runtime") == loaded.get("runtime"), "Final runtime differs from loaded event")
        _require(final.get("loader_metadata") == loaded.get("loader_metadata"), "Final loader metadata differs from loaded event")
    elif row.get("outcome") == "completed":
        raise ValueError("Completed row lacks checked worker final metadata")

    return {
        "samples": len(samples),
        "monitoring_available": samples_present,
        "worker_samples": observed_worker_samples,
        "accepted_worker_samples": accepted_worker_samples,
        "exit_policy_observations": exit_policy_observations,
        "rss_stop_observed": rss_stopped,
        "power_sources": power_sources,
        "worker": worker_summary,
    }
