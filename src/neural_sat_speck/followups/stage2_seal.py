"""Close an interrupted SIMON Stage 2 ledger without dispatching any work.

The sealer only validates immutable run metadata and already-written outputs.
It never starts a worker, loads a model, ranks candidates, builds a new queue,
or joins any key/plaintext truth.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from neural_sat_speck.backend.guard import atomic_json, read_checked_json
from neural_sat_speck.followups.stage2_control import (
    PROFILE,
    fixed_jobs,
    public_case,
    streaming_sha256,
    validate_manifest,
)
from neural_sat_speck.followups.stage2_runner import (
    normalize_job,
    public_payload,
    validate_execution_identity,
)
from neural_sat_speck.followups.stage2_watchdog import live_groups

_TOKEN = re.compile(r"[0-9a-f]{32}\Z")
_START_FIELDS = {
    "job",
    "invocation",
    "directory",
    "source_commit",
    "manifest_sha256",
    "started_utc",
}
_LEDGER_IDENTITY_FIELDS = (
    "schema",
    "mode",
    "source_commit",
    "manifest_sha256",
    "profile",
    "started_utc",
    "campaign_started_monotonic",
    "campaign_deadline_monotonic",
    "sleep_assertion",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _finite(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _unique_json(raw: bytes, label: str) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON value in {label}: {value}")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject_constant)


def _ledger_identity(
    ledger: object,
    campaign_start: dict,
    manifest: dict,
    manifest_sha256: str,
    jobs: list[dict],
) -> dict:
    if not isinstance(ledger, dict):
        raise ValueError("Campaign ledger must be an object")
    for field in _LEDGER_IDENTITY_FIELDS:
        if ledger.get(field) != campaign_start.get(field):
            raise ValueError(f"Ledger identity differs from campaign start: {field}")
    if (
        ledger.get("schema") != "stage2_control_results_v1"
        or ledger.get("mode") != manifest.get("mode")
        or ledger.get("mode") not in {"pilot", "full"}
        or ledger.get("source_commit") != manifest.get("source_commit")
        or ledger.get("manifest_sha256") != manifest_sha256
        or ledger.get("profile") != PROFILE
        or ledger.get("profile") != manifest.get("profile")
        or ledger.get("status") not in {"ACTIVE", "COMPLETE", "STOPPED_ALARM"}
        or not isinstance(ledger.get("starts"), dict)
        or not isinstance(ledger.get("records"), dict)
        or not isinstance(ledger.get("sleep_assertion"), dict)
    ):
        raise ValueError("Campaign ledger is not bound to this frozen manifest")
    started = ledger.get("campaign_started_monotonic")
    deadline = ledger.get("campaign_deadline_monotonic")
    if (
        not _finite(started)
        or not _finite(deadline)
        or abs(float(deadline) - float(started) - PROFILE["campaign_limit_seconds"]) > 1e-6
    ):
        raise ValueError("Campaign monotonic ceiling differs from the fixed profile")
    job_ids = {job["job_id"] for job in jobs}
    starts, records = ledger["starts"], ledger["records"]
    if not set(starts) <= job_ids or not set(records) <= job_ids:
        raise ValueError("Ledger contains a job outside its fixed grid")
    for job_id, start in starts.items():
        if (
            not isinstance(start, dict)
            or set(start) != {"invocation", "directory", "started_utc"}
            or not isinstance(start.get("invocation"), str)
            or _TOKEN.fullmatch(start["invocation"]) is None
            or not isinstance(start.get("directory"), str)
            or not Path(start["directory"]).is_absolute()
            or not isinstance(start.get("started_utc"), str)
        ):
            raise ValueError(f"Malformed saved invocation start: {job_id}")
    for job_id, row in records.items():
        job = next(item for item in jobs if item["job_id"] == job_id)
        if not isinstance(row, dict) or any(row.get(key) != value for key, value in job.items()):
            raise ValueError(f"Ledger row differs from its fixed job: {job_id}")
    ordered_job_ids = [job["job_id"] for job in jobs]
    record_positions = sorted(ordered_job_ids.index(job_id) for job_id in records)
    if record_positions != list(range(len(record_positions))):
        raise ValueError("Saved ledger records are not a prefix of the fixed serial grid")
    return ledger


def _history_prefix(
    output: Path,
    campaign_start: dict,
    manifest: dict,
    manifest_sha256: str,
    jobs: list[dict],
) -> tuple[int, dict, str]:
    """Return the highest contiguous checked snapshot, or the checked start."""
    history = output.parent / "ledger_history"
    base_sha = streaming_sha256(output.parent / "campaign_start.json")
    selected_sequence = -1
    selected = campaign_start
    selected_sha = base_sha
    prior = None
    sequence = 0
    while True:
        path = history / f"{sequence:04d}.json"
        if not path.exists():
            break
        try:
            candidate = read_checked_json(path)
            raw_sha = streaming_sha256(path)
            _ledger_identity(candidate, campaign_start, manifest, manifest_sha256, jobs)
        except (OSError, ValueError, json.JSONDecodeError):
            break
        if sequence == 0 and candidate != campaign_start:
            break
        if prior is not None:
            if not set(prior["starts"]) <= set(candidate["starts"]):
                break
            if not set(prior["records"]) <= set(candidate["records"]):
                break
            if any(candidate["starts"].get(key) != value for key, value in prior["starts"].items()):
                break
            if any(candidate["records"].get(key) != value for key, value in prior["records"].items()):
                break
        prior = candidate
        selected_sequence = sequence
        selected = candidate
        selected_sha = raw_sha
        sequence += 1
    return selected_sequence, selected, selected_sha


def _checked_original_ledger_hash(path: Path, baseline: dict, manifest: dict, manifest_sha: str, jobs: list[dict]) -> str | None:
    try:
        ledger = read_checked_json(path)
        _ledger_identity(ledger, baseline, manifest, manifest_sha, jobs)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return streaming_sha256(path)


def _canonical_invocation_path(parent: Path, job_id: str, token: str) -> Path:
    expected = parent / "jobs" / job_id / token
    if expected.is_symlink() or expected.resolve(strict=False) != expected.absolute():
        raise ValueError("Invocation directory is not the fixed, non-symlink path")
    return expected


def _scan_job_starts(
    parent: Path,
    jobs: list[dict],
    source_commit: str,
    manifest_sha256: str,
) -> dict[str, dict]:
    """Read checked per-invocation starts, including pre-dispatch orphan starts."""
    starts: dict[str, dict] = {}
    for job in jobs:
        job_root = parent / "jobs" / job["job_id"]
        if not job_root.exists():
            continue
        if job_root.is_symlink() or not job_root.is_dir():
            raise ValueError("Invocation job path is not a regular directory")
        for directory in sorted(job_root.iterdir()):
            if directory.is_symlink() or not directory.is_dir() or _TOKEN.fullmatch(directory.name) is None:
                raise ValueError("Unexpected invocation directory in the fixed job tree")
            start_path = directory / "campaign_job_start.json"
            sidecar = start_path.with_name(start_path.name + ".sha256")
            if not start_path.exists() and not sidecar.exists():
                continue
            start = read_checked_json(start_path)
            if (
                not isinstance(start, dict)
                or set(start) != _START_FIELDS
                or start.get("job") != job
                or start.get("invocation") != directory.name
                or start.get("directory") != str(directory)
                or start.get("source_commit") != source_commit
                or start.get("manifest_sha256") != manifest_sha256
                or not isinstance(start.get("started_utc"), str)
            ):
                raise ValueError("Checked per-job start does not match its source/job path")
            if job["job_id"] in starts:
                raise ValueError("More than one immutable invocation start exists for a job")
            starts[job["job_id"]] = start
    return starts


def _merge_starts(
    parent: Path,
    jobs: list[dict],
    snapshot: dict,
    job_files: dict[str, dict],
) -> tuple[dict[str, dict], set[str]]:
    starts = dict(snapshot["starts"])
    orphan_ids: set[str] = set()
    ordered_ids = [job["job_id"] for job in jobs]
    for job_id, saved in starts.items():
        expected_directory = _canonical_invocation_path(
            parent, job_id, saved["invocation"]
        )
        if saved["directory"] != str(expected_directory):
            raise ValueError("Ledger invocation path differs from its fixed job/token")
        artifact = job_files.get(job_id)
        if artifact is not None and (
            saved["invocation"] != artifact["invocation"]
            or saved["directory"] != artifact["directory"]
            or saved["started_utc"] != artifact["started_utc"]
        ):
            raise ValueError("Ledger and immutable per-job start disagree")
    for job_id, artifact in job_files.items():
        if job_id in starts:
            continue
        orphan_ids.add(job_id)
        starts[job_id] = {
            "invocation": artifact["invocation"],
            "directory": artifact["directory"],
            "started_utc": artifact["started_utc"],
        }
    positions = sorted(ordered_ids.index(job_id) for job_id in starts)
    if positions != list(range(len(positions))):
        raise ValueError("Invocation starts are not a prefix of the fixed serial grid")
    record_positions = sorted(
        ordered_ids.index(job_id) for job_id in snapshot["records"]
    )
    if len(positions) > len(record_positions) + 1:
        raise ValueError("More than one invocation is beyond the recorded row prefix")
    if len(orphan_ids) > 1 or (orphan_ids and next(iter(orphan_ids)) != ordered_ids[max(positions)]):
        raise ValueError("Only the final serial invocation may have an orphan start")
    if any(job_id not in snapshot["records"] for job_id in starts if job_id not in orphan_ids):
        # There can be one active checked start at the end of a saved snapshot.
        missing_rows = [job_id for job_id in starts if job_id not in snapshot["records"]]
        if len(missing_rows) != 1 or missing_rows[0] != ordered_ids[max(positions)]:
            raise ValueError("Saved starts do not form a closed serial prefix")
    return starts, orphan_ids


def _row_base(job: dict, outcome: str, *, dispatched: bool = False, start: dict | None = None) -> dict:
    row = {
        **job,
        "outcome": outcome,
        "dispatched": dispatched,
        "calls": [],
        "load_wall_seconds": None,
        "worker_wall_seconds": None,
        "queue_measurements": {},
        "completed_utc": _utc_now(),
    }
    if start is not None:
        row["directory"] = start["directory"]
        row["invocation"] = start["invocation"]
    return row


def _unknown_invocation(job: dict, start: dict, reason: Exception | str) -> dict:
    """Keep a started path but make no result or timing claim without full proof."""
    row = _row_base(job, "interrupted", dispatched=True, start=start)
    label = type(reason).__name__ if isinstance(reason, Exception) else reason
    row["error"] = f"Invocation evidence untrusted; no rank calls or durations claimed ({label})"
    return row


def _expected_parents(job: dict, records: dict[str, dict]) -> list:
    if job["operation"] == "stage1":
        return []
    parent_job = next(
        item
        for item in fixed_jobs()
        if item["key_index"] == job["key_index"] and item["operation"] == "stage1"
    )
    parent_row = records.get(parent_job["job_id"])
    if not parent_row or not parent_row.get("calls"):
        raise ValueError("Stage 2 invocation has no persisted Stage 1 parent ranking")
    return parent_row["calls"][0]["ordered_words"][:8]


def _read_source_row(directory: Path, name: str, expected: dict | None = None) -> dict | None:
    path = directory / name
    if not path.exists() and not path.with_name(path.name + ".sha256").exists():
        return None
    value = read_checked_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a checked JSON object")
    if expected is not None and value != expected:
        raise ValueError(f"{name} differs from the selected campaign row")
    return value


def _validate_guardian_artifacts(directory: Path, row: dict) -> list[dict]:
    """Bind accepted receipt entries back to their checked immutable event files."""
    receipt = row.get("guardian_receipt")
    if not isinstance(receipt, dict):
        raise ValueError("Normalized worker row has no guardian receipt")
    events = receipt.get("accepted_events")
    if not isinstance(events, list):
        raise ValueError("Guardian receipt has no accepted-event list")
    for index, event in enumerate(events):
        if not isinstance(event, dict) or event.get("event_path") != f"events/{index:04d}.json":
            raise ValueError("Guardian event path/index is not sequential")
        event_path = directory / event["event_path"]
        source_event = read_checked_json(event_path)
        if streaming_sha256(event_path) != event.get("event_sha256"):
            raise ValueError("Guardian event hash differs from its checked raw event")
        if not isinstance(source_event, dict) or any(
            event.get(key) != value for key, value in source_event.items()
        ):
            raise ValueError("Guardian accepted event differs from its checked raw event")
        if index == 0:
            if event.get("event") != "loaded" or "ordinal" in event:
                raise ValueError("First raw guardian event is not model-loaded evidence")
            continue
        ordinal = index - 1
        start_path = directory / f"calls/{ordinal:04d}_start.json"
        result_path = directory / f"calls/{ordinal:04d}_result.json"
        if (
            event.get("event") != "call_complete"
            or event.get("ordinal") != ordinal
            or event.get("start_path") != f"calls/{ordinal:04d}_start.json"
            or event.get("result_path") != f"calls/{ordinal:04d}_result.json"
            or streaming_sha256(start_path) != event.get("start_sha256")
            or streaming_sha256(result_path) != event.get("result_sha256")
        ):
            raise ValueError("Accepted call event is not bound to its checked start/result")
        read_checked_json(start_path)
        read_checked_json(result_path)
    return events


def _validate_phase_deadlines(
    directory: Path, job: dict, row: dict, campaign_deadline: float
) -> None:
    """Validate worker-local monotonic phases without comparing across boots."""
    guardian = read_checked_json(directory / "guardian.json")
    receipt = row.get("guardian_receipt")
    if not isinstance(receipt, dict):
        raise ValueError("Guardian receipt is missing from normalized row")
    expected_calls = 1 if job["operation"] == "stage1" else 8
    started = guardian.get("job_started_monotonic")
    if (
        guardian.get("invocation") != directory.name
        or guardian.get("job_id") != job["job_id"]
        or guardian.get("directory") != str(directory)
        or guardian.get("campaign_deadline_monotonic") != campaign_deadline
        or guardian.get("load_seconds") != 120.0
        or guardian.get("call_seconds") != 180.0
        or guardian.get("expected_calls") != expected_calls
        or not _finite(started)
        or started < 0
        or started >= campaign_deadline
    ):
        raise ValueError("Guardian phase configuration differs from the fixed campaign")
    events = receipt.get("accepted_events")
    if not isinstance(events, list):
        raise ValueError("Guardian accepted events are malformed")
    phase_deadline = min(float(started) + 120.0, campaign_deadline)
    if events:
        loaded = events[0]
        if loaded.get("event") != "loaded" or not _finite(loaded.get("completed_monotonic")):
            raise ValueError("Guardian load event is malformed")
        if loaded["completed_monotonic"] >= phase_deadline:
            raise ValueError("Loaded event completed at or after its phase deadline")
        phase_deadline = min(loaded["completed_monotonic"] + 180.0, campaign_deadline)
        for ordinal, event in enumerate(events[1:]):
            if (
                event.get("event") != "call_complete"
                or event.get("ordinal") != ordinal
                or not _finite(event.get("completed_monotonic"))
                or event["completed_monotonic"] >= phase_deadline
            ):
                raise ValueError("Rank event completed at or after its phase deadline")
            if ordinal + 1 < expected_calls:
                phase_deadline = min(
                    event["completed_monotonic"] + 180.0, campaign_deadline
                )
    if receipt.get("effective_deadline_monotonic") != phase_deadline:
        raise ValueError("Guardian final deadline differs from accepted phase journal")


def _validate_queue_measurements(
    job: dict, directory: Path, invocation: str, values: object, *, partial: bool
) -> dict[str, dict]:
    """Retain only measurements whose queue file was already checked and persisted."""
    if not isinstance(values, dict):
        return {}
    retained: dict[str, dict] = {}
    allowed = {"1"} if partial else {"1", "8"}
    from neural_sat_speck.followups.stage2_control import ordered_hash

    for width_key, measurement in values.items():
        if width_key not in allowed or not isinstance(measurement, dict):
            continue
        try:
            width = int(width_key)
            seconds = measurement["seconds"]
            queue_hash = measurement["queue_sha256"]
            queue = read_checked_json(directory / f"queue_m{width}.json")
            if (
                width not in {1, 8}
                or not _finite(seconds)
                or float(seconds) < 0
                or queue.get("invocation") != invocation
                or queue.get("job_id") != job["job_id"]
                or queue.get("width") != width
                or not isinstance(queue.get("candidate_tuples"), list)
                or ordered_hash(queue["candidate_tuples"]) != queue_hash
            ):
                continue
            retained[width_key] = {"seconds": seconds, "queue_sha256": queue_hash}
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return retained


def _validate_worker_result(
    job: dict,
    directory: Path,
    invocation: str,
    parents: list,
    calls: list,
    source_commit: str,
    assets: dict,
    events: list[dict],
) -> None:
    result = read_checked_json(directory / "worker_result.json")
    expected_count = 1 if job["operation"] == "stage1" else 8
    expected_assets = {
        name: record["identity"] for name, record in assets.items()
    }
    loaded_event = events[0] if events else {}
    last_event = events[-1] if events else {}
    if (
        not isinstance(result, dict)
        or result.get("schema") != "stage2_rank_worker_result_v1"
        or result.get("status") != "complete"
        or result.get("invocation") != invocation
        or result.get("job_id") != job["job_id"]
        or result.get("job") != job
        or result.get("operation") != job["operation"]
        or result.get("scorer") != job["scorer"]
        or result.get("key_index") != job["key_index"]
        or result.get("parents") != parents
        or result.get("completed_call_count") != expected_count
        or len(calls) != expected_count
        or not isinstance(result.get("telemetry"), dict)
        or result["telemetry"].get("git_commit") != source_commit
        or not isinstance(result.get("calls"), list)
        or len(result["calls"]) != expected_count
        or result.get("asset_identity_at_load") != expected_assets
        or result.get("asset_identity_at_exit") != expected_assets
        or result.get("runtime") != loaded_event.get("runtime")
        or result.get("telemetry") != loaded_event.get("telemetry")
        or result.get("loader_metadata") != loaded_event.get("loader_metadata")
        or result.get("loaded_completed_monotonic") != loaded_event.get("completed_monotonic")
        or not _finite(result.get("worker_started_monotonic"))
        or not _finite(result.get("worker_completed_monotonic"))
        or result["worker_started_monotonic"] > result["loaded_completed_monotonic"]
        or result["worker_completed_monotonic"]
        < calls[-1].get("call_completed_monotonic", 0)
        or result["worker_completed_monotonic"] > last_event.get("completed_monotonic", -1)
    ):
        raise ValueError("Worker final metadata is not bound to the complete fixed job")
    for ordinal, call in enumerate(calls):
        ref = result["calls"][ordinal]
        result_path = directory / f"calls/{ordinal:04d}_result.json"
        if (
            ref.get("ordinal") != ordinal
            or ref.get("parent") != call.get("parent")
            or ref.get("result") != f"calls/{ordinal:04d}_result.json"
            or ref.get("result_sha256") != streaming_sha256(result_path)
        ):
            raise ValueError("Worker final metadata call references differ from accepted results")


def _check_complete_row(
    manifest: dict,
    job: dict,
    row: dict,
    public: dict,
    parents: list,
    deadline: float,
) -> dict:
    directory = Path(row["directory"])
    if row.get("invocation") != directory.name or not _TOKEN.fullmatch(directory.name):
        raise ValueError("Completed row invocation path is malformed")
    supervisor = _read_source_row(directory, "supervisor_result.json", row)
    if supervisor is None:
        raise ValueError("Completed row lacks its checked supervisor result")
    guardian = read_checked_json(directory / "guardian.json")
    normalized = normalize_job(job, directory, guardian["job_started_monotonic"], None)
    events = _validate_guardian_artifacts(directory, normalized)
    _validate_phase_deadlines(directory, job, normalized, deadline)
    for field in (
        "outcome",
        "calls",
        "load_wall_seconds",
        "worker_wall_seconds",
        "accepted_events",
        "guardian_receipt",
    ):
        if normalized.get(field) != row.get(field):
            raise ValueError(f"Completed ledger row differs from normalized guardian evidence: {field}")
    validate_execution_identity(manifest, row, public, parents, deadline)
    _validate_worker_result(
        job,
        directory,
        row["invocation"],
        parents,
        row["calls"],
        manifest["source_commit"],
        manifest["assets"],
        events,
    )
    row = dict(row)
    expected_widths = set() if job["operation"] == "stage1" else {"1", "8"}
    if set(row.get("queue_measurements", {})) != expected_widths:
        raise ValueError("Completed row lacks its fixed persisted queue measurements")
    retained = _validate_queue_measurements(
        job, directory, row["invocation"], row.get("queue_measurements"), partial=False
    )
    if retained != row["queue_measurements"]:
        raise ValueError("Completed queue measurements lack matching checked queue files")
    row["queue_measurements"] = retained
    return row


def _recover_started_row(
    manifest: dict,
    job: dict,
    start: dict,
    public: dict,
    parents: list,
    deadline: float,
    source_row: dict | None,
    *,
    must_be_interrupted: bool,
) -> dict:
    directory = Path(start["directory"])
    invocation = start["invocation"]
    base_outcome = "interrupted"
    if source_row is not None and source_row.get("outcome") not in {"completed", "interrupted"}:
        base_outcome = source_row["outcome"]
    receipt_path = directory / "guardian_receipt.json"
    if not receipt_path.exists():
        row = _row_base(job, "interrupted", dispatched=True, start=start)
        row["error"] = "Guardian receipt unavailable; no rank calls or durations claimed"
        return row
    try:
        guardian = read_checked_json(directory / "guardian.json")
        receipt = read_checked_json(receipt_path)
        if source_row is not None and source_row.get("guardian_receipt") is not None:
            if source_row["guardian_receipt"] != receipt:
                raise ValueError("Saved row and checked guardian receipt differ")
        _validate_phase_deadlines(directory, job, {"guardian_receipt": receipt}, deadline)
        if receipt.get("worker_not_started") is True:
            expected_payload = public_payload(
                manifest, job, public, parents, invocation, directory
            )
            python = manifest["files"][
                "python" if job["scorer"] == "ddt7" else "tf_python"
            ]["path"]
            expected_arguments = [
                python,
                str(Path(manifest["source_root"]) / "scripts/run_simon_matched_stage2_worker.py"),
                str(directory / "payload.json"),
            ]
            payload = read_checked_json(directory / "payload.json")
            if (
                receipt.get("worker_pid") is not None
                or receipt.get("process_group") is not None
                or receipt.get("process_group_absent") is not True
                or receipt.get("accepted_events") != []
                or receipt.get("config_sha256") != streaming_sha256(directory / "guardian.json")
                or (directory / "process.json").exists()
                or payload != expected_payload
                or guardian.get("repo_root") != manifest["source_root"]
                or guardian.get("worker_arguments") != expected_arguments
                or guardian.get("receipt") != str(receipt_path)
                or guardian.get("coordinator_pid", 0) <= 1
            ):
                raise ValueError("No-worker guardian receipt is not a checked clean close")
            if source_row is not None and (
                source_row.get("directory") != str(directory)
                or source_row.get("invocation") != invocation
                or source_row.get("calls") != []
                or source_row.get("load_wall_seconds") is not None
                or source_row.get("worker_wall_seconds") is not None
                or source_row.get("queue_measurements") != {}
                or source_row.get("accepted_events") not in (None, [])
            ):
                raise ValueError("Saved no-worker row claims outputs absent from its receipt")
            outcome = "interrupted" if must_be_interrupted else base_outcome
            row = _row_base(job, outcome, dispatched=False, start=start)
            row["guardian_receipt"] = receipt
            row["accepted_events"] = []
            if source_row is not None and source_row.get("error"):
                row["error"] = source_row["error"]
            return row
        normalized = normalize_job(
            job, directory, guardian["job_started_monotonic"], "interrupted" if must_be_interrupted else base_outcome
        )
        _validate_guardian_artifacts(directory, normalized)
        _validate_phase_deadlines(directory, job, normalized, deadline)
        validate_execution_identity(manifest, normalized, public, parents, deadline)
        existing_result = _read_source_row(directory, "supervisor_result.json")
        measurements = {}
        if existing_result is not None:
            if (
                existing_result.get("job_id") != job["job_id"]
                or existing_result.get("invocation") != invocation
                or existing_result.get("directory") != str(directory)
            ):
                raise ValueError("Persisted supervisor result identity differs")
            for field in (
                "calls",
                "load_wall_seconds",
                "worker_wall_seconds",
                "accepted_events",
                "guardian_receipt",
            ):
                if existing_result.get(field) != normalized.get(field):
                    raise ValueError("Persisted supervisor result differs from accepted guardian calls")
            measurements = _validate_queue_measurements(
                job,
                directory,
                invocation,
                existing_result.get("queue_measurements"),
                partial=True,
            )
        normalized["outcome"] = "interrupted" if must_be_interrupted else base_outcome
        normalized["queue_measurements"] = measurements
        if source_row is not None and source_row.get("error"):
            normalized["error"] = source_row["error"]
        return normalized
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        row = _row_base(job, "interrupted", dispatched=True, start=start)
        row["error"] = f"Guardian artifacts untrusted; no rank calls claimed ({type(error).__name__})"
        return row


def _validate_row_job(row: dict, job: dict, label: str) -> None:
    if not isinstance(row, dict) or any(row.get(key) != value for key, value in job.items()):
        raise ValueError(f"{label} differs from its fixed job")
    if not isinstance(row.get("outcome"), str) or type(row.get("dispatched")) is not bool:
        raise ValueError(f"{label} lacks an outcome or dispatch classification")


def seal(
    manifest_path: Path,
    ledger_path: Path,
    output_path: Path,
    source_root: Path,
) -> dict:
    """Create a new STOPPED_ALARM ledger from checked evidence; never resume it."""
    manifest_path = Path(manifest_path)
    ledger_path = Path(ledger_path)
    output_path = Path(output_path)
    source_root = Path(source_root)
    if output_path.exists() or output_path.with_name(output_path.name + ".sha256").exists():
        raise FileExistsError("Sealed ledger output must be new")
    if output_path.resolve(strict=False) == ledger_path.resolve(strict=False):
        raise ValueError("Sealer cannot overwrite the original ledger")

    lock_path = ledger_path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = read_checked_json(manifest_path)
        if not isinstance(manifest, dict):
            raise ValueError("Manifest must be a checked JSON object")
        if Path(manifest.get("source_root", "")).resolve(strict=True) != source_root.resolve(strict=True):
            raise ValueError("Manifest source root differs from the requested source root")
        paths = validate_manifest(manifest, source_root)
        manifest_sha256 = streaming_sha256(manifest_path)
        jobs = fixed_jobs()[:3] if manifest.get("mode") == "pilot" else fixed_jobs()
        if manifest.get("mode") not in {"pilot", "full"}:
            raise ValueError("Sealer accepts only the frozen pilot or full grid")
        campaign_start_path = ledger_path.parent / "campaign_start.json"
        campaign_start = read_checked_json(campaign_start_path)
        _ledger_identity(campaign_start, campaign_start, manifest, manifest_sha256, jobs)
        if campaign_start.get("starts") != {} or campaign_start.get("records") != {} or campaign_start.get("status") != "ACTIVE":
            raise ValueError("Immutable campaign start is not the empty initial ledger")

        sequence, snapshot, snapshot_sha = _history_prefix(
            ledger_path, campaign_start, manifest, manifest_sha256, jobs
        )
        original_ledger_sha = _checked_original_ledger_hash(
            ledger_path, campaign_start, manifest, manifest_sha256, jobs
        )
        job_files = _scan_job_starts(
            ledger_path.parent, jobs, manifest["source_commit"], manifest_sha256
        )
        starts, orphan_ids = _merge_starts(ledger_path.parent, jobs, snapshot, job_files)
        for start in starts.values():
            token = start["invocation"]
            if live_groups(token):
                raise RuntimeError(f"Invocation still has a live guardian or worker: {token}")

        # This is a ciphertext-only projection. It does not invoke analysis or
        # read the truth join used for key/rank evaluation.
        dataset_path = paths["files"]["cohort"]
        dataset = _unique_json(dataset_path.read_bytes(), "opened public cohort")
        public = {job["job_id"]: public_case(dataset, job) for job in jobs}
        records: dict[str, dict] = {}
        original_records = snapshot["records"]
        for job in jobs:
            job_id = job["job_id"]
            start = starts.get(job_id)
            saved_row = original_records.get(job_id)
            is_orphan = job_id in orphan_ids
            if is_orphan:
                # Runner order is checked-start, checked history snapshot, then dispatch.
                # A per-job start absent from the selected history cannot have dispatched.
                orphan_dir = Path(start["directory"])
                if (orphan_dir / "guardian_receipt.json").exists() or (orphan_dir / "guardian.json").exists():
                    raise ValueError("Pre-history orphan start has worker artifacts")
                row = _row_base(
                    job,
                    "unavailable_after_coordinator_interrupted",
                    dispatched=False,
                    start=start,
                )
                row["error"] = "Checked start was written before its ledger snapshot; dispatch was not reached"
                records[job_id] = row
                continue
            if saved_row is None and start is None:
                records[job_id] = _row_base(
                    job, "unavailable_after_coordinator_interrupted", dispatched=False
                )
                continue
            if saved_row is not None:
                _validate_row_job(saved_row, job, "Saved ledger row")
                if not saved_row["dispatched"]:
                    if start is None:
                        if saved_row.get("calls") or saved_row.get("directory") or saved_row.get("guardian_receipt"):
                            raise ValueError("Undispatched saved row has unbound invocation output")
                        records[job_id] = dict(saved_row)
                    else:
                        parents = _expected_parents(job, records) if job["operation"] == "stage2" else []
                        records[job_id] = _recover_started_row(
                            manifest,
                            job,
                            start,
                            public[job_id],
                            parents,
                            snapshot["campaign_deadline_monotonic"],
                            saved_row,
                            must_be_interrupted=False,
                        )
                    continue
                if start is None:
                    raise ValueError("Dispatched saved row has no checked invocation start")
                directory = Path(start["directory"])
                if saved_row.get("directory") != str(directory) or saved_row.get("invocation") != start["invocation"]:
                    raise ValueError("Saved row and checked invocation start differ")
                parents = _expected_parents(job, records)
                if saved_row.get("outcome") == "completed":
                    try:
                        records[job_id] = _check_complete_row(
                            manifest,
                            job,
                            saved_row,
                            public[job_id],
                            parents,
                            snapshot["campaign_deadline_monotonic"],
                        )
                    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
                        records[job_id] = _unknown_invocation(job, start, error)
                    continue
                source = _read_source_row(directory, "supervisor_alarm.json")
                if source is not None and source != saved_row:
                    records[job_id] = _unknown_invocation(
                        job, start, "supervisor_alarm_mismatch"
                    )
                    continue
                records[job_id] = _recover_started_row(
                    manifest,
                    job,
                    start,
                    public[job_id],
                    parents,
                    snapshot["campaign_deadline_monotonic"],
                    saved_row,
                    must_be_interrupted=False,
                )
                continue
            # A history snapshot with a checked start but no result is the last
            # job reached by the coordinator. Keep only guardian-accepted calls.
            parents = _expected_parents(job, records)
            records[job_id] = _recover_started_row(
                manifest,
                job,
                start,
                public[job_id],
                parents,
                snapshot["campaign_deadline_monotonic"],
                None,
                must_be_interrupted=True,
            )

        # There can be no started invocation beyond the serial position already
        # closed or interrupted above. Every remaining fixed slot is a tombstone.
        if set(records) != {job["job_id"] for job in jobs}:
            raise ValueError("Sealer did not close the entire fixed worker grid")
        sealed = {
            **snapshot,
            "status": "STOPPED_ALARM",
            "stop_reason": "coordinator_interrupted",
            "completed_utc": _utc_now(),
            "starts": starts,
            "records": records,
            "sealed_from": {
                "ledger_sha256": original_ledger_sha,
                "history_sequence": sequence,
                "history_sha256": snapshot_sha,
                "campaign_start_sha256": streaming_sha256(campaign_start_path),
            },
        }
        atomic_json(output_path, sealed)
        return sealed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args(argv)
    seal(args.manifest, args.ledger, args.output, args.source_root)
    print(args.output)


if __name__ == "__main__":
    main()
