"""Serial orchestration for a fixed rank-only opened-data diagnostic."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from neural_sat_speck.backend.environment import native_policy_sample, sleep_assertion
from neural_sat_speck.backend.guard import atomic_json, read_checked_json
from neural_sat_speck.followups.environment import power_sample, same_python_environment
from neural_sat_speck.followups.stage2_control import (
    PROFILE,
    assemble_queue,
    ordered_hash,
    public_case,
    streaming_sha256,
    validate_manifest,
    validate_order,
)


class CampaignCeilingReached(TimeoutError):
    """The fixed coordinator deadline elapsed; retain evidence and close slots."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def save_ledger(output: Path, ledger: dict) -> None:
    """Preserve checked versions before replacing the crash-sensitive latest pair."""
    history = output.parent / "ledger_history"
    history.mkdir(exist_ok=True)
    sequence = len(list(history.glob("*.json")))
    path = history / f"{sequence:04d}.json"
    if path.exists() or path.with_name(path.name + ".sha256").exists():
        raise FileExistsError("Never replace an archived ledger version")
    atomic_json(path, ledger)
    atomic_json(output, ledger)


def memory_sample(group: int) -> dict:
    sample = subprocess.run(
        ["/bin/ps", "-ax", "-o", "pgid=,rss="],
        capture_output=True,
        text=True,
        check=True,
        timeout=2,
    )
    members = [
        int(v[1]) * 1024
        for line in sample.stdout.splitlines()
        if len(v := line.split()) == 2 and int(v[0]) == group
    ]
    total = sum(members)
    return {
        "group_rss_bytes": total,
        "members_observed": len(members),
        "accepted": total <= 2684354560,
    }


def public_payload(
    manifest: dict,
    job: dict,
    public: dict,
    parents: list,
    invocation: str,
    directory: Path,
) -> dict:
    return {
        "schema": "stage2_rank_payload_v1",
        "job": job,
        "public_case": public,
        "parents": parents,
        "invocation": invocation,
        "directory": str(directory),
        "source_root": manifest["source_root"],
        "source_commit": manifest["source_commit"],
        "source_files": {
            key.removeprefix("source:"): value
            for key, value in manifest["files"].items()
            if key.startswith("source:")
        },
        "assets": manifest["assets"],
        "runtime": manifest["runtime"],
        "profile": PROFILE,
    }


def normalize_job(
    job: dict, directory: Path, started: float, monitor_stop: str | None
) -> dict:
    receipt = read_checked_json(directory / "guardian_receipt.json")
    if (
        receipt.get("invocation") != directory.name
        or receipt.get("job_id") != job["job_id"]
        or receipt.get("process_group_absent") is not True
    ):
        raise ValueError("Guardian identity or process cleanup was not verified")
    events = receipt.get("accepted_events", [])
    calls, load = [], None
    previous = started
    for position, event in enumerate(events):
        if (
            event.get("invocation") != directory.name
            or event.get("job_id") != job["job_id"]
        ):
            raise ValueError("Accepted event belongs to a different invocation")
        completed = event["completed_monotonic"]
        if type(completed) not in (int, float) or completed < previous:
            raise ValueError("Accepted phase clocks are invalid")
        if position == 0:
            if event["event"] != "loaded":
                raise ValueError("Missing initial loaded event")
            load = completed - started
        else:
            ordinal = position - 1
            if event.get("ordinal") != ordinal or event["event"] != "call_complete":
                raise ValueError("Missing or repeated accepted rank call")
            path = directory / f"calls/{ordinal:04d}_result.json"
            call = read_checked_json(path)
            start = read_checked_json(directory / f"calls/{ordinal:04d}_start.json")
            if (
                event.get("result") != str(path.relative_to(directory))
                or event.get("result_sha256") != streaming_sha256(path)
                or call.get("invocation") != directory.name
                or call.get("job_id") != job["job_id"]
                or call.get("scorer") != job["scorer"]
                or call.get("ordinal") != ordinal
                or start.get("invocation") != directory.name
                or start.get("job_id") != job["job_id"]
                or start.get("ordinal") != ordinal
                or start.get("scorer") != job["scorer"]
                or start.get("parent") != call.get("parent")
                or start.get("call_started_monotonic")
                != call.get("call_started_monotonic")
                or not previous
                <= call["call_started_monotonic"]
                <= call["call_completed_monotonic"]
                <= completed
            ):
                raise ValueError(
                    "Result/start/event clock or invocation identity differs"
                )
            validate_order(call["ordered_words"])
            calls.append({**call, "phase_wall_seconds": completed - previous})
        previous = completed
    expected = 1 if job["operation"] == "stage1" else 8
    status = receipt["status"]
    outcome = monitor_stop or {
        "worker_exit": "completed"
        if receipt.get("worker_returncode") == 0 and len(calls) == expected
        else "worker_exception",
        "load_timeout": "load_timeout",
        "rank_timeout": "rank_timeout",
        "campaign_timeout": "campaign_timeout",
        "coordinator_stop": "interrupted",
        "guardian_error": "integrity_alarm",
    }.get(status, "integrity_alarm")
    worker_not_started = receipt.get("worker_not_started") is True
    return {
        **job,
        "outcome": outcome,
        "dispatched": not worker_not_started,
        "directory": str(directory),
        "invocation": directory.name,
        "calls": calls,
        "load_wall_seconds": load,
        "worker_wall_seconds": None
        if worker_not_started
        else receipt["cleanup_completed_monotonic"] - started,
        "queue_measurements": {},
        "guardian_receipt": receipt,
        "accepted_events": events,
        "completed_utc": utc_now(),
    }


def run_job(
    manifest: dict,
    job: dict,
    public: dict,
    parents: list,
    directory: Path,
    deadline: float,
    environment: dict,
) -> dict:
    started = time.perf_counter()
    payload = public_payload(manifest, job, public, parents, directory.name, directory)
    atomic_json(directory / "payload.json", payload)
    python = manifest["files"]["python" if job["scorer"] == "ddt7" else "tf_python"][
        "path"
    ]
    atomic_json(
        directory / "guardian.json",
        {
            "invocation": directory.name,
            "directory": str(directory),
            "repo_root": manifest["source_root"],
            "coordinator_pid": os.getpid(),
            "worker_arguments": [
                python,
                str(
                    Path(manifest["source_root"])
                    / "scripts/run_simon_matched_stage2_worker.py"
                ),
                str(directory / "payload.json"),
            ],
            "receipt": str(directory / "guardian_receipt.json"),
            "job_id": job["job_id"],
            "job_started_monotonic": started,
            "campaign_deadline_monotonic": deadline,
            "load_seconds": 120.0,
            "call_seconds": 180.0,
            "expected_calls": 1 if job["operation"] == "stage1" else 8,
        },
    )
    stopped, samples = None, []
    with (
        (directory / "guardian_stdout.txt").open("w") as stdout,
        (directory / "guardian_stderr.txt").open("w") as stderr,
    ):
        guardian = subprocess.Popen(
            [
                manifest["files"]["python"]["path"],
                "-m",
                "neural_sat_speck.followups.stage2_watchdog",
                str(directory / "guardian.json"),
            ],
            cwd=manifest["source_root"],
            env=environment,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        last_sample = 0.0
        sampled_worker = False
        try:
            while guardian.poll() is None:
                now = time.perf_counter()
                process_path = directory / "process.json"
                process_ready = (
                    process_path.exists()
                    and process_path.with_name("process.json.sha256").exists()
                )
                if now - last_sample >= 10 or (process_ready and not sampled_worker):
                    sample = {"monotonic": now, "power": power_sample()}
                    if process_ready:
                        process = read_checked_json(process_path)
                        if (
                            process.get("invocation") != directory.name
                            or type(process.get("pid")) is not int
                            or process["pid"] <= 1
                            or process.get("process_group") != process["pid"]
                        ):
                            raise ValueError("Resource sample process identity differs")
                        sample["memory"] = memory_sample(process["process_group"])
                        sample["process_policy"] = native_policy_sample(process["pid"])
                        sampled_worker = True
                    samples.append(sample)
                    atomic_json(directory / "environment_samples.json", samples)
                    if sample["power"].get("accepted") is not True:
                        stopped = "environment_alarm"
                    elif sample.get("memory", {}).get("accepted") is False:
                        stopped = "rss_stop"
                    elif (
                        sample.get("memory", {}).get("members_observed", 0) > 0
                        and sample.get("process_policy", {}).get("accepted") is not True
                    ):
                        # Recheck exit before treating an exited process as a policy alarm.
                        if not (directory / "guardian_receipt.json").exists():
                            stopped = "environment_alarm"
                    last_sample = now
                if stopped:
                    guardian.terminate()
                    break
                time.sleep(0.05)
            guardian.wait(timeout=10)
        except BaseException:
            if guardian.poll() is None:
                guardian.terminate()
                guardian.wait(timeout=10)
            raise
    row = normalize_job(job, directory, started, stopped)
    if job["operation"] == "stage2":
        complete = {call["ordinal"]: call for call in row["calls"]}
        for width in (1, 8):
            if time.perf_counter() >= deadline:
                row["outcome"] = "campaign_timeout"
                break
            if not all(index in complete for index in range(width)):
                continue
            assemble_started = time.perf_counter()
            queue = assemble_queue(
                parents,
                {index: complete[index]["ordered_words"] for index in range(width)},
                width,
            )
            atomic_json(
                directory / f"queue_m{width}.json",
                {
                    "invocation": directory.name,
                    "job_id": job["job_id"],
                    "width": width,
                    "candidate_tuples": queue,
                },
            )
            if time.perf_counter() >= deadline:
                # Preserve the late file but do not count it as an available queue.
                row["outcome"] = "campaign_timeout"
                break
            row["queue_measurements"][str(width)] = {
                "seconds": time.perf_counter() - assemble_started,
                "queue_sha256": ordered_hash(queue),
            }
    atomic_json(directory / "supervisor_result.json", row)
    return row


def unavailable(job: dict, reason: str) -> dict:
    return {
        **job,
        "outcome": reason,
        "dispatched": False,
        "calls": [],
        "load_wall_seconds": None,
        "worker_wall_seconds": None,
        "queue_measurements": {},
        "completed_utc": utc_now(),
    }


def validate_execution_identity(
    manifest: dict, row: dict, public: dict, parents: list, deadline: float
) -> None:
    """Bind detached artifacts to the intended payload, caps and live invocation."""
    from neural_sat_speck.followups.stage2_watchdog import live_groups

    directory = Path(row["directory"])
    job = next(job for job in manifest["jobs"] if job["job_id"] == row["job_id"])
    expected = public_payload(manifest, job, public, parents, directory.name, directory)
    payload = read_checked_json(directory / "payload.json")
    guardian = read_checked_json(directory / "guardian.json")
    receipt = row["guardian_receipt"]
    no_worker = receipt.get("worker_not_started") is True
    if no_worker:
        if (
            (directory / "process.json").exists()
            or receipt.get("worker_pid") is not None
            or row["calls"]
        ):
            raise ValueError("Pre-dispatch stop contradicts process or call artifacts")
        process = {"invocation": directory.name, "pid": None, "process_group": None}
    else:
        process = read_checked_json(directory / "process.json")
    python = manifest["files"]["python" if job["scorer"] == "ddt7" else "tf_python"][
        "path"
    ]
    arguments = [
        python,
        str(
            Path(manifest["source_root"]) / "scripts/run_simon_matched_stage2_worker.py"
        ),
        str(directory / "payload.json"),
    ]
    if (
        payload != expected
        or guardian.get("worker_arguments") != arguments
        or guardian.get("invocation") != directory.name
        or guardian.get("job_id") != job["job_id"]
        or guardian.get("campaign_deadline_monotonic") != deadline
        or guardian.get("load_seconds") != 120.0
        or guardian.get("call_seconds") != 180.0
        or guardian.get("expected_calls") != (1 if job["operation"] == "stage1" else 8)
        or receipt.get("config_sha256") != streaming_sha256(directory / "guardian.json")
        or process.get("invocation") != directory.name
        or process.get("pid") != receipt.get("worker_pid")
        or process.get("process_group") != receipt.get("process_group")
        or process.get("pid") != process.get("process_group")
        or live_groups(directory.name)
    ):
        raise ValueError("Execution payload, cap, process or invocation binding failed")
    for call in row["calls"]:
        expected_parent = (
            None if job["operation"] == "stage1" else parents[call["ordinal"]]
        )
        if call["parent"] != expected_parent or call["phase_wall_seconds"] > 180.0:
            raise ValueError(
                "Rank call used a different parent or exceeded its phase cap"
            )
    if row["load_wall_seconds"] is not None and row["load_wall_seconds"] > 120.0:
        raise ValueError("Loaded output exceeded the startup phase cap")
    for event in row["accepted_events"]:
        event_path = directory / event["event_path"]
        raw_event = read_checked_json(event_path)
        if streaming_sha256(event_path) != event["event_sha256"] or any(
            event.get(key) != value for key, value in raw_event.items()
        ):
            raise ValueError("Accepted event differs from the persisted source event")
    expected_runtime = manifest["runtime"][
        "python" if job["scorer"] == "ddt7" else "tensorflow"
    ]
    expected_assets = {
        name: record["identity"] for name, record in manifest["assets"].items()
    }
    if row["accepted_events"]:
        loaded = row["accepted_events"][0]
        loader = loaded.get("loader_metadata", {})
        if (
            loaded.get("runtime") != expected_runtime
            or loaded.get("asset_identity") != expected_assets
            or loaded.get("telemetry", {}).get("git_commit")
            != manifest["source_commit"]
        ):
            raise ValueError(
                "Partial worker loaded metadata differs from pinned runtime or assets"
            )
        if job["scorer"] in {"vv9", "vv8"}:
            if (
                loader.get("rounds") != (9 if job["scorer"] == "vv9" else 8)
                or loader.get("sha256") != manifest["assets"][job["scorer"]]["sha256"]
            ):
                raise ValueError(
                    "Partial worker loaded a different checkpoint or depth"
                )
        elif (
            loader.get("loaded_rounds") != 7
            or loader.get("sha256_preflight") != manifest["assets"]["ddt"]["sha256"]
        ):
            raise ValueError("Partial worker mapped a different DDT or depth")
    if row["outcome"] == "completed":
        final = read_checked_json(directory / "worker_result.json")
        expected_refs = [
            {
                "ordinal": call["ordinal"],
                "parent": call["parent"],
                "result": f"calls/{call['ordinal']:04d}_result.json",
                "result_sha256": streaming_sha256(
                    directory / f"calls/{call['ordinal']:04d}_result.json"
                ),
            }
            for call in row["calls"]
        ]
        if (
            final.get("status") != "complete"
            or final.get("job") != job
            or final.get("job_id") != job["job_id"]
            or final.get("invocation") != directory.name
            or final.get("scorer") != job["scorer"]
            or final.get("parents") != parents
            or final.get("calls") != expected_refs
            or final.get("completed_call_count") != len(row["calls"])
            or final.get("runtime") != expected_runtime
            or final.get("asset_identity_at_load") != expected_assets
            or final.get("asset_identity_at_exit") != expected_assets
            or final.get("telemetry", {}).get("git_commit") != manifest["source_commit"]
        ):
            raise ValueError("Final worker metadata differs from its pinned execution")


def run(
    manifest_path: Path,
    output: Path,
    root: Path,
    *,
    campaign_started: float | None = None,
) -> dict:
    campaign_started = (
        time.perf_counter() if campaign_started is None else campaign_started
    )
    deadline = campaign_started + 28800.0
    if (
        output.exists()
        or output.with_name(output.name + ".sha256").exists()
        or (output.parent / "campaign_start.json").exists()
        or any((output.parent / "ledger_history").glob("*.json"))
    ):
        raise FileExistsError("No retry or resume: preserve existing measurement")
    manifest = read_checked_json(manifest_path)
    paths = validate_manifest(manifest, root)
    if not same_python_environment(sys.executable, paths["files"]["python"]):
        raise ValueError("Coordinator interpreter environment drift")
    if time.perf_counter() >= deadline:
        raise RuntimeError("Campaign ceiling exhausted during preflight")
    dataset = json.loads(paths["files"]["cohort"].read_bytes())
    from neural_sat_speck.backend.artifacts import validate_cohort

    validate_cohort(dataset)
    # Validate every projection before any worker starts.
    public = {job["job_id"]: public_case(dataset, job) for job in manifest["jobs"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if output.exists():
            raise FileExistsError("Another coordinator already owns this measurement")
        environment = {
            **os.environ,
            **PROFILE["numeric_thread_env"],
            "PYTHONPATH": str(root / "src"),
            "TMPDIR": str(root / "runtime" / "stage2_control_tmp"),
            "CUDA_VISIBLE_DEVICES": "-1",
        }
        Path(environment["TMPDIR"]).mkdir(parents=True, exist_ok=True)
        ledger = {
            "schema": "stage2_control_results_v1",
            "mode": manifest["mode"],
            "source_commit": manifest["source_commit"],
            "manifest_sha256": streaming_sha256(manifest_path),
            "profile": PROFILE,
            "started_utc": utc_now(),
            "campaign_started_monotonic": campaign_started,
            "campaign_deadline_monotonic": deadline,
            "status": "ACTIVE",
            "starts": {},
            "records": {},
        }
        with sleep_assertion() as assertion:
            ledger["sleep_assertion"] = assertion
            for _ in range(20):
                if power_sample().get("accepted") is True:
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError("Power/sleep telemetry rejected before dispatch")
            atomic_json(output.parent / "campaign_start.json", ledger)
            save_ledger(output, ledger)
            stop_reason = None
            for job in manifest["jobs"]:
                name = job["job_id"]
                if time.perf_counter() >= deadline:
                    stop_reason = stop_reason or "campaign_timeout"
                stage1 = next(
                    (
                        record
                        for record in ledger["records"].values()
                        if record["key_index"] == job["key_index"]
                        and record["operation"] == "stage1"
                    ),
                    None,
                )
                if stop_reason:
                    row = unavailable(job, "unavailable_after_" + stop_reason)
                elif job["operation"] == "stage2" and not (stage1 and stage1["calls"]):
                    row = unavailable(job, "dependency_unavailable")
                else:
                    parents = (
                        []
                        if job["operation"] == "stage1"
                        else stage1["calls"][0]["ordered_words"][:8]
                    )
                    token = uuid.uuid4().hex
                    directory = output.parent / "jobs" / name / token
                    directory.mkdir(parents=True, exist_ok=False)
                    ledger["starts"][name] = {
                        "invocation": token,
                        "directory": str(directory),
                        "started_utc": utc_now(),
                    }
                    atomic_json(
                        directory / "campaign_job_start.json",
                        {
                            **ledger["starts"][name],
                            "job": job,
                            "source_commit": manifest["source_commit"],
                            "manifest_sha256": ledger["manifest_sha256"],
                        },
                    )
                    save_ledger(output, ledger)
                    try:
                        row = run_job(
                            manifest,
                            job,
                            public[name],
                            parents,
                            directory,
                            deadline,
                            environment,
                        )
                        validate_execution_identity(
                            manifest, row, public[name], parents, deadline
                        )
                    except Exception as error:
                        stop_reason = (
                            "campaign_timeout"
                            if isinstance(error, CampaignCeilingReached)
                            else "integrity_alarm"
                        )
                        row = {
                            **unavailable(job, stop_reason),
                            "dispatched": True,
                            "directory": str(directory),
                            "invocation": token,
                            "error": f"{type(error).__name__}: {error}",
                        }
                        # The independent guardian retains prior completed calls
                        # even if the coordinator alarm interrupted postprocessing.
                        if (
                            stop_reason == "campaign_timeout"
                            and (directory / "guardian_receipt.json").exists()
                        ):
                            saved = read_checked_json(directory / "guardian.json")
                            row = normalize_job(
                                job,
                                directory,
                                saved["job_started_monotonic"],
                                "campaign_timeout",
                            )
                        atomic_json(directory / "supervisor_alarm.json", row)
                    if row["outcome"] not in {
                        "completed",
                        "load_timeout",
                        "rank_timeout",
                        "rss_stop",
                    }:
                        stop_reason = row["outcome"]
                ledger["records"][name] = row
                if time.perf_counter() >= deadline:
                    stop_reason = stop_reason or "campaign_timeout"
                save_ledger(output, ledger)
                print(
                    json.dumps(
                        {
                            "completed_jobs": len(ledger["records"]),
                            "planned_jobs": len(manifest["jobs"]),
                            "job_id": name,
                            "outcome": row["outcome"],
                            "completed_calls": len(row["calls"]),
                        }
                    ),
                    flush=True,
                )
            if time.perf_counter() >= deadline:
                stop_reason = stop_reason or "campaign_timeout"
            ledger.update(
                status="STOPPED_ALARM" if stop_reason else "COMPLETE",
                stop_reason=stop_reason,
                completed_utc=utc_now(),
                campaign_wall_seconds=time.perf_counter() - campaign_started,
            )
            save_ledger(output, ledger)
    return ledger
