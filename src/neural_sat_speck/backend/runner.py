"""Serial backend coordinator with durable starts and a parent process watchdog."""

from __future__ import annotations

import contextlib
import fcntl
import json
import math
import os
import re
import signal
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from .environment import native_policy_sample, power_sample, sleep_assertion
from .guard import (
    THREAD_ENVIRONMENT,
    atomic_json,
    read_checked_json,
    sha256_file,
    validate_authorization,
    validate_manifest,
    validate_record,
)


class EnvironmentViolation(RuntimeError):
    pass


def now() -> str:
    return datetime.now(UTC).isoformat()


def failure(cell: dict, outcome: str, elapsed: float, **details) -> dict:
    return {
        **cell,
        "outcome": outcome,
        "online_seconds": elapsed,
        "verified_known": False,
        "verified_fresh": False,
        "assigned_key_match": False,
        "recovered_key": None,
        "solver_status": {
            "unknown": "unknown",
            "unsat_alarm": "unsat",
            "recovered": "sat",
            "different_verified_key": "sat",
            "verification_failed": "sat",
        }.get(outcome),
        **details,
    }


def _group_absent(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return True
    return False


def validate_ledger(ledger: dict, manifest: dict, manifest_sha: str) -> None:
    if (
        ledger.get("schema") != "backend_ledger_v1"
        or ledger.get("manifest_sha256") != manifest_sha
    ):
        raise ValueError("Resume execution identity mismatch")
    cells = {cell["cell_id"]: cell for cell in manifest["cells"]}
    starts, records = ledger.get("starts"), ledger.get("records")
    if (
        not isinstance(starts, dict)
        or not isinstance(records, dict)
        or not records.keys() <= starts.keys() <= cells.keys()
    ):
        raise ValueError("Malformed/duplicate start or result grid")
    for ident, start in starts.items():
        if (
            start.get("cell_id") != ident
            or re.fullmatch(r"[0-9a-f]{32}", str(start.get("invocation", ""))) is None
        ):
            raise ValueError("Malformed start identity")
    if len({start["invocation"] for start in starts.values()}) != len(starts):
        raise ValueError("Duplicate invocation identity")
    for ident, record in records.items():
        validate_record(record, cells[ident])
        if record.get("invocation") != starts[ident]["invocation"]:
            raise ValueError("Result invocation does not match its durable start")


def interrupted_phases(directory: Path, observed: float) -> dict:
    path = directory / "journal.json"
    if not path.exists():
        return {
            "completed_seconds": {},
            "active_phase": None,
            "inflight_seconds": None,
            "journal_available": False,
        }
    try:
        journal = read_checked_json(path)
        active = journal["phase"]
        started = journal["phase_started_monotonic"]
        return {
            "completed_seconds": journal["completed_seconds"],
            "active_phase": active,
            "inflight_seconds": max(0.0, observed - started)
            if active is not None
            else 0.0,
            "journal_available": True,
        }
    except (OSError, ValueError, KeyError, TypeError):
        return {
            "completed_seconds": {},
            "active_phase": None,
            "inflight_seconds": None,
            "journal_available": False,
        }


def _kill_and_reap(process: subprocess.Popen) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=5)


def _confirm_group_absence(group: int) -> bool:
    for _ in range(100):
        if _group_absent(group):
            return True
        time.sleep(0.01)
    return False


def run_cell(
    payload: dict, environment: dict[str, str], *, timeout: float = 300.0
) -> dict:
    """A separate supervisor owns the child and survives coordinator death."""
    started = time.perf_counter()
    deadline = started + timeout
    directory = Path(payload["directory"])
    guardian = None
    group = None
    observed = started
    result = None
    directory.mkdir(parents=True, exist_ok=False)
    payload_path = directory / "payload.json"
    atomic_json(payload_path, payload)
    guardian_path = directory / "guardian.json"
    atomic_json(
        guardian_path,
        {
            "deadline_monotonic": deadline,
            "invocation": payload["invocation"],
            "directory": str(directory),
            "repo_root": payload["repo_root"],
            "worker_arguments": [
                payload["python_path"],
                "-m",
                "neural_sat_speck.backend.worker",
                str(payload_path),
            ],
            "receipt": str(directory / "guardian_receipt.json"),
        },
    )

    def persist_identity():
        nonlocal group
        path = directory / "process.json"
        if not path.exists():
            return
        identity = read_checked_json(path)
        candidate = identity.get("process_group")
        if (
            identity.get("invocation") != payload["invocation"]
            or identity.get("supervisor_pid") != guardian.pid
            or type(candidate) is not int
            or candidate <= 1
            or identity.get("pid") != candidate
        ):
            raise ValueError("Supervisor child identity mismatch")
        if group is not None and candidate != group:
            raise ValueError("Supervisor child group changed")
        if group is None:
            group = candidate
            if payload.get("record_process"):
                ledger_path = Path(payload["record_process"])
                ledger = read_checked_json(ledger_path)
                ledger["starts"][payload["cell"]["cell_id"]]["process_group"] = group
                atomic_json(ledger_path, ledger)

    try:
        with (directory / "guardian_stderr.txt").open("w") as stderr:
            guardian = subprocess.Popen(
                [
                    payload["python_path"],
                    "-m",
                    "neural_sat_speck.backend.watchdog",
                    str(guardian_path),
                ],
                env=environment,
                cwd=payload["repo_root"],
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=stderr,
            )
        observations = []
        last_sample = started
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(guardian.args, timeout)
            try:
                guardian.wait(timeout=min(remaining, 0.05 if group is None else 1.0))
                persist_identity()
                break
            except subprocess.TimeoutExpired:
                if time.perf_counter() >= deadline:
                    raise
                # A two-file replacement may be momentarily incomplete while
                # the supervisor publishes its initial identity. Only defer
                # that read until its next wait; the final read is strict.
                try:
                    persist_identity()
                except (FileNotFoundError, ValueError):
                    if guardian.poll() is not None:
                        raise
                if time.perf_counter() - last_sample < 10.0:
                    continue
                last_sample = time.perf_counter()
                power = power_sample()
                atomic_json(directory / "power_latest.json", power)
                if power.get("accepted") is False:
                    raise EnvironmentViolation("Observed power/sleep policy violation")
                native_identity = directory / "native_process.json"
                if native_identity.exists():
                    native = read_checked_json(native_identity)
                    policy = native_policy_sample(native["pid"])
                    observations.append(
                        {
                            "utc": now(),
                            "pid": native["pid"],
                            "policy_sample": policy,
                            "power_sample": power,
                        }
                    )
                    atomic_json(directory / "native_process_samples.json", observations)
                    # ESRCH means the observed child has already exited, not a
                    # degraded process policy. Retain it as an unavailable sample.
                    errors = [
                        v["errno"] for v in policy.get("observations", {}).values()
                    ]
                    if policy.get("accepted") is False and 3 not in errors:
                        raise EnvironmentViolation(
                            "Observed native process policy violation"
                        )
        observed = time.perf_counter()
        elapsed = observed - started
        if elapsed >= timeout:
            result = failure(payload["cell"], "global_deadline", elapsed)
        elif guardian.returncode != 0 or not (directory / "result.json").exists():
            result = failure(
                payload["cell"],
                "worker_exception",
                elapsed,
                guardian_returncode=guardian.returncode,
            )
        else:
            child_receipt = read_checked_json(directory / "guardian_receipt.json")
            if (
                child_receipt.get("status") != "worker_exit"
                or child_receipt.get("worker_returncode") != 0
            ):
                result = failure(
                    payload["cell"],
                    "worker_exception",
                    elapsed,
                    worker_returncode=child_receipt.get("worker_returncode"),
                )
            else:
                result = read_checked_json(directory / "result.json")
                result["online_seconds"] = elapsed
                validate_record(result, payload["cell"])
    except subprocess.TimeoutExpired:
        observed = time.perf_counter()
        result = failure(payload["cell"], "global_deadline", observed - started)
    except EnvironmentViolation as error:
        observed = time.perf_counter()
        result = failure(
            payload["cell"],
            "environment_rejected",
            observed - started,
            error=str(error),
        )
    finally:
        # Includes worker startup and PID/ledger persistence failures. SIGTERM
        # requests supervisor cleanup; it is not a kill of the owning guardian.
        if guardian is not None:
            if guardian.poll() is None:
                if time.perf_counter() >= deadline:
                    try:
                        guardian.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        guardian.terminate()
                else:
                    guardian.terminate()
            try:
                guardian.wait(timeout=5)
            except subprocess.TimeoutExpired:
                guardian.kill()
                guardian.wait(timeout=5)
            if group is None:
                try:
                    persist_identity()
                except (OSError, ValueError):
                    if guardian.returncode == 0:
                        raise
            # If the guardian crashed, coordinator cleanup uses the recorded
            # child group; uncertain absence always blocks another dispatch.
            if guardian.returncode != 0:
                from .watchdog import invocation_groups

                # A crashed supervisor cannot vouch for a saved recyclable PID.
                # Additional cleanup needs a positively identified live command.
                for identified in invocation_groups(payload["invocation"]):
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(identified, signal.SIGKILL)
                    if not _confirm_group_absence(identified):
                        raise RuntimeError(
                            "Supervisor-crash orphan cleanup is unconfirmed"
                        )
    result["started_utc"] = payload["started_utc"]
    result["completed_utc"] = now()
    result["invocation"] = payload["invocation"]
    result["process_group_absent"] = group is not None and _confirm_group_absence(group)
    result["phase_accounting"] = interrupted_phases(directory, observed)
    result["guardian_returncode"] = guardian.returncode
    result["guardian_healthy"] = guardian.returncode == 0
    try:
        receipt = read_checked_json(directory / "guardian_receipt.json")
        result["guardian_receipt"] = receipt
        stamp = receipt.get("observed_monotonic")
        status = receipt.get("status")
        expected = {"worker_exit", "deadline_enforced", "coordinator_stop"}
        if (
            status not in expected
            or receipt.get("watchdog_pid") != guardian.pid
            or receipt.get("invocation") != payload["invocation"]
            or receipt.get("deadline_monotonic") != deadline
            or receipt.get("process_group") != group
            or receipt.get("process_group_absent") is not True
            or type(stamp) not in (int, float)
            or not math.isfinite(stamp)
            or (status == "deadline_enforced" and stamp < deadline)
            or (status == "worker_exit" and stamp >= deadline)
            or (
                status == "coordinator_stop"
                and result["outcome"] != "environment_rejected"
            )
        ):
            result["guardian_healthy"] = False
    except (OSError, ValueError) as error:
        result["guardian_receipt_unavailable"] = str(error)
        result["guardian_healthy"] = False
    result["cleanup_seconds"] = max(0.0, time.perf_counter() - observed)
    result["worker_directory"] = str(directory)
    return result


def _verify_recovery(record: dict, paths: dict[str, Path]) -> None:
    if record["outcome"] not in {
        "recovered",
        "different_verified_key",
        "verification_failed",
    }:
        return
    from .artifacts import verify_key

    ident = record["case_id"]
    public = json.loads(paths[f"{ident}:public"].read_bytes())
    truth = json.loads(paths[f"{ident}:truth"].read_bytes())
    flags = verify_key(
        record["cipher"], record["rounds"], record["recovered_key"], public, truth
    )
    if any(record.get(field) != flag for field, flag in flags.items()):
        raise ValueError("Resume/final key flags failed independent verification")


def run(
    manifest_path: Path,
    output: Path,
    *,
    authorization_path: Path | None = None,
    resume: bool = False,
    dispatch=run_cell,
) -> dict:
    output = output.resolve()
    run_started = time.perf_counter()
    run_started_utc = now()
    manifest = read_checked_json(manifest_path)
    root = Path(__file__).resolve().parents[3]
    paths = validate_manifest(manifest, root)
    manifest_sha = sha256_file(manifest_path)
    if authorization_path is None:
        raise ValueError("Separate reviewed launch authorization receipt is required")
    validate_authorization(authorization_path, manifest_sha, root, output=output)
    # Authorization/identity validation occurs before cohort loading or dispatch.
    from .artifacts import validate_case_inputs, validate_cohort
    from .evidence import audit_cell_evidence

    cohort = json.loads(paths["cohort"].read_bytes())
    validate_cohort(cohort)
    validate_case_inputs(cohort, paths)
    if output.exists() and not resume:
        raise FileExistsError("Existing output requires explicit --resume")
    if resume and not output.exists():
        raise FileNotFoundError("No ledger to resume")
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        output.with_name(output.name + ".lock").open("a") as lock,
        sleep_assertion() as sleep_receipt,
    ):
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another backend coordinator holds the lock") from error
        ledger = (
            read_checked_json(output)
            if resume
            else {
                "schema": "backend_ledger_v1",
                "manifest_sha256": manifest_sha,
                "created_utc": now(),
                "starts": {},
                "records": {},
                "status": "incomplete",
            }
        )
        validate_ledger(ledger, manifest, manifest_sha)
        if ledger.get("status") == "safety_stop":
            raise RuntimeError("Prior safety stop requires independent review")
        for ident, record in ledger["records"].items():
            _verify_recovery(record, paths)
            audit_cell_evidence(
                record, ledger["starts"][ident], manifest, paths, output
            )
        for ident, start in ledger["starts"].items():
            group = start.get("process_group")
            if group is None:
                # Coordinator may have died between spawn and PID persistence.
                raise RuntimeError(
                    "Unidentified prior process group: independent orphan audit required"
                )
            if type(group) is not int or group <= 1 or not _group_absent(group):
                raise RuntimeError(
                    "Prior worker group still exists or is unverified; no new dispatch"
                )
            if ident not in ledger["records"]:
                cell = next(c for c in manifest["cells"] if c["cell_id"] == ident)
                ledger["records"][ident] = failure(
                    cell,
                    "infrastructure_interrupted",
                    0.0,
                    duration_available=False,
                    invocation=start["invocation"],
                    interruption_review="prior_process_group_absent",
                )
        if any(
            r["outcome"]
            in {
                "unsat_alarm",
                "verification_failed",
                "solver_error",
                "environment_rejected",
                "worker_exception",
                "audit_rejected",
            }
            for r in ledger["records"].values()
        ):
            raise RuntimeError(
                "Prior correctness/environment/parser failure requires review"
            )
        atomic_json(output, ledger)
        ledger.setdefault("run_receipts", []).append(
            {
                "started_utc": run_started_utc,
                "shared_preflight_seconds": time.perf_counter() - run_started,
                "sleep_assertion": sleep_receipt,
            }
        )

        def finish(status: str) -> dict:
            ledger["status"] = status
            receipt = ledger["run_receipts"][-1]
            receipt.update(
                completed_utc=now(),
                total_awake_seconds=time.perf_counter() - run_started,
            )
            atomic_json(output, ledger)
            return ledger

        for cell in manifest["cells"]:
            ident = cell["cell_id"]
            if ident in ledger["records"]:
                continue
            preparation_started = time.perf_counter()
            sample = power_sample()
            if sample.get("accepted") is False:
                ledger.setdefault("environment_stops", []).append(sample)
                return finish("safety_stop")
            # Recheck source/runtime/artifact identity before every dispatch.
            validate_manifest(manifest, root)
            invocation = uuid.uuid4().hex
            started_utc = now()
            ledger["starts"][ident] = {
                "cell_id": ident,
                "invocation": invocation,
                "started_utc": started_utc,
                "process_group": None,
            }
            atomic_json(output, ledger)
            directory = output.parent / "workers" / f"{ident}_{invocation}"
            tmpdir = root / "runtime" / "backend_tmp" / invocation
            tmpdir.mkdir(parents=True, exist_ok=False)
            payload = {
                "cell": cell,
                "directory": str(directory),
                "repo_root": str(root),
                "invocation": invocation,
                "started_utc": started_utc,
                "record_process": str(output),
                "public_path": str(paths[f"{cell['case_id']}:public"]),
                "truth_path": str(paths[f"{cell['case_id']}:truth"]),
                "native_sha256": sha256_file(paths[f"{cell['case_id']}:native"]),
                "cnf_sha256": sha256_file(paths[f"{cell['case_id']}:cnf"]),
                "cms_path": str(paths["cms"]),
                "cadical_path": str(paths["cadical"]),
                "python_path": str(paths["python"]),
                "source_commit": manifest["source_commit_at_preparation"],
            }
            environment = {
                "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
                "LANG": "C",
                "LC_ALL": "C",
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPYCACHEPREFIX": str(tmpdir / "fresh_pycache"),
                **THREAD_ENVIRONMENT,
                "TMPDIR": str(tmpdir),
                "PYTHONPATH": str(root / "src"),
            }
            dispatch_started = time.perf_counter()
            pre_dispatch_seconds = dispatch_started - preparation_started
            post_dispatch_started = None
            try:
                result = dispatch(payload, environment)
                post_dispatch_started = time.perf_counter()
                atomic_json(directory / "coordinator_received.json", result)
                validate_record(result, cell)
                if result.get("invocation") != invocation:
                    raise ValueError("Worker result belongs to another invocation")
                _verify_recovery(result, paths)
                directory_files = {
                    p.name: sha256_file(p)
                    for p in directory.iterdir()
                    if p.is_file() and not p.name.endswith(".tmp")
                }
                result["raw_artifacts"] = directory_files
                result["pre_dispatch_seconds"] = pre_dispatch_seconds
                result["post_dispatch_audit_seconds"] = 0.0
                if result["outcome"] in {
                    "recovered",
                    "different_verified_key",
                    "unknown",
                    "global_deadline",
                    "unsat_alarm",
                }:
                    start_record = read_checked_json(output)["starts"][ident]
                    audit_cell_evidence(result, start_record, manifest, paths, output)
            except Exception as error:
                # An audit alarm is durable and prevents resume from laundering it
                # into an ordinary infrastructure interruption.
                result = failure(
                    cell,
                    "audit_rejected",
                    time.perf_counter() - dispatch_started,
                    audit_error=f"{type(error).__name__}: {error}",
                    audit_elapsed_includes_cleanup=True,
                    invocation=invocation,
                    started_utc=started_utc,
                    completed_utc=now(),
                )
            if post_dispatch_started is None:
                post_dispatch_started = time.perf_counter()
            # Reload the coordinator's persisted PID; never lose the start receipt.
            persisted = read_checked_json(output)
            ledger["starts"][ident] = persisted["starts"][ident]
            validate_record(result, cell)
            result["pre_dispatch_seconds"] = pre_dispatch_seconds
            result["post_dispatch_audit_seconds"] = (
                time.perf_counter() - post_dispatch_started
            )
            ledger["records"][ident] = result
            if (
                result.get("process_group_absent") is not True
                or result.get("guardian_healthy") is not True
                or result["outcome"]
                in {
                    "unsat_alarm",
                    "verification_failed",
                    "solver_error",
                    "environment_rejected",
                    "worker_exception",
                    "audit_rejected",
                }
            ):
                return finish("safety_stop")
            atomic_json(output, ledger)
        if set(ledger["records"]) != {c["cell_id"] for c in manifest["cells"]}:
            raise ValueError("Incomplete final cell grid")
        ledger["status"], ledger["completed_utc"] = "complete", now()
        return finish("complete")
