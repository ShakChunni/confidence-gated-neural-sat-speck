"""Serial conditioning-interface campaign using the existing independent guardian."""

from __future__ import annotations

import fcntl
import os
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from neural_sat_speck.backend.artifacts import verify_key
from neural_sat_speck.backend.environment import native_policy_sample, sleep_assertion
from neural_sat_speck.backend.formula import build_formula
from neural_sat_speck.backend.guard import (
    THREAD_ENVIRONMENT,
    atomic_json,
    checked_path,
    read_checked_json,
    sha256_file,
)
from neural_sat_speck.backend.worker import _complete_assignment
from neural_sat_speck.followups.conditioning_control import (
    PROFILE,
    SCIENTIFIC_OUTCOMES,
    digest,
    formula_identity,
    load_manifest,
    seconds,
    verify_cases,
)
from neural_sat_speck.followups.environment import power_sample
from neural_sat_speck.followups.runner import memory_sample
from neural_sat_speck.followups.stage2_receipts import (
    _validate_policy,
    _validate_power,
    _validate_tmpdir,
)


def now() -> str:
    return datetime.now(UTC).isoformat()


def controlled_environment(root: Path, tmpdir: Path) -> dict[str, str]:
    """Preserve ordinary environment while excluding interpreter/loader injection."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("DYLD_", "LD_", "PYTHON"))
    }
    return {
        **env,
        **THREAD_ENVIRONMENT,
        "PYTHONPATH": str(root / "src"),
        "PYTHONNOUSERSITE": "1",
        "TMPDIR": str(tmpdir.resolve()),
    }


def phase_accounting(directory: Path, observed: float) -> dict | None:
    path = directory / "journal.json"
    if not path.exists():
        return None
    journal = read_checked_json(path)
    completed = {
        key: seconds(value) for key, value in journal["completed_seconds"].items()
    }
    active = journal.get("phase")
    inflight = seconds(observed - journal["phase_started_monotonic"]) if active else 0.0
    return {
        "completed_seconds": completed,
        "inflight_phase": active,
        "inflight_seconds": inflight,
    }


def validate_cell(row: dict, manifest: dict, paths: dict[str, Path]) -> None:
    """Audit the invocation, original model and independent cipher verification."""
    cell = row["cell"]
    if cell not in manifest["cells"]:
        raise ValueError("Unexpected completed cell")
    directory = Path(row["directory"])
    payload = read_checked_json(directory / "payload.json")
    config = read_checked_json(directory / "guardian.json")
    receipt = read_checked_json(directory / "guardian_receipt.json")
    if row.get("supervisor_artifact") == "supervisor_sealed_result.json":
        from neural_sat_speck.followups.conditioning_seal import _guardian_is_running

        loss = read_checked_json(directory / "coordinator_loss.json")
        samples_path = directory / "environment_samples.json"
        sealed_samples = (
            read_checked_json(samples_path) if samples_path.exists() else []
        )
        if (
            row.get("outcome") != "infrastructure_interrupted"
            or row.get("coordinator_loss") != loss
            or receipt.get("status") != "coordinator_stop"
            or type(config.get("coordinator_pid")) is not int
            or loss.get("coordinator_pid") != config.get("coordinator_pid")
            or not seconds(row.get("started_monotonic"))
            <= seconds(loss.get("observed_monotonic"))
            <= seconds(receipt.get("observed_monotonic"))
            or row.get("environment_samples") != sealed_samples
            or _guardian_is_running(receipt["watchdog_pid"], directory.name)
        ):
            raise ValueError(
                "Derived interrupted row lacks independent coordinator-loss proof"
            )
    if row.get("dispatched") is False:
        if (
            row.get("outcome") != "infrastructure_interrupted"
            or receipt.get("status") != "coordinator_stop"
            or receipt.get("worker_not_started") is not True
            or receipt.get("process_group_absent") is not True
            or (directory / "process.json").exists()
            or "worker_pid" in receipt
            or "process_group" in receipt
            or config.get("invocation") != directory.name
            or payload.get("invocation_id") != directory.name
            or receipt.get("invocation") != directory.name
            or payload.get("cell") != cell
            or payload.get("source_commit") != manifest["source_commit"]
            or payload.get("repo_root") != manifest["source_root"]
            or payload.get("directory") != str(directory)
            or payload.get("public_sha256")
            != manifest["cases"][cell["case_id"]]["public"]["sha256"]
            or payload.get("truth_sha256")
            != manifest["cases"][cell["case_id"]]["truth"]["sha256"]
            or config.get("directory") != str(directory)
            or row.get("guardian_receipt") != receipt
            or config.get("deadline_monotonic") != payload.get("deadline_monotonic")
            or receipt.get("deadline_monotonic") != payload.get("deadline_monotonic")
        ):
            raise ValueError("Unproved pre-dispatch interruption")
        return
    process = read_checked_json(directory / "process.json")
    token = directory.name
    if (
        payload.get("cell") != cell
        or payload.get("invocation_id") != token
        or payload.get("directory") != str(directory)
        or payload.get("repo_root") != manifest["source_root"]
        or payload.get("source_commit") != manifest["source_commit"]
        or payload.get("formula_sha256")
        != manifest["cases"][cell["case_id"]]["formula_sha256"]
        or payload.get("public_path") != str(paths[cell["case_id"] + ":public"])
        or payload.get("truth_path") != str(paths[cell["case_id"] + ":truth"])
        or payload.get("public_sha256")
        != manifest["cases"][cell["case_id"]]["public"]["sha256"]
        or payload.get("truth_sha256")
        != manifest["cases"][cell["case_id"]]["truth"]["sha256"]
        or config.get("invocation") != token
        or process.get("invocation") != token
        or receipt.get("invocation") != token
        or config.get("directory") != str(directory)
        or config.get("repo_root") != manifest["source_root"]
        or config.get("receipt") != str(directory / "guardian_receipt.json")
        or config.get("worker_arguments")
        != [
            str(paths["python"]),
            "-m",
            "neural_sat_speck.followups.conditioning_worker",
            str(directory / "payload.json"),
        ]
        or process.get("supervisor_pid") != receipt.get("watchdog_pid")
        or receipt.get("worker_pid") != process.get("pid")
        or type(process.get("pid")) is not int
        or process["pid"] <= 1
        or process["pid"] != process.get("process_group")
        or process["pid"] != receipt.get("process_group")
        or receipt.get("process_group_absent") is not True
    ):
        raise ValueError("Invocation/process/formula identity mismatch")
    start, deadline, observed = (
        seconds(row[name])
        for name in (
            "started_monotonic",
            "deadline_monotonic",
            "observed_monotonic",
        )
    )
    if (
        deadline != start + 300.0
        or config.get("deadline_monotonic") != deadline
        or payload.get("deadline_monotonic") != deadline
        or receipt.get("deadline_monotonic") != deadline
        or receipt.get("observed_monotonic") != observed
        or seconds(row["parent_wall_seconds"]) != observed - start
        or row.get("guardian_receipt") != receipt
    ):
        raise ValueError("Deadline/elapsed identity mismatch")
    status = receipt.get("status")
    if status not in {"worker_exit", "deadline_enforced", "coordinator_stop"}:
        raise ValueError("Guardian failed")
    if (
        status == "deadline_enforced"
        and observed < deadline
        or status == "worker_exit"
        and observed >= deadline
    ):
        raise ValueError("Guardian cutoff status contradicts clock")
    if row.get("phase_accounting") != phase_accounting(directory, observed):
        raise ValueError("Phase accounting differs from journal")
    samples = read_checked_json(directory / "environment_samples.json")
    if samples != row.get("environment_samples") or not samples:
        raise ValueError("Missing resource observations")
    for sample in samples:
        # Reject infrastructure rows before invoking strict positive validators.
        if row["outcome"] == "environment_rejected":
            continue
        _validate_power(sample["power"])
        if "worker_policy" in sample:
            _validate_policy(
                sample["worker_policy"], process["pid"], terminal_exit_proven=True
            )
            mem = sample["memory"]
            if (
                type(mem.get("group_rss_bytes")) is not int
                or mem["group_rss_bytes"] < 0
                or mem.get("accepted") is not True
                or mem["group_rss_bytes"] > PROFILE["process_group_rss_limit_bytes"]
            ):
                raise ValueError("Memory resource observation rejected")
    telemetry_path = directory / "python_worker_receipt.json"
    if telemetry_path.exists():
        telemetry = read_checked_json(telemetry_path)
        if (
            telemetry.get("is_certified") is not True
            or telemetry.get("pid") != process["pid"]
            or telemetry.get("parent_pid") != receipt["watchdog_pid"]
            or telemetry.get("git_commit") != manifest["source_commit"]
            or telemetry.get("qos_class") not in {21, 25, 33}
            or any(
                telemetry.get(n) != 0
                for n in ("darwin_priority", "darwin_bg_state", "io_policy_state")
            )
            or telemetry.get("python_executable") != str(paths["python"])
            or any(
                telemetry.get("thread_environment", {}).get(n) != v
                for n, v in THREAD_ENVIRONMENT.items()
            )
        ):
            raise ValueError("Worker execution receipt differs")
        _validate_tmpdir(telemetry, Path(manifest["source_root"]))
    elif row["outcome"] in SCIENTIFIC_OUTCOMES:
        raise ValueError("Scientific outcome lacks worker execution certification")
    if row["outcome"] == "global_deadline":
        if status != "deadline_enforced":
            raise ValueError("Deadline row lacks deadline receipt")
        return
    if row["outcome"] not in SCIENTIFIC_OUTCOMES:
        return
    raw = read_checked_json(directory / "worker_result.json")
    if (
        raw != row.get("worker_result")
        or raw.get("invocation_id") != token
        or raw.get("cell_id") != cell["cell_id"]
        or raw.get("outcome") != row["outcome"]
        or raw.get("formula_sha256") != payload["formula_sha256"]
        or seconds(raw.get("completed_at_monotonic")) >= deadline
        or status != "worker_exit"
        or receipt.get("worker_returncode") != 0
    ):
        raise ValueError("Finished result does not belong to on-time invocation")
    if raw["outcome"] == "solver_unknown":
        if raw.get("solver_status") != "unknown":
            raise ValueError("Unknown status differs")
        return
    public = read_checked_json(paths[cell["case_id"] + ":public"])
    truth = read_checked_json(paths[cell["case_id"] + ":truth"])
    formula = build_formula(cell["cipher"], public, 4, 1, include_cnf=False)
    if digest(formula_identity(formula)) != raw["formula_sha256"]:
        raise ValueError("Model's original formula differs")
    model = raw.get("model")
    if (
        not isinstance(model, list)
        or len(model) != formula["num_vars"] + 1
        or model[0] is not None
    ):
        raise ValueError("Incomplete original SAT model")
    assignment = dict(enumerate(model[1:], start=1))
    if not _complete_assignment(formula, assignment):
        raise ValueError("Returned model violates the submitted formula")
    key = [
        sum(int(assignment[v]) << bit for bit, v in enumerate(word))
        for word in formula["varmap"]["master_key"]
    ]
    verification = verify_key(cell["cipher"], cell["rounds"], key, public, truth)
    if (
        raw.get("recovered_key") != key
        or raw.get("formula_valid") is not True
        or raw.get("solver_status") != "sat"
        or verification["verified_known"] is not True
        or any(raw.get(k) is not v for k, v in verification.items())
    ):
        raise ValueError("Independent key verification differs")
    expected = (
        "recovered"
        if verification["verified_fresh"] and verification["assigned_key_match"]
        else "verified_different_key"
        if verification["verified_fresh"]
        else "sat_model_not_fresh_verified"
    )
    if raw["outcome"] != expected:
        raise ValueError("Recovery category differs from independent verification")


def run_cell(payload: dict, environment: dict) -> dict:
    directory = Path(payload["directory"])
    directory.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    deadline = start + 300.0
    payload["deadline_monotonic"] = deadline
    atomic_json(directory / "payload.json", payload)
    atomic_json(
        directory / "guardian.json",
        {
            "deadline_monotonic": deadline,
            "invocation": payload["invocation_id"],
            "directory": str(directory),
            "repo_root": payload["repo_root"],
            "coordinator_pid": os.getpid(),
            "worker_arguments": [
                payload["python_path"],
                "-m",
                "neural_sat_speck.followups.conditioning_worker",
                str(directory / "payload.json"),
            ],
            "receipt": str(directory / "guardian_receipt.json"),
        },
    )
    samples, rejected = [], False
    with (directory / "guardian_stderr.txt").open("w") as stderr:
        guardian = subprocess.Popen(
            [
                payload["python_path"],
                "-m",
                "neural_sat_speck.followups.watchdog",
                str(directory / "guardian.json"),
            ],
            env=environment,
            cwd=payload["repo_root"],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=stderr,
        )
    try:
        next_sample, sampled_worker = start, False
        while guardian.poll() is None:
            current = time.perf_counter()
            process_path = directory / "process.json"
            if current >= next_sample or (not sampled_worker and process_path.exists()):
                next_sample = current + 10.0
                sample = {
                    "utc": now(),
                    "observed_monotonic": current,
                    "power": power_sample(),
                }
                if process_path.exists():
                    identity = read_checked_json(process_path)
                    sample["worker_policy"] = native_policy_sample(identity["pid"])
                    sample["memory"] = memory_sample(identity["process_group"])
                    sampled_worker = True
                    try:
                        _validate_policy(
                            sample["worker_policy"],
                            identity["pid"],
                            terminal_exit_proven=True,
                        )
                    except ValueError:
                        rejected = True
                    rejected |= sample["memory"].get("accepted") is not True
                rejected |= sample["power"].get("accepted") is not True
                samples.append(sample)
                atomic_json(directory / "environment_samples.json", samples)
            if rejected and process_path.exists():
                guardian.terminate()
                break
            if current > deadline + 5:
                guardian.terminate()
                break
            time.sleep(0.05)
        guardian.wait(timeout=8)
    except BaseException:
        if guardian.poll() is None:
            guardian.terminate()
        guardian.wait(timeout=8)
        raise
    receipt = read_checked_json(directory / "guardian_receipt.json")
    if receipt.get("process_group_absent") is not True:
        raise RuntimeError("Guardian failed to prove cleanup")
    observed = receipt["observed_monotonic"]
    raw_path = directory / "worker_result.json"
    raw = read_checked_json(raw_path) if raw_path.exists() else None
    if rejected:
        outcome = "environment_rejected"
    elif receipt["status"] == "deadline_enforced":
        outcome = "global_deadline"
    elif (
        receipt["status"] != "worker_exit"
        or receipt.get("worker_returncode") != 0
        or raw is None
    ):
        outcome = "worker_exception"
    else:
        outcome = raw["outcome"]
    row = {
        "cell": payload["cell"],
        "directory": str(directory),
        "outcome": outcome,
        "started_monotonic": start,
        "deadline_monotonic": deadline,
        "observed_monotonic": observed,
        "parent_wall_seconds": observed - start,
        "guardian_receipt": receipt,
        "cleanup_wall_seconds": receipt["cleanup_completed_monotonic"] - observed,
        "phase_accounting": phase_accounting(directory, observed),
        "environment_samples": samples,
        "worker_result": raw,
        "completed_utc": now(),
    }
    atomic_json(directory / "supervisor_result.json", row)
    return row


def audit_ledger(path: Path, manifest_path: Path) -> tuple[dict, dict]:
    manifest, paths = load_manifest(manifest_path)
    verify_cases(manifest, paths)
    ledger = read_checked_json(path)
    cells = manifest["cells"]
    if (
        ledger.get("schema") != "conditioning_control_results_v1"
        or ledger.get("manifest_sha256") != sha256_file(manifest_path)
        or ledger.get("source_commit") != manifest["source_commit"]
        or ledger.get("mode") != manifest["mode"]
    ):
        raise ValueError("Ledger execution identity differs")
    rows, starts = ledger.get("records"), ledger.get("starts")
    if (
        not isinstance(rows, dict)
        or not isinstance(starts, dict)
        or set(rows) != {c["cell_id"] for c in cells}
        or not set(starts) <= set(rows)
    ):
        raise ValueError("Incomplete or duplicate fixed cell grid")
    schedule = [c["cell_id"] for c in cells]
    if set(starts) != set(schedule[: len(starts)]):
        raise ValueError("Starts violate the fixed serial schedule")
    for cell in cells:
        name, row = cell["cell_id"], rows[cell["cell_id"]]
        if row.get("cell") != cell:
            raise ValueError("Stored row belongs to another cell")
        if row["outcome"] == "unavailable":
            if name in starts or not row.get("reason"):
                raise ValueError("Unavailable row has a dispatch or lacks reason")
            continue
        if name not in starts or starts[name].get("directory") != row.get("directory"):
            raise ValueError("Executed row lacks recorded start")
        artifact = row.get("supervisor_artifact", "supervisor_result.json")
        if artifact not in {
            "supervisor_result.json",
            "supervisor_sealed_result.json",
        } or (
            artifact == "supervisor_sealed_result.json"
            and ledger.get("status") != "INTERRUPTED"
        ):
            raise ValueError("Unexpected derived supervisor artifact")
        if starts[name].get("invocation_id") != Path(row["directory"]).name:
            raise ValueError("Stored invocation token differs")
        if read_checked_json(Path(row["directory"]) / artifact) != row:
            raise ValueError("Ledger row differs from supervisor evidence")
        validate_cell(row, manifest, paths)
    all_scientific = all(r["outcome"] in SCIENTIFIC_OUTCOMES for r in rows.values())
    if ledger.get("status") == "COMPLETE" and not all_scientific:
        raise ValueError("Campaign completion hides an infrastructure/correctness stop")
    return ledger, manifest


def require_pilot(manifest: dict, root: Path) -> None:
    pin = manifest.get("pilot_ledger")
    if pin is None:
        raise ValueError("Full launch lacks a pilot")
    pilot_path = checked_path(root, pin)
    pilot_ledger = read_checked_json(pilot_path)
    ledger, pilot = audit_ledger(pilot_path, Path(pilot_ledger["manifest_path"]))
    if (
        pilot["mode"] != "pilot"
        or ledger["status"] != "COMPLETE"
        or pilot["source_commit"] != manifest["source_commit"]
        or pilot["files"] != manifest["files"]
    ):
        raise ValueError("Pilot did not pass under the same pinned runtime/source")
    for row in ledger["records"].values():
        phases = row["phase_accounting"]
        if phases is None or (
            "solving" not in phases["completed_seconds"]
            and phases["inflight_phase"] != "solving"
        ):
            raise ValueError("Pilot failed to establish actual solver dispatch")
        request = read_checked_json(Path(row["directory"]) / "solver_dispatch.json")
        cell = row["cell"]
        public = read_checked_json(
            Path(pilot["cases"][cell["case_id"]]["public"]["path"])
        )
        formula = build_formula(cell["cipher"], public, 4, 1, include_cnf=False)
        expected_assumptions = (
            formula["assumption_literals"]
            if cell["arm"] == "temporary_assumptions"
            else []
        )
        if (
            request.get("invocation_id") != Path(row["directory"]).name
            or request.get("cell_id") != cell["cell_id"]
            or request.get("arm") != cell["arm"]
            or request.get("formula_sha256")
            != pilot["cases"][cell["case_id"]]["formula_sha256"]
            or request.get("assumptions") != expected_assumptions
            or request.get("permanent_unit_count")
            != (16 if cell["arm"] == "permanent_units" else 0)
            or request.get("solver_configuration") != {"verbose": 0, "threads": 1}
            or not row["started_monotonic"]
            <= seconds(request.get("requested_at_monotonic"))
            <= row["deadline_monotonic"] - 1
        ):
            raise ValueError("Pilot lacks a bound on-time frozen solver request")


def execute(manifest_path: Path, output: Path) -> dict:
    manifest, paths = load_manifest(manifest_path)
    verify_cases(manifest, paths)
    if manifest["mode"] == "full":
        require_pilot(manifest, Path(manifest["source_root"]))
    if output.exists():
        raise ValueError("Campaign directory already exists; no retries or resume")
    output.mkdir(parents=True)
    ledger_path = output / "ledger.json"
    ledger = {
        "schema": "conditioning_control_results_v1",
        "mode": manifest["mode"],
        "manifest_path": str(manifest_path.absolute()),
        "manifest_sha256": sha256_file(manifest_path),
        "source_commit": manifest["source_commit"],
        "started_utc": now(),
        "status": "RUNNING",
        "starts": {},
        "records": {},
    }
    revision = 0

    def persist():
        nonlocal revision
        atomic_json(output / "history" / f"{revision:04d}.json", ledger)
        atomic_json(ledger_path, ledger)
        revision += 1

    root = Path(manifest["source_root"])
    tmpdir = root / "runtime" / "conditioning_tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    env = controlled_environment(root, tmpdir)
    begun = time.perf_counter()
    stop_reason = None
    with (
        (output / "coordinator.lock").open("w") as lock,
        sleep_assertion() as assertion,
    ):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger["sleep_assertion"] = assertion
        persist()
        for cell in manifest["cells"]:
            if stop_reason is not None:
                ledger["records"][cell["cell_id"]] = {
                    "cell": cell,
                    "outcome": "unavailable",
                    "reason": stop_reason,
                }
                persist()
                continue
            if time.perf_counter() - begun > manifest["campaign_ceiling_seconds"] - 308:
                stop_reason = "campaign_ceiling_before_dispatch"
                ledger["records"][cell["cell_id"]] = {
                    "cell": cell,
                    "outcome": "unavailable",
                    "reason": stop_reason,
                }
                persist()
                continue
            ident, token = cell["case_id"], uuid.uuid4().hex
            directory = output / "cells" / cell["cell_id"] / token
            payload = {
                "cell": cell,
                "invocation_id": token,
                "directory": str(directory),
                "repo_root": str(root),
                "source_commit": manifest["source_commit"],
                "python_path": str(paths["python"]),
                "public_path": str(paths[ident + ":public"]),
                "truth_path": str(paths[ident + ":truth"]),
                "public_sha256": manifest["cases"][ident]["public"]["sha256"],
                "truth_sha256": manifest["cases"][ident]["truth"]["sha256"],
                "formula_sha256": manifest["cases"][ident]["formula_sha256"],
            }
            ledger["starts"][cell["cell_id"]] = {
                "directory": str(directory),
                "invocation_id": token,
                "utc": now(),
            }
            persist()
            row = run_cell(payload, env)
            ledger["records"][cell["cell_id"]] = row
            try:
                validate_cell(row, manifest, paths)
            except Exception as error:
                stop_reason = f"audit_rejected: {type(error).__name__}: {error}"
                ledger["audit_rejection"] = stop_reason
            if row["outcome"] not in SCIENTIFIC_OUTCOMES:
                stop_reason = stop_reason or row["outcome"]
            persist()
            print(
                f"{len(ledger['records'])}/{len(manifest['cells'])} {cell['cell_id']} {row['outcome']} {row['parent_wall_seconds']:.2f}s",
                flush=True,
            )
        ledger["status"] = "STOPPED" if stop_reason else "COMPLETE"
        ledger["stop_reason"] = stop_reason
        ledger["completed_utc"] = now()
        persist()
    if ledger["status"] == "COMPLETE":
        audit_ledger(ledger_path, manifest_path)
    return ledger
