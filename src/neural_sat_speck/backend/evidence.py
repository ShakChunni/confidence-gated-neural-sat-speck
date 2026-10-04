"""Fail-closed audit of saved evidence for one backend cell.

This checks the raw process chain and the solver transcript. Formula/cipher
reconstruction, source closure, and fresh-key verification remain the callers'
responsibility; this module does not launch a worker or solver.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path
from typing import Any

from .guard import canonical_bytes, read_checked_json, validate_record
from .solver_io import parse_solver_output, solver_arguments

_INVOCATION = re.compile(r"[0-9a-f]{32}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SCIENTIFIC_OUTCOMES = {
    "recovered",
    "different_verified_key",
    "verification_failed",
    "unsat_alarm",
    "unknown",
    "solver_error",
}
_NATIVE_PROFILES = {
    "cms_native_default",
    "cms_native_gauss",
    "cms_native_no_gauss",
    "cms_cnf_no_gauss",
    "cadical_cnf",
}
_FINAL_BOOKKEEPING = {
    "pre_dispatch_seconds",
    "post_dispatch_audit_seconds",
    "raw_artifacts",
}
_COMMON_RAW_FILES = {
    "process.json",
    "guardian.json",
    "guardian_receipt.json",
    "coordinator_received.json",
}


def _fail(message: str) -> None:
    raise ValueError(f"Cell evidence audit failed: {message}")


def _same(left: Any, right: Any) -> bool:
    """Compare JSON values without coercing bools, ints, floats, or containers."""
    try:
        return canonical_bytes(left) == canonical_bytes(right)
    except (TypeError, ValueError):
        return False


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        _fail(f"{name} is not a finite number")
    result = float(value)
    if result < 0 or (positive and result <= 0):
        _fail(f"{name} is outside its allowed range")
    return result


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _no_symlink(path: Path, *, must_exist: bool = True) -> Path:
    absolute = _absolute(path)
    try:
        resolved = absolute.resolve(strict=must_exist)
    except (OSError, RuntimeError) as error:
        _fail(f"cannot resolve evidence path {absolute}: {error}")
    if resolved != absolute:
        _fail(f"symlink or non-canonical evidence path: {absolute}")
    if must_exist:
        try:
            mode = absolute.lstat().st_mode
        except OSError as error:
            _fail(f"cannot stat evidence path {absolute}: {error}")
        if stat.S_ISLNK(mode):
            _fail(f"symlink evidence path: {absolute}")
    return absolute


def _raw_path(worker_directory: Path, filename: str) -> Path:
    if (
        not isinstance(filename, str)
        or not filename
        or filename in {".", ".."}
        or Path(filename).name != filename
        or "/" in filename
        or "\\" in filename
    ):
        _fail(f"raw artifact name is not a basename: {filename!r}")
    return worker_directory / filename


def _verify_raw_artifacts(
    worker_directory: Path,
    raw_artifacts: Any,
    required: set[str],
    *,
    required_present: bool,
) -> set[str]:
    if not isinstance(raw_artifacts, dict):
        if required_present:
            _fail("raw_artifacts is missing or malformed")
        return set()
    if not required <= raw_artifacts.keys():
        _fail(
            f"raw_artifacts omits required files: {sorted(required - raw_artifacts.keys())}"
        )
    verified: set[str] = set()
    for filename, expected in raw_artifacts.items():
        path = _no_symlink(_raw_path(worker_directory, filename))
        try:
            mode = path.lstat().st_mode
            if not stat.S_ISREG(mode):
                _fail(f"raw artifact is not a regular file: {filename}")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            _fail(f"cannot read raw artifact {filename}: {error}")
        if not isinstance(expected, str) or _SHA256.fullmatch(expected) is None:
            _fail(f"raw artifact has an invalid SHA-256: {filename}")
        if actual != expected:
            _fail(f"raw artifact SHA-256 mismatch: {filename}")
        verified.add(filename)
    return verified


def _checked_json(
    worker_directory: Path, filename: str, verified: set[str]
) -> dict[str, Any]:
    if filename not in verified:
        _fail(f"checked JSON is not sealed in raw_artifacts: {filename}")
    path = _no_symlink(_raw_path(worker_directory, filename))
    sidecar = _no_symlink(_raw_path(worker_directory, filename + ".sha256"))
    try:
        if not stat.S_ISREG(sidecar.lstat().st_mode):
            _fail(f"JSON sidecar is not a regular file: {filename}.sha256")
        value = read_checked_json(path)
    except (OSError, ValueError) as error:
        _fail(f"checked JSON failed validation ({filename}): {error}")
    if not isinstance(value, dict):
        _fail(f"checked JSON is not an object: {filename}")
    return value


def _cell_for(record: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    cell_id = record.get("cell_id")
    cells = manifest.get("cells")
    if not isinstance(cell_id, str) or not isinstance(cells, list):
        _fail("record or manifest has no cell identity grid")
    matches = [
        cell
        for cell in cells
        if isinstance(cell, dict) and cell.get("cell_id") == cell_id
    ]
    if len(matches) != 1:
        _fail("cell identity is absent or duplicated in manifest")
    cell = matches[0]
    try:
        validate_record(record, cell)
    except (KeyError, TypeError, ValueError) as error:
        _fail(f"final record is structurally invalid: {error}")
    return cell


def _audit_interrupted(
    record: dict[str, Any],
    start: dict[str, Any],
    ledger_path: Path,
) -> dict[str, Any]:
    invocation = start.get("invocation")
    group = start.get("process_group")
    if (
        start.get("cell_id") != record.get("cell_id")
        or not isinstance(invocation, str)
        or _INVOCATION.fullmatch(invocation) is None
        or record.get("invocation") != invocation
        or type(group) is not int
        or group <= 1
        or record.get("interruption_review") != "prior_process_group_absent"
    ):
        _fail("interrupted record is not bound to its unique durable start/group")
    if record.get("duration_available") is not False:
        _fail("interrupted cell must mark its duration unavailable")
    if (
        any(
            record.get(field) is not False
            for field in ("verified_known", "verified_fresh", "assigned_key_match")
        )
        or record.get("recovered_key") is not None
    ):
        _fail("interrupted cell must not claim a recovered or verified key")
    _number(record.get("online_seconds"), "interrupted placeholder duration")

    expected_directory = (
        _absolute(ledger_path).parent / "workers" / f"{record['cell_id']}_{invocation}"
    )
    supplied = record.get("worker_directory")
    if supplied is not None and supplied != str(expected_directory):
        _fail("interrupted worker directory is not canonical")
    directory = _no_symlink(expected_directory, must_exist=False)
    raw = record.get("raw_artifacts")
    verified: set[str] = set()
    if raw is not None:
        if raw and not directory.is_dir():
            _fail("interrupted raw artifacts name a missing worker directory")
        if directory.is_dir():
            _no_symlink(directory)
            verified = _verify_raw_artifacts(
                directory, raw, set(), required_present=True
            )
        elif not isinstance(raw, dict) or raw:
            _fail("interrupted raw artifact map is malformed")
    return {
        "schema": "backend_cell_evidence_audit_v1",
        "cell_id": record["cell_id"],
        "invocation": invocation,
        "outcome": "infrastructure_interrupted",
        "scope": "interrupted_start_only",
        "process_group": group,
        "retry_status": "not_retried_in_recorded_start_grid",
        "raw_artifacts_verified": len(verified),
        "timing": {
            "duration_available": False,
            "online_seconds": None,
            "unavailable_reason": "interrupted duration is a placeholder, not a measurement",
        },
        "solver_evidence": {
            "status": "unavailable",
            "model_claimed": False,
            "late_result_claimed": False,
        },
        "limits": [
            "No complete worker/guardian receipt is required for an interrupted start",
            "No-retry uniqueness is established by the caller's ledger-grid audit",
            "The current process-group absence event is not re-created by this audit",
        ],
    }


def _audit_process_chain(
    record: dict[str, Any],
    start: dict[str, Any],
    manifest: dict[str, Any],
    paths: dict[str, Path],
    worker_directory: Path,
    ledger_path: Path,
    verified: set[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    process = _checked_json(worker_directory, "process.json", verified)
    guardian = _checked_json(worker_directory, "guardian.json", verified)
    receipt = _checked_json(worker_directory, "guardian_receipt.json", verified)
    payload = _checked_json(worker_directory, "payload.json", verified)

    invocation = start["invocation"]
    group = process.get("process_group")
    worker_pid = process.get("pid")
    supervisor_pid = process.get("supervisor_pid")
    if (
        process.get("invocation") != invocation
        or type(worker_pid) is not int
        or worker_pid <= 1
        or type(group) is not int
        or group <= 1
        or worker_pid != group
        or start.get("process_group") != group
        or type(supervisor_pid) is not int
        or supervisor_pid <= 1
    ):
        _fail("process.json does not match the durable worker start")

    deadline = guardian.get("deadline_monotonic")
    deadline_value = _number(deadline, "guardian deadline", positive=True)
    if (
        guardian.get("invocation") != invocation
        or guardian.get("directory") != str(worker_directory)
        or guardian.get("receipt") != str(worker_directory / "guardian_receipt.json")
        or not isinstance(guardian.get("repo_root"), str)
        or guardian.get("worker_arguments")
        != [
            str(paths["python"]),
            "-m",
            "neural_sat_speck.backend.worker",
            str(worker_directory / "payload.json"),
        ]
    ):
        _fail("guardian.json is not bound to this cell worker")

    source_commit = manifest.get("source_commit_at_preparation")
    if (
        not isinstance(source_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
    ):
        _fail("manifest source commit is malformed")
    cell = next(
        cell for cell in manifest["cells"] if cell["cell_id"] == record["cell_id"]
    )
    expected_payload = {
        "cell": cell,
        "directory": str(worker_directory),
        "invocation": invocation,
        "record_process": str(_absolute(ledger_path)),
        "source_commit": source_commit,
        "cms_path": str(paths["cms"]),
        "cadical_path": str(paths["cadical"]),
        "python_path": str(paths["python"]),
        "public_path": str(paths[f"{cell['case_id']}:public"]),
        "truth_path": str(paths[f"{cell['case_id']}:truth"]),
        "started_utc": start.get("started_utc"),
        "native_sha256": hashlib.sha256(
            paths[f"{cell['case_id']}:native"].read_bytes()
        ).hexdigest(),
        "cnf_sha256": hashlib.sha256(
            paths[f"{cell['case_id']}:cnf"].read_bytes()
        ).hexdigest(),
    }
    if any(
        not _same(payload.get(key), value) for key, value in expected_payload.items()
    ):
        _fail("payload.json differs from durable cell/runtime/artifact identities")

    watchdog_pid = receipt.get("watchdog_pid")
    worker_returncode = receipt.get("worker_returncode")
    if (
        receipt.get("invocation") != invocation
        or type(watchdog_pid) is not int
        or watchdog_pid != supervisor_pid
        or not _same(receipt.get("deadline_monotonic"), deadline)
        or type(receipt.get("worker_pid")) is not int
        or receipt.get("worker_pid") != worker_pid
        or type(receipt.get("process_group")) is not int
        or receipt.get("process_group") != group
        or type(worker_returncode) is not int
    ):
        _fail("guardian receipt does not match process and invocation identities")
    observed = _number(receipt.get("observed_monotonic"), "guardian observation")
    cleanup = _number(receipt.get("cleanup_completed_monotonic"), "guardian cleanup")
    if cleanup < observed:
        _fail("guardian cleanup precedes its observation")
    status = receipt.get("status")
    if not isinstance(status, str) or status not in {
        "worker_exit",
        "deadline_enforced",
        "coordinator_stop",
    }:
        _fail("guardian status is missing or unsuccessful")
    if status == "deadline_enforced" and observed < deadline_value:
        _fail("guardian claims deadline enforcement before its deadline")
    if status == "worker_exit" and worker_returncode != 0:
        _fail("normal worker exit has a nonzero child return code")
    if status == "worker_exit" and observed >= deadline_value:
        _fail("guardian claims an on-time worker exit after its deadline")
    if status == "coordinator_stop" and record.get("outcome") != "environment_rejected":
        _fail("guardian was stopped for a non-environment result")
    if (
        record.get("guardian_receipt") is None
        or not _same(record.get("guardian_receipt"), receipt)
        or record.get("guardian_returncode") != 0
        or record.get("guardian_healthy") is not True
        or record.get("process_group_absent") is not True
        or receipt.get("process_group_absent") is not True
    ):
        _fail("final coordinator record does not certify guardian cleanup")
    if record.get("started_utc") != start.get("started_utc"):
        _fail("final record start timestamp differs from durable start")

    if record["outcome"] == "global_deadline":
        if status == "coordinator_stop":
            _fail("global deadline conflicts with coordinator-stop receipt")
        if (
            any(
                record.get(field) is not False
                for field in ("verified_known", "verified_fresh", "assigned_key_match")
            )
            or record.get("recovered_key") is not None
        ):
            _fail("deadline record claims a key or verification")
    elif record["outcome"] != "environment_rejected" and status != "worker_exit":
        _fail("non-deadline result lacks a normal worker-exit receipt")

    return process, guardian, receipt, payload


def _audit_python_receipt(
    record: dict[str, Any],
    manifest: dict[str, Any],
    worker_directory: Path,
    verified: set[str],
    process: dict[str, Any],
) -> bool:
    name = "python_worker_receipt.json"
    exists = name in verified
    required = record.get("outcome") in _SCIENTIFIC_OUTCOMES
    if required and not exists:
        _fail("scientific result lacks a sealed Python worker receipt")
    if not exists:
        return False
    receipt = _checked_json(worker_directory, name, verified)
    if (
        receipt.get("pid") != process.get("pid")
        or receipt.get("git_commit") != manifest.get("source_commit_at_preparation")
        or receipt.get("is_certified") is not True
    ):
        _fail("Python worker receipt PID/source/certification mismatch")
    return True


def _worker_result(
    record: dict[str, Any],
    cell: dict[str, Any],
    worker_directory: Path,
    verified: set[str],
) -> dict[str, Any] | None:
    name = "result.json"
    if name not in verified:
        if record["outcome"] == "global_deadline":
            return None
        _fail("non-deadline record lacks a sealed worker result")
    result = _checked_json(worker_directory, name, verified)
    for key, value in cell.items():
        if key not in result or not _same(result[key], value):
            _fail(f"raw worker cell identity differs: {key}")
    if record["outcome"] != "global_deadline":
        for key in (
            "outcome",
            "verified_known",
            "verified_fresh",
            "assigned_key_match",
            "recovered_key",
        ):
            if not _same(record.get(key), result.get(key)):
                _fail(f"final record differs from raw worker result: {key}")
        if "solver_status" in result and not _same(
            record.get("solver_status"), result.get("solver_status")
        ):
            _fail("final solver status differs from raw worker result")
    return result


def _audit_native_output(
    record: dict[str, Any],
    cell: dict[str, Any],
    manifest: dict[str, Any],
    paths: dict[str, Path],
    worker_directory: Path,
    verified: set[str],
    process: dict[str, Any],
    worker_result: dict[str, Any],
) -> dict[str, Any]:
    profile = cell["profile"]
    pure = profile in {"cms_cnf_no_gauss", "cadical_cnf"}
    input_name = "input.cnf" if pure else "input.dimacs"
    artifact_kind = "cnf" if pure else "native"
    required = {
        "native_launch.json",
        "native_launch_receipt.json",
        "native_process.json",
        "solver_stdout.txt",
        "solver_stderr.txt",
        input_name,
    }
    if not required <= verified:
        _fail(
            f"completed native outcome lacks raw files: {sorted(required - verified)}"
        )

    varmap_path = paths.get(f"{cell['case_id']}:varmap")
    if not isinstance(varmap_path, Path):
        _fail("validated variable map path is missing")
    try:
        varmap = json.loads(varmap_path.read_bytes())
    except (OSError, ValueError) as error:
        _fail(f"cannot read validated variable map: {error}")
    if not isinstance(varmap, dict):
        _fail("variable map is not an object")
    native_num_vars = varmap.get("native_num_vars")
    submitted_num_vars = varmap.get("pure_cnf_num_vars") if pure else native_num_vars
    if (
        type(native_num_vars) is not int
        or native_num_vars < 1
        or type(submitted_num_vars) is not int
        or submitted_num_vars < 1
        or worker_result.get("native_num_vars") != native_num_vars
        or worker_result.get("submitted_num_vars") != submitted_num_vars
    ):
        _fail("worker variable counts differ from the checked variable map")

    input_path = _no_symlink(_raw_path(worker_directory, input_name))
    artifact_path = paths.get(f"{cell['case_id']}:{artifact_kind}")
    if not isinstance(artifact_path, Path):
        _fail("validated solver formula path is missing")
    if (
        hashlib.sha256(input_path.read_bytes()).digest()
        != hashlib.sha256(artifact_path.read_bytes()).digest()
    ):
        _fail("solver input bytes differ from the validated canonical formula")

    expected_arguments = solver_arguments(
        profile,
        str(paths["cms"]),
        str(paths["cadical"]),
        str(input_path),
    )
    launch = _checked_json(worker_directory, "native_launch.json", verified)
    receipt = _checked_json(worker_directory, "native_launch_receipt.json", verified)
    native_process = _checked_json(worker_directory, "native_process.json", verified)
    if (
        launch.get("arguments") != expected_arguments
        or launch.get("receipt") != str(worker_directory / "native_launch_receipt.json")
        or receipt.get("solver_arguments") != expected_arguments
        or native_process.get("argv") != expected_arguments
        or worker_result.get("solver_arguments") != expected_arguments
        or not _same(worker_result.get("native_launch_receipt"), receipt)
        or (
            record["outcome"] != "global_deadline"
            and record.get("solver_arguments") != expected_arguments
        )
    ):
        _fail("native argv differs across launch request, receipt, process, and result")
    native_pid = native_process.get("pid")
    group = process["process_group"]
    source_commit = manifest["source_commit_at_preparation"]
    if (
        type(native_pid) is not int
        or native_pid <= 1
        or native_process.get("process_group") != group
        or receipt.get("pid") != native_pid
        or receipt.get("git_commit") != source_commit
        or receipt.get("is_certified") is not True
    ):
        _fail("native launch PID/group/source/certification mismatch")
    if record["outcome"] != "global_deadline" and not _same(
        record.get("native_launch_receipt"), receipt
    ):
        _fail("final record differs from native launch receipt")

    solver_returncode = worker_result.get("solver_returncode")
    if type(solver_returncode) is not int:
        _fail("completed native result lacks an integer solver exit code")
    try:
        stdout = _raw_path(worker_directory, "solver_stdout.txt").read_text()
        _raw_path(worker_directory, "solver_stderr.txt").read_text()
    except (OSError, UnicodeError) as error:
        _fail(f"cannot read native solver logs: {error}")
    parsed = parse_solver_output(
        "cadical" if profile == "cadical_cnf" else "cms",
        solver_returncode,
        stdout,
        required_variables=list(range(1, submitted_num_vars + 1)),
        num_vars=submitted_num_vars,
    )
    if parsed["status"] != worker_result.get("solver_status"):
        _fail("native stdout/exit code does not reproduce worker solver status")
    if record["outcome"] != "global_deadline" and (
        record.get("solver_returncode") != solver_returncode
        or record.get("solver_status") != parsed["status"]
        or not _same(record.get("gauss_evidence"), parsed.get("gauss_evidence"))
    ):
        _fail("final native status/log evidence differs from raw solver output")
    if not _same(worker_result.get("gauss_evidence"), parsed.get("gauss_evidence")):
        _fail("worker Gauss diagnostics differ from reparsed solver stdout")

    assignment = parsed["assignment"]
    return {
        "source": "solver_stdout.txt and solver exit code",
        "status": parsed["status"],
        "model_claimed": parsed["status"] == "sat",
        "assigned_variable_count": len(assignment),
        "required_variable_count": submitted_num_vars,
        "model_complete_for_submitted_variables": parsed["status"] == "sat"
        and len(assignment) == submitted_num_vars,
        "formula_satisfaction_rechecked": False,
        "matrix_count_samples": parsed["gauss_evidence"]["matrix_count_samples"],
        "gauss_log_is_proof_of_use": False,
    }


def audit_cell_evidence(
    record: dict[str, Any],
    start: dict[str, Any],
    manifest: dict[str, Any],
    paths: dict[str, Path],
    ledger_path: Path,
) -> dict[str, Any]:
    """Audit one final record against its durable start and raw worker files.

    Any identity, digest, sidecar, output/status, or cell-field mismatch raises
    ``ValueError``. `infrastructure_interrupted` returns an explicitly limited
    start-only audit because that outcome has no complete worker artifact
    guarantee. The function never starts a process and never promotes late
    worker evidence into a recovered final record.
    """
    if not all(isinstance(value, dict) for value in (record, start, manifest, paths)):
        _fail("record/start/manifest/paths must be mappings")
    cell = _cell_for(record, manifest)
    invocation = start.get("invocation")
    if (
        start.get("cell_id") != cell["cell_id"]
        or not isinstance(invocation, str)
        or _INVOCATION.fullmatch(invocation) is None
        or record.get("invocation") != invocation
    ):
        _fail("record is not bound to its durable unique start")
    if record["outcome"] == "infrastructure_interrupted":
        return _audit_interrupted(record, start, ledger_path)

    expected_directory = (
        _absolute(ledger_path).parent / "workers" / f"{cell['cell_id']}_{invocation}"
    )
    supplied_directory = record.get("worker_directory")
    if supplied_directory != str(expected_directory):
        _fail("worker_directory is not the canonical ledger-owned path")
    worker_directory = _no_symlink(expected_directory)
    if not worker_directory.is_dir():
        _fail("worker_directory is not a directory")
    raw_artifacts = record.get("raw_artifacts")
    required_files = _COMMON_RAW_FILES | {"payload.json"}
    verified = _verify_raw_artifacts(
        worker_directory, raw_artifacts, required_files, required_present=True
    )

    coordinator = _checked_json(worker_directory, "coordinator_received.json", verified)
    comparable_record = {
        key: value for key, value in record.items() if key not in _FINAL_BOOKKEEPING
    }
    if not _same(comparable_record, coordinator):
        _fail("final record differs from coordinator_received.json")
    for name in ("pre_dispatch_seconds", "post_dispatch_audit_seconds"):
        if name not in record:
            _fail(f"final record lacks coordinator bookkeeping: {name}")
        _number(record[name], name)

    process, guardian, guardian_receipt, _payload = _audit_process_chain(
        record, start, manifest, paths, worker_directory, ledger_path, verified
    )
    worker_result = _worker_result(record, cell, worker_directory, verified)
    deadline_outcome = record["outcome"] == "global_deadline"
    if worker_result is not None and deadline_outcome:
        # Validate its cell identity, but never let late worker keys or success
        # flags alter the coordinator's deadline outcome.
        late_result = {
            "present": True,
            "outcome": worker_result.get("outcome"),
            "recovery_claim_credited": False,
        }
    else:
        late_result = {
            "present": False,
            "outcome": None,
            "recovery_claim_credited": False,
        }

    python_certified = _audit_python_receipt(
        record, manifest, worker_directory, verified, process
    )
    solver_evidence: dict[str, Any]
    native_completed = worker_result is not None and (
        "solver_returncode" in worker_result
        or (not deadline_outcome and record["outcome"] in _SCIENTIFIC_OUTCOMES)
    )
    if cell["profile"] in _NATIVE_PROFILES and native_completed:
        solver_evidence = _audit_native_output(
            record,
            cell,
            manifest,
            paths,
            worker_directory,
            verified,
            process,
            worker_result,
        )
        if deadline_outcome:
            solver_evidence["recovery_claim_credited"] = False
    else:
        status = (
            worker_result.get("solver_status") if worker_result is not None else None
        )
        solver_evidence = {
            "source": "checked worker result only",
            "status": status if not deadline_outcome else "not_claimed",
            "model_claimed": False,
            "native_model_captured": False,
            "recovery_claim_credited": False,
        }

    duration_available = record.get("duration_available", True)
    if duration_available is not True:
        _fail("completed coordinator record must have an available online duration")
    phase_accounting = record.get("phase_accounting")
    if not isinstance(phase_accounting, dict):
        _fail("coordinator phase accounting is missing")
    journal_available = phase_accounting.get("journal_available") is True
    inflight = phase_accounting.get("inflight_seconds") if journal_available else None
    if inflight is not None:
        inflight = _number(inflight, "inflight phase duration")
    online_seconds = _number(record.get("online_seconds"), "online duration")
    if deadline_outcome and online_seconds < float(
        manifest.get("deadline_seconds", 300.0)
    ):
        _fail("global deadline record precedes the frozen cutoff")

    return {
        "schema": "backend_cell_evidence_audit_v1",
        "cell_id": cell["cell_id"],
        "invocation": invocation,
        "outcome": record["outcome"],
        "scope": "complete_raw_worker_audit",
        "raw_artifacts_verified": len(verified),
        "python_worker_certified": python_certified,
        "late_worker_result": late_result,
        "timing": {
            "duration_available": True,
            "online_seconds": online_seconds,
            "phase_journal_available": journal_available,
            "inflight_phase_seconds": inflight,
        },
        "solver_evidence": solver_evidence,
        "process_identity": {
            "worker_pid": process["pid"],
            "process_group": process["process_group"],
            "guardian_pid": guardian_receipt["watchdog_pid"],
            "deadline_monotonic": guardian["deadline_monotonic"],
            "group_absence_claim": record["process_group_absent"],
        },
        "limits": [
            "Formula satisfaction and fresh-key verification are checked by callers",
            "Native SAT assignment completeness is not a second formula check",
            "No all-thread or continuous scheduling claim is made by this audit",
        ],
    }
