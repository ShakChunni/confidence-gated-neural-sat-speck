"""Seal, without resuming, a conditioning campaign after coordinator loss.

The sealer creates a new ledger beside the original RUNNING ledger. It accepts
only a complete, checked guardian stop receipt for the one serial in-flight
cell. Crash windows without that terminal evidence require manual reconciliation
and are deliberately outside this automatic path.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import math
import re
import subprocess
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from neural_sat_speck.backend.guard import (
    atomic_json,
    canonical_bytes,
    read_checked_json,
    sha256_file,
)
from neural_sat_speck.followups.conditioning_control import (
    load_manifest,
    seconds,
    verify_cases,
)
from neural_sat_speck.followups.conditioning_runner import (
    audit_ledger,
    phase_accounting,
    validate_cell,
)

TOKEN_RE = re.compile(r"[0-9a-f]{32}\Z")
WORKER_MODULE = "neural_sat_speck.followups.conditioning_worker"
GUARDIAN_MODULE = "neural_sat_speck.followups.watchdog"


def now() -> str:
    return datetime.now(UTC).isoformat()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@contextmanager
def _exclusive_campaign_lock(campaign: Path):
    """Prove the coordinator no longer owns its campaign lock."""
    path = campaign / "coordinator.lock"
    _require(
        path.is_file() and not path.is_symlink(), "Missing coordinator lock evidence"
    )
    with path.open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Coordinator still owns the campaign lock") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _same_snapshot_history(campaign: Path, ledger: dict) -> None:
    """Reject the narrow persist() crash window between history and ledger writes."""
    history = campaign / "history"
    _require(history.is_dir(), "Missing archived ledger history")
    if any(path.name.endswith(".tmp") for path in history.iterdir()):
        raise ValueError("Incomplete history write requires manual reconciliation")
    revisions = sorted(
        path for path in history.iterdir() if re.fullmatch(r"[0-9]{4}\.json", path.name)
    )
    _require(bool(revisions), "Missing archived ledger revisions")
    expected_names = [f"{index:04d}.json" for index in range(len(revisions))]
    _require(
        [path.name for path in revisions] == expected_names,
        "Non-contiguous archived revisions require manual reconciliation",
    )
    snapshots = [read_checked_json(path) for path in revisions]
    _require(
        canonical_bytes(snapshots[-1]) == canonical_bytes(ledger),
        "Latest archived ledger differs; manual reconciliation is required",
    )


def _guardian_is_running(pid: int, token: str) -> bool:
    """Look for this invocation's guardian, avoiding PID-only reuse guesses."""
    try:
        result = subprocess.run(
            ["/bin/ps", "-p", str(pid), "-o", "pid=,stat=,command="],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError("Cannot establish guardian termination") from error
    if result.returncode not in {0, 1}:
        raise ValueError("Cannot establish guardian termination")
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) != 3:
            continue
        try:
            observed_pid = int(fields[0])
        except ValueError:
            continue
        state, command = fields[1], fields[2]
        if observed_pid == pid and not state.startswith(("Z", "X")):
            if token in command and GUARDIAN_MODULE in command:
                return True
    return False


def _checked_start(
    *,
    campaign: Path,
    cell: dict,
    start_record: dict,
    manifest: dict,
    paths: dict[str, Path],
) -> dict:
    """Validate the checked invocation/guardian identity before reconstruction."""
    token = start_record.get("invocation_id")
    _require(
        isinstance(token, str) and TOKEN_RE.fullmatch(token) is not None,
        "Malformed in-flight invocation token",
    )
    directory = Path(start_record.get("directory", ""))
    expected = campaign / "cells" / cell["cell_id"] / token
    _require(
        directory.is_absolute()
        and directory == expected
        and directory.resolve() == directory
        and directory.name == token
        and directory.is_dir(),
        "In-flight directory does not match its fixed cell/token",
    )
    payload = read_checked_json(directory / "payload.json")
    config = read_checked_json(directory / "guardian.json")
    receipt = read_checked_json(directory / "guardian_receipt.json")
    loss = read_checked_json(directory / "coordinator_loss.json")
    expected_formula = manifest["cases"][cell["case_id"]]["formula_sha256"]
    expected_worker = [
        str(paths["python"]),
        "-m",
        WORKER_MODULE,
        str(directory / "payload.json"),
    ]
    _require(
        payload.get("cell") == cell
        and payload.get("invocation_id") == token
        and payload.get("directory") == str(directory)
        and payload.get("repo_root") == manifest["source_root"]
        and payload.get("source_commit") == manifest["source_commit"]
        and payload.get("python_path") == str(paths["python"])
        and payload.get("public_path") == str(paths[cell["case_id"] + ":public"])
        and payload.get("truth_path") == str(paths[cell["case_id"] + ":truth"])
        and payload.get("public_sha256")
        == manifest["cases"][cell["case_id"]]["public"]["sha256"]
        and payload.get("truth_sha256")
        == manifest["cases"][cell["case_id"]]["truth"]["sha256"]
        and payload.get("formula_sha256") == expected_formula
        and config.get("invocation") == token
        and config.get("directory") == str(directory)
        and config.get("repo_root") == manifest["source_root"]
        and config.get("receipt") == str(directory / "guardian_receipt.json")
        and config.get("worker_arguments") == expected_worker,
        "Payload/guardian invocation identity differs from the fixed campaign",
    )
    coordinator_pid = config.get("coordinator_pid")
    _require(
        type(coordinator_pid) is int and coordinator_pid > 1,
        "Invalid recorded coordinator identity",
    )
    deadline = seconds(config.get("deadline_monotonic"))
    _require(
        payload.get("deadline_monotonic") == deadline
        and receipt.get("deadline_monotonic") == deadline
        and math.isclose(deadline - 300.0 + 300.0, deadline, abs_tol=1e-8),
        "In-flight cutoff differs from the fixed 300-second deadline",
    )
    started = deadline - 300.0
    loss_at = seconds(loss.get("observed_monotonic"))
    observed = seconds(receipt.get("observed_monotonic"))
    cleanup = seconds(receipt.get("cleanup_completed_monotonic"))
    _require(
        loss.get("coordinator_pid") == coordinator_pid
        and loss_at >= started
        and loss_at <= observed
        and cleanup >= observed
        and receipt.get("status") == "coordinator_stop"
        and receipt.get("invocation") == token
        and type(receipt.get("watchdog_pid")) is int
        and receipt["watchdog_pid"] > 1
        and receipt.get("process_group_absent") is True,
        "No matching terminal coordinator-loss/guardian cleanup proof",
    )
    _require(
        not _guardian_is_running(receipt["watchdog_pid"], token),
        "Guardian is still running",
    )
    return {
        "token": token,
        "directory": directory,
        "payload": payload,
        "config": config,
        "receipt": receipt,
        "loss": loss,
        "started": started,
        "deadline": deadline,
        "observed": observed,
        "cleanup": cleanup,
    }


def _read_optional_checked(path: Path, default):
    if not path.exists():
        _require(
            not path.with_name(path.name + ".sha256").exists(),
            f"Orphan checksum sidecar: {path.name}",
        )
        return default
    return read_checked_json(path)


def _interrupted_row(
    *,
    campaign: Path,
    cell: dict,
    start_record: dict,
    manifest: dict,
    paths: dict[str, Path],
) -> dict:
    identity = _checked_start(
        campaign=campaign,
        cell=cell,
        start_record=start_record,
        manifest=manifest,
        paths=paths,
    )
    directory = identity["directory"]
    receipt = identity["receipt"]
    token = identity["token"]
    process_path = directory / "process.json"
    process = _read_optional_checked(process_path, None)
    worker_not_started = receipt.get("worker_not_started") is True
    if worker_not_started:
        _require(
            process is None
            and "worker_pid" not in receipt
            and "process_group" not in receipt
            and not (directory / "python_worker_receipt.json").exists()
            and not (directory / "python_worker_receipt.json.sha256").exists()
            and not (directory / "worker_result.json").exists(),
            "No-child receipt contradicts worker artifacts",
        )
        dispatched = False
    else:
        _require(
            isinstance(process, dict), "Dispatched guardian lacks process identity"
        )
        pid = process.get("pid")
        _require(
            type(pid) is int
            and pid > 1
            and process.get("process_group") == pid
            and process.get("invocation") == token
            and process.get("supervisor_pid") == receipt["watchdog_pid"]
            and receipt.get("worker_pid") == pid
            and receipt.get("process_group") == pid
            and type(receipt.get("worker_returncode")) is int,
            "Worker and guardian terminal identities differ",
        )
        dispatched = True

    samples = _read_optional_checked(directory / "environment_samples.json", [])
    _require(isinstance(samples, list), "Environment observations are not an array")
    if dispatched:
        _require(bool(samples), "Dispatched interrupted cell lacks resource samples")
    for sample in samples:
        _require(
            isinstance(sample, dict)
            and seconds(sample.get("observed_monotonic"))
            <= identity["loss"].get("observed_monotonic"),
            "Resource sample occurs after coordinator-loss observation",
        )

    journal_path = directory / "journal.json"
    journal = _read_optional_checked(journal_path, None)
    if journal is not None:
        _require(isinstance(journal, dict), "Worker journal is malformed")
    phases = phase_accounting(directory, identity["observed"])
    raw = _read_optional_checked(directory / "worker_result.json", None)
    if raw is not None:
        _require(
            isinstance(raw, dict)
            and raw.get("invocation_id") == token
            and raw.get("cell_id") == cell["cell_id"]
            and raw.get("formula_sha256") == identity["payload"]["formula_sha256"],
            "Partial worker result belongs to another invocation/formula",
        )

    row = {
        "cell": cell,
        "directory": str(directory),
        "outcome": "infrastructure_interrupted",
        "reason": "coordinator_lost_before_supervisor_result_was_ledgered",
        "dispatched": dispatched,
        "supervisor_artifact": "supervisor_sealed_result.json",
        "started_monotonic": identity["started"],
        "deadline_monotonic": identity["deadline"],
        "observed_monotonic": identity["observed"],
        "parent_wall_seconds": identity["observed"] - identity["started"],
        "guardian_receipt": receipt,
        "coordinator_loss": identity["loss"],
        "cleanup_wall_seconds": identity["cleanup"] - identity["observed"],
        "phase_accounting": phases,
        "environment_samples": samples,
        "worker_result": raw,
        "sealed_utc": now(),
    }
    return row


def _validate_source_ledger(
    ledger_path: Path, manifest_path: Path, manifest: dict, paths: dict[str, Path]
) -> dict:
    campaign = ledger_path.parent
    ledger = read_checked_json(ledger_path)
    _require(
        ledger.get("schema") == "conditioning_control_results_v1"
        and ledger.get("status") == "RUNNING"
        and ledger.get("manifest_sha256") == sha256_file(manifest_path)
        and ledger.get("manifest_path") == str(manifest_path.absolute())
        and ledger.get("source_commit") == manifest["source_commit"]
        and ledger.get("mode") == manifest["mode"],
        "Source ledger is not the matching checked RUNNING campaign",
    )
    rows, starts = ledger.get("records"), ledger.get("starts")
    cells = manifest["cells"]
    names = {cell["cell_id"] for cell in cells}
    _require(
        isinstance(rows, dict)
        and isinstance(starts, dict)
        and set(rows) <= names
        and set(starts) <= names,
        "Ledger contains a duplicate or unexpected cell identity",
    )
    schedule = [cell["cell_id"] for cell in cells]
    _require(
        set(starts) == set(schedule[: len(starts)]),
        "Recorded starts do not follow the fixed serial schedule",
    )
    pending = []
    for cell in cells:
        name = cell["cell_id"]
        row = rows.get(name)
        start = starts.get(name)
        if row is None:
            if start is not None:
                pending.append((cell, start))
            continue
        _require(row.get("cell") == cell, "Stored row belongs to another fixed cell")
        if row.get("outcome") == "unavailable":
            _require(
                start is None and isinstance(row.get("reason"), str) and row["reason"],
                "Unavailable cell has a dispatch or lacks a reason",
            )
            continue
        _require(isinstance(start, dict), "Completed row lacks its recorded start")
        token = start.get("invocation_id")
        _require(
            isinstance(token, str)
            and TOKEN_RE.fullmatch(token) is not None
            and start.get("directory") == str(campaign / "cells" / name / token)
            and Path(start["directory"]).resolve() == Path(start["directory"])
            and row.get("directory") == start.get("directory"),
            "Completed start path/token differs from the fixed campaign",
        )
        artifact_name = row.get("supervisor_artifact", "supervisor_result.json")
        _require(
            artifact_name == "supervisor_result.json",
            "A RUNNING source ledger cannot contain a sealed supervisor artifact",
        )
        _require(
            start.get("directory") == row.get("directory")
            and start.get("invocation_id") == Path(row["directory"]).name
            and read_checked_json(Path(row["directory"]) / artifact_name) == row,
            "Completed supervisor row differs from its checked artifact",
        )
        validate_cell(row, manifest, paths)
    _require(
        len(pending) <= 1, "More than one in-flight cell violates serial execution"
    )
    for name, start in starts.items():
        _require(
            isinstance(start, dict)
            and isinstance(start.get("directory"), str)
            and isinstance(start.get("invocation_id"), str),
            "Malformed recorded start",
        )
    return ledger


def seal_running_ledger(
    ledger_path: Path, manifest_path: Path, output_path: Path
) -> dict:
    """Write a separate INTERRUPTED ledger after proven coordinator cleanup.

    This function never resumes execution, dispatches a worker, or overwrites the
    original RUNNING ledger. It refuses source/history gaps and any started cell
    without a matching terminal coordinator-stop receipt.
    """
    ledger_path = ledger_path.absolute()
    manifest_path = manifest_path.absolute()
    output_path = output_path.absolute()
    campaign = ledger_path.parent
    _require(
        output_path.parent == campaign
        and output_path != ledger_path
        and not output_path.exists(),
        "Sealed ledger must be a new file beside the original RUNNING ledger",
    )
    manifest, paths = load_manifest(manifest_path)
    verify_cases(manifest, paths)
    with _exclusive_campaign_lock(campaign):
        ledger = _validate_source_ledger(ledger_path, manifest_path, manifest, paths)
        _same_snapshot_history(campaign, ledger)
        sealed = copy.deepcopy(ledger)
        rows = sealed["records"]
        pending = [
            (cell, sealed["starts"][cell["cell_id"]])
            for cell in manifest["cells"]
            if cell["cell_id"] not in rows and cell["cell_id"] in sealed["starts"]
        ]
        if pending:
            cell, start = pending[0]
            directory = Path(start["directory"])
            completed_path = directory / "supervisor_result.json"
            if completed_path.exists():
                row = read_checked_json(completed_path)
                token = start.get("invocation_id")
                _require(
                    row.get("cell") == cell
                    and row.get("directory") == str(directory)
                    and row.get("supervisor_artifact", "supervisor_result.json")
                    == "supervisor_result.json"
                    and row.get("outcome") != "unavailable"
                    and row.get("guardian_receipt")
                    == read_checked_json(directory / "guardian_receipt.json"),
                    "Unledgered supervisor result identity differs",
                )
                validate_cell(row, manifest, paths)
                receipt = row["guardian_receipt"]
                _require(
                    not _guardian_is_running(receipt["watchdog_pid"], token),
                    "Guardian for the completed supervisor result is still running",
                )
            else:
                row = _interrupted_row(
                    campaign=campaign,
                    cell=cell,
                    start_record=start,
                    manifest=manifest,
                    paths=paths,
                )
                sealed_artifact = directory / "supervisor_sealed_result.json"
                _require(
                    not sealed_artifact.exists()
                    and not sealed_artifact.with_name(
                        sealed_artifact.name + ".sha256"
                    ).exists(),
                    "Sealed supervisor artifact already exists; manual review required",
                )
                atomic_json(sealed_artifact, row)
            rows[cell["cell_id"]] = row
        for cell in manifest["cells"]:
            name = cell["cell_id"]
            if name not in rows:
                _require(
                    name not in sealed["starts"],
                    "Started cell lacks terminal evidence; refusing to seal",
                )
                rows[name] = {
                    "cell": cell,
                    "outcome": "unavailable",
                    "reason": "coordinator_terminated_before_dispatch",
                }
        sealed["status"] = "INTERRUPTED"
        sealed["stop_reason"] = "coordinator_lost_campaign_not_resumed"
        sealed["sealed_from_sha256"] = sha256_file(ledger_path)
        sealed["sealed_utc"] = now()
        atomic_json(output_path, sealed)
        audited, _ = audit_ledger(output_path, manifest_path)
        _require(
            audited["status"] == "INTERRUPTED",
            "Sealed ledger did not pass the independent campaign audit",
        )
    return sealed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sealed = seal_running_ledger(args.ledger, args.manifest, args.output)
    print(
        f"Sealed interrupted ledger: {args.output.absolute()} ({len(sealed['records'])} cells)"
    )


if __name__ == "__main__":
    main()
