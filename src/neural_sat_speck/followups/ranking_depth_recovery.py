"""D-057's one-cell replacement plus fixed D-056 schedule suffix.

This module never edits the original campaign. Preparation writes a separate,
hash-bound manifest; execution refuses an existing execution tree and never
resumes a partial segment. Historical rows receive mechanical identity checks
only. The combined scientific audit is intentionally left to post-completion.
"""

from __future__ import annotations

import copy
import fcntl
import math
import os
import re
import time
import uuid
from pathlib import Path

from neural_sat_speck.backend.environment import sleep_assertion
from neural_sat_speck.backend.guard import atomic_json, read_checked_json, sha256_file
from neural_sat_speck.followups.conditioning_control import (
    digest,
    source_commit,
    validate_source_root,
)
from neural_sat_speck.followups.environment import same_python_environment
from neural_sat_speck.followups.ranking_depth_control import (
    D056_DESIGN,
    file_identity,
    fixed_cells,
    load_manifest,
    policy,
    source_names,
    stat_identity,
    streaming_hash,
)

ORIGINAL_DIRECTORY = "backend_ranking_depth_20261008_full_d056"
ORIGINAL_MANIFEST_SHA256 = (
    "0e7074541ea612f4277a600ce759a221924c0d934b05c9e29cb09a944f6f46af"
)
ORIGINAL_LAUNCH_RECEIPT = "runtime/d056_full_launch_receipt_20261008.json"
ORIGINAL_LAUNCH_RECEIPT_SHA256 = (
    "06d7bbf14908d38151f781fd0c6508c13701d1ab67cf0f28419f6d1966d8e2f2"
)
ORIGINAL_SOURCE_COMMIT = "b5a0aee02abce43aa1f5f30165df40a79fdcd1af"
ORIGINAL_CAMPAIGN_CEILING_SECONDS = 90_000.0
ORIGINAL_TERMINAL_RECORDS = 141
ORIGINAL_STARTS = 142
CONTINUATION_START_ORDINAL = 142
REPLACEMENT_CELL_ID = "speck32_64_17_m1_cms_native_no_gauss"
CELLS_PER_SECOND_ALLOWANCE = 300.0
SHUTDOWN_RESERVE_SECONDS = 30.0
PREPARATION_WRITE_RESERVE_SECONDS = 2.0
INCIDENT_TREE_MAX_BYTES = 64 * 1024 * 1024
INCIDENT_TREE_MAX_FILES = 10_000

RECOVERY_MODULE = "src/neural_sat_speck/followups/ranking_depth_recovery.py"
CONTINUATION_SCRIPT = "scripts/resume_backend_ranking_depth_d056.py"
GUARDIAN_MODULE = "src/neural_sat_speck/backend/watchdog.py"
REPAIRED_GUARDIAN_SHA256 = (
    "239f88f61975cf3848ebf8b0459d4dce03d0a72a4677005e8f641806e25465ce"
)
MANIFEST_NAME = "continuation_manifest.json"

SCIENTIFIC_OUTCOMES = {
    "recovered",
    "queue_exhausted",
    "global_deadline",
    "solver_unknown",
    "verification_failure",
    "distinct_verified_key",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha_pin(path: Path, expected: str, label: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"Malformed {label} SHA-256")
    value = read_checked_json(path)
    if sha256_file(path) != expected:
        raise ValueError(f"{label} SHA-256 differs from its required pin")
    return value


def _identity_path(pin: dict, label: str) -> Path:
    _require(isinstance(pin, dict), f"Malformed {label} identity")
    raw_path = pin.get("path")
    _require(isinstance(raw_path, str) and raw_path, f"Missing {label} path")
    path = Path(raw_path).absolute()
    _require(path.is_file(), f"Missing {label}: {path}")
    if "identity" in pin:
        _require(
            stat_identity(path) == pin["identity"],
            f"{label} stat identity changed",
        )
    return path


def _check_unlocked(path: Path) -> None:
    _require(path.is_file(), f"Missing original coordinator lock: {path}")
    with path.open("r") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Original D-056 coordinator is still active") from error
        fcntl.flock(lock, fcntl.LOCK_UN)


def _tree_inventory(directory: Path) -> list[dict]:
    """Hash-bind the small failed invocation without parsing scientific files."""
    _require(directory.is_dir(), "Missing original interrupted invocation")
    inventory = []
    total_bytes = 0
    for path in sorted(directory.rglob("*")):
        _require(not path.is_symlink(), "Symlink in interrupted invocation")
        if not path.is_file():
            continue
        size = path.stat().st_size
        total_bytes += size
        _require(total_bytes <= INCIDENT_TREE_MAX_BYTES, "Incident tree exceeds bound")
        _require(
            len(inventory) < INCIDENT_TREE_MAX_FILES, "Incident tree has too many files"
        )
        if path.suffix == ".json":
            sidecar = path.with_name(path.name + ".sha256")
            _require(sidecar.is_file(), f"Missing incident JSON sidecar: {path.name}")
            expected = f"{streaming_hash(path)}  {path.name}\n"
            _require(
                sidecar.read_text() == expected,
                f"Incident sidecar mismatch: {path.name}",
            )
        inventory.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "size_bytes": size,
                "sha256": streaming_hash(path),
            }
        )
    _require(inventory, "Interrupted invocation has no preserved evidence")
    return inventory


def _validate_start(
    original_root: Path,
    cell: dict,
    start: dict,
) -> tuple[Path, str]:
    _require(isinstance(start, dict), "Malformed start record")
    token = start.get("invocation_id")
    _require(
        isinstance(token, str) and re.fullmatch(r"[0-9a-f]{32}", token) is not None,
        "Malformed invocation token",
    )
    directory = Path(start.get("directory", "")).absolute()
    expected = original_root / "execution" / "cells" / cell["cell_id"] / token
    _require(directory == expected.absolute(), "Start directory is redirected")
    _require(directory.is_dir(), "Started invocation directory is missing")
    return directory, token


def _check_terminal_prefix(
    original_root: Path,
    cells: list[dict],
    ledger: dict,
) -> None:
    prefix = cells[:ORIGINAL_TERMINAL_RECORDS]
    expected_ids = [cell["cell_id"] for cell in prefix]
    _require(
        ledger.get("execution_order")
        == [cell["cell_id"] for cell in cells[:ORIGINAL_STARTS]],
        "Original execution order is not the exact 142-cell schedule prefix",
    )
    starts, records = ledger.get("starts"), ledger.get("records")
    _require(
        isinstance(starts, dict) and isinstance(records, dict), "Malformed ledger maps"
    )
    _require(
        len(starts) == ORIGINAL_STARTS
        and set(starts) == set(ledger["execution_order"]),
        "Original start set differs from the exact 142-cell prefix",
    )
    _require(
        len(records) == ORIGINAL_TERMINAL_RECORDS and set(records) == set(expected_ids),
        "Original terminal record set differs from the exact 141-cell prefix",
    )
    tokens = set()
    directories = set()
    for cell in prefix:
        cell_id = cell["cell_id"]
        directory, token = _validate_start(original_root, cell, starts[cell_id])
        _require(
            token not in tokens and directory not in directories,
            "Duplicate original invocation",
        )
        tokens.add(token)
        directories.add(directory)
        row = records[cell_id]
        result_path = directory / "supervisor_result.json"
        supervisor_row = read_checked_json(result_path)
        _require(
            row == supervisor_row,
            f"Saved row does not match {cell_id} supervisor artifact",
        )
        _require(row.get("cell") == cell, f"Saved row cell identity changed: {cell_id}")
        guardian = read_checked_json(directory / "guardian.json")
        receipt = read_checked_json(directory / "guardian_receipt.json")
        _require(
            guardian.get("invocation") == token
            and guardian.get("directory") == str(directory)
            and guardian.get("receipt") == str(directory / "guardian_receipt.json"),
            f"Guardian configuration does not bind {cell_id}",
        )
        _require(
            row.get("guardian_receipt") == receipt
            and receipt.get("invocation") == token
            and receipt.get("process_group_absent") is True,
            f"Guardian cleanup proof does not bind {cell_id}",
        )


def _validate_failed_invocation(
    original_root: Path,
    cell: dict,
    start: dict,
    ledger: dict,
) -> dict:
    directory, token = _validate_start(original_root, cell, start)
    _require(cell["cell_id"] == REPLACEMENT_CELL_ID, "D-057 replacement cell changed")
    _require(
        cell["cell_id"] not in ledger["records"],
        "Ordinal 142 already has a terminal row",
    )
    _require(
        not (directory / "supervisor_result.json").exists(),
        "Ordinal 142 has an undeclared supervisor result",
    )
    guardian = read_checked_json(directory / "guardian.json")
    receipt = read_checked_json(directory / "guardian_receipt.json")
    payload = read_checked_json(directory / "payload.json")
    _require(
        guardian.get("invocation") == token
        and guardian.get("directory") == str(directory)
        and guardian.get("receipt") == str(directory / "guardian_receipt.json"),
        "Failed guardian configuration does not bind the recorded invocation",
    )
    _require(
        payload.get("cell") == cell
        and payload.get("invocation_id") == token
        and payload.get("directory") == str(directory)
        and payload.get("source_commit") == ORIGINAL_SOURCE_COMMIT,
        "Failed payload does not bind the original scheduled cell",
    )
    _require(
        receipt.get("invocation") == token
        and receipt.get("status") == "guardian_error"
        and receipt.get("worker_returncode") == -9
        and receipt.get("process_group_absent") is False,
        "Ordinal 142 is not the authorized guardian-error incident",
    )
    from neural_sat_speck.followups.ranking_depth_runner import live_invocation_groups

    _require(
        not live_invocation_groups(token),
        "Original ordinal-142 process group is still live",
    )
    inventory = _tree_inventory(directory)
    return {
        "cell_id": cell["cell_id"],
        "ordinal": CONTINUATION_START_ORDINAL,
        "prior_invocation_id": token,
        "prior_directory": str(directory),
        "prior_terminal_outcome": "unavailable",
        "guardian_status": "guardian_error",
        "worker_returncode": -9,
        "process_group_absence_proven": False,
        "file_inventory": inventory,
        "file_inventory_sha256": digest(inventory),
    }


def _validate_source_rebase(
    original_manifest: dict,
    source_root: Path,
) -> tuple[dict, dict, str]:
    old_files = original_manifest["files"]
    old_names = {
        name.removeprefix("source:") for name in old_files if name.startswith("source:")
    }
    new_names = set(source_names(source_root))
    _require(
        new_names - old_names == {RECOVERY_MODULE} and old_names <= new_names,
        "Source inventory changed beyond the D-057 recovery module",
    )
    validate_source_root(source_root, sorted(new_names | {CONTINUATION_SCRIPT}))
    new_commit = source_commit(source_root)
    _require(
        new_commit != ORIGINAL_SOURCE_COMMIT,
        "Recovery snapshot is not a new source commit",
    )
    old_root = Path(original_manifest["source_root"])
    _require(
        old_root.is_dir() and source_commit(old_root) == ORIGINAL_SOURCE_COMMIT,
        "Original D-056 source snapshot is unavailable or changed",
    )
    validate_source_root(old_root, sorted(old_names))

    new_files = copy.deepcopy(old_files)
    source_changes = []
    for name in sorted(old_names):
        key = "source:" + name
        target = source_root / name
        _require(
            target.is_file() and not target.is_symlink(), f"Missing source file: {name}"
        )
        before = stat_identity(target)
        checksum = streaming_hash(target)
        if name == GUARDIAN_MODULE:
            _require(
                checksum == REPAIRED_GUARDIAN_SHA256,
                "Guardian source differs from reviewed repair commit be703d0",
            )
            _require(
                checksum != old_files[key]["sha256"],
                "Reviewed guardian repair is absent from the recovery snapshot",
            )
            source_changes.append(
                {
                    "path": name,
                    "old_sha256": old_files[key]["sha256"],
                    "new_sha256": checksum,
                }
            )
        else:
            _require(
                checksum == old_files[key]["sha256"], f"Scientific/source drift: {name}"
            )
        new_files[key] = {
            "path": str(target.absolute()),
            "sha256": checksum,
            "identity": before,
        }
        _require(
            stat_identity(target) == before, f"Source changed while hashing: {name}"
        )

    recovery_path = source_root / RECOVERY_MODULE
    recovery_identity = file_identity(recovery_path)
    new_files["source:" + RECOVERY_MODULE] = recovery_identity
    script_identity = file_identity(source_root / CONTINUATION_SCRIPT)
    for name, pin in old_files.items():
        if name.startswith("source:"):
            continue
        path = _identity_path(pin, name)
        if name in original_manifest["profile"].get("files", {}):
            expected_hash = original_manifest["profile"]["files"][name]["sha256"]
            _require(
                pin.get("sha256") == expected_hash, f"Frozen asset pin changed: {name}"
            )
        if stat_identity(path) != pin.get("identity"):
            raise ValueError(f"Frozen asset/runtime identity changed: {name}")
    _require(
        policy(source_root, D056_DESIGN) == original_manifest["profile"],
        "D-056 profile differs in the recovery snapshot",
    )
    return (
        new_files,
        {
            "root": str(source_root),
            "commit": new_commit,
            "source_changes": source_changes,
            "recovery_module": recovery_identity,
            "continuation_script": script_identity,
        },
        new_commit,
    )


def _validated_schedule(cells: list[dict]) -> list[dict]:
    _require(
        cells == fixed_cells("full", num_keys=32) and len(cells) == 256,
        "Original 256-cell schedule differs from D-056",
    )
    suffix = cells[ORIGINAL_TERMINAL_RECORDS:]
    _require(
        suffix[0]["cell_id"] == REPLACEMENT_CELL_ID
        and len(suffix) == 115
        and len({cell["cell_id"] for cell in suffix}) == 115,
        "Authorized schedule suffix is not exactly ordinal 142 plus 114 unstarted cells",
    )
    return suffix


def _remaining_original_budget(launch: dict) -> tuple[float, float]:
    elapsed = launch.get("whole_launch_wall_seconds")
    _require(
        isinstance(elapsed, (int, float))
        and not isinstance(elapsed, bool)
        and math.isfinite(elapsed)
        and 0 <= elapsed < ORIGINAL_CAMPAIGN_CEILING_SECONDS,
        "Invalid original active elapsed time",
    )
    remaining = ORIGINAL_CAMPAIGN_CEILING_SECONDS - float(elapsed)
    online_allowance = 115 * CELLS_PER_SECOND_ALLOWANCE
    _require(
        remaining > online_allowance + SHUTDOWN_RESERVE_SECONDS,
        "Remaining original budget cannot cover the fixed suffix",
    )
    return float(elapsed), remaining


def _load_original(
    original_root: Path,
    expected_ledger_sha256: str,
) -> tuple[dict, dict, dict, str, Path, dict]:
    original_root = original_root.resolve()
    _require(
        original_root.name == ORIGINAL_DIRECTORY, "Unexpected D-056 original directory"
    )
    manifest_path = original_root / "inputs" / "manifest.json"
    ledger_path = original_root / "execution" / "ledger.json"
    original = _sha_pin(manifest_path, ORIGINAL_MANIFEST_SHA256, "original manifest")
    ledger = _sha_pin(ledger_path, expected_ledger_sha256, "original ledger")
    _require(
        original.get("schema") == "backend_ranking_depth_manifest_v1"
        and original.get("status") == "APPROVED"
        and original.get("author_decision") == "D-056"
        and original.get("execution_authorized") is True
        and original.get("mode") == "full"
        and original.get("source_commit") == ORIGINAL_SOURCE_COMMIT
        and original.get("campaign_ceiling_seconds")
        == ORIGINAL_CAMPAIGN_CEILING_SECONDS,
        "Original manifest is not the frozen D-056 full campaign",
    )
    suffix = _validated_schedule(original.get("cells", []))
    _require(
        ledger.get("schema") == "backend_ranking_depth_results_v1"
        and ledger.get("mode") == "full"
        and ledger.get("manifest_sha256") == ORIGINAL_MANIFEST_SHA256
        and Path(ledger.get("manifest_path", "")).absolute() == manifest_path.absolute()
        and ledger.get("source_commit") == ORIGINAL_SOURCE_COMMIT
        and ledger.get("status") == "INTERRUPTED",
        "Original ledger identity/state differs from the authorized incident",
    )
    _check_unlocked(original_root / "execution" / "coordinator.lock")
    _check_terminal_prefix(original_root, original["cells"], ledger)
    failed_cell = original["cells"][ORIGINAL_TERMINAL_RECORDS]
    _require(
        failed_cell["cell_id"] == REPLACEMENT_CELL_ID,
        "Ordinal-142 schedule identity changed",
    )
    incident = _validate_failed_invocation(
        original_root,
        failed_cell,
        ledger["starts"][failed_cell["cell_id"]],
        ledger,
    )
    project_root = original_root.parent.parent
    launch_path = project_root / ORIGINAL_LAUNCH_RECEIPT
    launch = _sha_pin(
        launch_path, ORIGINAL_LAUNCH_RECEIPT_SHA256, "original launch receipt"
    )
    _require(
        launch.get("schema") == "d056_persistent_launch_v1"
        and launch.get("status") == "STOPPED"
        and launch.get("whole_launch_ceiling_seconds")
        == ORIGINAL_CAMPAIGN_CEILING_SECONDS
        and Path(launch.get("output_root", "")).resolve() == original_root
        and launch.get("source_commit") == ORIGINAL_SOURCE_COMMIT,
        "Original launch receipt does not bind this campaign",
    )
    elapsed, remaining = _remaining_original_budget(launch)
    online_allowance = len(suffix) * CELLS_PER_SECOND_ALLOWANCE
    return (
        original,
        ledger,
        launch,
        manifest_path,
        project_root,
        {
            "original_root": str(original_root),
            "manifest_path": str(manifest_path),
            "manifest_sha256": ORIGINAL_MANIFEST_SHA256,
            "ledger_path": str(ledger_path),
            "ledger_sha256": expected_ledger_sha256,
            "launch_receipt_path": str(launch_path),
            "launch_receipt_sha256": ORIGINAL_LAUNCH_RECEIPT_SHA256,
            "launch_active_elapsed_seconds": float(elapsed),
            "remaining_original_ceiling_seconds": remaining,
            "online_allowance_seconds": online_allowance,
            "start_ordinal": CONTINUATION_START_ORDINAL,
            "completed_prefix_count": ORIGINAL_TERMINAL_RECORDS,
            "started_prefix_count": ORIGINAL_STARTS,
            "execution_order_prefix_sha256": digest(ledger["execution_order"]),
            "failed_invocation": incident,
        },
    )


def prepare_continuation(
    *,
    source_root: Path,
    original_root: Path,
    output_root: Path,
    expected_ledger_sha256: str,
) -> Path:
    """Write a reviewed continuation manifest only; this function never launches."""
    started = time.perf_counter()
    source_root, original_root, output_root = (
        source_root.resolve(),
        original_root.resolve(),
        output_root.resolve(),
    )
    _require(not output_root.exists(), "Continuation output already exists")
    _require(
        output_root != original_root and original_root not in output_root.parents,
        "Continuation output must be separate from the immutable original",
    )
    original, _ledger, _launch, _manifest_path, _project, binding = _load_original(
        original_root, expected_ledger_sha256
    )
    new_files, source_binding, new_commit = _validate_source_rebase(
        original, source_root
    )
    suffix = _validated_schedule(original["cells"])
    recovery = {
        "schema": "d056_authorized_continuation_v1",
        **binding,
        "output_root": str(output_root),
        "suffix_cells": copy.deepcopy(suffix),
        "suffix_sha256": digest(suffix),
        "continuation_source": source_binding,
        "prepared_utc": None,
        "preparation_elapsed_seconds": None,
        "preparation_write_reserve_seconds": PREPARATION_WRITE_RESERVE_SECONDS,
        "shutdown_reserve_seconds": SHUTDOWN_RESERVE_SECONDS,
    }
    manifest = copy.deepcopy(original)
    manifest["source_root"] = str(source_root)
    manifest["source_commit"] = new_commit
    manifest["files"] = new_files
    manifest["recovery"] = recovery
    manifest_path = output_root / "inputs" / MANIFEST_NAME
    output_root.mkdir(parents=True)
    recovery["prepared_utc"] = _now()
    # The first write gives the repository loader a normal D-056 manifest. Its
    # 256-cell grid/profile/cases remain byte-identical to the original.
    atomic_json(manifest_path, manifest)
    checked, paths = load_manifest(manifest_path, hashes=True)
    _require(
        checked["recovery"]["suffix_cells"] == suffix, "Loader changed suffix binding"
    )
    from neural_sat_speck.followups.ranking_depth_runner import check_dispatch_identity

    check_dispatch_identity(checked)
    _require(
        same_python_environment(os.sys.executable, paths["python"]),
        "Wrong execution Python",
    )
    recovery["preparation_elapsed_seconds"] = (
        time.perf_counter() - started + PREPARATION_WRITE_RESERVE_SECONDS
    )
    atomic_json(manifest_path, manifest)
    return manifest_path


def _now() -> str:
    from neural_sat_speck.followups.ranking_depth_runner import now

    return now()


def _validate_prepared_output(
    output_root: Path,
    expected_manifest_sha256: str,
    expected_ledger_sha256: str,
) -> tuple[dict, Path]:
    output_root = output_root.resolve()
    manifest_path = output_root / "inputs" / MANIFEST_NAME
    _require(
        output_root.is_dir() and manifest_path.is_file(),
        "Missing prepared continuation",
    )
    _require(
        not (output_root / "execution").exists(),
        "Continuation execution already exists; no resume or retry",
    )
    tree = list(output_root.rglob("*"))
    _require(
        not any(path.is_symlink() for path in tree), "Symlink in prepared continuation"
    )
    items = {p.relative_to(output_root).as_posix() for p in tree if p.is_file()}
    _require(
        items == {f"inputs/{MANIFEST_NAME}", f"inputs/{MANIFEST_NAME}.sha256"},
        "Prepared output contains unexpected files",
    )
    manifest = _sha_pin(
        manifest_path, expected_manifest_sha256, "continuation manifest"
    )
    recovery = manifest.get("recovery", {})
    _require(
        recovery.get("schema") == "d056_authorized_continuation_v1"
        and recovery.get("ledger_sha256") == expected_ledger_sha256
        and recovery.get("output_root") == str(output_root),
        "Prepared continuation identity differs",
    )
    _require(
        isinstance(recovery.get("preparation_elapsed_seconds"), (int, float))
        and recovery["preparation_elapsed_seconds"] >= 0,
        "Missing charged preparation duration",
    )
    return manifest, manifest_path


def execute_continuation(
    *,
    output_root: Path,
    expected_manifest_sha256: str,
    expected_ledger_sha256: str,
) -> dict:
    """Execute each explicitly bound suffix cell once in the existing serial runner."""
    execution_started = time.perf_counter()
    output_root = output_root.resolve()
    manifest, manifest_path = _validate_prepared_output(
        output_root, expected_manifest_sha256, expected_ledger_sha256
    )
    recovery = manifest["recovery"]
    original_root = Path(recovery["original_root"])
    # Re-run the saved-evidence and source checks immediately before dispatch.
    _original, _ledger, _launch, _old_manifest_path, _project, current_binding = (
        _load_original(original_root, expected_ledger_sha256)
    )
    _require(
        current_binding == {k: recovery[k] for k in current_binding},
        "Original manifest, ledger, launch receipt, or incident changed after preparation",
    )
    _require(
        manifest.get("source_root") == recovery["continuation_source"].get("root"),
        "Continuation source root changed",
    )
    # A fresh full loader pass confirms runtime aliases and every asset/case SHA.
    checked_manifest, paths = load_manifest(manifest_path, hashes=True)
    _require(checked_manifest == manifest, "Prepared manifest changed during loading")
    from neural_sat_speck.followups.ranking_depth_audit import validate_cell
    from neural_sat_speck.followups.ranking_depth_runner import (
        check_dispatch_identity,
        controlled_environment,
        now,
        run_cell,
    )

    check_dispatch_identity(manifest)
    script_pin = recovery["continuation_source"]["continuation_script"]
    _require(
        streaming_hash(Path(script_pin["path"])) == script_pin["sha256"]
        and stat_identity(Path(script_pin["path"])) == script_pin["identity"],
        "Continuation script identity changed",
    )
    _require(
        same_python_environment(os.sys.executable, paths["python"]),
        "Wrong execution Python",
    )
    suffix = recovery.get("suffix_cells")
    _require(
        suffix == _original["cells"][ORIGINAL_TERMINAL_RECORDS:]
        and len(suffix) == 115
        and suffix[0]["cell_id"] == REPLACEMENT_CELL_ID
        and recovery.get("suffix_sha256") == digest(suffix),
        "Continuation schedule drift",
    )

    preparation_elapsed = float(recovery["preparation_elapsed_seconds"])
    remaining_original = float(recovery["remaining_original_ceiling_seconds"])
    execution_ceiling = (
        remaining_original - preparation_elapsed - SHUTDOWN_RESERVE_SECONDS
    )
    online_allowance = len(suffix) * CELLS_PER_SECOND_ALLOWANCE
    _require(
        execution_ceiling > len(suffix) * 308.0,
        "Charged preparation leaves insufficient fixed online allowance",
    )
    execution_root = output_root / "execution"
    _require(not execution_root.exists(), "Execution root exists; no resume")
    execution_root.mkdir(parents=True)
    ledger_out = {
        "schema": "backend_ranking_depth_d056_continuation_results_v1",
        "status": "RUNNING",
        "manifest_path": str(manifest_path),
        "manifest_sha256": expected_manifest_sha256,
        "original_manifest_path": recovery["manifest_path"],
        "original_manifest_sha256": recovery["manifest_sha256"],
        "original_ledger_path": recovery["ledger_path"],
        "original_ledger_sha256": recovery["ledger_sha256"],
        "original_launch_receipt_path": recovery["launch_receipt_path"],
        "original_launch_receipt_sha256": recovery["launch_receipt_sha256"],
        "source_root": manifest["source_root"],
        "source_commit": manifest["source_commit"],
        "original_source_commit": ORIGINAL_SOURCE_COMMIT,
        "start_ordinal": CONTINUATION_START_ORDINAL,
        "original_prefix_records": ORIGINAL_TERMINAL_RECORDS,
        "original_prefix_starts": ORIGINAL_STARTS,
        "failed_cell_replacement_binding": recovery["failed_invocation"],
        "preparation_elapsed_seconds": preparation_elapsed,
        "execution_ceiling_seconds": execution_ceiling,
        "online_allowance_seconds": online_allowance,
        "started_utc": now(),
        "starts": {},
        "records": {},
        "execution_order": [],
        "cells": copy.deepcopy(suffix),
    }
    history = execution_root / "history"
    revision = 0

    def persist() -> None:
        nonlocal revision
        atomic_json(history / f"{revision:04d}.json", ledger_out)
        atomic_json(execution_root / "ledger.json", ledger_out)
        revision += 1

    root = Path(manifest["source_root"])
    tmpdir = root / "runtime" / "ranking_depth_tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    environment = {
        **controlled_environment(root, tmpdir),
        "TF_NUM_INTRAOP_THREADS": "8",
        "TF_NUM_INTEROP_THREADS": "1",
    }
    assets = {
        name: str(paths[name])
        for name in (
            "cms",
            "tf_python",
            "bao_vv9",
            "simon_ddt",
            "speck_stage1",
            "speck_classical",
        )
    }
    stop_reason = None
    begun = execution_started
    try:
        with (
            (execution_root / "coordinator.lock").open("x") as lock,
            sleep_assertion() as assertion,
        ):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            ledger_out["sleep_assertion"] = assertion
            ledger_out["execution_started_utc"] = now()
            persist()
            try:
                for offset, cell in enumerate(suffix):
                    if time.perf_counter() - begun > execution_ceiling - 308.0:
                        stop_reason = (
                            stop_reason or "remaining_original_ceiling_before_dispatch"
                        )
                    if stop_reason:
                        ledger_out["records"][cell["cell_id"]] = {
                            "cell": cell,
                            "outcome": "unavailable",
                            "reason": stop_reason,
                        }
                        persist()
                        continue
                    check_dispatch_identity(manifest)
                    invocation = uuid.uuid4().hex
                    directory = execution_root / "cells" / cell["cell_id"] / invocation
                    case_id = cell["case_id"]
                    payload = {
                        "cell": cell,
                        "directory": str(directory),
                        "invocation_id": invocation,
                        "repo_root": str(root),
                        "source_commit": manifest["source_commit"],
                        "python_path": str(paths["python"]),
                        "paths": assets,
                        "public_path": str(paths[case_id + ":public"]),
                        "public_sha256": manifest["cases"][case_id]["public"]["sha256"],
                        "verification_path": str(paths[case_id + ":verification"]),
                        "verification_sha256": manifest["cases"][case_id][
                            "verification"
                        ]["sha256"],
                        "base_formula_sha256": manifest["cases"][case_id][
                            "base_formula_sha256"
                        ],
                    }
                    ledger_out["starts"][cell["cell_id"]] = {
                        "directory": str(directory),
                        "invocation_id": invocation,
                        "utc": now(),
                        "ordinal": CONTINUATION_START_ORDINAL + offset,
                    }
                    ledger_out["execution_order"].append(cell["cell_id"])
                    persist()
                    row = run_cell(payload, environment)
                    check_dispatch_identity(manifest)
                    ledger_out["records"][cell["cell_id"]] = row
                    try:
                        _require(
                            row.get("cell") == cell
                            and row.get("directory") == str(directory)
                            and isinstance(row.get("outcome"), str),
                            "New cell lacks a terminal supervised result",
                        )
                        saved = read_checked_json(directory / "supervisor_result.json")
                        receipt = read_checked_json(directory / "guardian_receipt.json")
                        _require(
                            row == saved
                            and row.get("guardian_receipt") == receipt
                            and receipt.get("process_group_absent") is True,
                            "New row lacks matching supervisor/guardian cleanup proof",
                        )
                        validate_cell(row, manifest, paths)
                    except Exception as error:
                        stop_reason = f"audit_rejected: {type(error).__name__}: {error}"
                    if row["outcome"] not in SCIENTIFIC_OUTCOMES:
                        stop_reason = stop_reason or row["outcome"]
                    persist()
                    print(
                        f"{offset + 1}/{len(suffix)} {cell['cell_id']} "
                        f"{row['outcome']} {row['parent_wall_seconds']:.2f}s",
                        flush=True,
                    )
            except BaseException as error:
                ledger_out.update(
                    status="INTERRUPTED",
                    stop_reason=f"{type(error).__name__}: {error}",
                    completed_utc=now(),
                )
                persist()
                raise
            ledger_out.update(
                status="STOPPED" if stop_reason else "COMPLETE",
                stop_reason=stop_reason,
                completed_utc=now(),
            )
            persist()
    except BaseException:
        raise
    return ledger_out
