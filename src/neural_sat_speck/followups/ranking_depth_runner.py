"""Serial D-055 execution, with an independent process-group deadline guardian."""

from __future__ import annotations

import fcntl
import os
import shlex
import signal
import subprocess
import time
import uuid
from pathlib import Path

from neural_sat_speck.backend.environment import native_policy_sample, sleep_assertion
from neural_sat_speck.backend.guard import atomic_json, read_checked_json, sha256_file
from neural_sat_speck.followups.conditioning_runner import (
    controlled_environment,
    now,
    phase_accounting,
)
from neural_sat_speck.followups.environment import power_sample
from neural_sat_speck.followups.ranking_depth_control import (
    SCIENTIFIC_OUTCOMES,
    digest,
    load_manifest,
    optional_terminal_json,
    source_commit,
    source_names,
    stat_identity,
    validate_source_root,
    verify_cases,
)
from neural_sat_speck.followups.runner import memory_sample
from neural_sat_speck.followups.stage2_receipts import (
    _validate_policy,
    is_process_exit_policy,
)


def check_dispatch_identity(manifest: dict) -> None:
    root = Path(manifest["source_root"])
    validate_source_root(root, source_names(root))
    if source_commit(root) != manifest["source_commit"]:
        raise ValueError("Source commit changed during campaign")
    pins = list(manifest["files"].values()) + [
        case[role]
        for case in manifest["cases"].values()
        for role in ("public", "verification", "truth")
    ]
    if any(stat_identity(Path(pin["path"])) != pin["identity"] for pin in pins):
        raise ValueError(
            "Asset/runtime/source/input stat identity changed during campaign"
        )


def live_invocation_groups(token: str) -> set[int]:
    """Locate actual unique invocation commands; never signal a saved PID."""
    if len(token) != 32 or any(c not in "0123456789abcdef" for c in token):
        raise ValueError("Invalid invocation token")
    observed = subprocess.run(
        ["/bin/ps", "-ax", "-o", "pgid=,command="],
        capture_output=True,
        text=True,
        check=True,
        timeout=2,
    )
    groups = set()
    for line in observed.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or token not in parts[1]:
            continue
        argv = shlex.split(parts[1])
        matches = len(argv) >= 4 and argv[1:3] in (
            ["-m", "neural_sat_speck.followups.ranking_depth_worker"],
            ["-m", "neural_sat_speck.followups.ranking_depth_ranking"],
            ["-m", "neural_sat_speck.backend.native_launcher"],
        )
        matches |= len(argv) >= 2 and Path(argv[0]).name == "cryptominisat5"
        if matches and any(token in Path(a).parts for a in argv if "/" in a):
            group = int(parts[0])
            if group <= 1 or group == os.getpgrp():
                raise RuntimeError("Unsafe live invocation process group")
            groups.add(group)
    return groups


def run_cell(payload: dict, environment: dict) -> dict:
    directory = Path(payload["directory"])
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    deadline = started + 300.0
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
                "neural_sat_speck.followups.ranking_depth_worker",
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
        next_sample, sampled_worker = started, False
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
                        if is_process_exit_policy(
                            sample["worker_policy"], identity["pid"]
                        ):
                            sample["policy_exit_pending"] = True
                        else:
                            _validate_policy(
                                sample["worker_policy"],
                                identity["pid"],
                                terminal_exit_proven=False,
                            )
                    except ValueError:
                        rejected = True
                    rejected |= sample["memory"].get("accepted") is not True
                rejected |= sample["power"].get("accepted") is not True
                samples.append(sample)
                atomic_json(directory / "environment_samples.json", samples)
            if rejected or current > deadline + 5:
                guardian.terminate()
                break
            time.sleep(0.05)
        guardian.wait(timeout=8)
    except BaseException:
        if guardian.poll() is None:
            guardian.terminate()
        guardian.wait(timeout=8)
        raise
    if guardian.returncode != 0 or not (directory / "guardian_receipt.json").exists():
        for group in live_invocation_groups(payload["invocation_id"]):
            os.killpg(group, signal.SIGKILL)
        raise RuntimeError(
            "Independent guardian failed; live invocation groups terminated"
        )
    receipt = read_checked_json(directory / "guardian_receipt.json")
    if receipt.get("process_group_absent") is not True:
        raise RuntimeError("Guardian did not prove descendant cleanup")
    observed = receipt["observed_monotonic"]
    deadline_proven = receipt["status"] == "deadline_enforced"
    raw, raw_uncommitted = optional_terminal_json(
        directory / "worker_result.json", deadline_proven
    )
    journal, journal_uncommitted = optional_terminal_json(
        directory / "journal.json", deadline_proven
    )
    phases = phase_accounting(directory, observed) if journal is not None else None
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
        "started_monotonic": started,
        "deadline_monotonic": deadline,
        "observed_monotonic": observed,
        "parent_wall_seconds": observed - started,
        "guardian_receipt": receipt,
        "guardian_returncode": guardian.returncode,
        "cleanup_wall_seconds": receipt["cleanup_completed_monotonic"] - observed,
        "phase_accounting": phases,
        "component_accounting_available": journal is not None,
        "uncommitted_terminal_json": [
            name
            for name, broken in (
                ("worker_result.json", raw_uncommitted),
                ("journal.json", journal_uncommitted),
            )
            if broken
        ],
        "environment_samples": samples,
        "worker_result": raw,
        "completed_utc": now(),
    }
    atomic_json(directory / "supervisor_result.json", row)
    return row


def require_pilot(manifest: dict) -> None:
    from neural_sat_speck.followups.ranking_depth_audit import audit_ledger

    pin = manifest.get("pilot_audit")
    if not isinstance(pin, dict) or sha256_file(Path(pin["path"])) != pin["sha256"]:
        raise ValueError("Full launch lacks a pinned independently audited pilot")
    audit = read_checked_json(Path(pin["path"]))
    if audit.get("passed") is not True or audit.get("mode") != "pilot":
        raise ValueError("Pilot did not pass mechanical gates")
    replay = audit_ledger(
        Path(audit["ledger_path"]), Path(audit["manifest_path"]), hashes=False
    )
    prior_manifest, _ = load_manifest(Path(audit["manifest_path"]), hashes=False)
    if (
        prior_manifest["files"] != manifest["files"]
        or prior_manifest["runtime"] != manifest["runtime"]
    ):
        raise ValueError("Pilot/full asset or runtime identities differ")
    if replay != audit or audit["source_commit"] != manifest["source_commit"]:
        raise ValueError("Pilot audit does not reproduce under the same source")
    if audit["profile_sha256"] != digest(manifest["profile"]):
        raise ValueError("Pilot scientific profile differs")


def execute(manifest_path: Path, output: Path) -> dict:
    from neural_sat_speck.followups.ranking_depth_audit import (
        audit_ledger,
        validate_cell,
    )

    manifest, paths = load_manifest(manifest_path)
    verify_cases(manifest, paths)
    if manifest["mode"] == "full":
        require_pilot(manifest)
    if output.exists():
        raise ValueError("Campaign output exists; no retries/resume/splicing")
    output.mkdir(parents=True)
    ledger = {
        "schema": "backend_ranking_depth_results_v1",
        "mode": manifest["mode"],
        "manifest_path": str(manifest_path.absolute()),
        "manifest_sha256": sha256_file(manifest_path),
        "source_commit": manifest["source_commit"],
        "started_utc": now(),
        "status": "RUNNING",
        "starts": {},
        "records": {},
        "execution_order": [],
    }
    revision = 0

    def persist():
        nonlocal revision
        atomic_json(output / "history" / f"{revision:04d}.json", ledger)
        atomic_json(output / "ledger.json", ledger)
        revision += 1

    root = Path(manifest["source_root"])
    tmpdir = root / "runtime" / "ranking_depth_tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    env = {
        **controlled_environment(root, tmpdir),
        "TF_NUM_INTRAOP_THREADS": "8",
        "TF_NUM_INTEROP_THREADS": "1",
    }
    begun, stop_reason = time.perf_counter(), None
    assets = {
        k: str(paths[k])
        for k in (
            "cms",
            "tf_python",
            "bao_vv9",
            "simon_ddt",
            "speck_stage1",
            "speck_classical",
        )
    }
    with (
        (output / "coordinator.lock").open("w") as lock,
        sleep_assertion() as assertion,
    ):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger["sleep_assertion"] = assertion
        persist()
        try:
            for cell in manifest["cells"]:
                if (
                    time.perf_counter() - begun
                    > manifest["campaign_ceiling_seconds"] - 308
                ):
                    stop_reason = stop_reason or "campaign_ceiling_before_dispatch"
                if stop_reason:
                    ledger["records"][cell["cell_id"]] = {
                        "cell": cell,
                        "outcome": "unavailable",
                        "reason": stop_reason,
                    }
                    persist()
                    continue
                ident, token = cell["case_id"], uuid.uuid4().hex
                check_dispatch_identity(manifest)
                directory = output / "cells" / cell["cell_id"] / token
                payload = {
                    "cell": cell,
                    "directory": str(directory),
                    "invocation_id": token,
                    "repo_root": str(root),
                    "source_commit": manifest["source_commit"],
                    "python_path": str(paths["python"]),
                    "paths": assets,
                    "public_path": str(paths[ident + ":public"]),
                    "public_sha256": manifest["cases"][ident]["public"]["sha256"],
                    "verification_path": str(paths[ident + ":verification"]),
                    "verification_sha256": manifest["cases"][ident]["verification"][
                        "sha256"
                    ],
                    "base_formula_sha256": manifest["cases"][ident][
                        "base_formula_sha256"
                    ],
                }
                ledger["starts"][cell["cell_id"]] = {
                    "directory": str(directory),
                    "invocation_id": token,
                    "utc": now(),
                }
                ledger["execution_order"].append(cell["cell_id"])
                persist()
                row = run_cell(payload, env)
                check_dispatch_identity(manifest)
                ledger["records"][cell["cell_id"]] = row
                try:
                    validate_cell(row, manifest, paths)
                except Exception as error:
                    stop_reason = f"audit_rejected: {type(error).__name__}: {error}"
                if row["outcome"] not in SCIENTIFIC_OUTCOMES:
                    stop_reason = stop_reason or row["outcome"]
                persist()
                print(
                    f"{len(ledger['records'])}/{len(manifest['cells'])} {cell['cell_id']} {row['outcome']} {row['parent_wall_seconds']:.2f}s",
                    flush=True,
                )
        except BaseException as error:
            ledger.update(
                status="INTERRUPTED",
                stop_reason=f"{type(error).__name__}: {error}",
                completed_utc=now(),
            )
            persist()
            raise
        ledger.update(
            status="STOPPED" if stop_reason else "COMPLETE",
            stop_reason=stop_reason,
            completed_utc=now(),
        )
        persist()
    if ledger["status"] == "COMPLETE":
        atomic_json(
            output / "audit.json",
            audit_ledger(output / "ledger.json", manifest_path, hashes=True),
        )
    return ledger
