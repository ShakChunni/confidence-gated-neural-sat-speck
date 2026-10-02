#!/usr/bin/env python3
"""Operational continuation of the frozen runner; no attack or analysis choices."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

FATAL_OUTCOMES = {"exception", "verification_failure", "sat_unverified"}
OS_INTERRUPTION_CODES = {
    -signal.SIGKILL,
    -signal.SIGTERM,
    -signal.SIGHUP,
    137,
    143,
}


def guarantee_ordinary_scheduling(pid: int = 0) -> None:
    """Guarantee that the process and its threads are in ordinary (standard, NOT Background) scheduling.

    Raises ValueError if Darwin background scheduling cannot be cleared or verified.
    """
    if sys.platform != "darwin":
        return
    import ctypes

    libc = ctypes.CDLL(None)
    PRIO_DARWIN_THREAD = 3
    PRIO_DARWIN_PROCESS = 4
    IOPOL_TYPE_DISK = 0
    IOPOL_SCOPE_PROCESS = 0
    IOPOL_DEFAULT = 0

    # Reset disk I/O policy to default
    try:
        libc.setiopolicy_np(IOPOL_TYPE_DISK, IOPOL_SCOPE_PROCESS, IOPOL_DEFAULT)
    except Exception:
        pass

    # Clear thread and process background scheduling
    if pid == 0:
        libc.setpriority(PRIO_DARWIN_THREAD, 0, 0)
    libc.setpriority(PRIO_DARWIN_PROCESS, pid, 0)

    # Verify background status is not active (1 means PRIO_DARWIN_BG is set)
    bg_state = libc.getpriority(PRIO_DARWIN_PROCESS, pid)
    if bg_state == 1 or (pid == 0 and bg_state != 0):
        target_pid = os.getpid() if pid == 0 else pid
        raise ValueError(
            f"Process {target_pid} scheduling policy is Background (PRIO_DARWIN_BG={bg_state}); "
            "ordinary task scheduling cannot be guaranteed."
        )


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def checked_ledger(path):
    path = Path(path)
    expected = path.with_name(path.name + ".sha256").read_text().split()[0]
    if sha(path) != expected:
        raise ValueError(
            "Ledger/sidecar mismatch: do not repair or regenerate checksum"
        )
    return json.loads(path.read_text())


def check_reviewed_records(ledger, baseline):
    """Require the reviewed prefix unchanged and reject every new fatal cell."""

    def cell(row):
        return row["instance_id"], row["method"]

    reviewed = {cell(row): row for row in baseline["records"]}
    current = {cell(row): row for row in ledger["records"]}
    if len(current) != len(ledger["records"]):
        raise ValueError("Duplicate completed cell")
    if any(current.get(key) != row for key, row in reviewed.items()):
        raise ValueError("Previously reviewed record missing or changed")
    if any(
        row["outcome"] in FATAL_OUTCOMES and reviewed.get(cell(row)) != row
        for row in ledger["records"]
    ):
        raise ValueError(
            "New infrastructure or verification failure requires reviewer clearance"
        )


def refresh_storage(config):
    """Restore missing invocation aliases only; never replace unexpected files."""
    repo_root = Path(config.get("cwd", Path.cwd()))
    permanent_runtime = repo_root / "runtime"
    permanent_tf = permanent_runtime / "simon_tf_venv"
    if not permanent_tf.exists():
        backup_tf = repo_root / "results" / "simon_confirmation" / "runtime_backup" / "tensorflow"
        if backup_tf.exists():
            permanent_runtime.mkdir(parents=True, exist_ok=True)
            permanent_tf.symlink_to(backup_tf, target_is_directory=True)

    for alias_name, target_name in config.get("storage_aliases", {}).items():
        alias, target = Path(alias_name), Path(target_name)
        if not target.exists():
            raise ValueError(f"Missing persistent runtime storage: {target}")
        if alias.is_symlink():
            if alias.resolve() != target.resolve():
                raise ValueError(f"Runtime alias points elsewhere: {alias}")
        elif alias.exists():
            raise ValueError(f"Unexpected object at runtime alias: {alias}")
        else:
            alias.parent.mkdir(parents=True, exist_ok=True)
            alias.symlink_to(target, target_is_directory=target.is_dir())
        os.utime(alias, None, follow_symlinks=False)
    for destination, original in config.get("invocation_files", {}).items():
        path, source = Path(destination), Path(original)
        if path.exists() and path.read_bytes() != source.read_bytes():
            raise ValueError(f"Invocation file changed: {path}")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(source.read_bytes())
        os.utime(path, None)


def processes():
    output = subprocess.check_output(
        ["/bin/ps", "-axo", "pid,ppid,pgid,command"], text=True
    )
    return [
        line.strip().split(None, 3) for line in output.splitlines()[1:] if line.strip()
    ]


def matching_runners(argv):
    # macOS ps reports Python.app's resolved executable instead of virtualenv
    # argv[0]. The exact script path and entire argument suffix remain stable.
    expected = " ".join(argv[1:])
    return [
        int(pid)
        for pid, _parent, _group, command in processes()
        if command.partition(" ")[2] == expected
    ]


def clean_orphans(config, emit):
    """After confirming no runner is alive, stop its detached worker groups."""
    expected = "-m neural_sat_speck.simon.method_worker --worker"
    for pid, parent, group, command in processes():
        if command.partition(" ")[2] == expected:
            if parent != "1" or pid != group:
                raise ValueError(
                    "A worker belongs to another live parent; refuse a second runner"
                )
            cwd = subprocess.check_output(
                ["/usr/sbin/lsof", "-a", "-p", pid, "-d", "cwd", "-Fn"], text=True
            )
            if f"n{config['cwd']}" not in cwd.splitlines():
                raise ValueError(
                    "An orphan worker belongs to another directory; refuse cleanup"
                )
            emit("orphan_worker_cleanup", pid=int(pid), process_group=int(group))
            os.killpg(int(group), signal.SIGKILL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = Path(config["supervisor_directory"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / "singleton.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        return supervise(config, root)


def supervise(config, root):
    guarantee_ordinary_scheduling(0)
    state_path = root / "state.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {"status": "READY", "os_restarts": 0, "invocations": 0}
    )
    if state["status"] in {"HOLD", "COMPLETE"}:
        return 0
    if state["status"] == "ANALYZING":
        state.update(
            status="HOLD",
            updated_at=datetime.now(UTC).isoformat(),
            reason="Previous final analysis may still be alive; review its process/output before retrying",
        )
        atomic(state_path, state)
        return 0

    def emit(event, **fields):
        row = dict(timestamp=datetime.now(UTC).isoformat(), event=event, **fields)
        with (root / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def save(status, **fields):
        state.update(status=status, updated_at=datetime.now(UTC).isoformat(), **fields)
        atomic(state_path, state)

    idle_guard = None
    try:
        guarantee_ordinary_scheduling(0)
        env = os.environ.copy()
        env.update(config.get("thread_environment", {}))
        env["PYTHONUNBUFFERED"] = "1"

        repo_root = Path(config.get("cwd", Path.cwd()))
        persistent_tmpdir = config.get("persistent_tmpdir")
        default_worker_tmp = repo_root / "results" / "simon_confirmation_replacement" / "worker_tmp"

        if persistent_tmpdir:
            p_tmp = Path(persistent_tmpdir)
            if str(p_tmp) in ("/tmp", "/private/tmp") or str(p_tmp).startswith(("/tmp/", "/private/tmp/")):
                raise ValueError(
                    f"persistent_tmpdir must be in a persistent project directory, not ephemeral /tmp: {p_tmp}"
                )
            resolved_tmpdir = str(p_tmp)
        else:
            resolved_tmpdir = str(default_worker_tmp)

        env["TMPDIR"] = resolved_tmpdir
        Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)

        argv = list(config["argv"])
        if "--resume" not in argv:
            argv.append("--resume")
        if any("evaluate_simon_confirmation.py" in arg for arg in argv):
            if "--runtime-dir" not in argv and (repo_root / "runtime").is_dir():
                argv.extend(["--runtime-dir", str(repo_root / "runtime")])

        idle_guard = subprocess.Popen(
            ["/usr/bin/caffeinate", "-d", "-i", "-m", "-s", "-u", "-w", str(os.getpid())]
        )
        baseline = checked_ledger(config["reviewed_baseline_ledger"])
        if (
            sha(config["reviewed_baseline_ledger"])
            != config["reviewed_baseline_sha256"]
        ):
            raise ValueError("Reviewer baseline has changed")
        while True:
            refresh_storage(config)
            live = matching_runners(argv)
            if len(live) > 1 or (live and state.get("runner_pid") != live[0]):
                raise ValueError("Unmanaged/duplicate runner; refuse competing launch")
            if live:
                # A previous supervisor can die while its detached runner survives.
                # Adopt monitoring; never launch a second runner.
                if state.get("supervisor_pid") != os.getpid():
                    emit(
                        "adopt_existing_runner",
                        runner_pid=live[0],
                        prior_supervisor_pid=state.get("supervisor_pid"),
                    )
                save("RUNNING", runner_pid=live[0], supervisor_pid=os.getpid())
                time.sleep(30)
                continue
            ledger = checked_ledger(config["ledger"])
            check_reviewed_records(ledger, baseline)
            if len(ledger["records"]) == config["total_cells"]:
                # Frozen analyzer independently validates the full grid/identity.
                save("ANALYZING", completed_cells=config["total_cells"])
                with (root / "analysis.log").open("a") as log:
                    result = subprocess.run(
                        config["analysis_argv"],
                        cwd=config["cwd"],
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                if result.returncode:
                    raise ValueError(
                        f"Final analysis rejected evidence: exit {result.returncode}"
                    )
                emit("completed_and_analyzed", cells=config["total_cells"])
                save("COMPLETE", runner_pid=None, completed_cells=config["total_cells"])
                return 0
            if state["status"] == "RUNNING":
                state["os_restarts"] += 1
                emit(
                    "os_interruption",
                    prior_runner_pid=state.get("runner_pid"),
                    completed_cells=len(ledger["records"]),
                    prior_log=state.get("log_path"),
                    accounting="Uncommitted in-flight work may be repeated; retain invocation logs and audit separately from final-cell online times",
                )
            if state["os_restarts"] > config.get("max_os_restarts", 3):
                raise ValueError("Bounded OS restart allowance exhausted")
            clean_orphans(config, emit)
            state["invocations"] += 1
            attempt = root / f"invocation_{state['invocations']:03d}"
            attempt.mkdir(exist_ok=False)
            # Preserve the prior ledger at each restart; no completed cell retries.
            atomic(attempt / "ledger_before.json", ledger)
            with (attempt / "runner.log").open("xb") as log:
                runner = subprocess.Popen(
                    argv,
                    cwd=config["cwd"],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                guarantee_ordinary_scheduling(runner.pid)
                save(
                    "RUNNING",
                    runner_pid=runner.pid,
                    supervisor_pid=os.getpid(),
                    log_path=str(attempt / "runner.log"),
                    completed_cells=len(ledger["records"]),
                )
                receipt = dict(
                    argv=argv,
                    started_at=datetime.now(UTC).isoformat(),
                    runner_pid=runner.pid,
                    supervisor_pid=os.getpid(),
                    supervisor_sha256=sha(__file__),
                    config_sha256=sha(config["config_path"]),
                    completed_cells_before=len(ledger["records"]),
                    thread_environment=config.get("thread_environment", {}),
                    persistent_tmpdir=env["TMPDIR"],
                    task_scheduling="standard_ordinary",
                    runtime_directory=str(repo_root / "runtime"),
                )
                atomic(attempt / "receipt.json", receipt)
                emit("launch", **receipt)
                while runner.poll() is None:
                    refresh_storage(config)
                    try:
                        runner.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        pass
                atomic(
                    attempt / "exit_status.json",
                    dict(
                        exit_code=runner.returncode,
                        finished_at=datetime.now(UTC).isoformat(),
                    ),
                )
            emit("runner_exit", exit_code=runner.returncode)
            if runner.returncode == 0:
                save("READY", runner_pid=None)
            elif runner.returncode not in OS_INTERRUPTION_CODES:
                raise ValueError(
                    f"Runner suspended/rejected execution (exit {runner.returncode}); reviewer clearance required"
                )
            # OS interruptions retain RUNNING so the next loop records the lost
            # invocation and counts its bounded restart. No source/ledger edits.
    except Exception as exc:
        emit("hold", error=f"{type(exc).__name__}: {exc}")
        save("HOLD", reason=f"{type(exc).__name__}: {exc}")
        return 0  # launchd must not loop on scientific/integrity rejection.
    finally:
        if idle_guard is not None:
            idle_guard.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
