"""Independent deadline guardian; persists across coordinator termination."""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

from .guard import atomic_json


def invocation_groups(token: str) -> set[int]:
    """Require the unique invocation path in a live command, not a recycled PID."""
    if not isinstance(token, str) or re.fullmatch(r"[0-9a-f]{32}", token) is None:
        raise ValueError("Invalid invocation token")
    result = subprocess.run(
        ["/bin/ps", "-ax", "-o", "pgid=,command="],
        capture_output=True,
        text=True,
        timeout=1,
    )
    if result.returncode:
        raise RuntimeError("Cannot establish live invocation identity")
    groups = set()
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) != 2 or token not in fields[1]:
            continue
        argv = shlex.split(fields[1])
        python_worker = (
            len(argv) == 4
            and argv[1] == "-m"
            and argv[2]
            in {
                "neural_sat_speck.backend.worker",
                "neural_sat_speck.backend.native_launcher",
            }
            and Path(argv[3]).name in {"payload.json", "native_launch.json"}
        )
        native_solver = (
            len(argv) >= 2
            and Path(argv[0]).name in {"cryptominisat5", "cadical"}
            and Path(argv[-1]).name in {"input.dimacs", "input.cnf"}
        )
        if (python_worker or native_solver) and re.search(
            r"(?:^|[_/])" + token + r"(?:/|$)", argv[-1]
        ):
            group = int(fields[0])
            if group <= 1 or group == os.getpgrp():
                raise RuntimeError("Unsafe invocation process group")
            groups.add(group)
    return groups


def main() -> None:
    payload = json.loads(Path(sys.argv[1]).read_bytes())
    deadline = payload["deadline_monotonic"]
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise ValueError("Invalid deadline")
    if re.fullmatch(r"[0-9a-f]{32}", str(payload.get("invocation", ""))) is None:
        raise ValueError("Invalid invocation")
    if not hasattr(os, "waitid") or not hasattr(os, "WNOWAIT"):
        raise RuntimeError("Non-reaping child observation is required")
    directory = Path(payload["directory"])
    stopped = False

    def stop_requested(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGINT, stop_requested)
    process = None
    receipt = {
        "watchdog_pid": os.getpid(),
        "invocation": payload["invocation"],
        "deadline_monotonic": deadline,
        "status": "guardian_error",
    }
    try:
        if stopped:
            raise RuntimeError("Supervisor stopped before dispatch")
        with (
            (directory / "worker_stdout.txt").open("w") as stdout,
            (directory / "worker_stderr.txt").open("w") as stderr,
        ):
            process = subprocess.Popen(
                payload["worker_arguments"],
                cwd=payload["repo_root"],
                start_new_session=True,
                stdout=stdout,
                stderr=stderr,
            )
        atomic_json(
            directory / "process.json",
            {
                "pid": process.pid,
                "process_group": process.pid,
                "invocation": payload["invocation"],
                "supervisor_pid": os.getpid(),
            },
        )
        # Never reap before group cleanup: the child PID remains reserved even
        # after a normal exit, so a process lookup or recycled-PID guess is not
        # needed if the coordinator dies.
        while not stopped:
            observed = time.perf_counter()
            if observed >= deadline:
                receipt["status"] = "deadline_enforced"
                break
            exited = os.waitid(
                os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT
            )
            if exited is not None:
                receipt["status"] = "worker_exit"
                break
            time.sleep(min(0.02, max(0.0, deadline - observed)))
        if stopped:
            receipt["status"] = "coordinator_stop"
        receipt["observed_monotonic"] = time.perf_counter()
        if not stopped and receipt["observed_monotonic"] >= deadline:
            receipt["status"] = "deadline_enforced"
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
    finally:
        if process is not None:
            receipt["worker_pid"] = receipt["process_group"] = process.pid
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as error:
                # Darwin can refuse a signal to an already-exited child held
                # as a zombie by WNOWAIT. Retain the error; only a subsequent
                # absence proof can make this cleanup successful.
                receipt["group_signal_error"] = f"{type(error).__name__}: {error}"
            try:
                receipt["worker_returncode"] = process.wait(timeout=5)
            except subprocess.TimeoutExpired as error:
                receipt["status"] = "guardian_error"
                receipt["error"] = f"Child did not terminate: {error}"
            absent = False
            for _ in range(100):
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    absent = True
                    break
                except OSError:
                    break
                time.sleep(0.01)
            receipt["process_group_absent"] = absent
            if not absent:
                receipt["status"] = "guardian_error"
                receipt["error"] = "Worker process group absence was not proven"
        receipt["cleanup_completed_monotonic"] = time.perf_counter()
        atomic_json(Path(payload["receipt"]), receipt)
    if receipt["status"] == "guardian_error":
        raise RuntimeError(receipt.get("error", "Guardian failed"))


if __name__ == "__main__":
    main()
