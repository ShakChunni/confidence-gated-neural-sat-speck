"""Narrow process/power observations; no all-thread or core-affinity claim."""

from __future__ import annotations

import contextlib
import ctypes
import os
import re
import subprocess
import sys
from datetime import UTC, datetime


def power_sample() -> dict:
    sample = {"utc": datetime.now(UTC).isoformat(), "platform": sys.platform}
    if sys.platform != "darwin":
        return {**sample, "supported": False, "accepted": None}
    try:
        assertions = subprocess.run(
            ["/usr/bin/pmset", "-g", "assertions"],
            capture_output=True,
            text=True,
            timeout=1,
        )
        battery = subprocess.run(
            ["/usr/bin/pmset", "-g", "batt"], capture_output=True, text=True, timeout=1
        )
        sleep = bool(
            re.search(
                r"\b(?:PreventUserIdleSystemSleep|PreventSystemSleep)\s+1\b",
                assertions.stdout,
            )
        )
        ac = "AC Power" in battery.stdout
        return {
            **sample,
            "supported": True,
            "raw_assertions": assertions.stdout,
            "raw_power": battery.stdout,
            "assertions_returncode": assertions.returncode,
            "power_returncode": battery.returncode,
            "ac_power_observed": ac,
            "sleep_assertion_observed": sleep,
            "accepted": assertions.returncode == battery.returncode == 0
            and ac
            and sleep,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {**sample, "supported": True, "accepted": False, "error": str(error)}


def native_policy_sample(pid: int) -> dict:
    sample = {
        "pid": pid,
        "scope": "process_priority_and_background_policy",
        "native_thread_qos": "unavailable",
    }
    if sys.platform != "darwin":
        return {**sample, "supported": False, "accepted": None}
    libc = ctypes.CDLL(None, use_errno=True)
    libc.getpriority.argtypes = [ctypes.c_int, ctypes.c_int]
    libc.getpriority.restype = ctypes.c_int
    values = {}
    for name, kind in (("nice", 0), ("background_policy", 4)):
        ctypes.set_errno(0)
        value = libc.getpriority(kind, pid)
        err = ctypes.get_errno()
        values[name] = {"value": value, "errno": err}
    return {
        **sample,
        "supported": True,
        "observations": values,
        "accepted": all(v["errno"] == 0 and v["value"] == 0 for v in values.values()),
    }


@contextlib.contextmanager
def sleep_assertion():
    if sys.platform != "darwin":
        yield {"supported": False}
        return
    process = subprocess.Popen(
        ["/usr/bin/caffeinate", "-d", "-i", "-m", "-s", "-w", str(os.getpid())],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        yield {
            "supported": True,
            "pid": process.pid,
            "coordinator_pid": os.getpid(),
            "arguments": process.args,
        }
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=3)
