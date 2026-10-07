"""Worker process execution receipts and Darwin QoS/scheduling certification.

Certifies that a worker process and its calling thread execute under standard,
unthrottled operating system scheduling, persistent non-ephemeral storage,
and verified git/python identity, strictly rejecting macOS Background QoS (9)
even when Darwin process priority is zero.
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Darwin QoS Classes (sys/qos.h)
QOS_CLASS_USER_INTERACTIVE = 33
QOS_CLASS_USER_INITIATED = 25
QOS_CLASS_DEFAULT = 21
QOS_CLASS_UTILITY = 17
QOS_CLASS_BACKGROUND = 9
QOS_CLASS_UNSPECIFIED = 0

# Darwin Priorities and Policies (sys/resource.h)
PRIO_PROCESS = 0
PRIO_DARWIN_PROCESS = 4
IOPOL_TYPE_DISK = 0
IOPOL_SCOPE_PROCESS = 0
IOPOL_NORMAL = 0

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TF_NUM_INTEROP_THREADS",
    "TF_NUM_INTRAOP_THREADS",
)


def _get_git_commit(repo_root: Path) -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN_COMMIT"


def _check_sleep_assertions() -> dict[str, Any]:
    """Inspect active macOS sleep assertions."""
    if sys.platform != "darwin":
        return {"supported": False}
    try:
        res = subprocess.run(
            ["pmset", "-g", "assertions"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        stdout = res.stdout
        prevent_sleep = bool(
            re.search(
                r"^\s*(?:PreventSystemSleep|PreventUserIdleSystemSleep)[ \t]+1[ \t]*$",
                stdout,
                re.MULTILINE,
            )
        )
        has_caffeinate = "caffeinate" in stdout.lower()
        return {
            "supported": True,
            "prevent_system_sleep": prevent_sleep,
            "caffeinate_active": has_caffeinate,
        }
    except Exception as exc:
        return {"supported": True, "error": str(exc), "prevent_system_sleep": False}


@dataclass(frozen=True)
class WorkerTelemetry:
    """Snapshot of worker process execution state and Darwin scheduling invariants."""

    observed_at: str
    pid: int
    parent_pid: int
    thread_id: int
    platform: str
    qos_class: int
    darwin_priority: int
    darwin_bg_state: int
    io_policy_state: int
    thread_environment: dict[str, str]
    tmpdir: str
    tmpdir_safe: bool
    git_commit: str
    python_executable: str
    command_line: list[str]
    sleep_assertions: dict[str, Any]
    is_certified: bool
    rejection_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkerTelemetry:
        return cls(**data)


def observe_worker_telemetry(
    repo_root: Path | None = None,
    *,
    libc_override: Any | None = None,
    tmpdir_override: str | None = None,
) -> WorkerTelemetry:
    """Collect an audited execution telemetry snapshot for the current worker thread."""
    if repo_root is None:
        repo_root = Path.cwd()

    observed_at = datetime.now(UTC).isoformat()
    pid = os.getpid()
    thread_id = threading.get_ident()
    platform = sys.platform

    qos_class = QOS_CLASS_UNSPECIFIED
    darwin_priority = 0
    darwin_bg_state = 0
    io_policy_state = 0

    if platform == "darwin":
        libc = libc_override if libc_override is not None else ctypes.CDLL(None)
        try:
            qos_class = libc.qos_class_self()
        except Exception:
            qos_class = -1

        try:
            darwin_priority = libc.getpriority(PRIO_PROCESS, 0)
        except Exception:
            darwin_priority = -999

        try:
            darwin_bg_state = libc.getpriority(PRIO_DARWIN_PROCESS, 0)
        except Exception:
            darwin_bg_state = -999

        try:
            io_policy_state = libc.getiopolicy_np(IOPOL_TYPE_DISK, IOPOL_SCOPE_PROCESS)
        except Exception:
            io_policy_state = -999

    thread_env = {var: os.environ.get(var, "") for var in THREAD_ENV_VARS if var in os.environ}

    parent_pid = os.getppid()

    # Determine persistent project directory for safe TMPDIR validation
    repo_path = repo_root.resolve()
    valid_tmp_roots = (
        (repo_path / "runtime").resolve(),
        (repo_path / "results").resolve(),
    )

    raw_tmp = tmpdir_override if tmpdir_override is not None else os.environ.get("TMPDIR", "")
    if raw_tmp:
        tmp_resolved = Path(raw_tmp).resolve()
        # Path must exist, be a directory, not reside in system ephemeral paths,
        # and be strictly contained within repo's runtime/ or results/
        is_in_project_dir = any(
            tmp_resolved == root or root in tmp_resolved.parents
            for root in valid_tmp_roots
        )
        is_ephemeral = str(tmp_resolved).startswith(("/tmp", "/private/tmp", "/var/folders", "/private/var"))
        tmpdir_safe = bool(tmp_resolved.is_dir() and is_in_project_dir and not is_ephemeral)
    else:
        tmp_resolved = Path("/tmp")
        tmpdir_safe = False

    git_commit = _get_git_commit(repo_root)
    py_exec = sys.executable
    cmd_line = list(sys.argv)
    sleep_info = _check_sleep_assertions()

    # Certification logic - strict fail-closed invariants
    rejections: list[str] = []
    if platform == "darwin":
        if qos_class == QOS_CLASS_BACKGROUND:
            rejections.append("Worker thread executing under Background QoS (class 9)")
        elif qos_class <= 0 or qos_class == QOS_CLASS_UNSPECIFIED:
            rejections.append(f"Worker thread QoS observation failed or unspecified (qos={qos_class})")
        elif qos_class not in (QOS_CLASS_USER_INTERACTIVE, QOS_CLASS_USER_INITIATED, QOS_CLASS_DEFAULT):
            rejections.append(f"Worker thread QoS class not accepted for standard execution (qos={qos_class})")

        if darwin_priority == -999:
            rejections.append("Failed to query Darwin process priority")
        elif darwin_priority != 0:
            rejections.append(f"Darwin process priority degraded or non-standard (prio={darwin_priority})")

        if darwin_bg_state == -999:
            rejections.append("Failed to query Darwin background state")
        elif darwin_bg_state != 0:
            rejections.append(f"Darwin process background policy active (bg_state={darwin_bg_state})")

        if io_policy_state == -999:
            rejections.append("Failed to query Darwin I/O throttle state")
        elif io_policy_state != IOPOL_NORMAL:
            rejections.append(f"Darwin I/O throttle active (iopolicy={io_policy_state})")

    if not tmpdir_safe:
        rejections.append(
            f"TMPDIR is non-persistent or outside project runtime/results ({raw_tmp!r} -> {str(tmp_resolved)!r})"
        )

    is_certified = len(rejections) == 0
    rejection_reason = "; ".join(rejections) if rejections else None

    return WorkerTelemetry(
        observed_at=observed_at,
        pid=pid,
        parent_pid=parent_pid,
        thread_id=thread_id,
        platform=platform,
        qos_class=qos_class,
        darwin_priority=darwin_priority,
        darwin_bg_state=darwin_bg_state,
        io_policy_state=io_policy_state,
        thread_environment=thread_env,
        tmpdir=str(tmp_resolved),
        tmpdir_safe=tmpdir_safe,
        git_commit=git_commit,
        python_executable=py_exec,
        command_line=cmd_line,
        sleep_assertions=sleep_info,
        is_certified=is_certified,
        rejection_reason=rejection_reason,
    )


def certify_worker_or_raise(
    repo_root: Path | None = None,
    *,
    libc_override: Any | None = None,
    tmpdir_override: str | None = None,
) -> WorkerTelemetry:
    """Observe worker telemetry and raise RuntimeError if execution fails certification."""
    telemetry = observe_worker_telemetry(
        repo_root=repo_root,
        libc_override=libc_override,
        tmpdir_override=tmpdir_override,
    )
    if not telemetry.is_certified:
        raise RuntimeError(
            f"Worker execution scheduling certification failed: {telemetry.rejection_reason}"
        )
    return telemetry


def save_worker_receipt(target_path: Path, telemetry: WorkerTelemetry) -> None:
    """Persist worker telemetry receipt atomically with sidecar SHA-256."""
    import hashlib

    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target.with_suffix(".tmp")
    data_bytes = json.dumps(telemetry.to_dict(), indent=2).encode("utf-8")

    temp_target.write_bytes(data_bytes)
    temp_target.replace(target)

    digest = hashlib.sha256(data_bytes).hexdigest()
    sidecar = target.with_name(f"{target.name}.sha256")
    sidecar.write_text(f"{digest}  {target.name}\n", encoding="utf-8")


def load_and_verify_worker_receipt(
    receipt_path: Path,
    *,
    expected_pid: int | None = None,
    expected_git_commit: str | None = None,
) -> WorkerTelemetry:
    """Load a persisted worker receipt, verify its SHA-256 sidecar, certification, and identity."""
    import hashlib

    receipt = Path(receipt_path)
    if not receipt.is_file():
        raise FileNotFoundError(f"Worker receipt not found: {receipt}")
    sidecar = receipt.with_name(f"{receipt.name}.sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Worker receipt sidecar checksum not found: {sidecar}")

    content = receipt.read_bytes()
    expected_digest = hashlib.sha256(content).hexdigest()
    sidecar_lines = sidecar.read_text(encoding="utf-8").strip().split()
    if not sidecar_lines or sidecar_lines[0] != expected_digest:
        raise ValueError(f"Worker receipt SHA-256 sidecar mismatch for {receipt}")

    data = json.loads(content.decode("utf-8"))
    telemetry = WorkerTelemetry.from_dict(data)

    if not telemetry.is_certified:
        raise ValueError(f"Worker receipt recorded uncertified execution: {telemetry.rejection_reason}")

    if expected_pid is not None and telemetry.pid != expected_pid:
        raise ValueError(
            f"Stale or mismatched receipt PID: expected {expected_pid}, got {telemetry.pid}"
        )

    if expected_git_commit is not None and telemetry.git_commit != expected_git_commit:
        raise ValueError(
            f"Source identity drift in receipt: expected commit {expected_git_commit}, got {telemetry.git_commit}"
        )

    return telemetry
