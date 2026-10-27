"""Unit tests for worker execution receipts, Darwin QoS certification, and adversarial rejection."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from neural_sat_speck.protocol import worker_telemetry
from neural_sat_speck.protocol.worker_telemetry import (
    QOS_CLASS_BACKGROUND,
    QOS_CLASS_DEFAULT,
    QOS_CLASS_UNSPECIFIED,
    QOS_CLASS_USER_INTERACTIVE,
    certify_worker_or_raise,
    load_and_verify_worker_receipt,
    observe_worker_telemetry,
    save_worker_receipt,
)


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        ("   PreventSystemSleep              1\n", True),
        ("\tPreventUserIdleSystemSleep\t1\t\n", True),
        ("PreventSystemSleep 0\nPreventUserIdleSystemSleep 1\n", True),
        ("PreventSystemSleep 1", True),
        ("PreventSystemSleep 0\nPreventUserIdleSystemSleep 0\n", False),
        ("PreventSystemSleep 10\n", False),
        ("OtherPreventSystemSleep 1\n", False),
        ("pid 42: PreventSystemSleep 1\n", False),
        ("PreventSystemSleep 1 owner detail\n", False),
        ("\n", False),
    ],
)
def test_sleep_assertion_summary_columns(monkeypatch, stdout, expected):
    monkeypatch.setattr(worker_telemetry.sys, "platform", "darwin")
    monkeypatch.setattr(
        worker_telemetry.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=stdout),
    )
    result = worker_telemetry._check_sleep_assertions()
    assert result["supported"] is True
    assert result["prevent_system_sleep"] is expected


def test_sleep_assertions_unsupported_off_darwin(monkeypatch):
    monkeypatch.setattr(worker_telemetry.sys, "platform", "linux")

    def reject_command(*args, **kwargs):
        raise AssertionError("pmset must not be called off Darwin")

    monkeypatch.setattr(worker_telemetry.subprocess, "run", reject_command)
    assert worker_telemetry._check_sleep_assertions() == {"supported": False}


# observe_worker_telemetry reads QoS, priority and I/O policy only on macOS.
darwin_only = pytest.mark.skipif(sys.platform != "darwin", reason="macOS QoS and priority checks")


class MockDarwinLibc:
    def __init__(
        self,
        qos: int = QOS_CLASS_USER_INTERACTIVE,
        prio: int = 0,
        bg_state: int = 0,
        iopolicy: int = 0,
    ) -> None:
        self._qos = qos
        self._prio = prio
        self._bg_state = bg_state
        self._iopolicy = iopolicy

    def qos_class_self(self) -> int:
        return self._qos

    def getpriority(self, which: int, who: int) -> int:
        if which == 0:  # PRIO_PROCESS
            return self._prio
        if which == 4:  # PRIO_DARWIN_PROCESS
            return self._bg_state
        return 0

    def getiopolicy_np(self, iotype: int, scope: int) -> int:
        return self._iopolicy


def test_live_worker_telemetry_observation():
    """Verify live telemetry collection captures valid system identifiers."""
    telemetry = observe_worker_telemetry()
    assert telemetry.pid > 0
    assert telemetry.parent_pid > 0
    assert telemetry.thread_id != 0
    assert isinstance(telemetry.observed_at, str)
    assert isinstance(telemetry.sleep_assertions, dict)
    assert telemetry.python_executable != ""


@darwin_only
def test_adversarial_qos9_zero_priority_rejected(tmp_path: Path):
    """Adversarial test: Background QoS (9) with zero priority MUST be rejected."""
    safe_runtime = tmp_path / "runtime"
    safe_runtime.mkdir(parents=True)
    safe_tmp = safe_runtime / "worker_tmp"
    safe_tmp.mkdir()

    libc = MockDarwinLibc(
        qos=QOS_CLASS_BACKGROUND,  # Class 9
        prio=0,                    # Priority 0
        bg_state=0,
        iopolicy=0,
    )

    t = observe_worker_telemetry(repo_root=tmp_path, libc_override=libc, tmpdir_override=str(safe_tmp))
    assert not t.is_certified
    assert "Background QoS (class 9)" in (t.rejection_reason or "")

    with pytest.raises(RuntimeError, match="Background QoS"):
        certify_worker_or_raise(repo_root=tmp_path, libc_override=libc, tmpdir_override=str(safe_tmp))


@darwin_only
def test_unspecified_or_failed_qos_rejected(tmp_path: Path):
    """Adversarial test: Unspecified (0) or failed (-1) QoS query MUST fail closed."""
    safe_runtime = tmp_path / "runtime"
    safe_runtime.mkdir(parents=True)
    safe_tmp = safe_runtime / "worker_tmp"
    safe_tmp.mkdir()

    for bad_qos in (QOS_CLASS_UNSPECIFIED, -1):
        libc = MockDarwinLibc(qos=bad_qos, prio=0, bg_state=0, iopolicy=0)
        with pytest.raises(RuntimeError, match="QoS observation failed or unspecified"):
            certify_worker_or_raise(repo_root=tmp_path, libc_override=libc, tmpdir_override=str(safe_tmp))


@darwin_only
def test_failed_priority_or_io_queries_rejected(tmp_path: Path):
    """Adversarial test: Failed libc queries for priority or IO MUST fail closed."""
    safe_runtime = tmp_path / "runtime"
    safe_runtime.mkdir(parents=True)
    safe_tmp = safe_runtime / "worker_tmp"
    safe_tmp.mkdir()

    # Failed priority query (-999)
    libc_prio_fail = MockDarwinLibc(qos=QOS_CLASS_USER_INTERACTIVE, prio=-999, bg_state=0, iopolicy=0)
    with pytest.raises(RuntimeError, match="Failed to query Darwin process priority"):
        certify_worker_or_raise(repo_root=tmp_path, libc_override=libc_prio_fail, tmpdir_override=str(safe_tmp))

    # Failed bg_state query (-999)
    libc_bg_fail = MockDarwinLibc(qos=QOS_CLASS_USER_INTERACTIVE, prio=0, bg_state=-999, iopolicy=0)
    with pytest.raises(RuntimeError, match="Failed to query Darwin background state"):
        certify_worker_or_raise(repo_root=tmp_path, libc_override=libc_bg_fail, tmpdir_override=str(safe_tmp))

    # Failed IO policy query (-999)
    libc_io_fail = MockDarwinLibc(qos=QOS_CLASS_USER_INTERACTIVE, prio=0, bg_state=0, iopolicy=-999)
    with pytest.raises(RuntimeError, match="Failed to query Darwin I/O throttle state"):
        certify_worker_or_raise(repo_root=tmp_path, libc_override=libc_io_fail, tmpdir_override=str(safe_tmp))


@darwin_only
def test_darwin_bg_state_rejected(tmp_path: Path):
    """Adversarial test: PRIO_DARWIN_BG active MUST be rejected."""
    safe_runtime = tmp_path / "runtime"
    safe_runtime.mkdir(parents=True)
    safe_tmp = safe_runtime / "worker_tmp"
    safe_tmp.mkdir()

    libc = MockDarwinLibc(
        qos=QOS_CLASS_DEFAULT,
        prio=0,
        bg_state=1,  # Background policy active
        iopolicy=0,
    )

    with pytest.raises(RuntimeError, match="Darwin process background policy active"):
        certify_worker_or_raise(repo_root=tmp_path, libc_override=libc, tmpdir_override=str(safe_tmp))


@darwin_only
def test_io_throttle_rejected(tmp_path: Path):
    """Adversarial test: I/O throttling policy MUST be rejected."""
    safe_runtime = tmp_path / "runtime"
    safe_runtime.mkdir(parents=True)
    safe_tmp = safe_runtime / "worker_tmp"
    safe_tmp.mkdir()

    libc = MockDarwinLibc(
        qos=QOS_CLASS_DEFAULT,
        prio=0,
        bg_state=0,
        iopolicy=1,  # Throttled I/O
    )

    with pytest.raises(RuntimeError, match="Darwin I/O throttle active"):
        certify_worker_or_raise(repo_root=tmp_path, libc_override=libc, tmpdir_override=str(safe_tmp))


@darwin_only
def test_degraded_process_priority_rejected(tmp_path: Path):
    """Adversarial test: Non-zero positive nice priority MUST be rejected."""
    safe_runtime = tmp_path / "runtime"
    safe_runtime.mkdir(parents=True)
    safe_tmp = safe_runtime / "worker_tmp"
    safe_tmp.mkdir()

    libc = MockDarwinLibc(
        qos=QOS_CLASS_USER_INTERACTIVE,
        prio=10,  # Niceness degraded
        bg_state=0,
        iopolicy=0,
    )

    with pytest.raises(RuntimeError, match="Darwin process priority degraded"):
        certify_worker_or_raise(repo_root=tmp_path, libc_override=libc, tmpdir_override=str(safe_tmp))


def test_unsafe_tmpdir_rejected(tmp_path: Path):
    """Adversarial test: TMPDIR in system ephemeral directories MUST be rejected."""
    libc = MockDarwinLibc(qos=QOS_CLASS_USER_INTERACTIVE, prio=0, bg_state=0, iopolicy=0)

    for bad_tmp in ("/tmp", "/private/tmp", "/var/folders", "/private/var"):
        with pytest.raises(RuntimeError, match="TMPDIR is non-persistent or outside project"):
            certify_worker_or_raise(repo_root=tmp_path, libc_override=libc, tmpdir_override=bad_tmp)


def test_symlink_escape_tmpdir_rejected(tmp_path: Path):
    """Adversarial test: Symlink pointing to an ephemeral path outside repo MUST be rejected."""
    libc = MockDarwinLibc(qos=QOS_CLASS_USER_INTERACTIVE, prio=0, bg_state=0, iopolicy=0)
    safe_runtime = tmp_path / "runtime"
    safe_runtime.mkdir(parents=True)
    symlink_tmp = safe_runtime / "escaped_tmp"
    # Create symlink pointing to /tmp
    os.symlink("/tmp", symlink_tmp)

    with pytest.raises(RuntimeError, match="TMPDIR is non-persistent or outside project"):
        certify_worker_or_raise(repo_root=tmp_path, libc_override=libc, tmpdir_override=str(symlink_tmp))


def test_valid_worker_certification_passes():
    """Positive test: Clean standard scheduling and safe project TMPDIR certifies successfully."""
    repo_root = Path(__file__).resolve().parents[2]
    safe_runtime = repo_root / "runtime"
    safe_runtime.mkdir(parents=True, exist_ok=True)
    safe_tmp = safe_runtime / f"_test_worker_tmp_{os.getpid()}"
    safe_tmp.mkdir(parents=True, exist_ok=True)

    try:
        libc = MockDarwinLibc(
            qos=QOS_CLASS_USER_INTERACTIVE,
            prio=0,
            bg_state=0,
            iopolicy=0,
        )

        telemetry = certify_worker_or_raise(repo_root=repo_root, libc_override=libc, tmpdir_override=str(safe_tmp))
        assert telemetry.is_certified
        assert telemetry.rejection_reason is None
        assert telemetry.tmpdir_safe is True
        assert telemetry.parent_pid > 0
    finally:
        if safe_tmp.exists():
            safe_tmp.rmdir()


def test_save_worker_receipt_with_sidecar(tmp_path: Path):
    """Verify atomic receipt persistence and sidecar SHA-256 generation."""
    safe_runtime = tmp_path / "runtime"
    safe_runtime.mkdir(parents=True)
    safe_tmp = safe_runtime / "worker_tmp"
    safe_tmp.mkdir()

    libc = MockDarwinLibc()
    telemetry = observe_worker_telemetry(repo_root=tmp_path, libc_override=libc, tmpdir_override=str(safe_tmp))

    receipt_file = tmp_path / "receipts" / "worker_receipt.json"
    save_worker_receipt(receipt_file, telemetry)

    assert receipt_file.is_file()
    sidecar_file = receipt_file.with_name("worker_receipt.json.sha256")
    assert sidecar_file.is_file()

    content = receipt_file.read_bytes()
    expected_digest = hashlib.sha256(content).hexdigest()
    sidecar_text = sidecar_file.read_text(encoding="utf-8")
    assert expected_digest in sidecar_text


def test_stale_or_mismatched_receipt_identity(tmp_path: Path):
    """Adversarial test: Receipt tampering, mismatched PID, git drift, or uncertified state MUST be rejected."""
    safe_runtime = tmp_path / "runtime"
    safe_runtime.mkdir(parents=True)
    safe_tmp = safe_runtime / "worker_tmp"
    safe_tmp.mkdir()

    libc = MockDarwinLibc(qos=QOS_CLASS_USER_INTERACTIVE, prio=0, bg_state=0, iopolicy=0)
    # Generate receipt
    telemetry = observe_worker_telemetry(repo_root=tmp_path, libc_override=libc, tmpdir_override=str(safe_tmp))
    # Override tmpdir_safe and is_certified for a valid baseline
    certified_telemetry = telemetry.__class__(
        **{**telemetry.to_dict(), "is_certified": True, "rejection_reason": None, "pid": 12345, "git_commit": "abcdef1"}
    )

    receipt_file = tmp_path / "receipts" / "worker_receipt.json"
    save_worker_receipt(receipt_file, certified_telemetry)

    # 1. Successful verification with matching identity
    loaded = load_and_verify_worker_receipt(receipt_file, expected_pid=12345, expected_git_commit="abcdef1")
    assert loaded.pid == 12345

    # 2. Tampered sidecar checksum
    sidecar_file = receipt_file.with_name("worker_receipt.json.sha256")
    sidecar_file.write_text("0" * 64 + "  worker_receipt.json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 sidecar mismatch"):
        load_and_verify_worker_receipt(receipt_file, expected_pid=12345)

    # Restore correct sidecar
    save_worker_receipt(receipt_file, certified_telemetry)

    # 3. Mismatched PID
    with pytest.raises(ValueError, match="Stale or mismatched receipt PID"):
        load_and_verify_worker_receipt(receipt_file, expected_pid=99999)

    # 4. Source identity drift (mismatched commit)
    with pytest.raises(ValueError, match="Source identity drift in receipt"):
        load_and_verify_worker_receipt(receipt_file, expected_git_commit="wrongcommit")

    # 5. Uncertified receipt
    uncertified = telemetry.__class__(
        **{**telemetry.to_dict(), "is_certified": False, "rejection_reason": "Background QoS", "pid": 12345}
    )
    save_worker_receipt(receipt_file, uncertified)
    with pytest.raises(ValueError, match="Worker receipt recorded uncertified execution"):
        load_and_verify_worker_receipt(receipt_file, expected_pid=12345)
