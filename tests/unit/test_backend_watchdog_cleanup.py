"""Adversarial cleanup contracts: only kernel ESRCH is an absence proof."""

import errno
import json
import signal
from types import SimpleNamespace

import pytest

from neural_sat_speck.backend import watchdog
from neural_sat_speck.backend.guard import read_checked_json


@pytest.fixture
def clock(monkeypatch):
    value = [100.0]
    monkeypatch.setattr(watchdog.time, "perf_counter", lambda: value[0])
    monkeypatch.setattr(
        watchdog.time, "sleep", lambda delay: value.__setitem__(0, value[0] + delay)
    )
    monkeypatch.setattr(watchdog.os, "getpgrp", lambda: 123)
    return value


@pytest.mark.parametrize("first_errno", [errno.EINTR, errno.EPERM, errno.EINVAL, None])
def test_transient_lookup_error_does_not_end_proof_window(
    clock, monkeypatch, first_errno
):
    events = []

    def probe(pgid, sig):
        events.append((pgid, sig))
        if len(events) == 1:
            raise OSError(first_errno, "transient lookup")
        raise OSError(errno.ESRCH, "group no longer exists")

    monkeypatch.setattr(watchdog.os, "killpg", probe)
    receipt = watchdog.confirm_group_absence(456)
    assert receipt["process_group_absent"] is True
    assert events == [(456, 0), (456, 0)]
    assert receipt["absence_probe_errors"][0]["errno"] == first_errno
    assert receipt["absence_proof"] == "killpg_signal_zero_ESRCH"


@pytest.mark.parametrize(
    "failure_errno", [errno.EPERM, errno.EINVAL, errno.EINTR, None]
)
def test_persistent_errors_are_bounded_and_fail_closed(
    clock, monkeypatch, failure_errno
):
    calls = []

    def probe(pgid, sig):
        calls.append(sig)
        raise OSError(failure_errno, "uncertain group state")

    monkeypatch.setattr(watchdog.os, "killpg", probe)
    receipt = watchdog.confirm_group_absence(456)
    assert receipt["process_group_absent"] is False
    assert 1 <= len(calls) <= 100
    assert set(calls) == {0}
    assert len(receipt["absence_probe_errors"]) == len(calls)
    assert receipt["absence_probe_wall_seconds"] <= 1.000001


def test_live_group_is_never_treated_as_absent(clock, monkeypatch):
    monkeypatch.setattr(watchdog.os, "killpg", lambda pgid, sig: None)
    receipt = watchdog.confirm_group_absence(456)
    assert receipt["process_group_absent"] is False
    assert receipt["absence_probe_errors"] == []


@pytest.mark.parametrize("group", [True, 0, 1, -5, 123])
def test_unsafe_group_rejected_before_probe(clock, monkeypatch, group):
    monkeypatch.setattr(
        watchdog.os, "killpg", lambda *args: pytest.fail("unsafe probe")
    )
    with pytest.raises(ValueError, match="Unsafe"):
        watchdog.confirm_group_absence(group)


@pytest.mark.parametrize("timeout", [True, 0, -1, float("nan"), float("inf")])
def test_invalid_timeout_rejected_before_probe(clock, monkeypatch, timeout):
    monkeypatch.setattr(
        watchdog.os, "killpg", lambda *args: pytest.fail("unsafe probe")
    )
    with pytest.raises(ValueError, match="timeout"):
        watchdog.confirm_group_absence(456, timeout=timeout)


@pytest.mark.parametrize("prove_absence", [True, False])
def test_main_preserves_error_receipt_and_never_signals_after_reaping(
    tmp_path, clock, monkeypatch, prove_absence
):
    payload = {
        "deadline_monotonic": 101.0,
        "invocation": "a" * 32,
        "directory": str(tmp_path),
        "repo_root": str(tmp_path),
        "worker_arguments": ["synthetic-worker"],
        "receipt": str(tmp_path / "receipt.json"),
    }
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps(payload))
    monkeypatch.setattr(watchdog.sys, "argv", ["guardian", str(payload_path)])
    monkeypatch.setattr(watchdog.signal, "signal", lambda *args: None)
    actions = []

    def reap(timeout):
        actions.append("reap")
        return 0

    child = SimpleNamespace(pid=456, wait=reap)
    monkeypatch.setattr(watchdog.subprocess, "Popen", lambda *args, **kwargs: child)
    monkeypatch.setattr(watchdog.os, "waitid", lambda *args: object())

    def probe(pgid, sig):
        actions.append(sig)
        if sig == signal.SIGKILL:
            assert "reap" not in actions
            raise PermissionError(errno.EPERM, "exited worker")
        assert "reap" in actions
        if prove_absence and actions.count(0) == 2:
            raise ProcessLookupError(errno.ESRCH, "absent")
        raise PermissionError(errno.EPERM, "lookup denied")

    monkeypatch.setattr(watchdog.os, "killpg", probe)
    if prove_absence:
        watchdog.main()
    else:
        with pytest.raises(RuntimeError, match="absence was not proven"):
            watchdog.main()
    receipt = read_checked_json(tmp_path / "receipt.json")
    assert receipt["group_signal_errno"] == errno.EPERM
    assert receipt["worker_returncode"] == 0
    assert receipt["process_group_absent"] is prove_absence
    assert receipt["absence_probe_errors"]
    assert actions.count(signal.SIGKILL) == 1
    assert receipt["status"] == ("worker_exit" if prove_absence else "guardian_error")
