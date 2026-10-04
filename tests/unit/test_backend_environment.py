"""Fail-closed policy observations without querying the host or running solvers."""

import ctypes
import subprocess
from types import SimpleNamespace

import pytest

from neural_sat_speck.backend import environment


@pytest.mark.parametrize(
    "power,assertions,returncode,accepted",
    [
        ("Now drawing from 'AC Power'", "PreventSystemSleep 1", 0, True),
        ("Now drawing from 'Battery Power'", "PreventSystemSleep 1", 0, False),
        ("Now drawing from 'AC Power'", "PreventSystemSleep 0", 0, False),
        ("Now drawing from 'AC Power'", "PreventSystemSleep 1", 1, False),
    ],
)
def test_power_observation_requires_successful_ac_and_sleep(
    monkeypatch, power, assertions, returncode, accepted
):
    monkeypatch.setattr(environment.sys, "platform", "darwin")
    calls = []

    def observe(argv, **kwargs):
        calls.append(argv)
        assert kwargs["timeout"] <= 1
        return SimpleNamespace(
            stdout=assertions if argv[-1] == "assertions" else power,
            returncode=returncode,
        )

    monkeypatch.setattr(environment.subprocess, "run", observe)
    sample = environment.power_sample()
    assert sample["accepted"] is accepted
    assert sample["raw_assertions"] == assertions
    assert sample["raw_power"] == power
    assert len(calls) == 2


def test_unavailable_power_query_rejects(monkeypatch):
    monkeypatch.setattr(environment.sys, "platform", "darwin")

    def unavailable(*args, **kwargs):
        raise subprocess.TimeoutExpired("pmset", 1)

    monkeypatch.setattr(environment.subprocess, "run", unavailable)
    sample = environment.power_sample()
    assert sample["accepted"] is False and "error" in sample


@pytest.mark.parametrize(
    "nice,background,errno,accepted",
    [(0, 0, 0, True), (0, 1, 0, False), (1, 0, 0, False), (0, 0, 3, False)],
)
def test_process_policy_does_not_substitute_for_thread_qos(
    monkeypatch, nice, background, errno, accepted
):
    monkeypatch.setattr(environment.sys, "platform", "darwin")

    class Priority:
        def __call__(self, kind, pid):
            assert pid == 12345
            ctypes.set_errno(errno)
            return nice if kind == 0 else background

    monkeypatch.setattr(
        environment.ctypes,
        "CDLL",
        lambda *args, **kwargs: SimpleNamespace(getpriority=Priority()),
    )
    sample = environment.native_policy_sample(12345)
    assert sample["accepted"] is accepted
    assert sample["native_thread_qos"] == "unavailable"
    assert sample["observations"]["background_policy"]["errno"] == errno


def test_unsupported_platform_is_explicit(monkeypatch):
    monkeypatch.setattr(environment.sys, "platform", "linux")
    assert environment.power_sample()["accepted"] is None
    assert environment.native_policy_sample(12345)["accepted"] is None
    with environment.sleep_assertion() as receipt:
        assert receipt == {"supported": False}
