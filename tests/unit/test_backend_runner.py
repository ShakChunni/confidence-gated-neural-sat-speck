"""Bounded synthetic process/watchdog tests; no cryptographic solver calls."""

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from neural_sat_speck.backend import guard, runner
from neural_sat_speck.backend.watchdog import invocation_groups
from neural_sat_speck.backend.worker import _complete_assignment


@pytest.mark.parametrize("phase", ["formula_construction", "solving", "verification"])
def test_parent_kills_hanging_phase(tmp_path, phase):
    """Real child ignores its own limit; parent retains measured in-flight work."""
    fake_src = tmp_path / "fake_source"
    package = fake_src / "neural_sat_speck" / "backend"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    actual_package = Path(runner.__file__).parent
    for module in ("watchdog.py", "guard.py"):
        (package / module).write_text((actual_package / module).read_text())
    (package / "worker.py").write_text(
        "import json,sys,time,hashlib\n"
        "from pathlib import Path\n"
        "p=json.loads(Path(sys.argv[1]).read_text())\n"
        "j=Path(p['directory'])/'journal.json'\n"
        f"raw=json.dumps({{'phase':{phase!r},'phase_started_monotonic':time.perf_counter(),'completed_seconds':{{}}}}).encode()\n"
        "j.write_bytes(raw)\n"
        "j.with_name(j.name+'.sha256').write_text(hashlib.sha256(raw).hexdigest()+'  '+j.name+'\\n')\n"
        "time.sleep(60)\n"
    )
    invocation = uuid.uuid4().hex
    directory = tmp_path / invocation
    payload = {
        "directory": str(directory),
        "cell": guard.fixed_grid()[0],
        "invocation": invocation,
        "started_utc": runner.now(),
        "python_path": sys.executable,
        "repo_root": str(tmp_path),
    }
    env = {**os.environ, "PYTHONPATH": str(fake_src)}
    started = time.perf_counter()
    result = runner.run_cell(payload, env, timeout=0.25)
    assert time.perf_counter() - started < 2
    assert result["outcome"] == "global_deadline"
    assert result["phase_accounting"]["active_phase"] == phase
    assert result["phase_accounting"]["inflight_seconds"] > 0
    assert result["guardian_healthy"] is True
    assert result["guardian_receipt"]["status"] == "deadline_enforced"
    identity = guard.read_checked_json(directory / "process.json")
    assert runner._group_absent(identity["process_group"])


def test_independent_assignment_validator_signed_xor_and_units():
    formula = {
        "num_vars": 2,
        "cnf_clauses": [[1, 2]],
        "assumption_literals": [-2],
        "xor_clauses": [{"literals": [-1, 2], "rhs": False}],
    }
    assert _complete_assignment(formula, {1: True, 2: False})
    assert not _complete_assignment(formula, {1: True})
    assert not _complete_assignment(formula, {1: True, 2: True})
    formula["cnf_clauses"].append([])
    assert not _complete_assignment(formula, {1: True, 2: False})


def test_corrupted_journal_time_is_unavailable(tmp_path):
    (tmp_path / "journal.json").write_text("interrupted partial write")
    record = runner.interrupted_phases(tmp_path, time.perf_counter())
    assert record["inflight_seconds"] is None
    assert not record["journal_available"]


def test_original_flags_cannot_claim_exact_recovery_without_key_match():
    cell = guard.fixed_grid()[0]
    result = runner.failure(cell, "different_verified_key", 1.0)
    result.update(
        verified_known=True,
        verified_fresh=True,
        assigned_key_match=False,
        recovered_key=[1, 2, 3, 4],
    )
    guard.validate_record(result, cell)
    result["outcome"] = "recovered"
    with pytest.raises(ValueError):
        guard.validate_record(result, cell)


def test_process_group_probe_never_kills_active_group(monkeypatch):
    calls = []
    monkeypatch.setattr(os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    assert not runner._group_absent(12345)
    assert calls == [(12345, 0)]


@pytest.mark.parametrize("token", ["", "a", "synthetic", "0" * 31, None])
def test_watchdog_rejects_ambiguous_invocation_identity(monkeypatch, token):
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **k: pytest.fail("Queried processes for unsafe token"),
    )
    with pytest.raises(ValueError, match="token"):
        invocation_groups(token)


def test_watchdog_identifies_token_in_current_command_not_saved_pid(monkeypatch):
    from types import SimpleNamespace

    token = "a" * 32
    output = f"12345 python -m neural_sat_speck.backend.worker /runtime/{token}/payload.json\n54321 cadical -q /runtime/{token}/input.cnf\n12346 reused unrelated command\n98765 cat /runtime/{token}/payload.json\n98766 python -m neural_sat_speck.backend.watchdog /runtime/{token}/guardian.json\n"
    monkeypatch.setattr(
        "subprocess.run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=output)
    )
    monkeypatch.setattr(os, "getpgrp", lambda: 99999)
    assert invocation_groups(token) == {12345, 54321}


def _supervisor_fixture(tmp_path):
    fake_src = tmp_path / "fake_source"
    package = fake_src / "neural_sat_speck" / "backend"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    for name in ("guard.py", "watchdog.py"):
        (package / name).write_text((Path(runner.__file__).parent / name).read_text())
    (package / "worker.py").write_text("import time\ntime.sleep(60)\n")
    token = uuid.uuid4().hex
    payload = {
        "directory": str(tmp_path / token),
        "cell": guard.fixed_grid()[0],
        "invocation": token,
        "started_utc": runner.now(),
        "python_path": sys.executable,
        "repo_root": str(tmp_path),
    }
    return payload, {**os.environ, "PYTHONPATH": str(fake_src)}


def test_supervisor_enforces_deadline_after_coordinator_is_killed(tmp_path):
    payload, environment = _supervisor_fixture(tmp_path)
    input_path = tmp_path / "coordinator_input.json"
    input_path.write_text(json.dumps({"payload": payload, "environment": environment}))
    actual_src = str(Path(runner.__file__).resolve().parents[2])
    code = (
        "import json,sys\n"
        f"sys.path.insert(0,{actual_src!r})\n"
        "from pathlib import Path\n"
        "from neural_sat_speck.backend.runner import run_cell\n"
        "v=json.loads(Path(sys.argv[1]).read_text())\n"
        "run_cell(v['payload'],v['environment'],timeout=0.6)\n"
    )
    coordinator = subprocess.Popen(
        [sys.executable, "-c", code, str(input_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    directory = Path(payload["directory"])
    identity = None
    try:
        end = time.perf_counter() + 2
        while time.perf_counter() < end:
            try:
                identity = guard.read_checked_json(directory / "process.json")
                break
            except (OSError, ValueError):
                time.sleep(0.01)
        assert identity is not None, "Supervisor never persisted its own child identity"
        coordinator.kill()
        coordinator.wait(timeout=2)
        receipt = None
        while time.perf_counter() < end:
            try:
                receipt = guard.read_checked_json(directory / "guardian_receipt.json")
                break
            except (OSError, ValueError):
                time.sleep(0.01)
        assert receipt is not None
        assert receipt["status"] == "deadline_enforced"
        assert receipt["invocation"] == payload["invocation"]
        assert receipt["worker_pid"] == identity["pid"]
        assert runner._confirm_group_absence(identity["process_group"])
    finally:
        if coordinator.poll() is None:
            coordinator.kill()
            coordinator.wait(timeout=2)
        if identity and not runner._group_absent(identity["process_group"]):
            os.killpg(identity["process_group"], 9)


def test_pid_persistence_failure_still_cleans_owned_child(tmp_path, monkeypatch):
    payload, environment = _supervisor_fixture(tmp_path)
    ledger_path = tmp_path / "ledger.json"
    cell_id = payload["cell"]["cell_id"]
    guard.atomic_json(ledger_path, {"starts": {cell_id: {"process_group": None}}})
    payload["record_process"] = str(ledger_path)
    original = runner.atomic_json

    def fail_persistence(path, value):
        if path == ledger_path:
            raise OSError("Synthetic journal write failure")
        return original(path, value)

    monkeypatch.setattr(runner, "atomic_json", fail_persistence)
    with pytest.raises(OSError, match="journal write failure"):
        runner.run_cell(payload, environment, timeout=2.0)
    identity = guard.read_checked_json(Path(payload["directory"]) / "process.json")
    assert runner._confirm_group_absence(identity["process_group"])
    receipt = guard.read_checked_json(
        Path(payload["directory"]) / "guardian_receipt.json"
    )
    assert receipt["status"] == "coordinator_stop"


def test_nonzero_worker_exit_cannot_credit_a_written_result(tmp_path):
    payload, environment = _supervisor_fixture(tmp_path)
    package = Path(environment["PYTHONPATH"]) / "neural_sat_speck" / "backend"
    (package / "worker.py").write_text(
        "import json,sys\nfrom pathlib import Path\n"
        "from .guard import atomic_json\n"
        "p=json.loads(Path(sys.argv[1]).read_text())\n"
        "atomic_json(Path(p['directory'])/'result.json',{**p['cell'],'outcome':'unknown','solver_status':'unknown','verified_known':False,'verified_fresh':False,'assigned_key_match':False,'recovered_key':None})\n"
        "sys.exit(7)\n"
    )
    result = runner.run_cell(payload, environment, timeout=2.0)
    assert result["outcome"] == "worker_exception"
    assert result["worker_returncode"] == 7
    assert result["process_group_absent"] is True


def test_supervisor_crash_before_identity_publication_cleans_identified_child(
    tmp_path, monkeypatch
):
    payload, environment = _supervisor_fixture(tmp_path)
    package = Path(environment["PYTHONPATH"]) / "neural_sat_speck" / "backend"
    marker = tmp_path / "public_fixture_child.json"
    (package / "watchdog.py").write_text(
        "import json,sys,subprocess,os,signal\nfrom pathlib import Path\n"
        "p=json.loads(Path(sys.argv[1]).read_text())\n"
        "child=subprocess.Popen(p['worker_arguments'],start_new_session=True,cwd=p['repo_root'])\n"
        f"Path({str(marker)!r}).write_text(json.dumps({{'pid':child.pid}}))\n"
        "os.kill(os.getpid(),signal.SIGKILL)\n"
    )
    calls = []

    def identified_groups(token):
        # Identity lookup is independently tested against ps text; this test
        # targets the actual crash/startup cleanup, not sandbox process access.
        calls.append(token)
        return {json.loads(marker.read_text())["pid"]}

    monkeypatch.setattr(
        "neural_sat_speck.backend.watchdog.invocation_groups", identified_groups
    )
    result = runner.run_cell(payload, environment, timeout=2.0)
    group = json.loads(marker.read_text())["pid"]
    assert calls == [payload["invocation"]]
    assert result["outcome"] == "worker_exception"
    assert runner._confirm_group_absence(group)
    # A recovered identity scan cannot invent the missing durable start receipt.
    assert result["process_group_absent"] is False
