import fcntl
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest

spec = importlib.util.spec_from_file_location(
    "supervisor", Path.cwd() / "scripts/supervise_simon_confirmation.py"
)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


def ledger(path, rows):
    path.write_text(json.dumps({"records": rows}))
    path.with_name(path.name + ".sha256").write_text(
        s.sha(path) + "  " + path.name + "\n"
    )


def row(i=1, outcome="exception"):
    return {
        "instance_id": str(i),
        "method": "toy",
        "outcome": outcome,
        "error_message": "fixture",
    }


def test_reviewed_exception_is_retained():
    base = {"records": [row()]}
    s.check_reviewed_records(base, base)


def test_new_fatal_is_blocked():
    with pytest.raises(ValueError):
        s.check_reviewed_records({"records": [row(), row(2)]}, {"records": [row()]})


def test_changed_completed_record_is_blocked():
    with pytest.raises(ValueError):
        s.check_reviewed_records(
            {"records": [row(outcome="queue_exhausted")]}, {"records": [row()]}
        )


def test_corrupt_sidecar_is_blocked(tmp_path):
    p = tmp_path / "ledger.json"
    ledger(p, [])
    p.write_text("changed")
    with pytest.raises(ValueError):
        s.checked_ledger(p)


def test_alias_missing_is_recreated_and_wrong_target_blocked(tmp_path):
    target = tmp_path / "target"
    target.write_text("frozen")
    alias = tmp_path / "alias"
    c = {"storage_aliases": {str(alias): str(target)}, "invocation_files": {}}
    s.refresh_storage(c)
    assert alias.resolve() == target
    alias.unlink()
    alias.write_text("wrong")
    with pytest.raises(ValueError):
        s.refresh_storage(c)


def test_singleton_blocks_second_supervisor(tmp_path):
    d = tmp_path / "service"
    d.mkdir()
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"supervisor_directory": str(d)}))
    with (d / "singleton.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (
            patch.object(s, "supervise") as run,
            patch("sys.argv", ["supervisor", "--config", str(config)]),
        ):
            assert s.main() == 0
        run.assert_not_called()


def test_orphan_cleanup_refuses_live_parent():
    with (
        patch.object(
            s,
            "processes",
            return_value=[
                [
                    "22",
                    "10",
                    "22",
                    "python -m neural_sat_speck.simon.method_worker --worker",
                ]
            ],
        ),
        patch.object(s.os, "killpg") as kill,
    ):
        with pytest.raises(ValueError):
            s.clean_orphans({"argv": ["python"]}, lambda *a, **k: None)
        kill.assert_not_called()


def test_integrity_exit_holds_without_retry(tmp_path):
    base = tmp_path / "baseline.json"
    ledger(base, [])
    out = tmp_path / "out.json"
    ledger(out, [])
    conf = tmp_path / "config.json"
    conf.write_text("{}")
    cfg = {
        "supervisor_directory": str(tmp_path),
        "thread_environment": {},
        "persistent_tmpdir": str(tmp_path / "tmp"),
        "reviewed_baseline_ledger": str(base),
        "reviewed_baseline_sha256": s.sha(base),
        "argv": ["python", "fixture"],
        "ledger": str(out),
        "total_cells": 9,
        "max_os_restarts": 3,
        "cwd": str(tmp_path),
        "config_path": str(conf),
        "storage_aliases": {},
        "invocation_files": {},
    }

    class Process:
        pid = 222
        returncode = 8

        def poll(self):
            return 8

        def terminate(self):
            pass

    with (
        patch.object(s, "processes", return_value=[]),
        patch.object(s.subprocess, "Popen", return_value=Process()) as popen,
    ):
        assert s.supervise(cfg, tmp_path) == 0
        assert popen.call_count == 2
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["status"] == "HOLD"
    assert state["invocations"] == 1
    with patch.object(s.subprocess, "Popen") as popen:
        assert s.supervise(cfg, tmp_path) == 0
        popen.assert_not_called()


def test_os_kill_restarts_once_then_integrity_exit_holds(tmp_path):
    base = tmp_path / "baseline.json"
    ledger(base, [])
    out = tmp_path / "out.json"
    ledger(out, [])
    conf = tmp_path / "config.json"
    conf.write_text("{}")
    cfg = {
        "thread_environment": {},
        "persistent_tmpdir": str(tmp_path / "tmp"),
        "reviewed_baseline_ledger": str(base),
        "reviewed_baseline_sha256": s.sha(base),
        "argv": ["python", "fixture"],
        "ledger": str(out),
        "total_cells": 9,
        "max_os_restarts": 3,
        "cwd": str(tmp_path),
        "config_path": str(conf),
        "storage_aliases": {},
        "invocation_files": {},
    }

    class Process:
        pid = 222

        def __init__(self, code):
            self.returncode = code

        def poll(self):
            return self.returncode

        def terminate(self):
            pass

    with (
        patch.object(s, "processes", return_value=[]),
        patch.object(
            s.subprocess, "Popen", side_effect=[Process(0), Process(-15), Process(8)]
        ) as popen,
    ):
        assert s.supervise(cfg, tmp_path) == 0
        assert popen.call_count == 3
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["status"] == "HOLD"
    assert state["os_restarts"] == 1
    assert state["invocations"] == 2
    assert (tmp_path / "invocation_001/ledger_before.json").exists()
    assert (tmp_path / "invocation_002/ledger_before.json").exists()
    assert s.checked_ledger(out)["records"] == []


def test_completed_grid_invokes_analysis_without_another_runner(tmp_path):
    base = tmp_path / "baseline.json"
    ledger(base, [row(outcome="queue_exhausted")])
    out = tmp_path / "out.json"
    ledger(out, [row(outcome="queue_exhausted")])
    cfg = {
        "thread_environment": {},
        "persistent_tmpdir": str(tmp_path / "tmp"),
        "reviewed_baseline_ledger": str(base),
        "reviewed_baseline_sha256": s.sha(base),
        "argv": ["python", "fixture"],
        "ledger": str(out),
        "total_cells": 1,
        "cwd": str(tmp_path),
        "storage_aliases": {},
        "invocation_files": {},
        "analysis_argv": ["python", "frozen_fixture_analyzer"],
    }
    with (
        patch.object(s, "processes", return_value=[]),
        patch.object(s.subprocess, "Popen") as popen,
        patch.object(s.subprocess, "run") as analysis,
    ):
        analysis.return_value.returncode = 0
        assert s.supervise(cfg, tmp_path) == 0
        assert popen.call_count == 1  # caffeinate only
        assert analysis.call_args.args[0] == cfg["analysis_argv"]
    assert json.loads((tmp_path / "state.json").read_text())["status"] == "COMPLETE"


def test_baseline_corruption_holds_before_dispatch(tmp_path):
    base = tmp_path / "baseline.json"
    ledger(base, [])
    base.write_text("corrupt")
    cfg = {
        "thread_environment": {},
        "persistent_tmpdir": str(tmp_path / "tmp"),
        "reviewed_baseline_ledger": str(base),
        "reviewed_baseline_sha256": "0" * 64,
        "argv": ["python", "fixture"],
    }
    with patch.object(s.subprocess, "Popen") as popen:
        assert s.supervise(cfg, tmp_path) == 0
        assert popen.call_count == 1  # caffeinate only
    assert json.loads((tmp_path / "state.json").read_text())["status"] == "HOLD"


def test_resolved_macos_python_command_adopts_same_runner():
    argv = ["/repo/.venv/bin/python", "/repo/scripts/runner.py", "--resume"]
    command = "/opt/homebrew/Python.app/Contents/MacOS/Python /repo/scripts/runner.py --resume"
    with patch.object(s, "processes", return_value=[["9050", "1", "9050", command]]):
        assert s.matching_runners(argv) == [9050]
        assert s.matching_runners([*argv, "--changed"]) == []


def test_interrupted_analysis_holds_before_duplicate_analyzer(tmp_path):
    (tmp_path / "state.json").write_text(json.dumps({"status": "ANALYZING"}))
    with patch.object(s.subprocess, "Popen") as popen:
        assert s.supervise({}, tmp_path) == 0
        popen.assert_not_called()
    assert json.loads((tmp_path / "state.json").read_text())["status"] == "HOLD"


def test_deliberate_ctrl_c_does_not_trigger_automatic_restart():
    assert -2 not in s.OS_INTERRUPTION_CODES
    assert 130 not in s.OS_INTERRUPTION_CODES


def test_caffeinate_assertion_includes_full_power_and_user_flags(tmp_path):
    cfg = {
        "thread_environment": {},
        "persistent_tmpdir": str(tmp_path / "tmp"),
        "reviewed_baseline_ledger": str(tmp_path / "base.json"),
        "reviewed_baseline_sha256": "0" * 64,
        "argv": ["python", "fixture"],
    }
    base = tmp_path / "base.json"
    base.write_text("corrupt")
    with patch.object(s.subprocess, "Popen") as popen:
        s.supervise(cfg, tmp_path)
        assert popen.call_count == 1
        caff_args = popen.call_args[0][0]
        assert caff_args[0] == "/usr/bin/caffeinate"
        for flag in ("-d", "-i", "-m", "-s", "-u", "-w"):
            assert flag in caff_args


def test_guarantee_ordinary_scheduling_behavior():
    # Calling on current process (ordinary) succeeds
    s.guarantee_ordinary_scheduling(0)

    # If background scheduling is reported and cannot be cleared, ValueError is raised
    class MockLibc:
        def setiopolicy_np(self, *args):
            return 0

        def setpriority(self, which, who, prio):
            return 0

        def getpriority(self, which, who):
            return 1  # Still in Background

    with patch("ctypes.CDLL", return_value=MockLibc()):
        with pytest.raises(ValueError, match="scheduling policy is Background"):
            s.guarantee_ordinary_scheduling(0)


def test_ephemeral_tmpdir_is_rejected(tmp_path):
    base = tmp_path / "base.json"
    ledger(base, [])
    cfg = {
        "thread_environment": {},
        "persistent_tmpdir": "/tmp/worker_tmp",
        "reviewed_baseline_ledger": str(base),
        "reviewed_baseline_sha256": s.sha(base),
        "argv": ["python", "fixture"],
    }
    with patch.object(s.subprocess, "Popen"):
        s.supervise(cfg, tmp_path)
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["status"] == "HOLD"
    assert "persistent_tmpdir must be in a persistent project directory" in state["reason"]

    cfg["persistent_tmpdir"] = "/private/tmp/worker_tmp"
    with patch.object(s.subprocess, "Popen"):
        s.supervise(cfg, tmp_path)
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["status"] == "HOLD"
    assert "persistent_tmpdir must be in a persistent project directory" in state["reason"]


def test_default_persistent_tmpdir_in_project_workspace(tmp_path):
    base = tmp_path / "base.json"
    ledger(base, [])
    out = tmp_path / "out.json"
    ledger(out, [])
    conf = tmp_path / "conf.json"
    conf.write_text("{}")
    cfg = {
        "thread_environment": {},
        # Omit persistent_tmpdir to test default resolution
        "reviewed_baseline_ledger": str(base),
        "reviewed_baseline_sha256": s.sha(base),
        "argv": ["python", "fixture"],
        "ledger": str(out),
        "total_cells": 1,
        "cwd": str(tmp_path),
        "config_path": str(conf),
        "storage_aliases": {},
        "invocation_files": {},
    }

    class Process:
        pid = 333
        returncode = 0

        def poll(self):
            return 0

        def terminate(self):
            pass

    with (
        patch.object(s, "processes", return_value=[]),
        patch.object(s.subprocess, "Popen", side_effect=[Process(), Process()]),
    ):
        assert s.supervise(cfg, tmp_path) == 0
    receipt = json.loads((tmp_path / "invocation_001" / "receipt.json").read_text())
    expected_tmp = str(tmp_path / "results" / "simon_confirmation_replacement" / "worker_tmp")
    assert receipt["persistent_tmpdir"] == expected_tmp
    assert Path(expected_tmp).is_dir()
    assert receipt["task_scheduling"] == "standard_ordinary"

