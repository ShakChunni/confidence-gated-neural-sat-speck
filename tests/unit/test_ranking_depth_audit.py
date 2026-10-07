"""Adversarial replay of raw toy SAT evidence, independent of attack rankers."""

import copy
import errno
import shutil
import sys
from pathlib import Path

import pytest

# This helper constructs three-round synthetic mappings and fixes a known key
# while generating the test assignment. It runs no outcome-bearing key search.
from test_ranking_depth_execution import model_for_key, toy

from neural_sat_speck.backend.formula import build_formula
from neural_sat_speck.backend.guard import atomic_json, read_checked_json, sha256_file
from neural_sat_speck.backend.solver_io import solver_arguments
from neural_sat_speck.followups import ranking_depth_audit as audit
from neural_sat_speck.followups.conditioning_control import digest
from neural_sat_speck.followups.ranking_depth_worker import conditioned_formula


@pytest.mark.parametrize("mutation", [None, "extra", "missing", "version", "origin"])
def test_tf_environment_lock_remains_strict_after_loaded_inventory(
    tmp_path, monkeypatch, mutation
):
    path = tmp_path / "tensorflow_stage1_worker.json"
    atomic_json(path, {"synthetic_telemetry": True})
    expected = {
        "packages": {"tensorflow": "2.16.2"},
        "origins": {"tensorflow": "/synthetic/tensorflow/__init__.py"},
        "installed_packages": {"tensorflow": "2.16.2", "vendored-package": "1.0"},
    }
    observed = {
        "python_executable": sys.executable,
        "packages": {
            "tensorflow": {
                "version": "2.16.2",
                "origin": expected["origins"]["tensorflow"],
            }
        },
        "installed_packages": copy.deepcopy(expected["installed_packages"]),
    }
    if mutation == "extra":
        observed["installed_packages"]["unexpected"] = "1.0"
    elif mutation == "missing":
        observed["installed_packages"].pop("vendored-package")
    elif mutation == "version":
        observed["installed_packages"]["vendored-package"] = "2.0"
    elif mutation == "origin":
        observed["packages"]["tensorflow"]["origin"] = "/redirected/__init__.py"
    ranker = {
        "tensorflow_child_pid": 2,
        "tensorflow_child_process_group": 1,
        "tensorflow_command": [
            sys.executable,
            "-m",
            "neural_sat_speck.followups.ranking_depth_ranking",
            "--tf-stage1",
            "--invocation-path",
            str(tmp_path),
        ],
        "runtime_receipt_sha256": sha256_file(path),
        "runtime": observed,
    }
    ranked = {
        "ranking_receipts": {
            "stage1": {"assets": {}, "code_dependencies": {}, "ranker": ranker},
            "stage2": None,
        }
    }
    monkeypatch.setattr(audit, "check_telemetry", lambda *_: None)
    args = (
        ranked,
        {"cipher": "simon32_64", "trailing_keys": 1},
        tmp_path,
        {"runtime": {"tensorflow": expected}},
        {"tf_python": Path(sys.executable)},
        1,
    )
    if mutation is None:
        audit.check_ranking_receipts(*args)
    else:
        with pytest.raises(ValueError, match="drift"):
            audit.check_ranking_receipts(*args)


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    public, fresh, key, keys = toy("speck32_64")
    root = tmp_path / "snapshot"
    root.mkdir()
    directory = tmp_path / ("a" * 32)
    directory.mkdir()
    public_path, fresh_path = tmp_path / "public.json", tmp_path / "fresh.json"
    atomic_json(public_path, public)
    atomic_json(fresh_path, fresh)
    paths = {
        "python": Path(sys.executable),
        "case:public": public_path,
        "case:verification": fresh_path,
    }
    for name in (
        "cms",
        "tf_python",
        "bao_vv9",
        "simon_ddt",
        "speck_stage1",
        "speck_classical",
    ):
        paths[name] = tmp_path / name
    cell = {
        "cell_id": "synthetic_m1_G",
        "case_id": "case",
        "cipher": "speck32_64",
        "key_index": 0,
        "rounds": 3,
        "trailing_keys": 1,
        "profile": "cms_native_gauss",
    }
    manifest = {
        "cells": [cell],
        "source_root": str(root),
        "source_commit": "b" * 40,
        "cases": {
            "case": {
                "public": {"sha256": sha256_file(public_path)},
                "verification": {"sha256": sha256_file(fresh_path)},
                "base_formula_sha256": "c" * 64,
            }
        },
    }
    payload = {
        "cell": cell,
        "directory": str(directory),
        "invocation_id": directory.name,
        "repo_root": str(root),
        "source_commit": manifest["source_commit"],
        "python_path": sys.executable,
        "paths": {
            k: str(paths[k])
            for k in (
                "cms",
                "tf_python",
                "bao_vv9",
                "simon_ddt",
                "speck_stage1",
                "speck_classical",
            )
        },
        "public_path": str(public_path),
        "public_sha256": sha256_file(public_path),
        "verification_path": str(fresh_path),
        "verification_sha256": sha256_file(fresh_path),
        "base_formula_sha256": "c" * 64,
        "deadline_monotonic": 1300.0,
    }
    guardian = {
        "worker_arguments": [
            sys.executable,
            "-m",
            "neural_sat_speck.followups.ranking_depth_worker",
            str(directory / "payload.json"),
        ],
        "directory": str(directory),
        "invocation": directory.name,
        "deadline_monotonic": 1300.0,
    }
    process = {"pid": 12345, "process_group": 12345, "supervisor_pid": 12346}
    receipt = {
        "worker_pid": 12345,
        "process_group": 12345,
        "watchdog_pid": 12346,
        "process_group_absent": True,
        "invocation": directory.name,
        "deadline_monotonic": 1300.0,
        "observed_monotonic": 1005.0,
        "cleanup_completed_monotonic": 1005.1,
        "status": "worker_exit",
        "worker_returncode": 0,
    }
    phases = {
        "completed_seconds": {
            "stage1_ranking": 0.5,
            "native_launch_parse_and_solve": 3.0,
        },
        "inflight_phase": None,
        "inflight_seconds": 0.0,
    }
    journal = {
        "completed_seconds": phases["completed_seconds"],
        "phase": None,
        "phase_started_monotonic": None,
        "worker_started_monotonic": 1000.5,
        "observed_monotonic": 1004.9,
    }
    samples = [
        {
            "observed_monotonic": 1000.5,
            "power": {},
            "worker_policy": {},
            "memory": {"group_rss_bytes": 1000000, "accepted": True},
        }
    ]
    parents = [keys[-1]] + [i for i in range(32) if i != keys[-1]]
    parents = parents[:32]
    queue = [[p] for p in parents]
    qrecord = {
        "queue": queue,
        "stage1_parents": parents,
        "queue_sha256": digest(queue),
        "stage1_sha256": digest(parents),
    }
    atomic_json(
        directory / "stage1.json",
        {"stage1_parents": parents, "stage1_prefix_sha256": digest(parents)},
    )
    base = build_formula(cell["cipher"], public, 4, 0, include_cnf=False)
    formula = conditioned_formula(base, queue[0])
    assignment = model_for_key(formula, key)
    attempt = directory / "attempts/0000"
    attempt.mkdir(parents=True)
    native = attempt / "input.dimacs"
    native.write_text(formula["native_dimacs"])
    (attempt / "solver_stdout.txt").write_text(
        "s SATISFIABLE\nv "
        + " ".join(str(v if b else -v) for v, b in assignment.items())
        + " 0\n"
    )
    (attempt / "solver_stderr.txt").write_text("")
    argv = solver_arguments(cell["profile"], str(paths["cms"]), "", str(native))
    launch = {"solver_arguments": argv}
    att = {
        "index": 0,
        "candidate": queue[0],
        "status": "sat",
        "returncode": 10,
        "native_sha256": sha256_file(native),
        "stdout_sha256": sha256_file(attempt / "solver_stdout.txt"),
        "stderr_sha256": sha256_file(attempt / "solver_stderr.txt"),
        "solver_arguments": argv,
        "native_launch_receipt": launch,
        "started_monotonic": 1001.0,
        "ended_monotonic": 1004.0,
        "solve_wall_seconds": 3.0,
        "recovered_key": list(key),
        "verified_known": True,
        "verified_fresh": True,
    }
    raw = {
        "outcome": "recovered",
        "attempt_count": 1,
        "recovered_key": list(key),
        "verified_known": True,
        "verified_fresh": True,
        "completed_phase_seconds": journal["completed_seconds"],
    }
    row = {
        "cell": cell,
        "directory": str(directory),
        "outcome": "recovered",
        "started_monotonic": 1000.0,
        "observed_monotonic": 1005.0,
        "deadline_monotonic": 1300.0,
        "parent_wall_seconds": 5.0,
        "cleanup_wall_seconds": receipt["cleanup_completed_monotonic"]
        - receipt["observed_monotonic"],
        "guardian_receipt": receipt,
        "guardian_returncode": 0,
        "worker_result": raw,
        "phase_accounting": phases,
        "component_accounting_available": True,
        "uncommitted_terminal_json": [],
        "environment_samples": samples,
    }
    for name, value in (
        ("payload", payload),
        ("guardian", guardian),
        ("process", process),
        ("guardian_receipt", receipt),
        ("journal", journal),
        ("environment_samples", samples),
        ("queue", qrecord),
        ("python_worker_receipt", {}),
        ("worker_result", raw),
        ("supervisor_result", row),
    ):
        atomic_json(directory / (name + ".json"), value)
    atomic_json(
        attempt / "native_process.json",
        {"pid": 54321, "process_group": 12345, "argv": argv},
    )
    atomic_json(attempt / "native_launch_receipt.json", launch)
    atomic_json(
        attempt / "attempt_start.json",
        {
            "candidate_literals": formula["assumption_literals"],
            "native_sha256": sha256_file(native),
            "formula_counts": formula["counts"],
        },
    )
    atomic_json(attempt / "attempt.json", att)
    # Platform collection itself is covered by worker telemetry/resource tests.
    monkeypatch.setattr(audit, "check_telemetry", lambda *_: None)
    monkeypatch.setattr(audit, "check_ranking_receipts", lambda *_: None)
    monkeypatch.setattr(audit, "_validate_policy", lambda *_, **__: None)
    monkeypatch.setattr(audit, "_validate_power", lambda *_: None)
    return row, manifest, paths, directory, attempt


def test_raw_toy_model_reproduces_fresh_verified_key(evidence):
    row, manifest, paths, _, _ = evidence
    checked = audit.validate_cell(row, manifest, paths)
    assert checked["recovered_key"] == row["worker_result"]["recovered_key"]
    assert checked["completed_attempts"] == 1


@pytest.mark.parametrize(
    "damage",
    [
        "truncated",
        "model_bit",
        "false_flag",
        "candidate",
        "solve_time",
        "cleanup",
        "phase_cost",
    ],
)
def test_adversarial_mutation_rejected_even_with_recomputed_sidecars(evidence, damage):
    row, manifest, paths, directory, attempt = evidence
    att = read_checked_json(attempt / "attempt.json")
    if damage in ("truncated", "model_bit"):
        p = attempt / "solver_stdout.txt"
        text = p.read_text()
        if damage == "truncated":
            text = text.rsplit(" 0", 1)[0] + "\n"
        else:
            lines = text.splitlines()
            words = lines[1].split()
            words[1] = str(-int(words[1]))
            text = lines[0] + "\n" + " ".join(words) + "\n"
        p.write_text(text)
        att["stdout_sha256"] = sha256_file(p)
    elif damage == "false_flag":
        att["verified_fresh"] = "true"
    elif damage == "candidate":
        att["candidate"] = [att["candidate"][0] ^ 1]
    elif damage == "solve_time":
        att["solve_wall_seconds"] = 0.0
    elif damage == "cleanup":
        row["guardian_receipt"]["process_group_absent"] = False
        atomic_json(directory / "guardian_receipt.json", row["guardian_receipt"])
    else:
        row["phase_accounting"]["completed_seconds"]["stage1_ranking"] = 0.0
    atomic_json(attempt / "attempt.json", att)
    atomic_json(directory / "supervisor_result.json", row)
    with pytest.raises(ValueError):
        audit.validate_cell(row, manifest, paths)


def test_deadline_overrides_a_late_raw_success(evidence):
    row, manifest, paths, directory, _ = evidence
    row.update(
        outcome="global_deadline",
        observed_monotonic=1300.01,
        parent_wall_seconds=300.01,
    )
    row["guardian_receipt"].update(
        status="deadline_enforced",
        observed_monotonic=1300.01,
        cleanup_completed_monotonic=1300.11,
        worker_returncode=-9,
    )
    row["cleanup_wall_seconds"] = 1300.11 - 1300.01
    atomic_json(directory / "guardian_receipt.json", row["guardian_receipt"])
    atomic_json(directory / "supervisor_result.json", row)
    checked = audit.validate_cell(row, manifest, paths)
    assert checked["recovered_key"] is not None
    assert row["outcome"] == "global_deadline"


def set_deadline(row, directory):
    row.update(
        outcome="global_deadline",
        observed_monotonic=1300.01,
        parent_wall_seconds=300.01,
        worker_result=None,
    )
    row["guardian_receipt"].update(
        status="deadline_enforced",
        observed_monotonic=1300.01,
        cleanup_completed_monotonic=1300.11,
        worker_returncode=-9,
    )
    row["cleanup_wall_seconds"] = 1300.11 - 1300.01
    (directory / "worker_result.json").unlink()
    atomic_json(directory / "guardian_receipt.json", row["guardian_receipt"])


@pytest.mark.parametrize("damage", ["missing_start", "partial_formula", "torn_start"])
def test_deadline_during_formula_io_retains_a_negative_cell(evidence, damage):
    row, manifest, paths, directory, attempt = evidence
    set_deadline(row, directory)
    for name in ("attempt.json", "native_process.json", "native_launch_receipt.json"):
        (attempt / name).unlink()
    (attempt / "input.dimacs").write_text("p cnf")
    if damage == "missing_start":
        (attempt / "attempt_start.json").unlink()
    elif damage == "torn_start":
        (attempt / "attempt_start.json").write_text('{"partial":')
    atomic_json(directory / "supervisor_result.json", row)
    checked = audit.validate_cell(row, manifest, paths)
    assert checked["completed_attempts"] == 0
    assert checked["recovered_key"] is None
    assert (attempt / "input.dimacs").read_text() == "p cnf"


@pytest.mark.parametrize("journal_damage", ["absent", "torn"])
def test_startup_deadline_without_committed_component_journal(evidence, journal_damage):
    row, manifest, paths, directory, _ = evidence
    set_deadline(row, directory)
    shutil.rmtree(directory / "attempts")
    for name in ("queue.json", "stage1.json", "python_worker_receipt.json"):
        (directory / name).unlink()
    journal = directory / "journal.json"
    if journal_damage == "absent":
        journal.unlink()
    else:
        journal.write_text('{"partial":')
        row["uncommitted_terminal_json"] = ["journal.json"]
    row["phase_accounting"] = None
    row["component_accounting_available"] = False
    atomic_json(directory / "supervisor_result.json", row)
    checked = audit.validate_cell(row, manifest, paths)
    assert checked["queue"] is None
    assert checked["completed_attempts"] == 0


def test_completed_cell_cannot_hide_a_corrupt_journal(evidence):
    row, manifest, paths, directory, _ = evidence
    (directory / "journal.json").write_text('{"partial":')
    with pytest.raises(ValueError):
        audit.validate_cell(row, manifest, paths)


def test_exit_policy_race_requires_proven_deadline_cleanup(evidence, monkeypatch):
    from neural_sat_speck.followups.stage2_receipts import _validate_policy

    row, manifest, paths, directory, _ = evidence
    set_deadline(row, directory)
    sample = row["environment_samples"][0]
    sample["worker_policy"] = {
        "pid": 12345,
        "scope": "process_priority_and_background_policy",
        "native_thread_qos": "unavailable",
        "supported": True,
        "observations": {
            "nice": {"value": -1, "errno": errno.ESRCH},
            "background_policy": {"value": 0, "errno": 0},
        },
        "accepted": False,
    }
    monkeypatch.setattr(audit, "_validate_policy", _validate_policy)
    atomic_json(directory / "environment_samples.json", row["environment_samples"])
    atomic_json(directory / "supervisor_result.json", row)
    assert audit.validate_cell(row, manifest, paths)["completed_attempts"] == 1
    row["guardian_receipt"]["process_group_absent"] = False
    atomic_json(directory / "guardian_receipt.json", row["guardian_receipt"])
    atomic_json(directory / "supervisor_result.json", row)
    with pytest.raises(ValueError, match="cleanup"):
        audit.validate_cell(row, manifest, paths)
