"""Adversarial contracts and actual guardian behavior on synthetic processes."""

import copy
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from neural_sat_speck.backend.guard import atomic_json, read_checked_json, sha256_file
from neural_sat_speck.followups import conditioning_control as control
from neural_sat_speck.followups.conditioning_runner import (
    controlled_environment,
    phase_accounting,
    require_pilot,
)


def test_schedule_is_paired_counterbalanced_and_fixed():
    cells = control.fixed_cells("full")
    assert len(cells) == len({c["cell_id"] for c in cells}) == 32
    for cipher, rounds in (("speck32_64", 7), ("simon32_64", 10)):
        subset = [c for c in cells if c["cipher"] == cipher]
        assert {c["rounds"] for c in subset} == {rounds}
        assert {c["key_index"] for c in subset} == set(range(8))
        first = [
            next(c for c in subset if c["key_index"] == i)["arm"] for i in range(8)
        ]
        assert first.count(control.ARMS[0]) == first.count(control.ARMS[1]) == 4
    assert cells[0]["cipher"] != cells[4]["cipher"]
    assert control.fixed_cells("pilot") == cells[:4]
    with pytest.raises(ValueError):
        control.fixed_cells("unseen")


@pytest.fixture
def manifest_fixture(tmp_path, monkeypatch):
    pytest.importorskip("pycryptosat")
    root = tmp_path / "snapshot"
    root.mkdir()
    names = [
        "src/toy.py",
        "scripts/prepare_conditioning_interface_control.py",
        "scripts/evaluate_conditioning_interface_control.py",
        "scripts/analyze_conditioning_interface_control.py",
        "scripts/run_conditioning_interface_campaign.py",
        "docs/CONDITIONING_INTERFACE_PROTOCOL_20261006.md",
    ]
    for name in names:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# synthetic fixture\n")
    cohort = tmp_path / "not_attack_data.json"
    atomic_json(cohort, {"fixture_only": True})
    monkeypatch.setattr(control, "COHORT_SHA256", sha256_file(cohort))
    monkeypatch.setattr(control, "source_commit", lambda _: "a" * 40)
    monkeypatch.setattr(control, "validate_source_root", lambda *_: None)
    solver = Path(importlib.util.find_spec("pycryptosat").origin)
    support = {name: tmp_path / name for name in control.runtime_support_paths()}
    for path in support.values():
        path.write_text("synthetic runtime identity\n")
    monkeypatch.setattr(control, "runtime_support_paths", lambda: support)
    public, truth = tmp_path / "public.json", tmp_path / "truth.json"
    atomic_json(public, {"fixture_only": True})
    atomic_json(truth, {"fixture_only": True})
    cells = control.fixed_cells("pilot")
    manifest = {
        "schema": "conditioning_control_manifest_v1",
        "status": "APPROVED",
        "author_decision": "D-054",
        "execution_authorized": True,
        "mode": "pilot",
        "source_root": str(root),
        "source_commit": "a" * 40,
        "profile": copy.deepcopy(control.PROFILE),
        "cells": cells,
        "campaign_ceiling_seconds": 1500.0,
        "python_version": sys.version,
        "solver_version": "5.14.7",
        "files": {
            "cohort": control.file_identity(cohort),
            "python": control.file_identity(Path(sys.executable)),
            "pycryptosat": control.file_identity(solver),
            **{name: control.file_identity(p) for name, p in support.items()},
            **{"source:" + n: control.file_identity(root / n) for n in names},
        },
        "cases": {
            c["case_id"]: {
                "public": control.file_identity(public),
                "truth": control.file_identity(truth),
                "formula_sha256": "b" * 64,
            }
            for c in cells
        },
        "pilot_ledger": None,
    }
    path = tmp_path / "manifest.json"
    atomic_json(path, manifest)
    return manifest, path


def test_manifest_positive_checks_without_loading_fixture_cohort(manifest_fixture):
    manifest, path = manifest_fixture
    checked, paths = control.load_manifest(path)
    assert checked == manifest
    assert paths["python"] == Path(sys.executable).absolute()


@pytest.mark.parametrize(
    "drift",
    [
        "seed",
        "deadline",
        "threads",
        "time_limit",
        "source_inventory",
        "cohort_hash",
        "case_duplication",
        "authorization",
        "solver_version",
        "source_escape",
    ],
)
def test_profile_drift_rejected_before_cohort_or_dispatch(manifest_fixture, drift):
    manifest, path = manifest_fixture
    if drift in {"seed", "deadline", "threads", "time_limit"}:
        field, value = {
            "seed": ("seed", 12345),
            "deadline": ("deadline_seconds", 301),
            "threads": ("solver_threads", 2),
            "time_limit": ("solver_time_limit_argument", 45),
        }[drift]
        manifest["profile"][field] = value
    elif drift == "source_inventory":
        manifest["files"].pop("source:src/toy.py")
    elif drift == "cohort_hash":
        manifest["files"]["cohort"]["sha256"] = "0" * 64
    elif drift == "case_duplication":
        manifest["cells"][1] = manifest["cells"][0]
    elif drift == "authorization":
        manifest["execution_authorized"] = "true"
    elif drift == "source_escape":
        manifest["files"]["source:src/toy.py"] = manifest["files"]["cohort"]
    else:
        manifest["solver_version"] = "other"
    atomic_json(path, manifest)
    with pytest.raises(ValueError):
        control.load_manifest(path)


def test_manifest_corruption_and_source_drift_rejected(manifest_fixture):
    manifest, path = manifest_fixture
    source = Path(manifest["files"]["source:src/toy.py"]["path"])
    source.write_text("# modified\n")
    with pytest.raises(ValueError, match="identity mismatch"):
        control.load_manifest(path)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        control.load_manifest(path)


def test_source_arms_have_identical_formula_identity():
    formula = {
        "num_vars": 2,
        "cnf_clauses": [[1]],
        "xor_clauses": [{"literals": [1, 2], "rhs": False}],
        "assumption_literals": [2],
        "varmap": {"master_key": [[1, 2]]},
        "arm": "permanent_units",
    }
    first = control.digest(control.formula_identity(formula))
    formula["arm"] = "temporary_assumptions"
    assert control.digest(control.formula_identity(formula)) == first
    formula["assumption_literals"] = [-2]
    assert control.digest(control.formula_identity(formula)) != first


def test_symlinked_source_directory_is_rejected_before_git_inspection(tmp_path):
    root, outside = tmp_path / "snapshot", tmp_path / "outside"
    (root / "src").mkdir(parents=True)
    outside.mkdir()
    (root / "src" / "untracked_package").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlinked source"):
        control.validate_source_root(root, [])


def test_interrupted_phase_time_is_preserved(tmp_path):
    atomic_json(
        tmp_path / "journal.json",
        {
            "completed_seconds": {"formula_construction": 0.3},
            "phase": "solving",
            "phase_started_monotonic": 100.0,
        },
    )
    assert phase_accounting(tmp_path, 399.0) == {
        "completed_seconds": {"formula_construction": 0.3},
        "inflight_phase": "solving",
        "inflight_seconds": 299.0,
    }
    with pytest.raises(ValueError):
        phase_accounting(tmp_path, 99.0)


def test_full_launch_requires_actual_checked_pilot(tmp_path):
    with pytest.raises(ValueError, match="lacks a pilot"):
        require_pilot({"pilot_ledger": None}, tmp_path)


def test_joint_order_is_crossed_and_loader_overrides_removed(tmp_path, monkeypatch):
    first = [control.fixed_cells("full")[i] for i in range(0, 32, 4)]
    assert {(c["cipher"], c["arm"]) for c in first} == {
        (cipher, arm) for cipher in ("speck32_64", "simon32_64") for arm in control.ARMS
    }
    monkeypatch.setenv("DYLD_LIBRARY_PATH", "/unapproved/library")
    monkeypatch.setenv("DYLD_INSERT_LIBRARIES", "/unapproved/injection")
    monkeypatch.setenv("LD_PRELOAD", "/unapproved/injection")
    monkeypatch.setenv("PYTHONHOME", "/unapproved/interpreter")
    monkeypatch.setenv("PYTHONTRACEMALLOC", "100")
    monkeypatch.setenv("PYTHONOPTIMIZE", "2")
    env = controlled_environment(tmp_path, tmp_path)
    assert not any(k.startswith(("DYLD_", "LD_")) for k in env)
    assert "PYTHONHOME" not in env
    assert {k for k in env if k.startswith("PYTHON")} == {
        "PYTHONPATH",
        "PYTHONNOUSERSITE",
    }
    assert env["PYTHONPATH"] == str(tmp_path / "src")
    assert env["PYTHONNOUSERSITE"] == "1"


@pytest.mark.parametrize(
    "sleep_seconds,cutoff,expected",
    [(0, 3.0, "worker_exit"), (5, 0.35, "deadline_enforced")],
)
def test_independent_guardian_deadline_and_cleanup(
    tmp_path, sleep_seconds, cutoff, expected
):
    root = Path(__file__).resolve().parents[2]
    directory = tmp_path / ("f" * 32)
    directory.mkdir()
    started = time.perf_counter()
    config = {
        "deadline_monotonic": started + cutoff,
        "invocation": directory.name,
        "directory": str(directory),
        "repo_root": str(root),
        "coordinator_pid": os.getpid(),
        "worker_arguments": [
            sys.executable,
            "-c",
            f"import time; time.sleep({sleep_seconds})",
        ],
        "receipt": str(directory / "guardian_receipt.json"),
    }
    atomic_json(directory / "guardian.json", config)
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "neural_sat_speck.followups.watchdog",
            str(directory / "guardian.json"),
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=8,
    )
    assert completed.returncode == 0, completed.stderr
    receipt = read_checked_json(directory / "guardian_receipt.json")
    assert receipt["status"] == expected
    assert receipt["process_group_absent"] is True
    assert receipt["worker_pid"] == receipt["process_group"]
    if expected == "deadline_enforced":
        assert receipt["observed_monotonic"] >= config["deadline_monotonic"]
    else:
        assert receipt["worker_returncode"] == 0
    with pytest.raises(ProcessLookupError):
        os.killpg(receipt["process_group"], 0)
