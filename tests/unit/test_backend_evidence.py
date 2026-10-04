"""Synthetic raw-receipt tests; no solver or protected data is used."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from neural_sat_speck.backend.evidence import audit_cell_evidence
from neural_sat_speck.backend.guard import atomic_json, fixed_grid
from neural_sat_speck.backend.solver_io import parse_solver_output, solver_arguments


def _seal_raw_artifacts(directory: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.iterdir()
        if path.is_file()
    }


def _fixture(
    tmp_path: Path,
    *,
    profile: str = "cms_native_default",
    outcome: str = "recovered",
    deadline: bool = False,
) -> tuple[dict, dict, dict, dict, Path]:
    cell = next(
        c for c in fixed_grid() if c["profile"] == profile and c["trailing_keys"] == 1
    )
    invocation = "a" * 32
    source_commit = "b" * 40
    started_utc = "2026-10-04T01:02:03+00:00"
    ledger_path = tmp_path / "ledger" / "cells.json"
    ledger_path.parent.mkdir(parents=True)
    worker_directory = (
        ledger_path.parent / "workers" / f"{cell['cell_id']}_{invocation}"
    )
    worker_directory.mkdir(parents=True)

    formula = "p cnf 2 1\n1 2 0\n"
    native_path = tmp_path / f"{cell['case_id']}.native.dimacs"
    cnf_path = tmp_path / f"{cell['case_id']}.cnf"
    native_path.write_text(formula)
    cnf_path.write_text(formula)
    varmap_path = tmp_path / f"{cell['case_id']}.varmap.json"
    varmap_path.write_text(
        json.dumps(
            {
                "mapping": {"master_key": [[1, 2]]},
                "native_num_vars": 2,
                "pure_cnf_num_vars": 2,
            }
        )
    )
    public_path = tmp_path / f"{cell['case_id']}.public.json"
    truth_path = tmp_path / f"{cell['case_id']}.truth.json"
    public_path.write_text("{}")
    truth_path.write_text("{}")
    paths = {
        "python": Path("/opt/fake/bin/python"),
        "cms": Path("/opt/fake/bin/cryptominisat5"),
        "cadical": Path("/opt/fake/bin/cadical"),
        f"{cell['case_id']}:native": native_path,
        f"{cell['case_id']}:cnf": cnf_path,
        f"{cell['case_id']}:varmap": varmap_path,
        f"{cell['case_id']}:public": public_path,
        f"{cell['case_id']}:truth": truth_path,
    }
    manifest = {
        "source_commit_at_preparation": source_commit,
        "deadline_seconds": 300.0,
        "cells": [cell],
    }
    start = {
        "cell_id": cell["cell_id"],
        "invocation": invocation,
        "started_utc": started_utc,
        "process_group": 12341,
    }

    worker_pid = 12341
    watchdog_pid = 12340
    deadline_monotonic = 400.0
    receipt_status = "deadline_enforced" if deadline else "worker_exit"
    observed_monotonic = 400.1 if deadline else 101.5
    guardian_receipt = {
        "watchdog_pid": watchdog_pid,
        "invocation": invocation,
        "deadline_monotonic": deadline_monotonic,
        "status": receipt_status,
        "observed_monotonic": observed_monotonic,
        "worker_pid": worker_pid,
        "process_group": worker_pid,
        "worker_returncode": -9 if deadline else 0,
        "cleanup_completed_monotonic": observed_monotonic + 0.1,
        "process_group_absent": True,
    }
    worker_directory_text = str(worker_directory)
    args = None
    native_receipt = None
    parsed = None
    solver_returncode = None
    if profile != "cms_wrapper":
        input_name = (
            "input.cnf"
            if profile in {"cms_cnf_no_gauss", "cadical_cnf"}
            else "input.dimacs"
        )
        input_path = worker_directory / input_name
        input_path.write_text(formula)
        args = solver_arguments(
            profile,
            str(paths["cms"]),
            str(paths["cadical"]),
            str(input_path),
        )
        solver_stdout = (
            "s UNSATISFIABLE\n"
            if outcome == "unsat_alarm"
            else "s SATISFIABLE\nv 1 2 0\n"
        )
        solver_returncode = 20 if outcome == "unsat_alarm" else 10
        parsed = parse_solver_output(
            "cadical" if profile == "cadical_cnf" else "cms",
            solver_returncode,
            solver_stdout,
            required_variables=[1, 2],
            num_vars=2,
        )
        native_receipt = {
            "pid": 23456,
            "git_commit": source_commit,
            "is_certified": True,
            "solver_arguments": args,
        }
        atomic_json(
            worker_directory / "native_launch.json",
            {
                "repo_root": "/fake/repo",
                "arguments": args,
                "receipt": str(worker_directory / "native_launch_receipt.json"),
            },
        )
        atomic_json(worker_directory / "native_launch_receipt.json", native_receipt)
        atomic_json(
            worker_directory / "native_process.json",
            {"pid": 23456, "process_group": worker_pid, "argv": args},
        )
        (worker_directory / "solver_stdout.txt").write_text(solver_stdout)
        (worker_directory / "solver_stderr.txt").write_text("")

    worker_outcome = "recovered" if deadline else outcome
    worker_result = {
        **cell,
        "outcome": worker_outcome,
        "verified_known": worker_outcome == "recovered",
        "verified_fresh": worker_outcome == "recovered",
        "assigned_key_match": worker_outcome == "recovered",
        "recovered_key": [1, 2, 3, 4] if worker_outcome == "recovered" else None,
        "solver_status": (
            parsed["status"]
            if parsed is not None
            else "sat"
            if worker_outcome == "recovered"
            else "unsat"
        ),
        "completed_phase_seconds": {"solving": 0.5},
        "worker_seconds": 1.0,
    }
    if profile != "cms_wrapper":
        worker_result.update(
            solver_returncode=solver_returncode,
            solver_arguments=args,
            native_launch_receipt=native_receipt,
            gauss_evidence=parsed["gauss_evidence"],
            native_num_vars=2,
            submitted_num_vars=2,
        )
    atomic_json(worker_directory / "result.json", worker_result)
    atomic_json(
        worker_directory / "python_worker_receipt.json",
        {"pid": worker_pid, "git_commit": source_commit, "is_certified": True},
    )
    atomic_json(
        worker_directory / "process.json",
        {
            "pid": worker_pid,
            "process_group": worker_pid,
            "invocation": invocation,
            "supervisor_pid": watchdog_pid,
        },
    )
    atomic_json(
        worker_directory / "guardian.json",
        {
            "deadline_monotonic": deadline_monotonic,
            "invocation": invocation,
            "directory": worker_directory_text,
            "repo_root": "/fake/repo",
            "worker_arguments": [
                str(paths["python"]),
                "-m",
                "neural_sat_speck.backend.worker",
                str(worker_directory / "payload.json"),
            ],
            "receipt": str(worker_directory / "guardian_receipt.json"),
        },
    )
    atomic_json(worker_directory / "guardian_receipt.json", guardian_receipt)
    atomic_json(
        worker_directory / "payload.json",
        {
            "cell": cell,
            "directory": worker_directory_text,
            "invocation": invocation,
            "started_utc": started_utc,
            "record_process": str(ledger_path),
            "source_commit": source_commit,
            "cms_path": str(paths["cms"]),
            "cadical_path": str(paths["cadical"]),
            "python_path": str(paths["python"]),
            "public_path": str(public_path),
            "truth_path": str(truth_path),
            "native_sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
            "cnf_sha256": hashlib.sha256(cnf_path.read_bytes()).hexdigest(),
        },
    )

    if deadline:
        record = {
            **cell,
            "outcome": "global_deadline",
            "online_seconds": 300.1,
            "verified_known": False,
            "verified_fresh": False,
            "assigned_key_match": False,
            "recovered_key": None,
            "solver_status": None,
        }
    else:
        record = {
            **worker_result,
            "online_seconds": 1.5,
        }
    record.update(
        {
            "worker_directory": worker_directory_text,
            "started_utc": started_utc,
            "completed_utc": "2026-10-04T01:02:04+00:00",
            "invocation": invocation,
            "process_group_absent": True,
            "phase_accounting": {
                "completed_seconds": {"solving": 0.5},
                "active_phase": None,
                "inflight_seconds": 0.0,
                "journal_available": True,
            },
            "guardian_returncode": 0,
            "guardian_healthy": True,
            "guardian_receipt": guardian_receipt,
        }
    )
    if deadline:
        record.update(
            {
                "cleanup_seconds": 0.1,
                "solver_status": None,
            }
        )
    elif profile != "cms_wrapper":
        record["solver_arguments"] = args
        record["solver_returncode"] = solver_returncode
        record["native_launch_receipt"] = native_receipt
        record["gauss_evidence"] = parsed["gauss_evidence"]

    atomic_json(worker_directory / "coordinator_received.json", record)
    record["pre_dispatch_seconds"] = 0.01
    record["post_dispatch_audit_seconds"] = 0.01
    record["raw_artifacts"] = _seal_raw_artifacts(worker_directory)
    return record, start, manifest, paths, ledger_path


def test_audit_reparses_complete_native_sat_model(tmp_path):
    record, start, manifest, paths, ledger_path = _fixture(tmp_path)
    audit = audit_cell_evidence(record, start, manifest, paths, ledger_path)
    assert audit["scope"] == "complete_raw_worker_audit"
    assert audit["solver_evidence"]["status"] == "sat"
    assert audit["solver_evidence"]["assigned_variable_count"] == 2
    assert audit["solver_evidence"]["model_complete_for_submitted_variables"] is True
    assert audit["solver_evidence"]["formula_satisfaction_rechecked"] is False
    assert audit["timing"]["online_seconds"] == 1.5


def test_audit_accepts_native_unsat_as_recorded_alarm(tmp_path):
    record, start, manifest, paths, ledger_path = _fixture(
        tmp_path, outcome="unsat_alarm"
    )
    audit = audit_cell_evidence(record, start, manifest, paths, ledger_path)
    assert audit["solver_evidence"]["status"] == "unsat"
    assert audit["solver_evidence"]["model_claimed"] is False


def test_wrapper_audit_does_not_claim_a_captured_native_model(tmp_path):
    record, start, manifest, paths, ledger_path = _fixture(
        tmp_path, profile="cms_wrapper"
    )
    audit = audit_cell_evidence(record, start, manifest, paths, ledger_path)
    assert audit["solver_evidence"]["status"] == "sat"
    assert audit["solver_evidence"]["native_model_captured"] is False


def test_late_worker_recovery_is_preserved_but_not_credited(tmp_path):
    record, start, manifest, paths, ledger_path = _fixture(tmp_path, deadline=True)
    audit = audit_cell_evidence(record, start, manifest, paths, ledger_path)
    assert audit["outcome"] == "global_deadline"
    assert audit["late_worker_result"] == {
        "present": True,
        "outcome": "recovered",
        "recovery_claim_credited": False,
    }
    assert audit["solver_evidence"]["recovery_claim_credited"] is False
    assert record["recovered_key"] is None
    assert audit["timing"]["online_seconds"] >= 300


def test_interruption_reports_unavailable_time_without_normal_receipts(tmp_path):
    cell = fixed_grid()[0]
    invocation = "c" * 32
    record = {
        **cell,
        "outcome": "infrastructure_interrupted",
        "online_seconds": 0.0,
        "duration_available": False,
        "verified_known": False,
        "verified_fresh": False,
        "assigned_key_match": False,
        "recovered_key": None,
        "solver_status": None,
        "invocation": invocation,
        "interruption_review": "prior_process_group_absent",
    }
    start = {
        "cell_id": cell["cell_id"],
        "invocation": invocation,
        "process_group": 23456,
    }
    audit = audit_cell_evidence(
        record, start, {"cells": [cell]}, {}, tmp_path / "ledger.json"
    )
    assert audit["scope"] == "interrupted_start_only"
    assert audit["timing"]["duration_available"] is False
    assert audit["timing"]["online_seconds"] is None
    assert audit["solver_evidence"]["model_claimed"] is False


def test_coordinator_record_mutation_is_rejected(tmp_path):
    record, start, manifest, paths, ledger_path = _fixture(tmp_path)
    record["recovered_key"] = [1, 2, 3, 5]
    with pytest.raises(ValueError, match="coordinator_received"):
        audit_cell_evidence(record, start, manifest, paths, ledger_path)


def test_nonzero_worker_exit_receipt_is_rejected_even_when_resealed(tmp_path):
    record, start, manifest, paths, ledger_path = _fixture(tmp_path)
    worker_directory = Path(record["worker_directory"])
    receipt = dict(record["guardian_receipt"])
    receipt["worker_returncode"] = 7
    atomic_json(worker_directory / "guardian_receipt.json", receipt)
    coordinator = json.loads(
        (worker_directory / "coordinator_received.json").read_text()
    )
    coordinator["guardian_receipt"] = receipt
    atomic_json(worker_directory / "coordinator_received.json", coordinator)
    record["guardian_receipt"] = receipt
    record["raw_artifacts"] = _seal_raw_artifacts(worker_directory)

    with pytest.raises(ValueError, match="nonzero child return code"):
        audit_cell_evidence(record, start, manifest, paths, ledger_path)


def test_unsealed_or_tampered_raw_solver_output_is_rejected(tmp_path):
    record, start, manifest, paths, ledger_path = _fixture(tmp_path)
    worker_directory = Path(record["worker_directory"])
    (worker_directory / "solver_stdout.txt").write_text("s UNSATISFIABLE\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        audit_cell_evidence(record, start, manifest, paths, ledger_path)


@pytest.mark.parametrize("absence", [False, None, "true"])
def test_guardian_must_itself_certify_group_absence(tmp_path, absence):
    record, start, manifest, paths, ledger_path = _fixture(tmp_path)
    directory = Path(record["worker_directory"])
    receipt = dict(record["guardian_receipt"])
    receipt["process_group_absent"] = absence
    atomic_json(directory / "guardian_receipt.json", receipt)
    coordinator = json.loads((directory / "coordinator_received.json").read_text())
    coordinator["guardian_receipt"] = receipt
    atomic_json(directory / "coordinator_received.json", coordinator)
    record["guardian_receipt"] = receipt
    record["raw_artifacts"] = _seal_raw_artifacts(directory)
    with pytest.raises(ValueError, match="certify guardian cleanup"):
        audit_cell_evidence(record, start, manifest, paths, ledger_path)


def test_resealed_solver_output_must_still_match_worker_status(tmp_path):
    record, start, manifest, paths, ledger_path = _fixture(tmp_path)
    worker_directory = Path(record["worker_directory"])
    (worker_directory / "solver_stdout.txt").write_text("s UNSATISFIABLE\n")
    record["raw_artifacts"] = _seal_raw_artifacts(worker_directory)
    with pytest.raises(ValueError, match="stdout/exit code"):
        audit_cell_evidence(record, start, manifest, paths, ledger_path)


def test_raw_worker_key_must_match_coordinator_result(tmp_path):
    record, start, manifest, paths, ledger_path = _fixture(tmp_path)
    worker_directory = Path(record["worker_directory"])
    worker_result = json.loads((worker_directory / "result.json").read_text())
    worker_result["recovered_key"] = [1, 2, 3, 5]
    atomic_json(worker_directory / "result.json", worker_result)
    record["raw_artifacts"] = _seal_raw_artifacts(worker_directory)
    with pytest.raises(ValueError, match="raw worker result"):
        audit_cell_evidence(record, start, manifest, paths, ledger_path)


def test_worker_directory_cannot_be_redirected_or_symlinked(tmp_path):
    record, start, manifest, paths, ledger_path = _fixture(tmp_path)
    record["worker_directory"] = str(tmp_path / "outside")
    with pytest.raises(ValueError, match="canonical"):
        audit_cell_evidence(record, start, manifest, paths, ledger_path)

    record, start, manifest, paths, ledger_path = _fixture(tmp_path / "second")
    worker_directory = Path(record["worker_directory"])
    target = worker_directory / "solver_stdout.txt"
    copy = worker_directory / "output-copy.txt"
    copy.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(copy)
    record["raw_artifacts"] = _seal_raw_artifacts(worker_directory)
    with pytest.raises(ValueError, match="symlink"):
        audit_cell_evidence(record, start, manifest, paths, ledger_path)


def test_raw_artifact_paths_must_be_basenames(tmp_path):
    record, start, manifest, paths, ledger_path = _fixture(tmp_path)
    record["raw_artifacts"]["../escape"] = "0" * 64
    with pytest.raises(ValueError, match="basename"):
        audit_cell_evidence(record, start, manifest, paths, ledger_path)


def test_interrupted_record_must_preserve_original_group_and_invocation(tmp_path):
    cell = fixed_grid()[0]
    invocation = "c" * 32
    record = {
        **cell,
        "outcome": "infrastructure_interrupted",
        "online_seconds": 0.0,
        "duration_available": False,
        "verified_known": False,
        "verified_fresh": False,
        "assigned_key_match": False,
        "recovered_key": None,
        "solver_status": None,
        "invocation": invocation,
        "interruption_review": "prior_process_group_absent",
    }
    start = {
        "cell_id": cell["cell_id"],
        "invocation": invocation,
        "process_group": None,
    }
    with pytest.raises(ValueError, match="durable start/group"):
        audit_cell_evidence(
            record, start, {"cells": [cell]}, {}, tmp_path / "ledger.json"
        )
