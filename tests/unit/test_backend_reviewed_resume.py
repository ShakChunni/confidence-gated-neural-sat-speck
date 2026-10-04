"""Adversarial AC-resume invariants; no solver or project cohort is loaded."""

from __future__ import annotations

import copy
from contextlib import nullcontext

import pytest

from neural_sat_speck.backend import reviewed_resume as resume
from neural_sat_speck.backend import runner
from neural_sat_speck.backend.guard import atomic_json, fixed_grid, sha256_file


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    # Full worktree/file guards are tested separately; isolate the resume plan.
    monkeypatch.setattr(resume, "validate_manifest", lambda *args: {"old": tmp_path})
    monkeypatch.setattr(resume, "validate_authorization", lambda *args, **kw: None)
    output = tmp_path / "ledger.json"
    binary = tmp_path / "binary"
    binary.write_text("fixed runtime")
    ident = {"path": str(binary), "sha256": sha256_file(binary)}
    cells = fixed_grid()
    old = {
        "files": {
            "python": copy.deepcopy(ident),
            "source:src/neural_sat_speck/backend/worker.py": copy.deepcopy(ident),
        },
        "cells": cells,
    }
    old_path = tmp_path / "original_manifest.json"
    auth = tmp_path / "original_authorization.json"
    atomic_json(old_path, old)
    atomic_json(auth, {})
    rows, starts = {}, {}
    for i, cell in enumerate(cells[:2]):
        row = {
            **cell,
            "outcome": "global_deadline" if i == 0 else "environment_rejected",
            "online_seconds": 300.01 if i == 0 else 90.0,
            "verified_known": False,
            "verified_fresh": False,
            "assigned_key_match": False,
            "recovered_key": None,
            "invocation": str(i) * 32,
        }
        rows[cell["cell_id"]] = row
        starts[cell["cell_id"]] = {
            "cell_id": cell["cell_id"],
            "invocation": row["invocation"],
            "process_group": 12340 + i,
        }
    stop = cells[1]["cell_id"]
    directory = tmp_path / "workers" / "stop"
    directory.mkdir(parents=True)
    power_path = directory / "power_latest.json"
    power = {
        "accepted": False,
        "ac_power_observed": False,
        "sleep_assertion_observed": True,
        "assertions_returncode": 0,
        "power_returncode": 0,
        "raw_power": "Now drawing from 'Battery Power'",
    }
    atomic_json(power_path, power)
    rows[stop].update(
        error="Observed power/sleep policy violation",
        guardian_healthy=True,
        process_group_absent=True,
        worker_directory=str(directory),
        raw_artifacts={power_path.name: sha256_file(power_path)},
    )
    archive = {
        "schema": "backend_ledger_v1",
        "status": "safety_stop",
        "manifest_sha256": sha256_file(old_path),
        "starts": starts,
        "records": rows,
    }
    archive_path = tmp_path / "archived.json"
    atomic_json(archive_path, archive)
    manifest = copy.deepcopy(old)
    manifest["reviewed_power_resume"] = {
        "schema": "backend_ac_resume_v1",
        "status": "AUTHOR_APPROVED",
        "decision": "D-045",
        "output_path": str(output),
        "prior_source_root": str(tmp_path),
        "prior_manifest": {"path": str(old_path), "sha256": sha256_file(old_path)},
        "prior_authorization": {"path": str(auth), "sha256": sha256_file(auth)},
        "stopped_ledger": {
            "path": str(archive_path),
            "sha256": sha256_file(archive_path),
        },
        "power_stop_cells": [stop],
    }
    return tmp_path, output, manifest, archive, archive_path, power_path, power


def reseal_archive(f):
    _, _, manifest, archive, path, _, _ = f
    atomic_json(path, archive)
    manifest["reviewed_power_resume"]["stopped_ledger"]["sha256"] = sha256_file(path)


def test_transition_preserves_records_and_old_source_provenance(fixture):
    root, output, manifest, archive, *_ = fixture
    prefix = resume.load_prefix(manifest, root, output)
    before = copy.deepcopy(archive)
    transitioned = resume.transition_identity(archive, prefix, "new_manifest_sha")
    assert transitioned["records"] == before["records"]
    assert transitioned["starts"] == before["starts"]
    assert archive == before
    assert transitioned["manifest_sha256"] == "new_manifest_sha"
    resume.validate_preserved_prefix(transitioned, prefix)
    assert resume.original_stop_only(transitioned, prefix)
    old_id = next(iter(archive["records"]))
    assert resume.provenance(old_id, manifest, {}, prefix)[0] == prefix["manifest"]
    assert (
        resume.provenance(fixed_grid()[2]["cell_id"], manifest, {}, prefix)[0]
        is manifest
    )


@pytest.mark.parametrize("section", ["records", "starts"])
def test_old_cells_cannot_be_mutated_or_deleted(fixture, section):
    root, output, manifest, archive, *_ = fixture
    prefix = resume.load_prefix(manifest, root, output)
    changed = copy.deepcopy(archive)
    ident = next(iter(changed[section]))
    changed[section][ident]["invocation"] = "f" * 32
    with pytest.raises(ValueError, match="changed a prior"):
        resume.validate_preserved_prefix(changed, prefix)
    changed[section].pop(ident)
    with pytest.raises(ValueError):
        resume.transition_identity(changed, prefix, "new_sha")


def test_review_does_not_clear_a_later_safety_stop(fixture):
    root, output, manifest, archive, *_ = fixture
    prefix = resume.load_prefix(manifest, root, output)
    changed = copy.deepcopy(archive)
    cell = fixed_grid()[2]
    changed["starts"][cell["cell_id"]] = {"invocation": "x" * 32}
    assert not resume.original_stop_only(changed, prefix)
    assert not resume.original_stop_only(archive, None)
    changed = copy.deepcopy(archive)
    changed["environment_stops"] = [{"accepted": False, "ac_power_observed": False}]
    assert not resume.original_stop_only(changed, prefix)


@pytest.mark.parametrize(
    "field",
    [
        "decision",
        "stopped_ledger_sha256",
        "prior_manifest_sha256",
        "continued_manifest_sha256",
        "retained_power_stop_cells",
    ],
)
def test_resume_review_history_must_bind_all_approved_identities(fixture, field):
    root, output, manifest, archive, *_ = fixture
    prefix = resume.load_prefix(manifest, root, output)
    changed = resume.transition_identity(archive, prefix, "new_sha")
    resume.validate_review_history(changed, prefix, "new_sha")
    changed["resume_reviews"][0][field] = "forged"
    with pytest.raises(ValueError, match="review history"):
        resume.validate_review_history(changed, prefix, "new_sha")


def test_restored_review_never_dispatches_while_current_power_is_rejected(
    fixture, monkeypatch
):
    root, output, manifest, archive, *_ = fixture
    prefix = resume.load_prefix(manifest, root, output)
    manifest_path = root / "continued_manifest.json"
    authorization_path = root / "continued_authorization.json"
    cohort = root / "cohort.json"
    atomic_json(manifest_path, manifest)
    atomic_json(authorization_path, {})
    atomic_json(cohort, {})
    atomic_json(output, archive)
    monkeypatch.setattr(runner, "validate_manifest", lambda *args: {"cohort": cohort})
    monkeypatch.setattr(runner, "validate_authorization", lambda *args, **kwargs: None)
    monkeypatch.setattr(resume, "load_prefix", lambda *args: prefix)
    monkeypatch.setattr(
        "neural_sat_speck.backend.artifacts.validate_cohort", lambda *args: None
    )
    monkeypatch.setattr(
        "neural_sat_speck.backend.artifacts.validate_case_inputs", lambda *args: None
    )
    monkeypatch.setattr(
        "neural_sat_speck.backend.evidence.audit_cell_evidence",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(runner, "sleep_assertion", lambda: nullcontext({"test": True}))
    monkeypatch.setattr(runner, "_group_absent", lambda group: True)
    monkeypatch.setattr(runner, "power_sample", lambda: {"accepted": False})
    result = runner.run(
        manifest_path,
        output,
        authorization_path=authorization_path,
        resume=True,
        dispatch=lambda *args: pytest.fail("Dispatched while current AC policy failed"),
    )
    assert result["status"] == "safety_stop"
    assert result["records"] == archive["records"]
    assert result["starts"] == archive["starts"]


@pytest.mark.parametrize(
    "defect",
    [
        "unapproved",
        "output",
        "empty_review",
        "new_identity",
        "runtime",
        "worker",
        "other_alarm",
        "incomplete",
        "schedule",
        "unsealed",
    ],
)
def test_rejects_unrelated_or_laundered_resume(fixture, defect):
    root, output, manifest, archive, path, power_path, _ = fixture
    plan = manifest["reviewed_power_resume"]
    if defect == "unapproved":
        plan["status"] = "PROPOSED"
    elif defect == "output":
        plan["output_path"] = str(output) + ".other"
    elif defect == "empty_review":
        plan["power_stop_cells"] = []
    elif defect == "new_identity":
        manifest["files"]["unexpected"] = manifest["files"]["python"]
    elif defect in {"runtime", "worker"}:
        key = (
            "python"
            if defect == "runtime"
            else "source:src/neural_sat_speck/backend/worker.py"
        )
        manifest["files"][key]["sha256"] = "a" * 64
    elif defect == "other_alarm":
        archive["records"][plan["power_stop_cells"][0]]["outcome"] = "solver_error"
        reseal_archive(fixture)
    elif defect == "incomplete":
        archive["starts"].pop(next(iter(archive["starts"])))
        reseal_archive(fixture)
    elif defect == "schedule":
        archive["records"]["alien"] = archive["records"].pop(
            next(iter(archive["records"]))
        )
        archive["starts"]["alien"] = archive["starts"].pop(
            next(iter(archive["starts"]))
        )
        reseal_archive(fixture)
    else:
        power_path.write_text("corrupted")
    with pytest.raises(ValueError):
        resume.load_prefix(manifest, root, output)


@pytest.mark.parametrize(
    "key,value",
    [
        ("ac_power_observed", True),
        ("ac_power_observed", "false"),
        ("sleep_assertion_observed", False),
        ("power_returncode", 1),
        ("raw_power", "AC Power"),
        ("accepted", True),
    ],
)
def test_rehashed_power_assertion_cannot_forge_ac_loss(fixture, key, value):
    root, output, manifest, archive, _, power_path, power = fixture
    power[key] = value
    atomic_json(power_path, power)
    stop = manifest["reviewed_power_resume"]["power_stop_cells"][0]
    archive["records"][stop]["raw_artifacts"][power_path.name] = sha256_file(power_path)
    reseal_archive(fixture)
    with pytest.raises(ValueError, match="establish.*AC loss"):
        resume.load_prefix(manifest, root, output)
