"""Synthetic D-057 continuation contracts; no opened keys or rankings."""

from __future__ import annotations

import copy
import hashlib
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from neural_sat_speck.backend.guard import atomic_json, read_checked_json, sha256_file
from neural_sat_speck.followups import ranking_depth_control as control
from neural_sat_speck.followups import ranking_depth_recovery as recovery
from neural_sat_speck.followups.ranking_depth_control import file_identity
from neural_sat_speck.followups.ranking_depth_runner import stat_identity


def _synthetic_prefix(tmp_path):
    original_root = (tmp_path / recovery.ORIGINAL_DIRECTORY).resolve()
    cells = control.fixed_cells("full", num_keys=32)
    starts, records, order = {}, {}, []
    for index, cell in enumerate(cells[: recovery.ORIGINAL_STARTS]):
        cell_id = cell["cell_id"]
        token = f"{index + 1:032x}"
        directory = original_root / "execution" / "cells" / cell_id / token
        directory.mkdir(parents=True)
        order.append(cell_id)
        starts[cell_id] = {
            "directory": str(directory),
            "invocation_id": token,
            "utc": "synthetic",
        }
        receipt = {
            "invocation": token,
            "process_group_absent": index < recovery.ORIGINAL_TERMINAL_RECORDS,
            "status": "worker_exit"
            if index < recovery.ORIGINAL_TERMINAL_RECORDS
            else "guardian_error",
            "worker_returncode": 0
            if index < recovery.ORIGINAL_TERMINAL_RECORDS
            else -9,
        }
        guardian = {
            "invocation": token,
            "directory": str(directory),
            "receipt": str(directory / "guardian_receipt.json"),
        }
        atomic_json(directory / "guardian.json", guardian)
        atomic_json(directory / "guardian_receipt.json", receipt)
        if index < recovery.ORIGINAL_TERMINAL_RECORDS:
            row = {
                "cell": cell,
                "directory": str(directory),
                "guardian_receipt": receipt,
                "opaque_worker_payload": {"synthetic": index},
            }
            atomic_json(directory / "supervisor_result.json", row)
            records[cell_id] = row
        else:
            atomic_json(
                directory / "payload.json",
                {
                    "cell": cell,
                    "invocation_id": token,
                    "directory": str(directory),
                    "source_commit": recovery.ORIGINAL_SOURCE_COMMIT,
                },
            )
    ledger = {
        "status": "INTERRUPTED",
        "execution_order": order,
        "starts": starts,
        "records": records,
    }
    return original_root, cells, ledger


def _integration_stubs(tmp_path, monkeypatch):
    original_root, cells, _ledger = _synthetic_prefix(tmp_path)
    output_root = tmp_path / "continuation"
    source_root = tmp_path / "source"
    (source_root / "scripts").mkdir(parents=True)
    script = source_root / recovery.CONTINUATION_SCRIPT
    script.write_text("synthetic continuation script\n")
    script_pin = {
        "path": str(script),
        "sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "identity": stat_identity(script),
    }
    profile = {"synthetic_profile_id": "frozen"}
    cases = {}
    for cell in cells:
        case_id = cell["case_id"]
        cases.setdefault(
            case_id,
            {
                "public": {"path": f"/opaque/{case_id}.public", "sha256": "a" * 64},
                "verification": {
                    "path": f"/opaque/{case_id}.verification",
                    "sha256": "b" * 64,
                },
                "truth": {"path": f"/opaque/{case_id}.truth", "sha256": "c" * 64},
                "base_formula_sha256": "d" * 64,
            },
        )
    original = {
        "schema": "backend_ranking_depth_manifest_v1",
        "status": "APPROVED",
        "author_decision": "D-056",
        "execution_authorized": True,
        "mode": "full",
        "profile": profile,
        "source_root": "/synthetic/original-source",
        "source_commit": recovery.ORIGINAL_SOURCE_COMMIT,
        "campaign_ceiling_seconds": recovery.ORIGINAL_CAMPAIGN_CEILING_SECONDS,
        "files": {},
        "cases": cases,
        "cells": cells,
    }
    prefix_binding = {
        "original_root": str(original_root),
        "manifest_path": str(original_root / "inputs/manifest.json"),
        "manifest_sha256": recovery.ORIGINAL_MANIFEST_SHA256,
        "ledger_path": str(original_root / "execution/ledger.json"),
        "ledger_sha256": "e" * 64,
        "launch_receipt_path": str(tmp_path / "launch.json"),
        "launch_receipt_sha256": recovery.ORIGINAL_LAUNCH_RECEIPT_SHA256,
        "launch_active_elapsed_seconds": 31_045.115611666995,
        "remaining_original_ceiling_seconds": 58_954.884388333005,
        "online_allowance_seconds": 34_500.0,
        "start_ordinal": 142,
        "completed_prefix_count": 141,
        "started_prefix_count": 142,
        "execution_order_prefix_sha256": "f" * 64,
        "failed_invocation": {"cell_id": recovery.REPLACEMENT_CELL_ID},
    }
    manifest_path = original_root / "inputs/manifest.json"
    new_source_binding = {
        "root": str(source_root),
        "commit": "c" * 40,
        "source_changes": [],
        "recovery_module": {"path": "synthetic"},
        "continuation_script": script_pin,
    }
    files = {
        "python": {"path": "/synthetic/python"},
        "tf_python": {"path": "/synthetic/tf-python"},
        "cms": {"path": "/synthetic/cms"},
        "bao_vv9": {"path": "/synthetic/bao"},
        "simon_ddt": {"path": "/synthetic/ddt"},
        "speck_stage1": {"path": "/synthetic/stage1"},
        "speck_classical": {"path": "/synthetic/classical"},
    }
    paths = {name: Path(pin["path"]) for name, pin in files.items()}
    for case_id in cases:
        for role in ("public", "verification", "truth"):
            paths[case_id + ":" + role] = Path(cases[case_id][role]["path"])

    monkeypatch.setattr(
        recovery,
        "_load_original",
        lambda _root, _sha: (
            original,
            _ledger,
            {},
            manifest_path,
            tmp_path,
            prefix_binding,
        ),
    )
    monkeypatch.setattr(
        recovery,
        "_validate_source_rebase",
        lambda _manifest, _root: (files, new_source_binding, "c" * 40),
    )
    monkeypatch.setattr(recovery, "same_python_environment", lambda *_: True)
    monkeypatch.setattr(
        recovery,
        "load_manifest",
        lambda path, hashes=True: (read_checked_json(path), paths),
    )
    monkeypatch.setattr(
        recovery, "sleep_assertion", lambda: nullcontext({"accepted": True})
    )

    import neural_sat_speck.followups.ranking_depth_audit as audit
    import neural_sat_speck.followups.ranking_depth_runner as runner

    monkeypatch.setattr(runner, "check_dispatch_identity", lambda _manifest: None)
    monkeypatch.setattr(
        runner,
        "controlled_environment",
        lambda *_: {"PYTHONPATH": "/synthetic/source/src"},
    )
    validations = []
    monkeypatch.setattr(
        audit,
        "validate_cell",
        lambda row, _manifest, _paths: validations.append(row["cell"]["cell_id"]),
    )
    dispatched = []

    def synthetic_run_cell(payload, _environment):
        dispatched.append(copy.deepcopy(payload))
        directory = Path(payload["directory"])
        directory.mkdir(parents=True)
        receipt = {"process_group_absent": True, "invocation": payload["invocation_id"]}
        row = {
            "cell": payload["cell"],
            "directory": str(directory),
            "outcome": "queue_exhausted",
            "guardian_receipt": receipt,
            "parent_wall_seconds": 0.1,
        }
        atomic_json(directory / "guardian_receipt.json", receipt)
        atomic_json(directory / "supervisor_result.json", row)
        return row

    monkeypatch.setattr(runner, "run_cell", synthetic_run_cell)
    return SimpleNamespace(
        original=original,
        cells=cells,
        original_root=original_root,
        output_root=output_root,
        source_root=source_root,
        ledger_sha="e" * 64,
        dispatched=dispatched,
        validations=validations,
    )


def test_prepare_then_execute_dispatches_only_the_exact_115_cell_suffix(
    tmp_path, monkeypatch
):
    fixture = _integration_stubs(tmp_path, monkeypatch)
    manifest_path = recovery.prepare_continuation(
        source_root=fixture.source_root,
        original_root=fixture.original_root,
        output_root=fixture.output_root,
        expected_ledger_sha256=fixture.ledger_sha,
    )
    prepared = read_checked_json(manifest_path)
    assert prepared["cells"] == fixture.cells
    assert prepared["profile"] == fixture.original["profile"]
    assert prepared["campaign_ceiling_seconds"] == 90_000.0
    assert prepared["recovery"]["suffix_cells"] == fixture.cells[141:]
    assert prepared["recovery"]["start_ordinal"] == 142
    assert prepared["recovery"]["online_allowance_seconds"] == 34_500.0

    result = recovery.execute_continuation(
        output_root=fixture.output_root,
        expected_manifest_sha256=sha256_file(manifest_path),
        expected_ledger_sha256=fixture.ledger_sha,
    )
    expected_ids = [cell["cell_id"] for cell in fixture.cells[141:]]
    assert result["status"] == "COMPLETE"
    assert len(result["records"]) == len(result["starts"]) == 115
    assert [
        payload["cell"]["cell_id"] for payload in fixture.dispatched
    ] == expected_ids
    assert fixture.validations == expected_ids
    assert all(
        payload["cell"]["profile"] == fixture.cells[141 + i]["profile"]
        and payload["paths"]["cms"] == "/synthetic/cms"
        for i, payload in enumerate(fixture.dispatched)
    )
    assert not (fixture.output_root / "execution/audit.json").exists()


def test_prepare_rejects_preexisting_output(tmp_path):
    output = tmp_path / "already-there"
    output.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        recovery.prepare_continuation(
            source_root=tmp_path,
            original_root=tmp_path / recovery.ORIGINAL_DIRECTORY,
            output_root=output,
            expected_ledger_sha256="0" * 64,
        )


def test_execute_rejects_existing_execution_tree_instead_of_resuming(
    tmp_path, monkeypatch
):
    fixture = _integration_stubs(tmp_path, monkeypatch)
    manifest_path = recovery.prepare_continuation(
        source_root=fixture.source_root,
        original_root=fixture.original_root,
        output_root=fixture.output_root,
        expected_ledger_sha256=fixture.ledger_sha,
    )
    (fixture.output_root / "execution").mkdir()
    with pytest.raises(ValueError, match="no resume or retry"):
        recovery.execute_continuation(
            output_root=fixture.output_root,
            expected_manifest_sha256=sha256_file(manifest_path),
            expected_ledger_sha256=fixture.ledger_sha,
        )


def test_old_prefix_requires_exact_supervisor_and_guardian_proofs(
    tmp_path, monkeypatch
):
    original_root, cells, ledger = _synthetic_prefix(tmp_path)
    import neural_sat_speck.followups.ranking_depth_runner as runner

    monkeypatch.setattr(runner, "live_invocation_groups", lambda _token: set())
    recovery._check_terminal_prefix(original_root, cells, ledger)
    incident = recovery._validate_failed_invocation(
        original_root,
        cells[141],
        ledger["starts"][cells[141]["cell_id"]],
        ledger,
    )
    assert incident["cell_id"] == recovery.REPLACEMENT_CELL_ID
    assert incident["process_group_absence_proven"] is False
    assert len(incident["file_inventory"]) >= 6

    corrupted = copy.deepcopy(ledger)
    corrupted["records"][cells[0]["cell_id"]]["opaque_worker_payload"] = {
        "changed": True
    }
    with pytest.raises(ValueError, match="supervisor artifact"):
        recovery._check_terminal_prefix(original_root, cells, corrupted)

    retried = copy.deepcopy(ledger)
    retried["records"][cells[141]["cell_id"]] = {"cell": cells[141]}
    with pytest.raises(ValueError, match="141-cell prefix"):
        recovery._check_terminal_prefix(original_root, cells, retried)


def test_old_prefix_rejects_nonterminal_or_unproved_cleanup(tmp_path, monkeypatch):
    original_root, cells, ledger = _synthetic_prefix(tmp_path)
    import neural_sat_speck.followups.ranking_depth_runner as runner

    monkeypatch.setattr(runner, "live_invocation_groups", lambda _token: set())
    first = cells[0]
    directory = Path(ledger["starts"][first["cell_id"]]["directory"])
    receipt = read_checked_json(directory / "guardian_receipt.json")
    receipt["process_group_absent"] = False
    atomic_json(directory / "guardian_receipt.json", receipt)
    row = ledger["records"][first["cell_id"]]
    row["guardian_receipt"] = receipt
    atomic_json(directory / "supervisor_result.json", row)
    with pytest.raises(ValueError, match="cleanup proof"):
        recovery._check_terminal_prefix(original_root, cells, ledger)

    failed = cells[141]
    failed_dir = Path(ledger["starts"][failed["cell_id"]]["directory"])
    failed_receipt = read_checked_json(failed_dir / "guardian_receipt.json")
    failed_receipt["status"] = "worker_exit"
    atomic_json(failed_dir / "guardian_receipt.json", failed_receipt)
    with pytest.raises(ValueError, match="authorized guardian-error"):
        recovery._validate_failed_invocation(
            original_root,
            failed,
            ledger["starts"][failed["cell_id"]],
            ledger,
        )


def test_recovery_budget_uses_original_launch_elapsed_not_a_new_ceiling():
    elapsed, remaining = recovery._remaining_original_budget(
        {"whole_launch_wall_seconds": 31_045.115611666995}
    )
    assert elapsed == 31_045.115611666995
    assert remaining == pytest.approx(58_954.884388333005)
    assert 115 * recovery.CELLS_PER_SECOND_ALLOWANCE == 34_500.0
    assert remaining < recovery.ORIGINAL_CAMPAIGN_CEILING_SECONDS
    with pytest.raises(ValueError, match="cannot cover"):
        recovery._remaining_original_budget({"whole_launch_wall_seconds": 80_000.0})


def test_schedule_and_checksum_corruption_fail_closed(tmp_path):
    cells = control.fixed_cells("full", num_keys=32)
    assert len(recovery._validated_schedule(cells)) == 115
    duplicated = copy.deepcopy(cells)
    duplicated[200] = copy.deepcopy(duplicated[199])
    with pytest.raises(ValueError, match="schedule"):
        recovery._validated_schedule(duplicated)

    checked = tmp_path / "pin.json"
    atomic_json(checked, {"synthetic": True})
    expected = sha256_file(checked)
    checked.write_text('{"synthetic": false}\n')
    with pytest.raises(ValueError, match="Checksum mismatch"):
        recovery._sha_pin(checked, expected, "synthetic")


def test_source_rebase_rejects_scientific_or_runtime_asset_drift(tmp_path, monkeypatch):
    old_root, new_root = tmp_path / "old", tmp_path / "new"
    worker_name = "src/neural_sat_speck/followups/ranking_depth_worker.py"
    guardian_name = recovery.GUARDIAN_MODULE
    for root in (old_root, new_root):
        (root / "src/neural_sat_speck/followups").mkdir(parents=True)
        (root / "src/neural_sat_speck/backend").mkdir(parents=True)
        (root / "scripts").mkdir(parents=True)
        (root / recovery.RECOVERY_MODULE).parent.mkdir(parents=True, exist_ok=True)
    (old_root / worker_name).write_text("frozen worker\n")
    (new_root / worker_name).write_text("changed worker\n")
    (old_root / guardian_name).write_text("old guardian\n")
    (new_root / guardian_name).write_text("repaired guardian\n")
    (new_root / recovery.RECOVERY_MODULE).write_text("recovery module\n")
    (new_root / recovery.CONTINUATION_SCRIPT).write_text("continuation script\n")
    asset = tmp_path / "asset.bin"
    asset.write_bytes(b"asset")
    runtime_file = tmp_path / "python"
    runtime_file.write_bytes(b"interpreter alias")
    asset_hash = hashlib.sha256(b"asset").hexdigest()
    profile = {"files": {"cms": {"sha256": asset_hash}}}
    old_files = {
        "source:" + worker_name: {
            "path": str(old_root / worker_name),
            "sha256": hashlib.sha256(b"frozen worker\n").hexdigest(),
            "identity": stat_identity(old_root / worker_name),
        },
        "source:" + guardian_name: {
            "path": str(old_root / guardian_name),
            "sha256": hashlib.sha256(b"old guardian\n").hexdigest(),
            "identity": stat_identity(old_root / guardian_name),
        },
        "cms": {
            **file_identity(asset),
        },
        "python": {**file_identity(runtime_file)},
    }
    old_manifest = {
        "source_root": str(old_root),
        "files": old_files,
        "profile": profile,
    }
    names = {worker_name, guardian_name, recovery.RECOVERY_MODULE}
    monkeypatch.setattr(recovery, "source_names", lambda _root: sorted(names))
    monkeypatch.setattr(recovery, "validate_source_root", lambda *_: None)
    monkeypatch.setattr(
        recovery,
        "source_commit",
        lambda root: (
            recovery.ORIGINAL_SOURCE_COMMIT if Path(root) == old_root else "c" * 40
        ),
    )
    monkeypatch.setattr(recovery, "policy", lambda *_: profile)
    monkeypatch.setattr(
        recovery,
        "streaming_hash",
        lambda path: (
            recovery.REPAIRED_GUARDIAN_SHA256
            if Path(path) == new_root / guardian_name
            else hashlib.sha256(Path(path).read_bytes()).hexdigest()
        ),
    )
    with pytest.raises(ValueError, match="Scientific/source drift"):
        recovery._validate_source_rebase(old_manifest, new_root)

    (new_root / worker_name).write_text("frozen worker\n")
    asset.write_bytes(b"changed asset")
    with pytest.raises(ValueError, match="stat identity changed"):
        recovery._validate_source_rebase(old_manifest, new_root)

    asset.write_bytes(b"asset")
    old_files["cms"]["identity"] = stat_identity(asset)
    runtime_file.write_bytes(b"changed interpreter alias")
    with pytest.raises(ValueError, match="stat identity changed"):
        recovery._validate_source_rebase(old_manifest, new_root)
