"""Reject fabricated segment joins before inspecting any key truth."""

import copy

import pytest

from neural_sat_speck.backend.guard import atomic_json, sha256_file
from neural_sat_speck.followups import ranking_depth_segment_audit as auditor
from neural_sat_speck.followups.conditioning_control import digest
from neural_sat_speck.followups.ranking_depth_control import fixed_cells
from neural_sat_speck.followups.ranking_depth_segment_audit import (
    validate_segment_layout,
)


def layout():
    cells = fixed_cells("full", num_keys=32)
    original = {"cells": cells, "profile": {"role": "development"}, "cases": {"a": 1}}
    continuation = copy.deepcopy(original)
    continuation["recovery"] = {
        "suffix_cells": cells[141:],
        "suffix_sha256": digest(cells[141:]),
    }
    old = {"status": "INTERRUPTED", "starts": {}, "records": {}, "execution_order": []}
    new = {
        "schema": "backend_ranking_depth_d056_continuation_results_v1",
        "status": "COMPLETE",
        "starts": {},
        "records": {},
        "execution_order": [],
        "cells": cells[141:],
    }
    for index, cell in enumerate(cells):
        ident = cell["cell_id"]
        start = {
            "directory": f"/synthetic/{index:032x}",
            "invocation_id": f"{index:032x}",
        }
        row = {
            "cell": cell,
            "directory": start["directory"],
            "started_monotonic": 10.0 * index,
            "guardian_receipt": {"cleanup_completed_monotonic": 10.0 * index + 5},
        }
        segment = old if index < 141 else new
        segment["starts"][ident] = start
        segment["records"][ident] = row
        segment["execution_order"].append(ident)
        if index == 141:
            old["starts"][ident] = {
                "directory": "/synthetic/unavailable",
                "invocation_id": "unavailable",
            }
            old["execution_order"].append(ident)
    return original, continuation, old, new


def test_complete_segment_join_keeps_one_original_unavailable_attempt():
    validate_segment_layout(*layout())


@pytest.mark.parametrize("flag", [False, "true", 1, None])
def test_distinct_or_unchecked_key_cannot_pass_exact_recovery_audit(flag):
    with pytest.raises(ValueError, match="assigned master key"):
        auditor.require_assigned_key({"assigned_key_match": flag})


def test_verified_assigned_key_passes_exact_recovery_gate():
    auditor.require_assigned_key({"assigned_key_match": True})


@pytest.mark.parametrize(
    "defect",
    [
        "duplicate",
        "missing",
        "extra",
        "incomplete",
        "order",
        "prefix",
        "token",
        "overlap",
        "profile",
        "case",
        "suffix",
        "cell",
    ],
)
def test_invalid_segment_join_is_rejected(defect):
    original, continuation, old, new = layout()
    first = new["execution_order"][0]
    second = new["execution_order"][1]
    if defect == "duplicate":
        original["cells"].append(original["cells"][0])
        continuation["cells"] = original["cells"]
    elif defect == "missing":
        del new["records"][first]
    elif defect == "extra":
        new["records"]["unknown"] = new["records"][first]
    elif defect == "incomplete":
        new["status"] = "RUNNING"
    elif defect == "order":
        new["execution_order"].reverse()
    elif defect == "prefix":
        old["records"][first] = new["records"][first]
    elif defect == "token":
        new["records"][second]["directory"] = new["records"][first]["directory"]
        new["starts"][second] = new["starts"][first]
    elif defect == "overlap":
        new["records"][first]["started_monotonic"] = 0
    elif defect == "profile":
        continuation["profile"]["role"] = "confirmation"
    elif defect == "case":
        continuation["cases"]["a"] = 2
    elif defect == "suffix":
        continuation["recovery"]["suffix_sha256"] = "0" * 64
    elif defect == "cell":
        new["records"][first]["cell"] = new["records"][second]["cell"]
    with pytest.raises(ValueError):
        validate_segment_layout(original, continuation, old, new)


@pytest.mark.parametrize("drift", ["original_ledger", "source", "resigned_manifest"])
def test_freeze_drift_rejected_before_loading_original(tmp_path, monkeypatch, drift):
    manifest = {
        "source_commit": auditor.CONTINUATION_SOURCE_COMMIT,
        "recovery": {"ledger_sha256": auditor.ORIGINAL_LEDGER_SHA256},
    }
    path, ledger = tmp_path / "manifest.json", tmp_path / "ledger.json"
    atomic_json(path, manifest)
    atomic_json(ledger, {})
    monkeypatch.setattr(auditor, "CONTINUATION_MANIFEST_SHA256", sha256_file(path))
    if drift == "original_ledger":
        manifest["recovery"]["ledger_sha256"] = "0" * 64
    elif drift == "source":
        manifest["source_commit"] = "0" * 40
    else:
        manifest["recovery"]["unexpected_authorization"] = True
    atomic_json(path, manifest)

    def prohibited(*_):
        pytest.fail("Freeze rejection must precede original evidence loading")

    monkeypatch.setattr(auditor, "_load_original", prohibited)
    with pytest.raises(ValueError, match="recorded launch freeze"):
        auditor.audit_continuation(ledger, path)
