"""D-062: process.json read race fix and the continuation of an interrupted full run."""

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

from neural_sat_speck.backend.guard import atomic_json, sha256_file
from neural_sat_speck.followups import ranking_depth_control as control
from neural_sat_speck.followups.ranking_depth_runner import process_identity_ready

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "analyze_prospective_queue_cost.py"
PROFILE = {"selected_instances": list(range(32)), "seed": 601001}
CELLS = control.prospective_cells("full", num_keys=32)
IDS = [c["cell_id"] for c in CELLS]
START = 299


def _analysis():
    spec = importlib.util.spec_from_file_location(
        "analyze_prospective_queue_cost_continuation", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _interrupted(start: int = START) -> tuple[dict, dict]:
    manifest = {
        "campaign": control.PROSPECTIVE_CAMPAIGN,
        "campaign_role": "full",
        "mode": "full",
        "profile": copy.deepcopy(PROFILE),
        "cells": copy.deepcopy(CELLS),
        "source_commit": "a" * 40,
    }
    ledger = {
        "schema": "backend_ranking_depth_results_v1",
        "status": "INTERRUPTED",
        "mode": "full",
        "manifest_sha256": "b" * 64,
        "source_commit": "a" * 40,
        "execution_order": IDS[: start + 1],
        "starts": {i: {"invocation_id": i[-8:]} for i in IDS[: start + 1]},
        "records": {i: {"outcome": "recovered"} for i in IDS[:start]},
    }
    return manifest, ledger


def _check(manifest: dict, ledger: dict) -> int:
    return control.validate_continuation(manifest, ledger, "b" * 64, PROFILE)


def test_process_identity_waits_for_the_sidecar(tmp_path):
    path = tmp_path / "process.json"
    path.write_text("{}")
    assert not process_identity_ready(path)
    atomic_json(path, {"pid": 1})
    assert process_identity_ready(path)


def test_continuation_resumes_at_the_interrupted_cell():
    manifest, ledger = _interrupted()
    assert _check(manifest, ledger) == START
    assert CELLS[START:][0]["cell_id"] == ledger["execution_order"][-1]
    assert len(CELLS[START:]) == 21


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m, ledger: ledger.update(status="COMPLETE"),
        lambda m, ledger: ledger.update(status="STOPPED"),
        lambda m, ledger: ledger.update(mode="pilot"),
        lambda m, ledger: ledger.update(manifest_sha256="c" * 64),
        lambda m, ledger: ledger.update(source_commit="d" * 40),
        lambda m, ledger: ledger["records"][IDS[0]].update(outcome="worker_exception"),
        lambda m, ledger: ledger["records"].update(
            {IDS[START]: {"outcome": "recovered"}}
        ),
        lambda m, ledger: ledger["records"].pop(IDS[5]),
        lambda m, ledger: ledger.update(execution_order=IDS[1 : START + 2]),
        lambda m, ledger: ledger["starts"].pop(IDS[START]),
        lambda m, ledger: m.update(campaign_role="pilot"),
        lambda m, ledger: m.update(continuation={"start_index": 1}),
        lambda m, ledger: m["profile"].update(seed=1),
        lambda m, ledger: m["cells"].reverse(),
    ],
)
def test_continuation_refuses_anything_but_the_d062_shape(mutate):
    manifest, ledger = _interrupted()
    mutate(manifest, ledger)
    with pytest.raises(ValueError):
        _check(manifest, ledger)


def test_continuation_refuses_an_empty_run():
    manifest, ledger = _interrupted()
    ledger.update(execution_order=[], starts={}, records={})
    with pytest.raises(ValueError):
        _check(manifest, ledger)


def test_continuation_record_pins_files_and_the_interrupted_invocation(tmp_path):
    manifest, ledger = _interrupted()
    atomic_json(tmp_path / "inputs" / "manifest.json", manifest)
    ledger["manifest_sha256"] = sha256_file(tmp_path / "inputs" / "manifest.json")
    atomic_json(tmp_path / "execution" / "ledger.json", ledger)
    record = control.continuation_record(tmp_path, PROFILE)
    assert record["author_decision"] == "D-062"
    assert record["start_index"] == START
    assert record["interrupted_cell"] == IDS[START]
    assert record["interrupted_invocation"] == IDS[START][-8:]
    assert record["original_ledger"]["sha256"] == sha256_file(
        tmp_path / "execution" / "ledger.json"
    )


@pytest.mark.parametrize("approval", [None, "D-060", "D-063"])
def test_continuation_preparation_needs_explicit_d062_approval(
    tmp_path, monkeypatch, approval
):
    monkeypatch.setattr(control, "validate_source_root", lambda *_args: None)
    with pytest.raises(ValueError, match="D-062"):
        control._prepare_prospective(
            REPO,
            tmp_path / "out",
            "full",
            Path(sys.executable),
            REPO,
            tmp_path / "pilot.json",
            control.PROSPECTIVE_FULL_DESIGN,
            control.policy(REPO, control.PROSPECTIVE_FULL_DESIGN),
            None,
            tmp_path,
            approval,
        )


def _bound(tmp_path):
    original = tmp_path / "full" / "execution"
    suffix_root = tmp_path / "continuation"
    atomic_json(original / "ledger.json", {"segment": "original"})
    atomic_json(suffix_root / "execution" / "ledger.json", {"segment": "suffix"})
    audit = {
        "schema": "backend_ranking_depth_continuation_audit_v1",
        "passed": True,
        "original_ledger_path": str((original / "ledger.json").absolute()),
        "original_ledger_sha256": sha256_file(original / "ledger.json"),
        "continuation_ledger_path": str(
            (suffix_root / "execution" / "ledger.json").absolute()
        ),
        "continuation_ledger_sha256": sha256_file(
            suffix_root / "execution" / "ledger.json"
        ),
    }
    return original, suffix_root, audit


def test_analysis_merges_both_segments_only_under_a_binding_audit(
    tmp_path, monkeypatch
):
    an = _analysis()
    original, suffix_root, audit = _bound(tmp_path)
    atomic_json(suffix_root / "continuation_audit.json", audit)
    monkeypatch.setattr(an, "load_campaign_rows", lambda d: [d.parent.name])
    assert an.load_continued_rows(original, suffix_root) == ["full", "continuation"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("passed", False),
        ("original_ledger_sha256", "0" * 64),
        ("continuation_ledger_sha256", "0" * 64),
        ("original_ledger_path", "/elsewhere/ledger.json"),
        ("schema", "backend_ranking_depth_audit_v1"),
    ],
)
def test_analysis_refuses_an_unbound_or_failed_audit(tmp_path, field, value):
    an = _analysis()
    original, suffix_root, audit = _bound(tmp_path)
    audit[field] = value
    atomic_json(suffix_root / "continuation_audit.json", audit)
    with pytest.raises(ValueError):
        an.load_continued_rows(original, suffix_root)
