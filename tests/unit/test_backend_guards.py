"""Independent adversarial identities and dispatch-boundary tests, no solvers."""

import copy
import json
from pathlib import Path

import pytest

from neural_sat_speck.backend import guard, runner


@pytest.fixture
def profile(tmp_path, monkeypatch):
    # Source-git validation is independently tested below. These fixtures target
    # data/artifact identities and dispatch, not a mock claim of a real worktree.
    monkeypatch.setattr(guard, "validate_source_tree", lambda manifest, root: None)
    root = tmp_path / "source_root"
    (root / "src" / "neural_sat_speck").mkdir(parents=True)
    (root / "scripts").mkdir()
    tiny = root / "tiny.txt"
    tiny.write_text("{}\n")
    rec = {"path": str(tiny), "sha256": guard.sha256_file(tiny)}
    catalog = {}
    for cell in guard.fixed_grid():
        ident = cell["case_id"]
        catalog[ident] = {
            "case_id": ident,
            "files": {k: rec for k in ("public", "truth", "native", "cnf", "varmap")},
        }
    catpath = root / "catalog.json"
    catpath.write_text(json.dumps(catalog))
    sources = {}
    for name in ("prepare", "evaluate", "analyze"):
        path = root / "scripts" / f"{name}_backend_comparison.py"
        path.write_text("# fixture only\n")
        sources[f"source:scripts/{path.name}"] = {
            "path": str(path),
            "sha256": guard.sha256_file(path),
        }
    src = root / "src" / "neural_sat_speck" / "fixture.py"
    src.write_text("# fixture only\n")
    sources["source:src/neural_sat_speck/fixture.py"] = {
        "path": str(src),
        "sha256": guard.sha256_file(src),
    }
    manifest = {
        "schema": "backend_demo_v1",
        "status": "APPROVED",
        "execution_authorized": True,
        "native_policy_review_passed": True,
        "source_commit_at_preparation": "a" * 40,
        "source_isolation": "git_worktree",
        "launch_blockers": [],
        "seed": 999999,
        "role": "opened_demo_diagnostic",
        "key_count": 4,
        "cohort_key_count": 8,
        "pair_count": 64,
        "fresh_block_count": 2,
        "deadline_seconds": 300.0,
        "threads": 1,
        "repetitions": 1,
        "max_cell_starts": 160,
        "worst_case_cutoff_seconds": 48000.0,
        "assumption_policy": "unit_clauses_all_profiles",
        "interruption_policy": "retain_started_without_retry",
        "thread_environment": guard.THREAD_ENVIRONMENT,
        "cells": guard.fixed_grid(),
        "runtime_versions": {
            "cms": "5.14.7",
            "pycryptosat": "5.14.7",
            "cadical": "3.0.1",
        },
        "files": {
            **{
                k: rec
                for k in (
                    "cohort",
                    "python",
                    "pycryptosat",
                    "cms",
                    "cadical",
                    "protocol",
                )
            },
            **sources,
            "catalog": {"path": str(catpath), "sha256": guard.sha256_file(catpath)},
        },
    }
    return root, manifest


def test_grid_independently_counted():
    cells = guard.fixed_grid()
    assert len(cells) == len({c["cell_id"] for c in cells}) == 160
    assert len({c["case_id"] for c in cells}) == 32
    assert sum(c["trailing_keys"] > 0 for c in cells) == 96
    assert sum(c["trailing_keys"] == 0 for c in cells) == 64
    for cipher, rounds in (("speck32_64", 7), ("simon32_64", 10)):
        group = [c for c in cells if c["cipher"] == cipher]
        assert len(group) == 80 and {c["rounds"] for c in group} == {rounds}


def test_approved_complete_fixture(profile):
    root, manifest = profile
    assert len(guard.validate_manifest(manifest, root)) > 160


@pytest.mark.parametrize(
    "key,value",
    [
        ("status", "PROPOSED"),
        ("execution_authorized", "true"),
        ("native_policy_review_passed", False),
        ("seed", 260926),
        ("deadline_seconds", 45.0),
        ("key_count", 8),
        ("threads", True),
        ("pair_count", 2),
        ("fresh_block_count", 4),
        ("max_cell_starts", 161),
        (
            "runtime_versions",
            {"cms": "5.14.7", "pycryptosat": "5.14.7", "cadical": "3.0.0"},
        ),
    ],
)
def test_rejects_drift_before_dispatch(profile, monkeypatch, tmp_path, key, value):
    root, manifest = profile
    manifest[key] = value
    path = tmp_path / "manifest.json"
    guard.atomic_json(path, manifest)
    loaded = []
    monkeypatch.setattr(
        "neural_sat_speck.backend.artifacts.validate_cohort", lambda d: loaded.append(d)
    )
    called = []
    with pytest.raises(ValueError):
        runner.run(path, tmp_path / "ledger.json", dispatch=lambda *a: called.append(a))
    assert not loaded and not called


def test_modified_artifact_and_missing_source_rejected(profile):
    root, manifest = profile
    path = Path(manifest["files"]["cms"]["path"])
    path.write_text("changed\n")
    with pytest.raises(ValueError, match="identity mismatch"):
        guard.validate_manifest(manifest, root)
    root, manifest = profile
    manifest["files"].pop("source:src/neural_sat_speck/fixture.py")
    with pytest.raises(ValueError, match="source identity closure"):
        guard.validate_manifest(manifest, root)


def test_changed_or_duplicate_grid(profile):
    root, manifest = profile
    manifest["cells"][1] = manifest["cells"][0]
    with pytest.raises(ValueError, match="grid/order"):
        guard.validate_manifest(manifest, root)


def test_checksum_corruption_is_not_repaired(tmp_path):
    path = tmp_path / "ledger.json"
    guard.atomic_json(path, {"records": {}})
    path.write_text('{"records":{"fabricated":true}}')
    with pytest.raises(ValueError, match="Checksum mismatch"):
        guard.read_checked_json(path)


def test_keeps_venv_executable_symlink(tmp_path):
    binary = tmp_path / "base_python"
    binary.write_text("fake executable")
    link = tmp_path / "venv_python"
    link.symlink_to(binary)
    assert (
        guard.checked_path(
            tmp_path, {"path": str(link), "sha256": guard.sha256_file(binary)}
        )
        == link
    )


@pytest.mark.parametrize(
    "mutation",
    [
        {"verified_fresh": "false"},
        {"assigned_key_match": True, "verified_known": True, "verified_fresh": True},
        {"outcome": "UNSAT"},
        {"online_seconds": float("nan")},
        {"online_seconds": -1},
    ],
)
def test_record_schema_flags(mutation):
    cell = guard.fixed_grid()[0]
    record = runner.failure(cell, "unknown", 0.5)
    record.update(mutation)
    with pytest.raises(ValueError):
        guard.validate_record(record, cell)


def test_unknown_preserved_ledger_and_duplicate_rejected():
    cell = guard.fixed_grid()[0]
    ledger = {
        "schema": "backend_ledger_v1",
        "manifest_sha256": "abc",
        "starts": {
            cell["cell_id"]: {"cell_id": cell["cell_id"], "invocation": "a" * 32}
        },
        "records": {
            cell["cell_id"]: runner.failure(cell, "unknown", 0.5, invocation="a" * 32)
        },
    }
    runner.validate_ledger(ledger, {"cells": guard.fixed_grid()}, "abc")
    drifted = copy.deepcopy(ledger)
    drifted["records"]["alien"] = drifted["records"][cell["cell_id"]]
    with pytest.raises(ValueError):
        runner.validate_ledger(drifted, {"cells": guard.fixed_grid()}, "abc")


def test_late_recovery_cannot_count():
    cell = guard.fixed_grid()[0]
    record = {
        **runner.failure(cell, "recovered", 300.0),
        "verified_known": True,
        "verified_fresh": True,
        "assigned_key_match": True,
        "recovered_key": [0, 1, 2, 3],
    }
    with pytest.raises(ValueError, match="Late"):
        guard.validate_record(record, cell)


def _authorization(tmp_path, root, manifest_path):
    path = tmp_path / "authorization.json"
    guard.atomic_json(
        path,
        {
            "schema": "backend_launch_authorization_v1",
            "status": "AUTHOR_APPROVED",
            "manifest_sha256": guard.sha256_file(manifest_path),
            "decision": "D-999",
            "source_root": str(root.resolve()),
            "isolated_snapshot": True,
            "depth_campaign_closed": True,
            "output_path": str((tmp_path / "ledger.json").resolve()),
        },
    )
    return path


def test_approval_is_separate_and_bound_to_manifest(tmp_path):
    path = tmp_path / "authorization.json"
    receipt = {
        "schema": "backend_launch_authorization_v1",
        "status": "AUTHOR_APPROVED",
        "manifest_sha256": "a" * 64,
        "decision": "D-999",
        "source_root": str(tmp_path.resolve()),
        "isolated_snapshot": True,
        "depth_campaign_closed": True,
    }
    guard.atomic_json(path, receipt)
    guard.validate_authorization(path, "a" * 64, tmp_path)
    with pytest.raises(ValueError, match="another manifest"):
        guard.validate_authorization(path, "b" * 64, tmp_path)
    receipt["depth_campaign_closed"] = False
    guard.atomic_json(path, receipt)
    with pytest.raises(ValueError, match="Depth campaign"):
        guard.validate_authorization(path, "a" * 64, tmp_path)


def _synthetic_runner_setup(profile, monkeypatch, tmp_path):
    import contextlib

    root, manifest = profile
    path = tmp_path / "manifest.json"
    guard.atomic_json(path, manifest)
    auth = _authorization(tmp_path, root, path)
    monkeypatch.setattr(
        runner,
        "__file__",
        str(root / "src" / "neural_sat_speck" / "backend" / "runner.py"),
    )
    # These tests target scheduling/persistence; cipher/cohort semantics have
    # their own independent fixtures and are deliberately outside this spy.
    monkeypatch.setattr(
        "neural_sat_speck.backend.artifacts.validate_cohort", lambda d: None
    )
    monkeypatch.setattr(
        "neural_sat_speck.backend.artifacts.validate_case_inputs", lambda d, p: None
    )
    monkeypatch.setattr(
        "neural_sat_speck.backend.evidence.audit_cell_evidence",
        lambda *args: {"scope": "scheduling_fixture_only"},
    )
    monkeypatch.setattr(
        runner, "power_sample", lambda: {"supported": False, "accepted": None}
    )
    monkeypatch.setattr(
        runner, "sleep_assertion", lambda: contextlib.nullcontext({"supported": False})
    )
    return root, manifest, path, auth


def test_full_synthetic_grid_resume_does_not_repeat_failures(
    profile, monkeypatch, tmp_path
):
    _, _, path, auth = _synthetic_runner_setup(profile, monkeypatch, tmp_path)
    output = tmp_path / "ledger.json"
    calls = []

    def dispatch(payload, environment):
        calls.append(payload["cell"]["cell_id"])
        assert "DYLD_INSERT_LIBRARIES" not in environment and "HOME" not in environment
        assert environment["OMP_NUM_THREADS"] == "1"
        ledger = guard.read_checked_json(output)
        ledger["starts"][calls[-1]]["process_group"] = 999999999
        guard.atomic_json(output, ledger)
        return runner.failure(
            payload["cell"],
            "unknown",
            0.01,
            process_group_absent=True,
            guardian_healthy=True,
            invocation=payload["invocation"],
        )

    ledger = runner.run(path, output, authorization_path=auth, dispatch=dispatch)
    assert ledger["status"] == "complete"
    assert len(calls) == len(set(calls)) == len(ledger["records"]) == 160
    resumed = runner.run(
        path,
        output,
        authorization_path=auth,
        resume=True,
        dispatch=lambda *args: pytest.fail("Repeated a completed failure"),
    )
    assert resumed["records"] == ledger["records"]


def test_audit_alarm_is_durable_and_blocks_resume(profile, monkeypatch, tmp_path):
    _, _, path, auth = _synthetic_runner_setup(profile, monkeypatch, tmp_path)
    output = tmp_path / "ledger.json"

    def dispatch(payload, environment):
        ledger = guard.read_checked_json(output)
        ident = payload["cell"]["cell_id"]
        ledger["starts"][ident]["process_group"] = 999999999
        guard.atomic_json(output, ledger)
        # Broken parser promises success without any verification.
        return runner.failure(payload["cell"], "recovered", 0.1)

    ledger = runner.run(path, output, authorization_path=auth, dispatch=dispatch)
    assert ledger["status"] == "safety_stop" and len(ledger["records"]) == 1
    assert next(iter(ledger["records"].values()))["outcome"] == "audit_rejected"
    with pytest.raises(RuntimeError, match="requires independent review"):
        runner.run(
            path,
            output,
            authorization_path=auth,
            resume=True,
            dispatch=lambda *args: pytest.fail("Dispatched after audit alarm"),
        )


def test_unconfirmed_cleanup_stops_further_dispatch(profile, monkeypatch, tmp_path):
    _, _, path, auth = _synthetic_runner_setup(profile, monkeypatch, tmp_path)
    output = tmp_path / "ledger.json"
    calls = []

    def dispatch(payload, environment):
        calls.append(payload["cell"]["cell_id"])
        ledger = guard.read_checked_json(output)
        ledger["starts"][calls[-1]]["process_group"] = 999999999
        guard.atomic_json(output, ledger)
        return runner.failure(
            payload["cell"],
            "global_deadline",
            300.01,
            process_group_absent=False,
            guardian_healthy=True,
            invocation=payload["invocation"],
        )

    ledger = runner.run(path, output, authorization_path=auth, dispatch=dispatch)
    assert ledger["status"] == "safety_stop" and len(calls) == 1


@pytest.mark.parametrize(
    "raw", ['{"records":{},"records":{}}', '{"duration":NaN}', '{"duration":Infinity}']
)
def test_valid_checksum_does_not_make_ambiguous_json_valid(tmp_path, raw):
    path = tmp_path / "ledger.json"
    path.write_text(raw)
    path.with_name(path.name + ".sha256").write_text(
        f"{guard.sha256_file(path)}  {path.name}\n"
    )
    with pytest.raises(ValueError, match="Duplicate|Nonfinite"):
        guard.read_checked_json(path)


def _unfinished_ledger(manifest, path, group):
    cell = manifest["cells"][0]
    return {
        "schema": "backend_ledger_v1",
        "manifest_sha256": guard.sha256_file(path),
        "status": "incomplete",
        "starts": {
            cell["cell_id"]: {
                "cell_id": cell["cell_id"],
                "invocation": "a" * 32,
                "process_group": group,
            }
        },
        "records": {},
    }


@pytest.mark.parametrize("group", [None, 12345])
def test_resume_cannot_dispatch_with_unidentified_or_active_group(
    profile, monkeypatch, tmp_path, group
):
    _, manifest, path, auth = _synthetic_runner_setup(profile, monkeypatch, tmp_path)
    output = tmp_path / "ledger.json"
    guard.atomic_json(output, _unfinished_ledger(manifest, path, group))
    monkeypatch.setattr(runner, "_group_absent", lambda pgid: False)
    with pytest.raises(RuntimeError, match="orphan audit|Prior worker group"):
        runner.run(
            path,
            output,
            authorization_path=auth,
            resume=True,
            dispatch=lambda *args: pytest.fail("Dispatched into a prior worker"),
        )


def test_resume_preserves_started_interruption_without_retry(
    profile, monkeypatch, tmp_path
):
    _, manifest, path, auth = _synthetic_runner_setup(profile, monkeypatch, tmp_path)
    output = tmp_path / "ledger.json"
    unfinished = manifest["cells"][0]["cell_id"]
    guard.atomic_json(output, _unfinished_ledger(manifest, path, 999999999))
    calls = []

    def dispatch(payload, environment):
        ident = payload["cell"]["cell_id"]
        assert ident != unfinished
        calls.append(ident)
        ledger = guard.read_checked_json(output)
        ledger["starts"][ident]["process_group"] = 999999999
        guard.atomic_json(output, ledger)
        return runner.failure(
            payload["cell"],
            "unknown",
            0.01,
            process_group_absent=True,
            guardian_healthy=True,
            invocation=payload["invocation"],
        )

    ledger = runner.run(
        path, output, authorization_path=auth, resume=True, dispatch=dispatch
    )
    assert len(calls) == 159 and ledger["status"] == "complete"
    interrupted = ledger["records"][unfinished]
    assert interrupted["outcome"] == "infrastructure_interrupted"
    assert interrupted["duration_available"] is False
    assert interrupted["invocation"] == "a" * 32


def _analysis_script(monkeypatch, root):
    import importlib.util

    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "analyze_backend_comparison.py"
    )
    spec = importlib.util.spec_from_file_location("backend_analysis_fixture", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "_verify_recovery", lambda record, paths: None)
    monkeypatch.setattr(
        "neural_sat_speck.backend.artifacts.validate_cohort", lambda data: None
    )
    monkeypatch.setattr(
        "neural_sat_speck.backend.artifacts.validate_case_inputs",
        lambda data, paths: None,
    )
    monkeypatch.setattr(
        "neural_sat_speck.backend.evidence.audit_cell_evidence",
        lambda *args: {"scope": "numeric_fixture_only"},
    )
    return module


def test_analysis_uses_all_failures_and_paired_keys(profile, monkeypatch, tmp_path):
    root, manifest = profile
    path = tmp_path / "manifest.json"
    guard.atomic_json(path, manifest)
    records, starts = {}, {}
    for index, cell in enumerate(manifest["cells"]):
        record = runner.failure(cell, "unknown", 20.0)
        # Numeric analysis fixture only: cryptographic validation has separate tests.
        if cell["profile"] == "cms_cnf_no_gauss" and cell["key_index"] < 2:
            record.update(
                outcome="recovered",
                solver_status="sat",
                online_seconds=5.0 + 2.0 * cell["key_index"],
                verified_known=True,
                verified_fresh=True,
                assigned_key_match=True,
                recovered_key=[1, 2, 3, 4],
            )
        if cell["profile"] == "cms_native_default" and cell["key_index"] == 0:
            record.update(
                outcome="recovered",
                solver_status="sat",
                online_seconds=11.0,
                verified_known=True,
                verified_fresh=True,
                assigned_key_match=True,
                recovered_key=[1, 2, 3, 4],
            )
        if cell["profile"] == "cadical_cnf" and cell["key_index"] == 2:
            record.update(
                outcome="infrastructure_interrupted",
                solver_status=None,
                online_seconds=0.0,
                duration_available=False,
            )
        records[cell["cell_id"]] = record
        record["invocation"] = f"{index:032x}"
        starts[cell["cell_id"]] = {
            "cell_id": cell["cell_id"],
            "invocation": f"{index:032x}",
            "process_group": 999999999,
        }
    ledger = {
        "schema": "backend_ledger_v1",
        "manifest_sha256": guard.sha256_file(path),
        "status": "complete",
        "starts": starts,
        "records": records,
    }
    output = tmp_path / "ledger.json"
    guard.atomic_json(output, ledger)
    module = _analysis_script(monkeypatch, root)
    analysis = module.analyze(path, output)
    summaries = analysis["summaries"]
    block = "speck32_64_o4_m1"
    assert summaries[f"{block}__cms_cnf_no_gauss"]["par2_seconds"] == 303.0
    assert (
        summaries[f"{block}__cms_cnf_no_gauss"]["median_available_online_seconds"]
        == 13.5
    )
    assert summaries[f"{block}__cadical_cnf"]["unavailable_durations"] == 1
    assert summaries[f"{block}__cadical_cnf"]["median_available_online_seconds"] == 20.0
    contrast = next(
        row
        for row in analysis["paired_descriptive_contrasts"][block]
        if row["first"] == "cms_cnf_no_gauss" and row["second"] == "cms_native_default"
    )
    assert (
        contrast["both"],
        contrast["first_only"],
        contrast["second_only"],
        contrast["neither"],
    ) == (1, 1, 0, 2)
    assert contrast["recovery_difference"] == 0.25
    assert contrast["mean_paired_par2_difference_seconds"] == -149.75
    ledger["records"].pop(next(iter(ledger["records"])))
    guard.atomic_json(output, ledger)
    with pytest.raises(ValueError, match="complete 160-cell"):
        module.analyze(path, output)


@pytest.mark.parametrize("drift", ["revision", "root", "dirty", "untracked", "none"])
def test_committed_source_worktree_is_enforced(tmp_path, monkeypatch, drift):
    from types import SimpleNamespace

    (tmp_path / ".git").write_text("gitdir: /fixture/review-worktree")
    manifest = {
        "source_isolation": "git_worktree",
        "source_commit_at_preparation": "a" * 40,
    }

    def git(argv, **kwargs):
        if argv[1:] == ["rev-parse", "HEAD"]:
            output = "b" * 40 if drift == "revision" else "a" * 40
        elif argv[1:] == ["rev-parse", "--show-toplevel"]:
            output = "/another/tree" if drift == "root" else str(tmp_path)
        else:
            output = {
                "dirty": " M src/encoder.py",
                "untracked": "?? src/encoder.py",
            }.get(drift, "")
        return SimpleNamespace(returncode=0, stdout=output)

    monkeypatch.setattr(guard.subprocess, "run", git)
    if drift == "none":
        guard.validate_source_tree(manifest, tmp_path)
    else:
        with pytest.raises(ValueError, match="source"):
            guard.validate_source_tree(manifest, tmp_path)


def test_ledger_start_and_result_cannot_share_or_swap_invocations():
    cells = guard.fixed_grid()[:2]
    starts = {
        cell["cell_id"]: {"cell_id": cell["cell_id"], "invocation": f"{index:032x}"}
        for index, cell in enumerate(cells)
    }
    records = {
        cell["cell_id"]: runner.failure(
            cell, "unknown", 1.0, invocation=f"{index:032x}"
        )
        for index, cell in enumerate(cells)
    }
    ledger = {
        "schema": "backend_ledger_v1",
        "manifest_sha256": "a" * 64,
        "starts": starts,
        "records": records,
    }
    runner.validate_ledger(ledger, {"cells": cells}, "a" * 64)
    starts[cells[1]["cell_id"]]["invocation"] = "0" * 32
    with pytest.raises(ValueError, match="Duplicate invocation"):
        runner.validate_ledger(ledger, {"cells": cells}, "a" * 64)
    starts[cells[1]["cell_id"]]["invocation"] = "1" * 32
    with pytest.raises(ValueError, match="does not match"):
        runner.validate_ledger(ledger, {"cells": cells}, "a" * 64)
