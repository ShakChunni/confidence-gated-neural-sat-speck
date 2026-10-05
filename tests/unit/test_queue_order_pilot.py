"""Synthetic readiness-pilot tests; no real cohort, ranking, or SAT is run."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest

from neural_sat_speck.backend.guard import atomic_json, read_checked_json
from neural_sat_speck.followups import pilot, queue_order
from tests.unit.test_queue_order_guards import _candidate_queue, _demo_dataset


def _pilot_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcomes=None):
    """Mirror runner dispatch fixtures while retaining pilot's real validators."""
    dataset = _demo_dataset()
    cohort = tmp_path / "synthetic-cohort.json"
    cohort.write_text(json.dumps(dataset), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    atomic_json(manifest_path, {"synthetic": True})

    tf_python = Path(sys.executable).absolute()
    manifest = {
        "source_commit": "fixture-commit",
        "execution_kind": "pilot",
        "pilot_gate": {
            "required": True,
            "authorization_decision": "D-050",
            "job_ids": [job["job_id"] for job in queue_order.fixed_jobs()[:12]],
        },
        "files": {
            "python": {"path": str(tf_python)},
            "tf_python": {"path": str(tf_python)},
        },
    }
    paths = {"cohort": cohort, "python": tf_python, "tf_python": tf_python}
    monkeypatch.setattr(pilot, "load_manifest", lambda *args: (manifest, paths))
    monkeypatch.setattr(pilot, "validate_runtime", lambda *args: None)

    def fake_check_output(arguments, **kwargs):
        if arguments[1] == "rev-parse":
            return "fixture-commit"
        return ""

    monkeypatch.setattr(pilot.subprocess, "check_output", fake_check_output)
    monkeypatch.setattr(
        pilot, "sleep_assertion", lambda: nullcontext({"synthetic": True})
    )
    monkeypatch.setattr(pilot, "power_sample", lambda: {"accepted": True})
    monkeypatch.setattr(pilot, "live_groups", lambda invocation: set())
    monkeypatch.setattr(pilot, "validate_execution_artifacts", lambda *args: None)
    monkeypatch.setattr(pilot, "verify_recovery", lambda *args: None)
    monkeypatch.setattr(
        "neural_sat_speck.backend.artifacts.validate_cohort", lambda data: None
    )

    calls: list[tuple[dict, float, dict]] = []
    outcome_map = outcomes or {}

    def fake_run_job(payload, environment, timeout):
        job = payload["job"]
        directory = Path(payload["directory"])
        directory.mkdir(parents=True)
        calls.append((job, timeout, payload))
        index = len(calls)
        atomic_json(directory / "process.json", {"pid": 10000 + index})
        atomic_json(
            directory / "worker_receipt.json",
            {"git_commit": "fixture-commit", "is_certified": True},
        )

        raw = {
            "job_id": job["job_id"],
            "job": job,
            "directory": str(directory),
            "completed_utc": f"synthetic-{index}",
            "deadline_seconds": timeout,
            "physical_worker_wall_seconds": 1.0
            if job["operation"] == "rank"
            else (
                timeout if outcome_map.get(job["job_id"]) == "global_deadline" else 2.0
            ),
            "success": False,
            "verified_known": False,
            "verified_fresh": False,
        }

        outcome = outcome_map.get(job["job_id"])
        if job["operation"] == "rank":
            raw["outcome"] = outcome or "queue_ready"
            if raw["outcome"] == "queue_ready":
                ranked = _candidate_queue()
                shuffled = queue_order.permutation(ranked, job)
                raw.update(
                    ranked=ranked,
                    shuffled=shuffled,
                    set_sha256=queue_order.ordered_hash(sorted(ranked)),
                    ranked_sha256=queue_order.ordered_hash(ranked),
                    shuffled_sha256=queue_order.ordered_hash(shuffled),
                )
        else:
            raw["outcome"] = outcome or "queue_exhausted"
            if raw["outcome"] == "verified_recovery":
                preparation = payload["preparation"]
                candidate = preparation[job["order"]][0]
                raw.update(
                    success=True,
                    verified_known=True,
                    verified_fresh=True,
                    recovered_key=[0, 1, 2, 3],
                    recovered_candidate_index=1,
                    attempts=[
                        {
                            "candidate_index": 1,
                            "candidate": list(candidate),
                            "solver_status": "sat",
                            "solve_wall_seconds": 0.1,
                            "verified_known": True,
                            "verified_fresh": True,
                        }
                    ],
                    candidates_tested=1,
                )

        atomic_json(directory / "supervisor_result.json", raw)
        if job["job_id"] == "simon32_64_k0_bao_ddt_hybrid_8parent__prepare":
            atomic_json(
                directory / "tensorflow_worker_receipt.json",
                {
                    "git_commit": "fixture-commit",
                    "is_certified": True,
                    "python_executable": str(tf_python),
                },
            )
        return read_checked_json(directory / "supervisor_result.json")

    monkeypatch.setattr(pilot, "run_job", fake_run_job)
    return manifest_path, manifest, dataset, calls


def test_tf_wrapper_smoke_blocks_protocol_and_sat_imports_in_real_subprocess():
    root = Path(__file__).resolve().parents[2]
    wrapper = root / "scripts/run_queue_order_neural_ranking.py"
    code = "\n".join(
        [
            "import importlib.abc, runpy, sys",
            "class BlockScientificImports(importlib.abc.MetaPathFinder):",
            "    def find_spec(self, fullname, path=None, target=None):",
            "        if fullname == 'pycryptosat' or fullname == 'neural_sat_speck.protocol' or fullname.startswith('neural_sat_speck.protocol.'):",
            "            raise AssertionError('forbidden import: ' + fullname)",
            "sys.meta_path.insert(0, BlockScientificImports())",
            "namespace = runpy.run_path(sys.argv[1])",
            "namespace['telemetry_module']()",
            "assert 'pycryptosat' not in sys.modules",
            "assert 'neural_sat_speck.protocol' not in sys.modules",
            "assert 'tensorflow' not in sys.modules",
            "assert 'neural_sat_speck.simon' not in sys.modules",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(wrapper)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr


def test_pilot_audit_rejects_start_invocation_detached_from_job_directory(
    tmp_path, monkeypatch
):
    manifest_path, manifest, dataset, _ = _pilot_fixture(tmp_path, monkeypatch)
    report = pilot.run(manifest_path, tmp_path / "evidence" / "pilot.json", tmp_path)
    first = pilot.pilot_jobs()[0]["job_id"]
    report["starts"][first]["invocation"] = "f" * 32
    with pytest.raises(ValueError, match="directory and invocation differ"):
        pilot.audit(report, manifest, dataset)


def test_tf_import_preflight_uses_the_actual_wrapper_without_running_main():
    root = Path(__file__).resolve().parents[2]
    from neural_sat_speck.followups.runner import validate_tf_import

    validate_tf_import(
        {
            "tf_python": Path(sys.executable),
            "source:scripts/run_queue_order_neural_ranking.py": root
            / "scripts/run_queue_order_neural_ranking.py",
        }
    )


def test_key0_pilot_dispatches_exact_four_preparations_and_eight_alternating_arms(
    tmp_path, monkeypatch
):
    manifest_path, _, _, calls = _pilot_fixture(tmp_path, monkeypatch)
    output = tmp_path / "pilot-evidence" / "pilot.json"

    report = pilot.run(manifest_path, output, tmp_path)

    expected = pilot.pilot_jobs()
    assert expected == queue_order.fixed_jobs()[:12]
    assert [entry[0]["job_id"] for entry in calls] == [
        job["job_id"] for job in expected
    ]
    assert report["status"] == "COMPLETE"
    assert report["readiness_passed"] is True
    assert len(report["records"]) == len(report["starts"]) == 12
    assert sum(job["operation"] == "rank" for job, _, _ in calls) == 4
    assert sum(job["operation"] == "solve" for job, _, _ in calls) == 8

    for offset in range(0, 12, 3):
        group = calls[offset : offset + 3]
        jobs = [entry[0] for entry in group]
        assert [job["operation"] for job in jobs] == ["rank", "solve", "solve"]
        assert len({job["case_id"] for job in jobs}) == 1
        assert [job["order"] for job in jobs[1:]] == ["ranked", "shuffled"]
        assert [entry[1] for entry in group] == [300.0, 299.0, 299.0]

    for job, _, payload in calls:
        assert set(payload["public"]) == {
            "cipher",
            "rounds",
            "instance_id",
            "ciphertext_pairs",
            "known_observations",
        }
        if job["operation"] == "rank":
            assert "fresh" not in payload and "preparation" not in payload
        else:
            assert len(payload["fresh"]) == 2


@pytest.mark.parametrize(
    "normal_outcomes",
    [
        ["queue_exhausted"] * 8,
        ["unresolved_candidates"] * 8,
        ["global_deadline"] * 8,
    ],
)
def test_normal_sat_outcomes_do_not_fail_pilot_or_shorten_schedule(
    tmp_path, monkeypatch, normal_outcomes
):
    jobs = pilot.pilot_jobs()
    outcomes = {
        job["job_id"]: outcome
        for job, outcome in zip(
            [job for job in jobs if job["operation"] == "solve"],
            normal_outcomes,
            strict=True,
        )
    }
    manifest_path, _, _, calls = _pilot_fixture(tmp_path, monkeypatch, outcomes)
    output = tmp_path / "pilot-evidence" / "pilot.json"

    report = pilot.run(manifest_path, output, tmp_path)

    assert report["status"] == "COMPLETE"
    assert report["readiness_passed"] is True
    assert len(calls) == 12
    assert {report["records"][name]["outcome"] for name in outcomes} == set(
        normal_outcomes
    )


def test_verified_recovery_does_not_short_circuit_or_tune_the_pilot_schedule(
    tmp_path, monkeypatch
):
    first_solve = next(job for job in pilot.pilot_jobs() if job["operation"] == "solve")
    manifest_path, _, _, calls = _pilot_fixture(
        tmp_path,
        monkeypatch,
        {first_solve["job_id"]: "verified_recovery"},
    )
    output = tmp_path / "pilot-evidence" / "pilot.json"

    report = pilot.run(manifest_path, output, tmp_path)

    assert report["records"][first_solve["job_id"]]["outcome"] == "verified_recovery"
    assert report["status"] == "COMPLETE"
    assert report["readiness_passed"] is True
    assert [entry[0]["job_id"] for entry in calls] == [
        job["job_id"] for job in pilot.pilot_jobs()
    ]


def test_worker_exception_stops_pilot_and_persists_readiness_false(
    tmp_path, monkeypatch
):
    alarm_job = pilot.pilot_jobs()[9]
    manifest_path, _, _, calls = _pilot_fixture(
        tmp_path, monkeypatch, {alarm_job["job_id"]: "worker_exception"}
    )
    output = tmp_path / "pilot-evidence" / "pilot.json"

    report = pilot.run(manifest_path, output, tmp_path)

    assert report["status"] == "FAILED"
    assert report["readiness_passed"] is False
    assert [entry[0]["job_id"] for entry in calls] == [
        job["job_id"] for job in pilot.pilot_jobs()[:10]
    ]
    assert alarm_job["job_id"] in report["records"]
    assert all(job["job_id"] not in report["starts"] for job in pilot.pilot_jobs()[10:])
    assert read_checked_json(output)["readiness_passed"] is False


def _gate_fixture(tmp_path: Path):
    base_files = {
        "cohort": {"path": "/fixture/cohort.json", "sha256": "a" * 64},
        "speck_stage1": {"path": "/fixture/model.pt", "sha256": "b" * 64},
    }
    previous = {
        "source_commit": "fixture-commit",
        "source_root": "/fixture/source",
        "profile": queue_order.PROFILE,
        "jobs": queue_order.fixed_jobs(),
        "runtime_versions": {"numpy": "fixture"},
        "python_version": "fixture-python",
        "tensorflow_runtime_versions": {"tensorflow": "fixture"},
        "files": base_files,
    }
    previous_path = tmp_path / "pilot-manifest.json"
    atomic_json(previous_path, previous)
    digest = hashlib.sha256(previous_path.read_bytes()).hexdigest()
    report = {
        "schema": "queue_order_actual_pilot_v1",
        "status": "COMPLETE",
        "readiness_passed": True,
        "source_commit": previous["source_commit"],
        "manifest_sha256": digest,
    }
    report_path = tmp_path / "pilot-report.json"
    atomic_json(report_path, report)
    report_record = {
        "path": str(report_path),
        "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    }
    previous_record = {
        "path": str(previous_path),
        "sha256": hashlib.sha256(previous_path.read_bytes()).hexdigest(),
    }
    current = {
        **previous,
        "execution_kind": "full",
        "pilot_gate": {
            "required": True,
            "authorization_decision": "D-050",
            "job_ids": [job["job_id"] for job in pilot.pilot_jobs()],
        },
        "files": {
            **base_files,
            "pilot_report": report_record,
            "pilot_manifest": previous_record,
        },
    }
    paths = {"pilot_report": report_path, "pilot_manifest": previous_path}
    return current, paths, report_path, previous_path


def test_full_gate_rejects_failed_pilot_report_before_audit(tmp_path, monkeypatch):
    current, paths, report_path, _ = _gate_fixture(tmp_path)
    failed = read_checked_json(report_path)
    failed["readiness_passed"] = False
    atomic_json(report_path, failed)
    paths["pilot_report"] = report_path
    audited = []
    monkeypatch.setattr(pilot, "audit", lambda *args: audited.append(args))

    with pytest.raises(ValueError, match="Pilot pass"):
        pilot.validate_gate(current, paths, {})

    assert audited == []


@pytest.mark.parametrize("drift", ["source", "asset"])
def test_full_gate_rejects_pilot_source_or_asset_drift_before_audit(
    tmp_path, monkeypatch, drift
):
    current, paths, _, _ = _gate_fixture(tmp_path)
    if drift == "source":
        current["source_commit"] = "different-source-commit"
        report = read_checked_json(paths["pilot_report"])
        report["source_commit"] = current["source_commit"]
        atomic_json(paths["pilot_report"], report)
    else:
        current["files"]["speck_stage1"] = {
            "path": "/fixture/other-model.pt",
            "sha256": "c" * 64,
        }
    audited = []
    monkeypatch.setattr(pilot, "audit", lambda *args: audited.append(args))

    with pytest.raises(ValueError, match="Pilot and replacement"):
        pilot.validate_gate(current, paths, {})

    assert audited == []


@pytest.mark.parametrize("missing", ["pilot_report", "pilot_manifest"])
def test_full_gate_rejects_missing_pilot_receipt_before_audit(
    tmp_path, monkeypatch, missing
):
    current, paths, _, _ = _gate_fixture(tmp_path)
    del paths[missing]
    audited = []
    monkeypatch.setattr(pilot, "audit", lambda *args: audited.append(args))

    with pytest.raises(ValueError, match="actual passed pilot"):
        pilot.validate_gate(current, paths, {})

    assert audited == []
