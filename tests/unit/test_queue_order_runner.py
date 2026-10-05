"""Synthetic dispatch, resume and physical watchdog checks; no attack cohort."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
import time
import uuid
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from neural_sat_speck.backend.guard import atomic_json, read_checked_json
from neural_sat_speck.followups import queue_order as protocol
from neural_sat_speck.followups import runner
from tests.unit.test_queue_order_guards import _candidate_queue, _demo_dataset


@pytest.fixture
def dispatch(tmp_path, monkeypatch):
    dataset = _demo_dataset()
    for values in dataset["ciphers"].values():
        for row in values["instances"]:
            for rd in row["rounds_data"].values():
                rd["round_keys"] = [0] * 12
    cohort = tmp_path / "fixture.json"
    cohort.write_text(json.dumps(dataset))
    manifest = {
        "source_commit": "fixture-commit",
        "profile": protocol.PROFILE,
        "jobs": protocol.fixed_jobs(),
    }
    manifest_path = tmp_path / "manifest.json"
    atomic_json(manifest_path, manifest)
    monkeypatch.setattr(
        runner,
        "load_manifest",
        lambda *a: (manifest, {"cohort": cohort, "python": Path(sys.executable)}),
    )
    monkeypatch.setattr(runner, "validate_runtime", lambda *a: None)
    monkeypatch.setattr(runner, "validate_execution_artifacts", lambda *a: None)
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda argv, **kw: SimpleNamespace(
            stdout="fixture-commit" if argv[1] == "rev-parse" else ""
        ),
    )
    monkeypatch.setattr(
        runner, "sleep_assertion", lambda: nullcontext({"synthetic": True})
    )
    monkeypatch.setattr(runner, "power_sample", lambda: {"accepted": True})
    monkeypatch.setattr(
        "neural_sat_speck.backend.artifacts.validate_cohort", lambda data: None
    )
    monkeypatch.setattr(runner, "live_groups", lambda token: set())
    calls = []

    def fake_run(payload, environment, timeout):
        job = payload["job"]
        calls.append((job["job_id"], timeout))
        directory = Path(payload["directory"])
        directory.mkdir(parents=True)
        raw = {
            "job_id": job["job_id"],
            "job": job,
            "directory": str(directory),
            "completed_utc": "synthetic",
            "physical_worker_wall_seconds": 1.0 if job["operation"] == "rank" else 2.0,
            "success": False,
            "deadline_seconds": timeout,
        }
        if job["operation"] == "rank":
            ranked = _candidate_queue()
            shuffled = protocol.permutation(ranked, job)
            raw.update(
                outcome="queue_ready",
                ranked=ranked,
                shuffled=shuffled,
                set_sha256=protocol.ordered_hash(sorted(ranked)),
                ranked_sha256=protocol.ordered_hash(ranked),
                shuffled_sha256=protocol.ordered_hash(shuffled),
            )
        else:
            assert "fresh" in payload and len(payload["fresh"]) == 2
            raw.update(outcome="queue_exhausted")
        atomic_json(directory / "supervisor_result.json", raw)
        # Match subprocess JSON round-trip types rather than in-memory tuples.
        return read_checked_json(directory / "supervisor_result.json")

    monkeypatch.setattr(runner, "run_job", fake_run)
    return (
        tmp_path,
        manifest_path,
        tmp_path / "evidence" / "ledger.json",
        manifest,
        dataset,
        calls,
    )


def test_complete_schedule_reuses_set_not_solver_and_charges_preparation(dispatch):
    root, manifest_path, output, _, _, calls = dispatch
    ledger = runner.run(manifest_path, output, root)
    assert ledger["status"] == "COMPLETE"
    assert len(ledger["records"]) == len(ledger["starts"]) == len(calls) == 96
    assert all(limit == (300 if "__prepare" in name else 299) for name, limit in calls)
    for record in ledger["records"].values():
        if record["job"]["operation"] == "solve":
            assert record["charged_online_seconds"] == 3
    calls.clear()
    resumed = runner.run(manifest_path, output, root, resume=True)
    assert resumed["records"] == ledger["records"]
    assert calls == []


def test_existing_output_needs_explicit_resume(dispatch):
    root, manifest_path, output, *_ = dispatch
    runner.run(manifest_path, output, root)
    with pytest.raises(FileExistsError):
        runner.run(manifest_path, output, root)


def test_started_without_result_is_unavailable_and_never_retried(dispatch):
    root, manifest_path, output, _, _, calls = dispatch
    complete = runner.run(manifest_path, output, root)
    jobs = protocol.fixed_jobs()
    interrupted = copy.deepcopy(complete)
    interrupted["records"] = {jobs[0]["job_id"]: complete["records"][jobs[0]["job_id"]]}
    interrupted["starts"] = {
        j["job_id"]: complete["starts"][j["job_id"]] for j in jobs[:2]
    }
    start = interrupted["starts"][jobs[1]["job_id"]]
    Path(start["directory"], "supervisor_result.json").unlink()
    interrupted["status"] = "ACTIVE"
    atomic_json(output, interrupted)
    # Remaining jobs were not actually started in this reconstructed fixture.
    for j in jobs[2:]:
        import shutil

        shutil.rmtree(Path(complete["starts"][j["job_id"]]["directory"]))
    calls.clear()
    result = runner.run(manifest_path, output, root, resume=True)
    missing = result["records"][jobs[1]["job_id"]]
    assert missing["outcome"] == "infrastructure_interrupted"
    assert missing["charged_online_seconds"] is None
    assert missing["common_preparation_seconds"] == 1
    assert jobs[0]["job_id"] not in [n for n, _ in calls]
    assert jobs[1]["job_id"] not in [n for n, _ in calls]
    assert result["status"] == "COMPLETE"


@pytest.mark.parametrize(
    "defect", ["hash", "start_path", "artifact", "schedule", "profile"]
)
def test_resume_rejects_corruption_before_dispatch(dispatch, defect):
    root, manifest_path, output, _, _, calls = dispatch
    ledger = runner.run(manifest_path, output, root)
    first = protocol.fixed_jobs()[0]["job_id"]
    if defect == "hash":
        output.write_text(output.read_text() + " ")
    elif defect == "start_path":
        ledger["starts"][first]["directory"] = str(root / "unrelated")
        atomic_json(output, ledger)
    elif defect == "artifact":
        path = Path(ledger["starts"][first]["directory"]) / "supervisor_result.json"
        path.write_text("{}")
    elif defect == "schedule":
        del ledger["records"][first]
        atomic_json(output, ledger)
    else:
        ledger["profile"] = {**ledger["profile"], "solver_threads": True}
        atomic_json(output, ledger)
    calls.clear()
    with pytest.raises(ValueError):
        runner.run(manifest_path, output, root, resume=True)
    assert calls == []


def test_exact_cutoff_recovery_is_retained_as_late_failure():
    job = next(j for j in protocol.fixed_jobs() if j["operation"] == "solve")
    prep = {"common_preparation_seconds": 20, "ranked_sha256": "a" * 64}
    raw = {
        "job_id": job["job_id"],
        "outcome": "verified_recovery",
        "success": True,
        "verified_known": True,
        "verified_fresh": True,
        "recovered_key": [1, 2, 3, 4],
        "physical_worker_wall_seconds": 280,
    }
    result = runner.normalize(job, raw, prep)
    assert result["outcome"] == "global_deadline"
    assert result["success"] is False and result["late_verified_result"] is True


@pytest.mark.parametrize(
    "phase",
    [
        "ranking",
        "initial_formula_construction",
        "solving",
        "verification",
        "rebuilding_formula",
    ],
)
@pytest.mark.parametrize("initial_rejection", [False, True])
def test_real_guardian_kills_hanging_followup_worker_in_every_phase(
    tmp_path, monkeypatch, phase, initial_rejection
):
    source = tmp_path / "source"
    package = source / "neural_sat_speck"
    for subpackage in ("backend", "followups"):
        (package / subpackage).mkdir(parents=True)
        (package / subpackage / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    actual = Path(runner.__file__).parents[1] / "backend"
    for name in ("guard.py", "watchdog.py"):
        (package / "backend" / name).write_text((actual / name).read_text())
    (package / "followups" / "watchdog.py").write_text(
        (Path(runner.__file__).parent / "watchdog.py").read_text()
    )
    (package / "followups" / "worker.py").write_text(
        "import sys,time\nfrom pathlib import Path\nfrom neural_sat_speck.backend.guard import read_checked_json,atomic_json\n"
        "p=read_checked_json(Path(sys.argv[1]))\n"
        f"atomic_json(Path(p['directory'])/'journal.json',{{'phase':{phase!r}}})\n"
        "time.sleep(60)\n"
    )
    monkeypatch.setattr(
        runner, "power_sample", lambda: {"accepted": not initial_rejection}
    )
    monkeypatch.setattr(runner, "native_policy_sample", lambda pid: {"accepted": True})
    monkeypatch.setattr(runner, "memory_sample", lambda group: {"accepted": True})
    invocation = uuid.uuid4().hex
    job = next(
        j
        for j in protocol.fixed_jobs()
        if j["operation"] == ("rank" if phase == "ranking" else "solve")
    )
    payload = {
        "directory": str(tmp_path / invocation),
        "invocation": invocation,
        "repo_root": str(source),
        "python_path": sys.executable,
        "job": job,
    }
    result = runner.run_job(
        payload, {**os.environ, "PYTHONPATH": str(source)}, timeout=0.4
    )
    assert result["outcome"] == (
        "environment_rejected"
        if initial_rejection
        else "ranking_timeout"
        if phase == "ranking"
        else "global_deadline"
    )
    assert result["physical_worker_wall_seconds"] < 1
    assert result["guardian_receipt"]["process_group_absent"] is True
    if not initial_rejection:
        assert result["journal"]["phase"] == phase


def test_frozen_analysis_counts_all_assigned_trials(dispatch):
    root, manifest_path, output, manifest, dataset, _ = dispatch
    ledger = runner.run(manifest_path, output, root)
    path = Path(__file__).parents[2] / "scripts/analyze_queue_order_diagnostic.py"
    spec = importlib.util.spec_from_file_location("order_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import hashlib

    result = module.analyze(
        ledger,
        manifest,
        hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        dataset,
    )
    assert len(result["cells"]) == 64 and len(result["summaries"]) == 4
    for row in result["summaries"]:
        assert row["assigned_pairs"] == row["completed_pairs"] == row["n00"] == 8
        assert row["ranked_verified_of8"] == row["shuffled_verified_of8"] == 0
        assert row["ranked_assigned_par2_seconds"] == 600


@pytest.mark.parametrize("before_dispatch", [False, True])
def test_coordinator_death_stops_child_before_its_original_deadline(
    tmp_path, before_dispatch
):
    source = tmp_path / "source"
    package = source / "neural_sat_speck"
    for subpackage in ("backend", "followups"):
        (package / subpackage).mkdir(parents=True)
        (package / subpackage / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    actual = Path(runner.__file__).parents[1] / "backend"
    for name in ("guard.py", "watchdog.py"):
        contents = (actual / name).read_text()
        if before_dispatch and name == "watchdog.py":
            contents = contents.replace(
                '        if stopped:\n            raise RuntimeError("Supervisor stopped before dispatch")',
                '        (directory / "guardian_ready").touch()\n        time.sleep(0.4)\n        if stopped:\n            raise RuntimeError("Supervisor stopped before dispatch")',
                1,
            )
        (package / "backend" / name).write_text(contents)
    (package / "followups" / "watchdog.py").write_text(
        (Path(runner.__file__).parent / "watchdog.py").read_text()
    )
    (package / "followups" / "worker.py").write_text("import time\ntime.sleep(60)\n")
    directory = tmp_path / uuid.uuid4().hex
    payload = {
        "directory": str(directory),
        "invocation": directory.name,
        "repo_root": str(source),
        "python_path": sys.executable,
        "job": protocol.fixed_jobs()[0],
    }
    input_path = tmp_path / "coordinator.json"
    input_path.write_text(
        json.dumps(
            {
                "payload": payload,
                "environment": {**os.environ, "PYTHONPATH": str(source)},
            }
        )
    )
    actual_source = str(Path(runner.__file__).parents[2])
    code = (
        "import json,sys\n"
        f"sys.path.insert(0,{actual_source!r})\n"
        "from neural_sat_speck.followups import runner\n"
        "runner.power_sample=lambda:{'accepted':True}\n"
        "runner.native_policy_sample=lambda pid:{'accepted':True}\n"
        "runner.memory_sample=lambda group:{'accepted':True}\n"
        "p=json.load(open(sys.argv[1]))\n"
        "runner.run_job(p['payload'],p['environment'],8)\n"
    )
    coordinator = subprocess.Popen(
        [sys.executable, "-c", code, str(input_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        started = time.monotonic()
        marker = directory / ("guardian_ready" if before_dispatch else "process.json")
        while not marker.exists() and time.monotonic() - started < 3:
            if coordinator.poll() is not None:
                pytest.fail(coordinator.stderr.read().decode())
            time.sleep(0.02)
        assert marker.exists()
        coordinator.kill()
        coordinator.wait(timeout=2)
        stopped = time.monotonic()
        receipt = None
        while time.monotonic() - stopped < 2:
            try:
                candidate = read_checked_json(directory / "guardian_receipt.json")
            except (OSError, ValueError):
                candidate = None
            # The JSON and sidecar are separate atomic replacements; the
            # wrapper may also replace a transient error with the stop receipt.
            if candidate is not None and candidate.get("status") == "coordinator_stop":
                receipt = candidate
                break
            time.sleep(0.02)
        assert receipt is not None, "No checked final stop receipt within deadline"
        assert receipt["status"] == "coordinator_stop"
        assert receipt["process_group_absent"] is True
        assert (
            read_checked_json(directory / "coordinator_loss.json")["coordinator_pid"]
            == coordinator.pid
        )
        assert receipt["observed_monotonic"] < receipt["deadline_monotonic"]
        restored = runner.observe_finished(
            directory,
            job=payload["job"],
            started=receipt["deadline_monotonic"] - 8,
            timeout=8,
        )
        restored.update(deadline_seconds=8)
        runner.validate_execution_artifacts(directory, payload["job"], restored)
        if before_dispatch:
            assert receipt["worker_not_started"] is True
            assert not (directory / "process.json").exists()
    finally:
        if coordinator.poll() is None:
            coordinator.kill()
        coordinator.wait(timeout=2)
        coordinator.stderr.close()


@pytest.mark.parametrize("cipher", ["speck32_64", "simon32_64"])
def test_recovery_audit_checks_trailing_keys_and_fresh_blocks(cipher):
    if cipher == "speck32_64":
        from neural_sat_speck.speck.cipher import encrypt_block
        from neural_sat_speck.speck.key_schedule import expand_key
    else:
        from neural_sat_speck.simon.cipher import encrypt_block
        from neural_sat_speck.simon.key_schedule import expand_key
    key = (0x1918, 0x1110, 0x0908, 0x0100)
    known = [[(i, 0), encrypt_block((i, 0), key, 4)] for i in range(128)]
    fresh = [[pt, encrypt_block(pt, key, 4)] for pt in ((40000, 1), (40001, 1))]
    candidate = list(expand_key(key, 4)[-2:])
    public = {"cipher": cipher, "rounds": 4, "known_observations": known}
    record = {
        "success": True,
        "recovered_key": list(key),
        "recovered_candidate_index": 1,
        "attempts": [
            {"solver_status": "sat", "candidate_index": 1, "candidate": candidate}
        ],
    }
    runner.verify_recovery(record, public, fresh)
    corrupted = copy.deepcopy(record)
    corrupted["attempts"][0]["candidate"][0] ^= 1
    with pytest.raises(ValueError, match="trailing"):
        runner.verify_recovery(corrupted, public, fresh)
    bad_fresh = copy.deepcopy(fresh)
    bad_fresh[0][1] = (bad_fresh[0][1][0] ^ 1, bad_fresh[0][1][1])
    with pytest.raises(ValueError, match="verify"):
        runner.verify_recovery(record, public, bad_fresh)


def test_dispatched_environment_alarm_blocks_resume_before_another_worker(
    dispatch, monkeypatch
):
    root, manifest_path, output, _, _, calls = dispatch
    original = runner.run_job

    def reject(payload, environment, timeout):
        raw = original(payload, environment, timeout)
        raw.update(outcome="environment_rejected", environment_rejection=True)
        atomic_json(Path(raw["directory"]) / "supervisor_result.json", raw)
        return raw

    monkeypatch.setattr(runner, "run_job", reject)
    ledger = runner.run(manifest_path, output, root)
    assert ledger["status"] == "STOPPED_ALARM"
    assert len(ledger["records"]) == len(calls) == 1
    calls.clear()
    with pytest.raises(RuntimeError, match="alarm"):
        runner.run(manifest_path, output, root, resume=True)
    assert calls == []


def test_native_extension_version_works_without_distribution_metadata(monkeypatch):
    def metadata(name):
        if name == "pycryptosat":
            raise runner.importlib.metadata.PackageNotFoundError(name)
        return "fixture-numeric-version"

    monkeypatch.setattr(runner.importlib.metadata, "version", metadata)
    monkeypatch.setitem(
        sys.modules,
        "pycryptosat",
        SimpleNamespace(__version__="fixture-native-version"),
    )
    assert runner.runtime_versions()["pycryptosat"] == "fixture-native-version"
