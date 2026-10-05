"""Actual fixed key0 readiness pilot, separate from the diagnostic analysis."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import uuid
from pathlib import Path

from neural_sat_speck.backend.environment import sleep_assertion
from neural_sat_speck.backend.guard import atomic_json, read_checked_json
from neural_sat_speck.followups.environment import power_sample, same_python_environment
from neural_sat_speck.followups.queue_order import (
    fixed_jobs,
    fresh_observations,
    load_manifest,
    ordered_hash,
    public_case,
    validate_preparation,
    validate_replay,
)
from neural_sat_speck.followups.runner import (
    live_groups,
    normalize,
    now,
    run_job,
    validate_execution_artifacts,
    validate_runtime,
    verify_recovery,
)


def pilot_jobs() -> list[dict]:
    return fixed_jobs()[:12]


def audit(report: dict, manifest: dict, dataset: dict) -> None:
    names = {j["job_id"] for j in pilot_jobs()}
    if report.get("status") != "COMPLETE" or (
        set(report.get("records", {})) != names
        or set(report.get("starts", {})) != names
        or report.get("source_commit") != manifest["source_commit"]
    ):
        raise ValueError("Incomplete pilot grid or source identity")
    pids = set()
    evidence_root = Path(report.get("evidence_root", ""))
    if not evidence_root.is_absolute():
        raise ValueError("Pilot evidence root must be absolute")
    for job in pilot_jobs():
        row = report["records"][job["job_id"]]
        start = report["starts"][job["job_id"]]
        token = start.get("invocation", "")
        if (
            not isinstance(token, str)
            or len(token) != 32
            or any(c not in "0123456789abcdef" for c in token)
        ):
            raise ValueError("Invalid pilot invocation")
        if start.get("directory") != str(
            evidence_root / "jobs" / job["job_id"] / token
        ):
            raise ValueError("Pilot directory and invocation differ")
        directory = Path(start["directory"])
        raw = read_checked_json(directory / "supervisor_result.json")
        validate_execution_artifacts(directory, job, raw)
        process = read_checked_json(directory / "process.json")
        if process["pid"] in pids:
            raise ValueError("Pilot worker PID reused")
        pids.add(process["pid"])
        receipt = read_checked_json(directory / "worker_receipt.json")
        if (
            receipt["git_commit"] != manifest["source_commit"]
            or receipt["is_certified"] is not True
        ):
            raise ValueError("Invalid actual pilot worker receipt")
        if live_groups(start["invocation"]):
            raise ValueError("Pilot worker group still live")
        prep = report["records"].get(job["case_id"] + "__prepare")
        if ordered_hash(
            normalize(job, raw, prep if job["operation"] == "solve" else None)
        ) != ordered_hash(row):
            raise ValueError("Pilot record differs from raw execution")
        if "environment_alarm" in row or row.get("verification_failures", 0) > 0:
            raise ValueError("Pilot environment or verification alarm")
        if job["operation"] == "rank":
            if row["outcome"] != "queue_ready":
                raise ValueError("Pilot did not exercise a complete queue")
            validate_preparation(row, job)
            if job["policy"] == "bao_ddt_hybrid_8parent":
                tf = read_checked_json(directory / "tensorflow_worker_receipt.json")
                if (
                    tf["is_certified"] is not True
                    or tf["git_commit"] != manifest["source_commit"]
                ):
                    raise ValueError("Invalid actual TensorFlow receipt")
                if not same_python_environment(
                    tf["python_executable"],
                    Path(manifest["files"]["tf_python"]["path"]),
                ):
                    raise ValueError("TensorFlow interpreter differs")
        else:
            validate_replay(row, job, prep)
            if row["outcome"] not in {
                "verified_recovery",
                "queue_exhausted",
                "unresolved_candidates",
                "global_deadline",
            }:
                raise ValueError("Pilot replay has a harness or verification failure")
            verify_recovery(
                row, public_case(dataset, job), fresh_observations(dataset, job)
            )


def validate_gate(manifest: dict, paths: dict[str, Path], dataset: dict) -> None:
    """Full replacement run requires the audited pilot from identical source."""
    if ordered_hash(manifest.get("pilot_gate")) != ordered_hash(
        {
            "required": True,
            "authorization_decision": "D-050",
            "job_ids": [j["job_id"] for j in pilot_jobs()],
        }
    ):
        raise ValueError("Missing fixed pilot authorization")
    if "pilot_report" not in paths or "pilot_manifest" not in paths:
        raise ValueError("Full run requires an actual passed pilot")
    report = read_checked_json(paths["pilot_report"])
    previous = read_checked_json(paths["pilot_manifest"])
    if (
        report.get("readiness_passed") is not True
        or report.get("manifest_sha256")
        != hashlib.sha256(paths["pilot_manifest"].read_bytes()).hexdigest()
    ):
        raise ValueError("Pilot pass or manifest binding differs")
    for field in (
        "source_commit",
        "source_root",
        "profile",
        "jobs",
        "runtime_versions",
        "python_version",
        "tensorflow_runtime_versions",
    ):
        if ordered_hash(previous.get(field)) != ordered_hash(manifest.get(field)):
            raise ValueError("Pilot and replacement scientific identity differs")
    if {
        k: v
        for k, v in manifest["files"].items()
        if k not in {"pilot_report", "pilot_manifest"}
    } != previous["files"]:
        raise ValueError("Pilot and replacement pinned files differ")
    audit(report, previous, dataset)


def run(manifest_path: Path, output: Path, root: Path) -> dict:
    if output.exists():
        raise FileExistsError("Pilot evidence cannot be overwritten or resumed")
    manifest, paths = load_manifest(manifest_path, root)
    if manifest.get("execution_kind") != "pilot":
        raise ValueError("Pilot requires a dedicated pilot manifest")
    if ordered_hash(manifest.get("pilot_gate")) != ordered_hash(
        {
            "required": True,
            "authorization_decision": "D-050",
            "job_ids": [j["job_id"] for j in pilot_jobs()],
        }
    ):
        raise ValueError("Pilot does not have fixed author authorization")
    validate_runtime(manifest, paths)
    if (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        != manifest["source_commit"]
    ):
        raise ValueError("Pilot executed source differs")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, text=True
    ):
        raise ValueError("Pilot source is not clean")
    dataset = json.loads(paths["cohort"].read_bytes())
    from neural_sat_speck.backend.artifacts import validate_cohort

    validate_cohort(dataset)
    report = {
        "schema": "queue_order_actual_pilot_v1",
        "role": "opened_demo_readiness_pilot",
        "source_commit": manifest["source_commit"],
        "evidence_root": str(output.parent.absolute()),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "status": "ACTIVE",
        "readiness_passed": False,
        "started_utc": now(),
        "starts": {},
        "records": {},
        "limits": [
            "No tuning or merging into replacement analysis",
            "Pilot may warm OS page caches before replacement",
        ],
    }
    env = {
        **os.environ,
        "PYTHONPATH": str(root / "src"),
        "TMPDIR": str(root / "runtime/queue_order_tmp"),
        "TF_NUM_INTRAOP_THREADS": "8",
        "TF_NUM_INTEROP_THREADS": "1",
    }
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[name] = "1"
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with sleep_assertion() as assertion:
        import time

        for _ in range(20):
            if power_sample().get("accepted") is True:
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("Pilot sleep/power preflight rejected")
        report["sleep_assertion"] = assertion
        atomic_json(output, report)
        for job in pilot_jobs():
            prep = report["records"].get(job["case_id"] + "__prepare")
            timeout = (
                300.0
                if job["operation"] == "rank"
                else 300.0 - prep["common_preparation_seconds"]
            )
            token = uuid.uuid4().hex
            directory = output.parent / "jobs" / job["job_id"] / token
            report["starts"][job["job_id"]] = {
                "invocation": token,
                "directory": str(directory),
            }
            atomic_json(output, report)
            payload = {
                "job": job,
                "public": public_case(dataset, job),
                "paths": {k: str(v) for k, v in paths.items()},
                "directory": str(directory),
                "invocation": token,
                "repo_root": str(root),
                "source_commit": manifest["source_commit"],
                "python_path": str(paths["python"]),
            }
            if job["operation"] == "solve":
                payload.update(preparation=prep, fresh=fresh_observations(dataset, job))
            raw = run_job(payload, env, timeout)
            row = normalize(job, raw, prep)
            report["records"][job["job_id"]] = row
            atomic_json(output, report)
            print(
                f"Pilot {len(report['records'])}/12 {job['job_id']}: {row['outcome']}",
                flush=True,
            )
            if (
                (job["operation"] == "rank" and row["outcome"] != "queue_ready")
                or row["outcome"]
                in {
                    "worker_exception",
                    "environment_rejected",
                    "verification_failure",
                }
                or row.get("verification_failures", 0) > 0
                or "environment_alarm" in row
            ):
                report.update(status="FAILED", stopped_utc=now())
                atomic_json(output, report)
                return report
        report.update(status="COMPLETE", completed_utc=now())
        audit(report, manifest, dataset)
        report["readiness_passed"] = True
        atomic_json(output, report)
    return report
