"""Serial, guarded execution of a fixed queue-order diagnostic grid."""

from __future__ import annotations

import fcntl
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from neural_sat_speck.backend.environment import (
    native_policy_sample,
    sleep_assertion,
)
from neural_sat_speck.backend.guard import atomic_json, read_checked_json
from neural_sat_speck.followups.environment import power_sample, same_python_environment
from neural_sat_speck.followups.queue_order import (
    FINAL_OUTCOMES,
    PROFILE,
    fixed_jobs,
    fresh_observations,
    load_manifest,
    ordered_hash,
    public_case,
    seconds,
    validate_preparation,
    validate_replay,
)


def now() -> str:
    return datetime.now(UTC).isoformat()


def runtime_versions() -> dict[str, str]:
    """Locally built solver extensions need not ship wheel metadata."""
    import pycryptosat

    versions = {n: importlib.metadata.version(n) for n in ("numpy", "torch")}
    native = getattr(pycryptosat, "__version__", None)
    if not isinstance(native, str) or not native:
        raise ValueError("Native solver extension does not expose a version")
    versions["pycryptosat"] = native
    return versions


def validate_tf_import(paths: dict[str, Path]) -> None:
    """Exercise the exact TF interpreter import boundary without inference."""
    probe = subprocess.run(
        [
            str(paths["tf_python"]),
            "-c",
            "import runpy,sys; m=runpy.run_path(sys.argv[1]); "
            "m['telemetry_module'](); "
            "assert 'pycryptosat' not in sys.modules; "
            "assert 'neural_sat_speck.protocol' not in sys.modules; "
            "import json; print(json.dumps({'executable':sys.executable}))",
            str(paths["source:scripts/run_queue_order_neural_ranking.py"]),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    if not same_python_environment(
        json.loads(probe.stdout)["executable"], paths["tf_python"]
    ):
        raise ValueError("Actual TensorFlow interpreter environment differs")


def validate_runtime(manifest: dict, paths: dict[str, Path]) -> None:
    if Path(sys.executable).absolute() != paths[
        "python"
    ] or sys.version != manifest.get("python_version"):
        raise ValueError("Executed interpreter differs from pinned runtime")
    observed = runtime_versions()
    if observed != manifest.get("runtime_versions"):
        raise ValueError("Numeric/solver runtime versions differ")
    spec = importlib.util.find_spec("pycryptosat")
    if spec is None or Path(spec.origin).absolute() != paths["pycryptosat"]:
        raise ValueError("Imported solver binary differs")
    probe = subprocess.run(
        [
            str(paths["tf_python"]),
            "-c",
            "import importlib.metadata,json;print(json.dumps({n:importlib.metadata.version(n) for n in ('numpy','tensorflow','keras','h5py')}))",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    if json.loads(probe.stdout) != manifest.get("tensorflow_runtime_versions"):
        raise ValueError("TensorFlow runtime versions differ")
    validate_tf_import(paths)


def memory_sample(group: int) -> dict:
    observed = subprocess.run(
        ["/bin/ps", "-ax", "-o", "pgid=,rss="],
        capture_output=True,
        text=True,
        check=True,
        timeout=2,
    )
    members = [
        int(v[1]) * 1024
        for line in observed.stdout.splitlines()
        if len(v := line.split()) == 2 and int(v[0]) == group
    ]
    total = sum(members)
    return {
        "group_rss_bytes": total,
        "members_observed": len(members),
        "accepted": total <= PROFILE["process_group_rss_limit_bytes"],
    }


def live_groups(invocation: str) -> set[int]:
    """Observe invocation identity; never signal a stored/recycled PID on resume."""
    if len(invocation) != 32 or any(c not in "0123456789abcdef" for c in invocation):
        raise ValueError("Invalid invocation token")
    observed = subprocess.run(
        ["/bin/ps", "-ax", "-o", "pgid=,command="],
        capture_output=True,
        text=True,
        check=True,
        timeout=2,
    )
    groups = set()
    for line in observed.stdout.splitlines():
        values = line.strip().split(None, 1)
        if (
            len(values) == 2
            and invocation in values[1]
            and (
                "neural_sat_speck.followups.worker" in values[1]
                or "neural_sat_speck.followups.watchdog" in values[1]
            )
        ):
            group = int(values[0])
            if group <= 1 or group == os.getpgrp():
                raise RuntimeError("Unsafe live invocation group")
            groups.add(group)
    return groups


def validate_execution_artifacts(directory: Path, job: dict, raw: dict) -> None:
    """Bind checked files to this start, rather than trusting detached hashes alone."""
    if (
        raw.get("job_id") != job["job_id"]
        or ordered_hash(raw.get("job")) != ordered_hash(job)
        or raw.get("directory") != str(directory)
    ):
        raise ValueError("Supervisor artifact belongs to another job/directory")
    if raw.get("timing_unavailable") is True:
        if raw.get("outcome") != "infrastructure_interrupted":
            raise ValueError("Only interrupted starts can lack a guardian receipt")
        return
    payload = read_checked_json(directory / "payload.json")
    guardian = read_checked_json(directory / "guardian.json")
    receipt = read_checked_json(directory / "guardian_receipt.json")
    no_child = receipt.get("worker_not_started") is True
    if no_child and (
        (directory / "process.json").exists()
        or receipt.get("status") != "coordinator_stop"
        or "worker_pid" in receipt
        or "process_group" in receipt
    ):
        raise ValueError("Contradictory pre-dispatch stop receipt")
    process = (
        {"invocation": directory.name, "pid": None, "process_group": None}
        if no_child
        else read_checked_json(directory / "process.json")
    )
    token = directory.name
    expected_arguments = [
        payload["python_path"],
        "-m",
        "neural_sat_speck.followups.worker",
        str(directory / "payload.json"),
    ]
    if (
        payload.get("invocation") != token
        or payload.get("directory") != str(directory)
        or ordered_hash(payload.get("job")) != ordered_hash(job)
        or guardian.get("invocation") != token
        or guardian.get("directory") != str(directory)
        or guardian.get("repo_root") != payload.get("repo_root")
        or guardian.get("worker_arguments") != expected_arguments
        or guardian.get("receipt") != str(directory / "guardian_receipt.json")
        or receipt.get("invocation") != token
        or process.get("invocation") != token
        or receipt.get("worker_pid") != process.get("pid")
        or receipt.get("process_group") != process.get("process_group")
        or (not no_child and type(process.get("pid")) is not int)
        or (not no_child and process["pid"] <= 1)
        or process["pid"] != process.get("process_group")
        or receipt.get("process_group_absent") is not True
        or receipt.get("status")
        not in {"worker_exit", "deadline_enforced", "coordinator_stop"}
        or raw.get("guardian_receipt") != receipt
    ):
        raise ValueError("Guardian/process/payload identity differs from started job")
    deadline = seconds(payload.get("deadline_monotonic"))
    if deadline != seconds(guardian.get("deadline_monotonic")) or deadline != seconds(
        receipt.get("deadline_monotonic")
    ):
        raise ValueError("Guardian deadline identity differs")
    observed = seconds(receipt.get("observed_monotonic"))
    if (receipt["status"] == "deadline_enforced" and observed < deadline) or (
        receipt["status"] == "worker_exit" and observed >= deadline
    ):
        raise ValueError("Guardian status contradicts deadline clock")
    if (
        not abs(
            seconds(raw.get("physical_worker_wall_seconds"))
            - (
                seconds(receipt.get("observed_monotonic"))
                - (deadline - seconds(raw.get("deadline_seconds")))
            )
        )
        < 1e-7
    ):
        raise ValueError("Physical duration differs from guardian clock")


def observe_finished(
    directory: Path, *, job: dict, started: float, timeout: float
) -> dict:
    guardian = read_checked_json(directory / "guardian_receipt.json")
    if (
        guardian.get("process_group_absent") is not True
        or guardian.get("status") == "guardian_error"
    ):
        raise RuntimeError("Guardian did not prove child process-group cleanup")
    elapsed = seconds(guardian["observed_monotonic"] - started)
    result_path = directory / "result.json"
    if guardian["status"] == "deadline_enforced" or elapsed >= timeout:
        outcome = "ranking_timeout" if job["operation"] == "rank" else "global_deadline"
        result = {"job_id": job["job_id"], "outcome": outcome, "success": False}
    elif (
        guardian["status"] != "worker_exit"
        or guardian.get("worker_returncode") != 0
        or not result_path.exists()
    ):
        result = {
            "job_id": job["job_id"],
            "outcome": "worker_exception",
            "success": False,
        }
    else:
        result = read_checked_json(result_path)
        if result.get("job_id") != job["job_id"]:
            raise ValueError("Result belongs to another job")
    result.update(
        physical_worker_wall_seconds=elapsed,
        guardian_receipt=guardian,
        job=job,
        completed_utc=now(),
        directory=str(directory),
    )
    for filename in (
        "journal.json",
        "candidate_progress.json",
        "worker_receipt.json",
        "tensorflow_worker_receipt.json",
        "environment_samples.json",
        "environment_rejection.json",
    ):
        p = directory / filename
        if p.exists():
            result[filename.removesuffix(".json")] = read_checked_json(p)
    return result


def run_job(payload: dict, environment: dict, timeout: float) -> dict:
    """The reused independent guardian, rather than coordinator lifetime, owns child."""
    directory = Path(payload["directory"])
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    payload["deadline_monotonic"] = started + timeout
    atomic_json(directory / "payload.json", payload)
    atomic_json(
        directory / "guardian.json",
        {
            "deadline_monotonic": payload["deadline_monotonic"],
            "invocation": payload["invocation"],
            "directory": str(directory),
            "repo_root": payload["repo_root"],
            "coordinator_pid": os.getpid(),
            "worker_arguments": [
                payload["python_path"],
                "-m",
                "neural_sat_speck.followups.worker",
                str(directory / "payload.json"),
            ],
            "receipt": str(directory / "guardian_receipt.json"),
        },
    )
    samples = []
    environment_rejected = False
    with (directory / "guardian_stderr.txt").open("w") as stderr:
        guardian = subprocess.Popen(
            [
                payload["python_path"],
                "-m",
                "neural_sat_speck.followups.watchdog",
                str(directory / "guardian.json"),
            ],
            env=environment,
            cwd=payload["repo_root"],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=stderr,
        )
    try:
        next_sample = started
        while guardian.poll() is None:
            if time.perf_counter() >= next_sample:
                next_sample = time.perf_counter() + 10
                power = power_sample()
                sample = {"observed_utc": now(), "power": power}
                process_path = directory / "process.json"
                if process_path.exists():
                    identity = read_checked_json(process_path)
                    sample["worker_policy"] = native_policy_sample(identity["pid"])
                    sample["memory"] = memory_sample(identity["process_group"])
                samples.append(sample)
                atomic_json(directory / "environment_samples.json", samples)
                policy = sample.get("worker_policy", {})
                errors = [v["errno"] for v in policy.get("observations", {}).values()]
                if (
                    power.get("accepted") is not True
                    or sample.get("memory", {}).get("accepted") is False
                    or (policy.get("accepted") is False and 3 not in errors)
                ):
                    environment_rejected = True
                    atomic_json(directory / "environment_rejection.json", sample)
            # Wait for the independent guardian to install handlers and own a
            # child before asking it to clean up an environment rejection.
            if environment_rejected and (directory / "process.json").exists():
                guardian.terminate()
                break
            if time.perf_counter() > started + timeout + 5:
                guardian.terminate()
                break
            time.sleep(0.05)
        guardian.wait(timeout=8)
    except BaseException:
        if guardian.poll() is None:
            guardian.terminate()
        guardian.wait(timeout=8)
        raise
    result = observe_finished(
        directory, job=payload["job"], started=started, timeout=timeout
    )
    if environment_rejected:
        result.update(
            outcome="environment_rejected",
            success=False,
            environment_rejection=True,
            discarded_verified_result=result.get("success") is True,
            verified_known=False,
            verified_fresh=False,
        )
    result["started_monotonic"] = started
    result["deadline_seconds"] = timeout
    validate_execution_artifacts(directory, payload["job"], result)
    atomic_json(directory / "supervisor_result.json", result)
    return result


def normalize(job: dict, result: dict, preparation: dict | None) -> dict:
    """Keep physical and charged replay cost separate; reject late positive results."""
    result = dict(result)
    result.setdefault("verified_known", False)
    result.setdefault("verified_fresh", False)
    journal = result.get("journal")
    if journal is not None:
        completed = {
            name: seconds(value) for name, value in journal["completed_seconds"].items()
        }
        active = journal.get("phase")
        inflight = (
            seconds(
                result["guardian_receipt"]["observed_monotonic"]
                - journal["phase_started_monotonic"]
            )
            if active is not None
            else 0.0
        )
        result["phase_accounting"] = {
            "completed_seconds": completed,
            "inflight_phase": active,
            "inflight_seconds": inflight,
        }
    else:
        result["phase_accounting"] = None
    if job["operation"] == "rank":
        result["common_preparation_seconds"] = result["physical_worker_wall_seconds"]
        if result.get("outcome") == "queue_ready":
            validate_preparation(result, job)
        return result
    common = preparation["common_preparation_seconds"] if preparation else 0.0
    if result.get("timing_unavailable") is True:
        result.update(
            common_preparation_seconds=common,
            replay_wall_seconds=None,
            charged_online_seconds=None,
            queue_sha256=preparation[job["order"] + "_sha256"] if preparation else None,
        )
        validate_replay(result, job, preparation)
        return result
    result.update(
        common_preparation_seconds=common,
        replay_wall_seconds=result["physical_worker_wall_seconds"],
        charged_online_seconds=common + result["physical_worker_wall_seconds"],
        queue_sha256=preparation[job["order"] + "_sha256"] if preparation else None,
    )
    if result.get("success") is True and result["charged_online_seconds"] >= 300:
        result.update(
            outcome="global_deadline", success=False, late_verified_result=True
        )
    validate_replay(result, job, preparation)
    return result


def inherited_failure(job: dict, preparation: dict) -> dict:
    outcome = preparation["outcome"]
    if outcome not in FINAL_OUTCOMES:
        raise ValueError("Invalid shared preparation failure")
    cost = preparation["common_preparation_seconds"]
    result = {
        "job_id": job["job_id"],
        "job": job,
        "outcome": outcome,
        "success": False,
        "dispatched": False,
        "common_preparation_seconds": cost,
        "replay_wall_seconds": 0.0,
        "charged_online_seconds": cost,
        "verified_known": False,
        "verified_fresh": False,
        "queue_sha256": None,
        "completed_utc": now(),
    }
    if preparation.get("timing_unavailable") is True:
        result.update(
            timing_unavailable=True,
            replay_wall_seconds=None,
            charged_online_seconds=None,
        )
    return result


def verify_recovery(record: dict, public: dict, fresh: list) -> None:
    if record.get("success") is not True:
        return
    from neural_sat_speck.followups.worker import verify_observations

    key = record["recovered_key"]
    if not record.get("attempts"):
        raise ValueError("Recovery lacks a completed candidate trace")
    attempt = record["attempts"][-1]
    if attempt.get("solver_status") != "sat" or attempt.get(
        "candidate_index"
    ) != record.get("recovered_candidate_index"):
        raise ValueError("Recovery candidate differs from final SAT attempt")
    if public["cipher"] == "speck32_64":
        from neural_sat_speck.speck.key_schedule import expand_key
    else:
        from neural_sat_speck.simon.key_schedule import expand_key
    if tuple(expand_key(tuple(key), public["rounds"])[-2:]) != tuple(
        attempt["candidate"]
    ):
        raise ValueError("Recovered trailing subkeys differ from tested assumptions")
    if not verify_observations(
        public["cipher"], public["rounds"], key, public["known_observations"]
    ) or not verify_observations(public["cipher"], public["rounds"], key, fresh):
        raise ValueError("Recovered key does not independently verify")


def validate_ledger(
    ledger: dict, manifest: dict, manifest_sha: str, dataset: dict
) -> None:
    if (
        ledger.get("manifest_sha256") != manifest_sha
        or ledger.get("source_commit") != manifest["source_commit"]
        or ledger.get("schema") != "queue_order_results_v1"
        or ordered_hash(ledger.get("profile")) != ordered_hash(manifest["profile"])
    ):
        raise ValueError("Resume identity differs")
    jobs = {j["job_id"]: j for j in fixed_jobs()}
    records, starts = ledger.get("records"), ledger.get("starts")
    if (
        not isinstance(records, dict)
        or not isinstance(starts, dict)
        or not set(records) <= jobs.keys()
        or not set(starts) <= jobs.keys()
    ):
        raise ValueError("Unexpected completed/started grid cells")
    completed = list(records)
    # JSON serialization sorts keys; test membership as a schedule prefix.
    schedule = list(jobs)
    if set(completed) != set(schedule[: len(completed)]):
        raise ValueError("Completed cells do not form the fixed serial schedule")
    if not set(starts) <= set(schedule[: len(completed) + 1]):
        raise ValueError("Start outside the completed serial prefix")
    evidence_root = Path(ledger.get("evidence_root", ""))
    if not evidence_root.is_absolute():
        raise ValueError("Missing absolute evidence root")
    for name, start in starts.items():
        token = start.get("invocation", "")
        if (
            not isinstance(token, str)
            or len(token) != 32
            or any(c not in "0123456789abcdef" for c in token)
        ):
            raise ValueError("Malformed stored invocation identity")
        if start.get("directory") != str(evidence_root / "jobs" / name / token):
            raise ValueError("Stored start directory differs from job identity")
    for name, row in records.items():
        job = jobs[name]
        if row.get("job_id") != name or row.get("job") != job:
            raise ValueError("Record/job identity differs")
        prep = (
            records.get(job["case_id"] + "__prepare")
            if job["operation"] == "solve"
            else None
        )
        if row.get("dispatched") is False:
            if name in starts or prep is None or prep["outcome"] == "queue_ready":
                raise ValueError("Invalid inherited preparation failure")
            expected = inherited_failure(job, prep)
            expected["completed_utc"] = row["completed_utc"]
        else:
            if name not in starts or row.get("directory") != starts[name]["directory"]:
                raise ValueError("Completed dispatch lacks its recorded start")
            raw = read_checked_json(
                Path(starts[name]["directory"]) / "supervisor_result.json"
            )
            validate_execution_artifacts(Path(starts[name]["directory"]), job, raw)
            if raw.get("timing_unavailable") is not True:
                allowance = (
                    300.0
                    if job["operation"] == "rank"
                    else 300.0 - prep["common_preparation_seconds"]
                )
                if seconds(raw.get("deadline_seconds")) != allowance:
                    raise ValueError(
                        "Completed artifact uses a different parent allowance"
                    )
            if raw.get("job_id") != name or raw.get("job") != job:
                raise ValueError("Completed artifact belongs to another job")
            expected = normalize(job, raw, prep)
        if ordered_hash(expected) != ordered_hash(row):
            raise ValueError("Completed ledger differs from checked execution artifact")
        if job["operation"] == "rank":
            if row.get("timing_unavailable") is True:
                if (
                    row.get("outcome") != "infrastructure_interrupted"
                    or row.get("common_preparation_seconds") is not None
                ):
                    raise ValueError("Invalid unavailable preparation timing")
            else:
                seconds(row.get("common_preparation_seconds"))
            if row.get("outcome") == "queue_ready":
                if row["common_preparation_seconds"] >= 300:
                    raise ValueError(
                        "Queue preparation completed beyond the parent cutoff"
                    )
                validate_preparation(row, job)
            elif row.get("outcome") not in FINAL_OUTCOMES:
                raise ValueError("Unknown preparation outcome")
        else:
            prep = records.get(job["case_id"] + "__prepare")
            if not prep:
                raise ValueError("Replay lacks completed shared preparation")
            available = prep if prep["outcome"] == "queue_ready" else None
            validate_replay(row, job, available)
            verify_recovery(
                row, public_case(dataset, job), fresh_observations(dataset, job)
            )


def run(manifest_path: Path, output: Path, root: Path, *, resume: bool = False) -> dict:
    import hashlib

    manifest, paths = load_manifest(manifest_path, root)
    if manifest.get("execution_kind") == "pilot":
        raise ValueError("Use the pilot entry point before the full replacement")
    validate_runtime(manifest, paths)
    source = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if source != manifest.get("source_commit"):
        raise ValueError("Executed git source differs")
    if subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout:
        raise ValueError("Executed source snapshot is dirty or untracked")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if output.exists() and not resume:
            raise FileExistsError("Existing evidence requires explicit --resume")
        dataset = json.loads(paths["cohort"].read_bytes())
        # Deterministic opened-family replay and all known/fresh encryptions.
        from neural_sat_speck.backend.artifacts import validate_cohort

        validate_cohort(dataset)
        if manifest.get("pilot_gate") is not None:
            from neural_sat_speck.followups.pilot import validate_gate

            validate_gate(manifest, paths, dataset)
        manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        ledger = (
            read_checked_json(output)
            if output.exists()
            else {
                "schema": "queue_order_results_v1",
                "manifest_sha256": manifest_sha,
                "source_commit": source,
                "profile": manifest["profile"],
                "evidence_root": str(output.parent.absolute()),
                "started_utc": now(),
                "records": {},
                "starts": {},
                "status": "ACTIVE",
            }
        )
        if ledger.get("evidence_root") != str(output.parent.absolute()):
            raise ValueError("Resume evidence directory differs")
        validate_ledger(ledger, manifest, manifest_sha, dataset)
        if ledger.get("status") == "STOPPED_ALARM":
            raise RuntimeError(
                "Dispatched alarm needs a recorded repair review before continuation"
            )
        environment = {
            **os.environ,
            **{
                k: "1"
                for k in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "PYTHONPATH": str(root / "src"),
            "TMPDIR": str(root / "runtime" / "queue_order_tmp"),
            "TF_NUM_INTRAOP_THREADS": "8",
            "TF_NUM_INTEROP_THREADS": "1",
        }
        Path(environment["TMPDIR"]).mkdir(parents=True, exist_ok=True)
        with sleep_assertion() as assertion:
            ledger["sleep_assertion"] = assertion
            # Allow the owned caffeinate child to install its assertions.
            for _ in range(20):
                if power_sample().get("accepted") is True:
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError(
                    "Observed sleep/power telemetry rejected before dispatch"
                )
            atomic_json(output, ledger)
            for job in manifest["jobs"]:
                name = job["job_id"]
                if name in ledger["records"]:
                    continue
                public = public_case(dataset, job)
                prep = ledger["records"].get(job["case_id"] + "__prepare")
                if job["operation"] == "solve" and prep["outcome"] != "queue_ready":
                    row = inherited_failure(job, prep)
                else:
                    timeout = (
                        300.0
                        if job["operation"] == "rank"
                        else max(0.0, 300 - prep["common_preparation_seconds"])
                    )
                    if timeout <= 0:
                        raise ValueError(
                            "Successful preparation consumed the entire deadline"
                        )
                    if name in ledger["starts"]:
                        start = ledger["starts"][name]
                        if live_groups(start["invocation"]):
                            raise RuntimeError(
                                "A previously started guardian/worker remains active; do not duplicate it"
                            )
                        directory = Path(start["directory"])
                        if (directory / "supervisor_result.json").exists():
                            raw = read_checked_json(
                                directory / "supervisor_result.json"
                            )
                        elif (directory / "guardian_receipt.json").exists():
                            saved_guardian = read_checked_json(
                                directory / "guardian.json"
                            )
                            raw = observe_finished(
                                directory,
                                job=job,
                                started=saved_guardian["deadline_monotonic"] - timeout,
                                timeout=timeout,
                            )
                            raw.update(
                                started_monotonic=saved_guardian["deadline_monotonic"]
                                - timeout,
                                deadline_seconds=timeout,
                            )
                            # No coordinator completion receipt means its power/RSS
                            # monitor was lost. Preserve artifacts but do not turn
                            # surviving child execution into a completed trial.
                            raw.update(
                                outcome="infrastructure_interrupted",
                                success=False,
                                verified_known=False,
                                verified_fresh=False,
                                coordinator_interruption=True,
                            )
                        else:
                            raw = {
                                "job_id": name,
                                "job": job,
                                "outcome": "infrastructure_interrupted",
                                "success": False,
                                "physical_worker_wall_seconds": None,
                                "timing_unavailable": True,
                                "directory": str(directory),
                            }
                        validate_execution_artifacts(directory, job, raw)
                        rejection_path = directory / "environment_rejection.json"
                        if rejection_path.exists():
                            raw["environment_alarm"] = read_checked_json(rejection_path)
                        atomic_json(directory / "supervisor_result.json", raw)
                        row = normalize(job, raw, prep)
                    else:
                        if power_sample().get("accepted") is not True:
                            ledger["status"] = "STOPPED_ENVIRONMENT"
                            atomic_json(output, ledger)
                            return ledger
                        invocation = uuid.uuid4().hex
                        directory = output.parent / "jobs" / name / invocation
                        ledger["starts"][name] = {
                            "invocation": invocation,
                            "directory": str(directory),
                            "started_utc": now(),
                        }
                        atomic_json(output, ledger)
                        payload = {
                            "job": job,
                            "public": public,
                            "paths": {k: str(p) for k, p in paths.items()},
                            "directory": str(directory),
                            "invocation": invocation,
                            "repo_root": str(root),
                            "source_commit": source,
                            "python_path": str(paths["python"]),
                        }
                        if job["operation"] == "solve":
                            payload.update(
                                preparation=prep, fresh=fresh_observations(dataset, job)
                            )
                        raw = run_job(payload, environment, timeout)
                        row = normalize(job, raw, prep)
                if job["operation"] == "solve":
                    validate_replay(
                        row, job, prep if prep["outcome"] == "queue_ready" else None
                    )
                    verify_recovery(row, public, fresh_observations(dataset, job))
                ledger["records"][name] = row
                atomic_json(output, ledger)
                print(
                    f"{len(ledger['records'])}/96 {name}: {row['outcome']}", flush=True
                )
                if (
                    row["outcome"]
                    in {
                        "environment_rejected",
                        "worker_exception",
                        "verification_failure",
                    }
                    or row.get("verification_failures", 0) > 0
                    or "environment_alarm" in row
                ):
                    ledger["status"] = "STOPPED_ALARM"
                    atomic_json(output, ledger)
                    return ledger
            validate_ledger(ledger, manifest, manifest_sha, dataset)
            if set(ledger["records"]) != {j["job_id"] for j in fixed_jobs()}:
                raise ValueError("Incomplete final cell grid")
            ledger.update(status="COMPLETE", completed_utc=now())
            atomic_json(output, ledger)
            return ledger
