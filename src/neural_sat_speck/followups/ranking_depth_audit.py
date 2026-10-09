"""Replay raw D-055 models and verification; no outcome-dependent launch gate."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

from neural_sat_speck.backend.artifacts import cipher_functions, verify_key
from neural_sat_speck.backend.formula import build_formula
from neural_sat_speck.backend.guard import read_checked_json, sha256_file
from neural_sat_speck.backend.solver_io import parse_solver_output, solver_arguments
from neural_sat_speck.backend.worker import _complete_assignment
from neural_sat_speck.followups.conditioning_control import digest, seconds
from neural_sat_speck.followups.conditioning_runner import phase_accounting
from neural_sat_speck.followups.environment import same_python_environment
from neural_sat_speck.followups.ranking_depth_control import (
    PROSPECTIVE_CAMPAIGN,
    SCIENTIFIC_OUTCOMES,
    asset_keys_for,
    load_manifest,
    optional_terminal_json,
    verify_cases,
)
from neural_sat_speck.followups.ranking_depth_worker import (
    conditioned_formula,
    verify_without_truth,
)
from neural_sat_speck.followups.stage2_receipts import (
    _validate_policy,
    _validate_power,
    _validate_sleep_assertion,
    _validate_tmpdir,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_telemetry(
    receipt: dict, pid: int, parent: int, manifest: dict, python: Path
) -> None:
    require(
        receipt.get("pid") == pid and receipt.get("parent_pid") == parent,
        "Receipt PID/parent does not bind the observed process",
    )
    require(
        receipt.get("git_commit") == manifest["source_commit"]
        and receipt.get("is_certified") is True
        and receipt.get("rejection_reason") is None,
        "Worker source/certification drift",
    )
    require(
        receipt.get("platform") == "darwin"
        and receipt.get("qos_class") in (33, 25, 21),
        "Unsupported or background calling thread",
    )
    require(
        all(
            type(receipt.get(k)) is int and receipt[k] == 0
            for k in ("darwin_priority", "darwin_bg_state", "io_policy_state")
        ),
        "Worker process policy drift",
    )
    env = receipt.get("thread_environment", {})
    require(
        all(
            env.get(n) == ("8" if n == "TF_NUM_INTRAOP_THREADS" else "1")
            for n in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "TF_NUM_INTEROP_THREADS",
                "TF_NUM_INTRAOP_THREADS",
            )
        ),
        "Thread environment drift",
    )
    _validate_tmpdir(receipt, Path(manifest["source_root"]))
    reported = receipt.get("python_executable")
    require(
        isinstance(reported, str) and same_python_environment(reported, python),
        "Wrong Python environment",
    )


def check_ranking_receipts(
    ranked: dict,
    cell: dict,
    directory: Path,
    manifest: dict,
    paths: dict[str, Path],
    worker_pid: int,
) -> None:
    receipts = ranked["ranking_receipts"]
    stages = [receipts["stage1"]]
    if cell["trailing_keys"] == 2:
        require(isinstance(receipts.get("stage2"), dict), "Missing Stage2 receipt")
        stages.append(receipts["stage2"])
    else:
        require(receipts.get("stage2") is None, "Stage1-only cell ran Stage2")
    for stage in stages:
        for name, pin in stage["assets"].items():
            require(
                pin["path"] == str(paths[name])
                and pin["size_bytes"] == manifest["files"][name]["identity"]["size"],
                "Ranker loaded an unpinned asset",
            )
        for name, checksum in stage["code_dependencies"].items():
            require(
                "source:" + name in manifest["files"]
                and manifest["files"]["source:" + name]["sha256"] == checksum,
                "Ranker imported unpinned repository code",
            )
    first = receipts["stage1"]["ranker"]
    if (
        cell["cipher"] == "speck32_64"
        and cell.get("stage1", "neural") == "classical_sparse_r6"
    ):
        require(
            first.get("ranker") == "production_rank_classical_subkeys"
            and first.get("table") == "sparseR6"
            and first.get("table_rounds") == 6
            and first.get("device") == "cpu",
            "Classical Stage1 ranker identity drift",
        )
    elif cell["cipher"] == "speck32_64":
        require(
            first.get("device") == "mps"
            and first.get("torch_intra_op_threads") == 1
            and first.get("torch_inter_op_threads") == 1,
            "SPECK inference device/thread profile drift",
        )
    else:
        telemetry = read_checked_json(directory / "tensorflow_stage1_worker.json")
        pid = first["tensorflow_child_pid"]
        check_telemetry(telemetry, pid, worker_pid, manifest, paths["tf_python"])
        require(
            first["tensorflow_child_process_group"] == worker_pid
            and first["tensorflow_command"]
            == [
                str(paths["tf_python"]),
                "-m",
                "neural_sat_speck.followups.ranking_depth_ranking",
                "--tf-stage1",
                "--invocation-path",
                str(directory),
            ]
            and first["runtime_receipt_sha256"]
            == sha256_file(directory / "tensorflow_stage1_worker.json"),
            "TF child invocation escaped the timed group or changed its receipt",
        )
        runtime = first["runtime"]
        expected = manifest["runtime"]["tensorflow"]
        require(
            same_python_environment(runtime["python_executable"], paths["tf_python"]),
            "TF child interpreter drift",
        )
        require(
            runtime["packages"]
            == {
                n: {"version": v, "origin": expected["origins"][n]}
                for n, v in expected["packages"].items()
            },
            "TF child package version/origin drift",
        )
        require(
            runtime["installed_packages"] == expected["installed_packages"],
            "TF installed environment lock drift",
        )


def validate_cell(row: dict, manifest: dict, paths: dict[str, Path]) -> dict:
    cell = row.get("cell")
    require(
        cell in manifest["cells"] and row.get("outcome") in SCIENTIFIC_OUTCOMES,
        "Invalid cell identity/outcome",
    )
    directory = Path(row["directory"])
    require(
        re.fullmatch(r"[0-9a-f]{32}", directory.name) is not None,
        "Invalid invocation token",
    )
    require(
        row == read_checked_json(directory / "supervisor_result.json"),
        "Supervisor row binding differs",
    )
    payload = read_checked_json(directory / "payload.json")
    guardian = read_checked_json(directory / "guardian.json")
    receipt = read_checked_json(directory / "guardian_receipt.json")
    process = read_checked_json(directory / "process.json")
    ident = cell["case_id"]
    require(
        payload["cell"] == cell
        and payload["invocation_id"] == directory.name
        and payload["repo_root"] == manifest["source_root"]
        and payload["source_commit"] == manifest["source_commit"],
        "Payload execution identity differs",
    )
    require(
        set(payload)
        == {
            "cell",
            "directory",
            "invocation_id",
            "repo_root",
            "source_commit",
            "python_path",
            "paths",
            "public_path",
            "public_sha256",
            "verification_path",
            "verification_sha256",
            "base_formula_sha256",
            "deadline_monotonic",
        },
        "Unexpected truth or payload field",
    )
    require(
        payload["public_path"] == str(paths[ident + ":public"])
        and payload["verification_path"] == str(paths[ident + ":verification"])
        and payload["public_sha256"] == manifest["cases"][ident]["public"]["sha256"]
        and payload["verification_sha256"]
        == manifest["cases"][ident]["verification"]["sha256"]
        and payload["base_formula_sha256"]
        == manifest["cases"][ident]["base_formula_sha256"],
        "Input provenance drift",
    )
    require(
        payload["paths"] == {k: str(paths[k]) for k in asset_keys_for(manifest)},
        "Asset/runtime dispatch drift",
    )
    require(
        guardian["worker_arguments"]
        == [
            str(paths["python"]),
            "-m",
            "neural_sat_speck.followups.ranking_depth_worker",
            str(directory / "payload.json"),
        ]
        and guardian["directory"] == str(directory)
        and guardian["invocation"] == directory.name,
        "Actual worker dispatch differs",
    )
    start, observed, deadline = (
        seconds(row[k])
        for k in ("started_monotonic", "observed_monotonic", "deadline_monotonic")
    )
    require(
        deadline == start + 300.0
        and guardian["deadline_monotonic"] == deadline
        and payload["deadline_monotonic"] == deadline
        and receipt["deadline_monotonic"] == deadline,
        "Parent clock/deadline drift",
    )
    require(
        receipt == row["guardian_receipt"]
        and receipt["observed_monotonic"] == observed
        and row["parent_wall_seconds"] == observed - start
        and observed >= start,
        "Guardian timing does not bind the row",
    )
    require(
        receipt.get("process_group_absent") is True
        and row.get("guardian_returncode") == 0
        and receipt["worker_pid"] == process["pid"]
        and receipt["process_group"] == process["process_group"] == process["pid"]
        and receipt["watchdog_pid"] == process["supervisor_pid"]
        and receipt["invocation"] == directory.name,
        "Process/cleanup identity differs",
    )
    require(
        math.isclose(
            row["cleanup_wall_seconds"],
            seconds(receipt["cleanup_completed_monotonic"]) - observed,
        ),
        "Cleanup accounting drift",
    )
    raw_path = directory / "worker_result.json"
    deadline_proven = receipt["status"] == "deadline_enforced"
    raw, raw_uncommitted = optional_terminal_json(raw_path, deadline_proven)
    journal, journal_uncommitted = optional_terminal_json(
        directory / "journal.json", deadline_proven
    )
    require(raw == row["worker_result"], "Worker result binding differs")
    if row["outcome"] == "global_deadline":
        require(
            receipt["status"] == "deadline_enforced" and observed >= deadline,
            "Timeout lacks independent deadline proof",
        )
    else:
        require(
            receipt["status"] == "worker_exit"
            and receipt["worker_returncode"] == 0
            and observed < deadline
            and raw is not None
            and raw["outcome"] == row["outcome"],
            "Completed worker lacks successful bounded exit",
        )
    samples = row["environment_samples"]
    require(
        samples == read_checked_json(directory / "environment_samples.json")
        and len(samples) >= 1,
        "Missing or changed resource samples",
    )
    sampled_worker = False
    for sample in samples:
        require(
            start <= seconds(sample["observed_monotonic"]) <= observed + 2.0,
            "Resource sample outside cell",
        )
        _validate_power(sample["power"])
        if "worker_policy" in sample:
            sampled_worker = True
            _validate_policy(
                sample["worker_policy"],
                process["pid"],
                terminal_exit_proven=(
                    receipt["status"] in {"worker_exit", "deadline_enforced"}
                    and receipt["process_group_absent"] is True
                ),
            )
            memory = sample["memory"]
            require(
                type(memory["group_rss_bytes"]) is int
                and memory["group_rss_bytes"] >= 0
                and memory.get("accepted") is True
                and memory["group_rss_bytes"] <= 2684354560,
                "Sampled RSS limit exceeded",
            )
    require(sampled_worker, "No process-group resource sample")
    worker_receipt = directory / "python_worker_receipt.json"
    if worker_receipt.exists():
        check_telemetry(
            read_checked_json(worker_receipt),
            process["pid"],
            receipt["watchdog_pid"],
            manifest,
            paths["python"],
        )
    else:
        require(row["outcome"] == "global_deadline", "Completed worker lacks receipt")
    phases = row["phase_accounting"]
    require(
        row.get("component_accounting_available") is (journal is not None)
        and row.get("uncommitted_terminal_json")
        == [
            name
            for name, broken in (
                ("worker_result.json", raw_uncommitted),
                ("journal.json", journal_uncommitted),
            )
            if broken
        ],
        "Interrupted metadata accounting differs",
    )
    require(
        phases
        == (phase_accounting(directory, observed) if journal is not None else None),
        "Phase journal binding differs",
    )
    if phases is None:
        require(
            deadline_proven and raw is None, "Completed worker lacks phase evidence"
        )
    else:
        accounted = sum(
            seconds(v) for v in phases["completed_seconds"].values()
        ) + seconds(phases["inflight_seconds"])
        require(
            accounted <= row["parent_wall_seconds"] + 0.02, "Phase costs double counted"
        )
    public = read_checked_json(paths[ident + ":public"])
    fresh = read_checked_json(paths[ident + ":verification"])
    base = build_formula(cell["cipher"], public, 4, 0, include_cnf=False)
    queue_path = directory / "queue.json"
    queue_record, _ = optional_terminal_json(queue_path, deadline_proven)
    stage1_path = directory / "stage1.json"
    prefix, _ = optional_terminal_json(stage1_path, deadline_proven)
    if prefix is not None:
        limit = 32 if cell["cipher"] == "speck32_64" else 8
        require(
            len(prefix["stage1_parents"]) == limit
            and len(set(prefix["stage1_parents"])) == limit
            and prefix["stage1_prefix_sha256"] == digest(prefix["stage1_parents"]),
            "Persisted Stage1 prefix is malformed",
        )
    if queue_record:
        queue, parents = queue_record["queue"], queue_record["stage1_parents"]
        limit = 32 if cell["cipher"] == "speck32_64" else 8
        require(
            len(parents) == limit
            and len(set(parents)) == limit
            and all(type(p) is int and 0 <= p < 65536 for p in parents),
            "Invalid parent prefix",
        )
        require(
            len(queue) == (limit if cell["trailing_keys"] == 1 else 1024)
            and len({tuple(q) for q in queue}) == len(queue)
            and all(
                len(q) == cell["trailing_keys"]
                and q[-1] in parents
                and all(type(w) is int and 0 <= w < 65536 for w in q)
                for q in queue
            ),
            "Invalid candidate queue",
        )
        require(
            queue_record["queue_sha256"] == digest(queue)
            and queue_record["stage1_sha256"] == digest(parents),
            "Queue fingerprint differs",
        )
        require(
            prefix is not None and prefix["stage1_parents"] == parents,
            "Queue parents differ from persisted Stage1 prefix",
        )
        check_ranking_receipts(
            queue_record, cell, directory, manifest, paths, process["pid"]
        )
        if cell["trailing_keys"] == 1:
            require(queue == [[p] for p in parents], "Stage1-only order differs")
    else:
        queue = []
        require(
            row["outcome"] == "global_deadline", "Completed worker lacks a ranked queue"
        )
    attempted = sorted((directory / "attempts").glob("[0-9][0-9][0-9][0-9]"))
    require(
        [p.name for p in attempted] == [f"{i:04d}" for i in range(len(attempted))],
        "Attempt gaps/duplicates",
    )
    recovered, completed, pids = None, 0, set()
    for index, attempt_dir in enumerate(attempted):
        require(index < len(queue), "Attempt outside candidate queue")
        formula = conditioned_formula(base, queue[index])
        native = attempt_dir / "input.dimacs"
        terminal_timeout = deadline_proven and index == len(attempted) - 1
        started_attempt, _ = optional_terminal_json(
            attempt_dir / "attempt_start.json", terminal_timeout
        )
        att, _ = optional_terminal_json(attempt_dir / "attempt.json", terminal_timeout)
        process_path = attempt_dir / "native_process.json"
        native_process, _ = optional_terminal_json(process_path, terminal_timeout)
        if terminal_timeout and att is None and native_process is None:
            if started_attempt is not None:
                require(
                    started_attempt["candidate_literals"]
                    == formula["assumption_literals"]
                    and started_attempt["native_sha256"]
                    == hashlib.sha256(formula["native_dimacs"].encode()).hexdigest()
                    and started_attempt["formula_counts"] == formula["counts"],
                    "Interrupted formula-start provenance differs",
                )
            # Retain partial bytes. A directory alone proves neither solver dispatch
            # nor SAT/UNSAT and cannot supply an observed candidate result.
            continue
        require(
            native.read_text() == formula["native_dimacs"],
            "Submitted formula differs from candidate/public input",
        )
        require(
            started_attempt is not None
            and started_attempt["candidate_literals"] == formula["assumption_literals"]
            and started_attempt["native_sha256"] == sha256_file(native)
            and started_attempt["formula_counts"] == formula["counts"],
            "Attempt-start provenance differs",
        )
        if native_process is not None:
            require(
                native_process["pid"] not in pids
                and native_process["process_group"] == process["process_group"],
                "Native candidate shares state or escaped process group",
            )
            pids.add(native_process["pid"])
            argv = solver_arguments(cell["profile"], str(paths["cms"]), "", str(native))
            require(native_process["argv"] == argv, "Native argv differs")
            launch_path = attempt_dir / "native_launch_receipt.json"
            launch, _ = optional_terminal_json(launch_path, terminal_timeout)
            if launch is not None:
                check_telemetry(
                    launch,
                    native_process["pid"],
                    process["pid"],
                    manifest,
                    paths["python"],
                )
                require(
                    launch.get("solver_arguments") == argv,
                    "Native pre-exec invocation differs",
                )
            else:
                # A capped kill may land before the launcher writes its receipt.
                require(
                    row["outcome"] == "global_deadline"
                    or (att is not None and att.get("status") == "capped"),
                    "Missing native pre-exec receipt",
                )
        if att is None:
            require(
                row["outcome"] == "global_deadline" and index == len(attempted) - 1,
                "Nonterminal/incomplete attempt without timeout",
            )
            continue
        completed += 1
        require(
            att["index"] == index
            and att["candidate"] == queue[index]
            and att["native_sha256"] == sha256_file(native)
            and att["stdout_sha256"] == sha256_file(attempt_dir / "solver_stdout.txt")
            and att["stderr_sha256"] == sha256_file(attempt_dir / "solver_stderr.txt"),
            "Attempt/raw-output binding differs",
        )
        require(
            native_process is not None
            and att["solver_arguments"] == native_process["argv"],
            "Completed native process missing",
        )
        if att["status"] == "capped":
            # Accepted only in a capped cell, only as a SIGKILL at or after the cap.
            # A capped attempt is a loss for its queue position: it is never parsed.
            cap = cell.get("candidate_cap_seconds")
            require(
                cap is not None and att.get("candidate_cap_seconds") == cap,
                "Capped attempt in uncapped cell or cap differs",
            )
            require(
                att["returncode"] == -9, "Capped attempt was not SIGKILL-terminated"
            )
            require(
                seconds(att["solve_wall_seconds"]) >= cap,
                "Capped attempt finished before its cap",
            )
            require(
                seconds(att["started_monotonic"]) >= start
                and seconds(att["ended_monotonic"]) <= observed
                and math.isclose(
                    seconds(att["solve_wall_seconds"]),
                    att["ended_monotonic"] - att["started_monotonic"],
                ),
                "Capped attempt timing outside parent or incorrectly accounted",
            )
            expected_receipt = (
                read_checked_json(attempt_dir / "native_launch_receipt.json")
                if (attempt_dir / "native_launch_receipt.json").exists()
                else None
            )
            require(
                att["native_launch_receipt"] == expected_receipt,
                "Capped attempt receipt changed",
            )
            continue
        require(
            att["native_launch_receipt"]
            == read_checked_json(attempt_dir / "native_launch_receipt.json"),
            "Native receipt changed",
        )
        require(
            seconds(att["started_monotonic"]) >= start
            and seconds(att["ended_monotonic"]) <= observed
            and math.isclose(
                seconds(att["solve_wall_seconds"]),
                att["ended_monotonic"] - att["started_monotonic"],
            ),
            "Attempt timing outside parent or incorrectly accounted",
        )
        parsed = parse_solver_output(
            "cms",
            att["returncode"],
            (attempt_dir / "solver_stdout.txt").read_text(),
            required_variables=list(range(1, base["num_vars"] + 1)),
            num_vars=base["num_vars"],
        )
        require(parsed["status"] != "error", "Malformed solver result")
        if parsed["status"] == "sat":
            require(
                _complete_assignment(formula, parsed["assignment"]),
                "Invalid original SAT model",
            )
            key = [
                sum(int(parsed["assignment"][v]) << b for b, v in enumerate(word))
                for word in base["varmap"]["master_key"]
            ]
            checked = verify_without_truth(
                cell["cipher"], cell["rounds"], key, public, fresh
            )
            require(
                att["recovered_key"] == key
                and all(att.get(k) is v for k, v in checked.items()),
                "Decoded-key verification differs",
            )
            if all(checked.values()):
                require(
                    att["status"] == "sat" and index == len(attempted) - 1,
                    "Search continued past verified recovery",
                )
                recovered = key
            else:
                require(
                    att["status"] == "verification_rejected",
                    "Unverified key misclassified",
                )
        else:
            require(att["status"] == parsed["status"], "UNSAT/UNKNOWN misclassified")
            if parsed["status"] == "unknown":
                require(
                    index == len(attempted) - 1,
                    "Unknown candidate was retried/continued",
                )
    if raw is not None:
        require(raw["attempt_count"] == completed, "Completed attempt count differs")
        require(
            raw["completed_phase_seconds"] == journal["completed_seconds"],
            "Worker and durable phase costs differ",
        )
        if raw["outcome"] == "recovered":
            require(
                recovered == raw["recovered_key"]
                and raw.get("verified_known") is True
                and raw.get("verified_fresh") is True,
                "Claimed worker recovery lacks raw model/fresh verification",
            )
    if row["outcome"] == "recovered":
        require(recovered is not None, "Recovered cell has no independent SAT model")
    if row["outcome"] == "queue_exhausted":
        require(
            len(attempted) == len(queue) == completed and recovered is None,
            "Queue exhaustion did not exhaust all candidates",
        )
    if row["outcome"] == "solver_unknown":
        require(
            completed > 0
            and read_checked_json(attempted[-1] / "attempt.json")["status"] == "unknown"
            and recovered is None,
            "Unresolved outcome lacks UNKNOWN proof",
        )
    return {
        "queue": queue_record,
        "stage1_sha256": prefix["stage1_prefix_sha256"] if prefix is not None else None,
        "completed_attempts": completed,
        "recovered_key": recovered,
    }


def audit_ledger(path: Path, manifest_path: Path, *, hashes: bool = True) -> dict:
    manifest, paths = load_manifest(manifest_path, hashes=hashes)
    verify_cases(manifest, paths)
    ledger = read_checked_json(path)
    require(
        ledger.get("schema") == "backend_ranking_depth_results_v1"
        and ledger.get("status") == "COMPLETE"
        and ledger.get("mode") == manifest["mode"]
        and ledger.get("source_commit") == manifest["source_commit"]
        and ledger.get("manifest_path") == str(manifest_path.absolute())
        and ledger.get("manifest_sha256") == sha256_file(manifest_path),
        "Ledger/manifest identity differs",
    )
    _validate_sleep_assertion(ledger["sleep_assertion"])
    ids = [c["cell_id"] for c in manifest["cells"]]
    require(
        set(ledger["records"]) == set(ids)
        and set(ledger["starts"]) == set(ids)
        and ledger.get("execution_order") == ids,
        "Incomplete or reordered cell grid",
    )
    inspected, assigned, prior_end = {}, 0, None
    for cell in manifest["cells"]:
        ident = cell["cell_id"]
        row = ledger["records"][ident]
        require(
            ledger["starts"][ident]["directory"] == row["directory"]
            and ledger["starts"][ident]["invocation_id"] == Path(row["directory"]).name,
            "Cell start differs/was retried",
        )
        if prior_end is not None:
            require(
                row["started_monotonic"] >= prior_end, "Overlapping scientific cells"
            )
        prior_end = row["guardian_receipt"]["cleanup_completed_monotonic"]
        inspected[ident] = validate_cell(row, manifest, paths)
        # Ground truth is joined only after the complete grid was validated above.
        truth = read_checked_json(paths[cell["case_id"] + ":truth"])
        _, expand = cipher_functions(cell["cipher"])
        true_tuple = list(
            expand(tuple(truth["master_key"]), cell["rounds"])[-cell["trailing_keys"] :]
        )
        for attempt in (Path(row["directory"]) / "attempts").glob(
            "[0-9][0-9][0-9][0-9]"
        ):
            record, _ = optional_terminal_json(
                attempt / "attempt.json", row["outcome"] == "global_deadline"
            )
            if record is not None:
                require(
                    not (
                        record["candidate"] == true_tuple
                        and record["status"] == "unsat"
                    ),
                    "Correct trailing-key tuple returned false UNSAT",
                )
        if row["outcome"] == "recovered":
            checked = verify_key(
                cell["cipher"],
                cell["rounds"],
                inspected[ident]["recovered_key"],
                read_checked_json(paths[cell["case_id"] + ":public"]),
                truth,
            )
            assigned += checked["assigned_key_match"] is True
    parity = []
    groups = list(
        dict.fromkeys(
            (c["case_id"], c.get("stage1", "neural")) for c in manifest["cells"]
        )
    )
    for case, stage in groups:
        members = [
            c
            for c in manifest["cells"]
            if c["case_id"] == case and c.get("stage1", "neural") == stage
        ]
        rows = [inspected[c["cell_id"]]["queue"] for c in members]
        available = [r for r in rows if r is not None]
        prefixes = [inspected[c["cell_id"]]["stage1_sha256"] for c in members]
        require(
            len({p for p in prefixes if p is not None}) <= 1,
            "Stage1 prefixes differ, including interrupted Stage2 cells",
        )
        require(
            len({r["stage1_sha256"] for r in available}) <= 1,
            "Stage1 prefixes differ across depth/backend",
        )
        for depth in (1, 2):
            qs = [
                inspected[c["cell_id"]]["queue"]
                for c in members
                if c["trailing_keys"] == depth
            ]
            require(
                len({q["queue_sha256"] for q in qs if q is not None}) <= 1,
                "G/X candidate queues differ",
            )
        entry = {
            "case_id": case,
            "complete_queues": len(available),
            "parity_checked": True,
        }
        if manifest.get("campaign") == PROSPECTIVE_CAMPAIGN:
            entry["stage1"] = stage
        parity.append(entry)
    return {
        "schema": "backend_ranking_depth_audit_v1",
        "passed": True,
        "mode": manifest["mode"],
        "ledger_path": str(path.absolute()),
        "ledger_sha256": sha256_file(path),
        "manifest_path": str(manifest_path.absolute()),
        "manifest_sha256": sha256_file(manifest_path),
        "source_commit": manifest["source_commit"],
        "profile_sha256": digest(manifest["profile"]),
        "cells": len(ids),
        "assigned_verified_recoveries": assigned,
        "completed_attempts": sum(r["completed_attempts"] for r in inspected.values()),
        "queue_parity": parity,
    }
