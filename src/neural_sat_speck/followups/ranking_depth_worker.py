"""Timed autonomous ranking followed by independent native candidate solves."""

from __future__ import annotations

import hashlib
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from neural_sat_speck.backend.artifacts import cipher_functions
from neural_sat_speck.backend.formula import build_formula, render_native_dimacs
from neural_sat_speck.backend.guard import atomic_json, read_checked_json, sha256_file
from neural_sat_speck.backend.solver_io import parse_solver_output, solver_arguments
from neural_sat_speck.backend.worker import Journal, _complete_assignment
from neural_sat_speck.followups.conditioning_control import digest, formula_identity


def conditioned_formula(base: dict, candidate: list[int]) -> dict:
    """Chronological tail words, in the original full-round/key-schedule formula."""
    if len(candidate) not in (1, 2) or any(
        type(w) is not int or not 0 <= w < 65536 for w in candidate
    ):
        raise ValueError("Invalid candidate words")
    words = base["varmap"]["round_keys"][-len(candidate) :]
    literals = [
        v if (w >> bit) & 1 else -v
        for w, variables in zip(candidate, words, strict=True)
        for bit, v in enumerate(variables)
    ]
    result = {
        **base,
        "assumption_literals": literals,
        "trailing_key_count": len(candidate),
    }
    result["counts"] = {
        **base["counts"],
        "assumption_units": len(literals),
        "native_records": len(base["cnf_clauses"])
        + len(base["xor_clauses"])
        + len(literals),
    }
    result["native_dimacs"] = render_native_dimacs(
        base["num_vars"], base["cnf_clauses"], base["xor_clauses"], literals
    )
    return result


def verify_without_truth(
    cipher: str, rounds: int, key: list[int], public: dict, fresh: dict
) -> dict:
    """Check all known/fresh blocks without loading an assigned master key."""
    if len(key) != 4 or any(type(w) is not int or not 0 <= w < 65536 for w in key):
        raise ValueError("Malformed decoded key")
    encrypt, _ = cipher_functions(cipher)
    known = all(
        encrypt(tuple(pair[f"plaintext{s}"]), tuple(key), rounds)
        == tuple(pair[f"ciphertext{s}"])
        for pair in public["pairs"]
        for s in (0, 1)
    )
    pairs = fresh.get("verification_pairs")
    if not isinstance(pairs, list) or len(pairs) != 2:
        raise ValueError("Exactly two independent fresh blocks required")
    used = {tuple(p[f"plaintext{s}"]) for p in public["pairs"] for s in (0, 1)}
    pts = [tuple(pt) for pt, _ in pairs]
    if len(set(pts)) != 2 or used.intersection(pts):
        raise ValueError("Fresh verification overlaps attack information")
    return {
        "verified_known": known,
        "verified_fresh": all(
            encrypt(tuple(pt), tuple(key), rounds) == tuple(ct) for pt, ct in pairs
        ),
    }


def solve_candidate(
    formula: dict, payload: dict, directory: Path, journal: Journal
) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    journal.transition("candidate_formula_io")
    path = directory / "input.dimacs"
    atomic_json(
        directory / "attempt_start.json",
        {
            "index": int(directory.name),
            "candidate_literals": formula["assumption_literals"],
            "native_sha256": hashlib.sha256(
                formula["native_dimacs"].encode()
            ).hexdigest(),
            "formula_counts": formula["counts"],
            "observed_monotonic": time.perf_counter(),
        },
    )
    path.write_text(formula["native_dimacs"])
    arguments = solver_arguments(
        payload["cell"]["profile"], payload["paths"]["cms"], "", str(path)
    )
    launch = directory / "native_launch.json"
    atomic_json(
        launch,
        {
            "repo_root": payload["repo_root"],
            "arguments": arguments,
            "receipt": str(directory / "native_launch_receipt.json"),
        },
    )
    journal.transition("native_launch_parse_and_solve")
    started = time.perf_counter()
    with (
        (directory / "solver_stdout.txt").open("w") as stdout,
        (directory / "solver_stderr.txt").open("w") as stderr,
    ):
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "neural_sat_speck.backend.native_launcher",
                str(launch),
            ],
            stdout=stdout,
            stderr=stderr,
        )
        atomic_json(
            directory / "native_process.json",
            {
                "pid": proc.pid,
                "process_group": os.getpgrp(),
                "argv": arguments,
                "started_monotonic": started,
            },
        )
        code = proc.wait()
    ended = time.perf_counter()
    journal.transition("model_parsing_and_decode")
    receipt = read_checked_json(directory / "native_launch_receipt.json")
    if (
        receipt.get("pid") != proc.pid
        or receipt.get("is_certified") is not True
        or receipt.get("solver_arguments") != arguments
        or receipt.get("git_commit") != payload["source_commit"]
    ):
        raise ValueError("Native launch receipt does not match actual child/source")
    parsed = parse_solver_output(
        "cms",
        code,
        (directory / "solver_stdout.txt").read_text(),
        required_variables=list(range(1, formula["num_vars"] + 1)),
        num_vars=formula["num_vars"],
    )
    if parsed["status"] == "error":
        raise ValueError("Invalid solver status/model: " + str(parsed.get("error")))
    if parsed["status"] == "sat" and not _complete_assignment(
        formula, parsed["assignment"]
    ):
        raise ValueError("SAT assignment violates original formula/candidate units")
    return {
        "parsed": parsed,
        "returncode": code,
        "solver_arguments": arguments,
        "solve_wall_seconds": ended - started,
        "started_monotonic": started,
        "ended_monotonic": ended,
        "native_launch_receipt": receipt,
        "native_sha256": sha256_file(path),
        "stdout_sha256": sha256_file(directory / "solver_stdout.txt"),
        "stderr_sha256": sha256_file(directory / "solver_stderr.txt"),
    }


def execute(payload: dict) -> dict:
    directory, cell = Path(payload["directory"]), payload["cell"]
    journal = Journal(directory / "journal.json")
    result = {
        "outcome": "worker_exception",
        "recovered_key": None,
        "verified_known": False,
        "verified_fresh": False,
        "attempt_count": 0,
        "verification_rejections": 0,
    }
    try:
        journal.transition("input_loading_and_environment")
        public = read_checked_json(Path(payload["public_path"]))
        if sha256_file(Path(payload["public_path"])) != payload["public_sha256"]:
            raise ValueError("Public input drift")
        from neural_sat_speck.protocol.worker_telemetry import certify_worker_or_raise

        receipt = certify_worker_or_raise(Path(payload["repo_root"]))
        atomic_json(directory / "python_worker_receipt.json", receipt.to_dict())
        if receipt.git_commit != payload["source_commit"]:
            raise ValueError("Ranking worker source revision drift")
        journal.transition("stage1_ranking")
        from neural_sat_speck.followups.ranking_depth_ranking import rank_queue

        def phase(name: str, detail: dict) -> None:
            if name == "stage1_start":
                journal.transition("stage1_ranking")
            elif name == "stage1_complete":
                atomic_json(directory / "stage1_receipt.json", detail)
            elif name == "stage2_start":
                journal.transition("stage2_ranking")

        ciphertexts = [
            (tuple(p["ciphertext0"]), tuple(p["ciphertext1"])) for p in public["pairs"]
        ]
        ranked = rank_queue(
            cell["cipher"],
            ciphertexts,
            cell["trailing_keys"],
            {k: Path(v) for k, v in payload["paths"].items()},
            directory,
            phase,
        )
        queue = [list(q) for q in ranked["queue"]]
        parents = list(ranked["stage1_parents"])
        limit = 32 if cell["cipher"] == "speck32_64" else 8
        if (
            len(parents) != limit
            or len(set(parents)) != limit
            or len(queue) != (limit if cell["trailing_keys"] == 1 else 1024)
            or len({tuple(q) for q in queue}) != len(queue)
            or any(
                len(q) != cell["trailing_keys"] or q[-1] not in parents for q in queue
            )
            or (cell["trailing_keys"] == 1 and queue != [[p] for p in parents])
        ):
            raise ValueError(
                "Ranking queue differs from frozen dimensions/parent prefix"
            )
        queue_record = {
            **ranked,
            "queue": queue,
            "stage1_parents": parents,
            "queue_sha256": digest(queue),
            "stage1_sha256": digest(parents),
        }
        atomic_json(directory / "queue.json", queue_record)
        result.update(queue_sha256=digest(queue), stage1_sha256=digest(parents))
        journal.transition("initial_formula_construction")
        base = build_formula(cell["cipher"], public, 4, 0, include_cnf=False)
        if digest(formula_identity(base)) != payload["base_formula_sha256"]:
            raise ValueError("Base formula differs from frozen public inputs")
        result["formula_counts"] = base["counts"]
        result["outcome"] = "queue_exhausted"
        for index, candidate in enumerate(queue):
            journal.transition("candidate_formula_construction")
            formula = conditioned_formula(base, candidate)
            attempt_dir = directory / "attempts" / f"{index:04d}"
            solved = solve_candidate(formula, payload, attempt_dir, journal)
            parsed = solved.pop("parsed")
            att = {
                **solved,
                "index": index,
                "candidate": candidate,
                "status": parsed["status"],
            }
            result["attempt_count"] = index + 1
            if parsed["status"] == "sat":
                key = [
                    sum(int(parsed["assignment"][v]) << b for b, v in enumerate(word))
                    for word in base["varmap"]["master_key"]
                ]
                journal.transition("verification")
                fresh_path = Path(payload["verification_path"])
                fresh = read_checked_json(fresh_path)
                if sha256_file(fresh_path) != payload["verification_sha256"]:
                    raise ValueError("Fresh verification input drift")
                checked = verify_without_truth(
                    cell["cipher"], cell["rounds"], key, public, fresh
                )
                att.update(recovered_key=key, **checked)
                if all(checked.values()):
                    result.update(outcome="recovered", recovered_key=key, **checked)
                else:
                    result["verification_rejections"] += 1
                    att["status"] = "verification_rejected"
            elif parsed["status"] == "unknown":
                result["outcome"] = "solver_unknown"
            journal.transition("attempt_persistence")
            atomic_json(attempt_dir / "attempt.json", att)
            if result["outcome"] in {"recovered", "solver_unknown"}:
                break
        journal.transition(None)
    except Exception as error:
        result.update(
            outcome="worker_exception", error=f"{type(error).__name__}: {error}"
        )
        journal.transition(None)
    result.update(
        completed_phase_seconds=journal.completed,
        worker_seconds=time.perf_counter() - journal.started,
    )
    return result


def main() -> None:
    payload = read_checked_json(Path(sys.argv[1]))
    parent = os.getppid()
    if parent <= 1 or os.getpgrp() != os.getpid():
        raise RuntimeError(
            "Cell worker requires a live guardian and its own process group"
        )
    done = threading.Event()

    def watch_guardian():
        while not done.wait(0.05):
            if os.getppid() != parent:
                atomic_json(
                    Path(payload["directory"]) / "guardian_loss.json",
                    {"guardian_pid": parent, "observed_monotonic": time.perf_counter()},
                )
                # This is our own live group, never a saved/recycled group ID.
                os.killpg(os.getpgrp(), signal.SIGKILL)
                return

    observer = threading.Thread(target=watch_guardian, daemon=True)
    observer.start()
    try:
        atomic_json(Path(payload["directory"]) / "worker_result.json", execute(payload))
    finally:
        done.set()
        observer.join(timeout=1)


if __name__ == "__main__":
    main()
