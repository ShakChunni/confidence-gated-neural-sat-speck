#!/usr/bin/env python3
"""Demo-only nine-round SIMON SAT-only check under a 300-second online cutoff."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from neural_sat_speck.data.seed_policy import CLI_DEMO_SEED
from neural_sat_speck.simon.attack_instance import DEFAULT_SIMON_INPUT_DIFFERENCE
from neural_sat_speck.simon.sat_model import SimonSatModel
from neural_sat_speck.simon.verification import verify_simon_master_key

if __package__:
    from scripts.adjudicate_simon_operating_window import (
        ROOT,
        _atomic_write,
        _git_head,
        _instances,
        _write_checksum_sidecar,
        known_observations,
        sha256_file,
    )
else:  # direct CLI execution places scripts/, not the repository root, on sys.path
    from adjudicate_simon_operating_window import (
        ROOT,
        _atomic_write,
        _git_head,
        _instances,
        _write_checksum_sidecar,
        known_observations,
        sha256_file,
    )

PROTOCOL = ROOT / "docs" / "SIMON_300S_SAT_CONTROL_PROTOCOL.md"
DEFAULT_OUTPUT = ROOT / "results" / "simon_feasibility" / "simon_sat_only_300s.json"
ROUNDS = 9
OBSERVATIONS = 4
INSTANCE_COUNT = 8
CUTOFF_SECONDS = 300.0


def configuration() -> dict[str, Any]:
    """Identify the fixed demo run and every source needed for replay."""
    return {
        "schema": "simon_sat_only_300s.v1",
        "evidence_class": "ENGINEERING_DEMO",
        "seed": CLI_DEMO_SEED,
        "rounds": ROUNDS,
        "observations": OBSERVATIONS,
        "instance_count": INSTANCE_COUNT,
        "cutoff_seconds": CUTOFF_SECONDS,
        "backend": "pycryptosat_threads_1",
        "input_difference": list(DEFAULT_SIMON_INPUT_DIFFERENCE),
        "protocol_sha256": sha256_file(PROTOCOL),
        "runner_sha256": sha256_file(Path(__file__)),
        "instance_generator_sha256": sha256_file(
            ROOT / "src/neural_sat_speck/simon/attack_instance.py"
        ),
        "cipher_sha256": sha256_file(ROOT / "src/neural_sat_speck/simon/cipher.py"),
        "key_schedule_sha256": sha256_file(
            ROOT / "src/neural_sat_speck/simon/key_schedule.py"
        ),
        "sat_model_sha256": sha256_file(ROOT / "src/neural_sat_speck/simon/sat_model.py"),
        "verification_sha256": sha256_file(
            ROOT / "src/neural_sat_speck/simon/verification.py"
        ),
    }


def worker_evaluate(payload: dict[str, Any]) -> dict[str, Any]:
    """Solve one instance from public attack observations, without ground truth."""
    if set(payload) != {"rounds", "known_pairs", "verification_pairs"}:
        raise ValueError("Worker payload has unexpected or missing fields")
    if payload["rounds"] != ROUNDS or len(payload["known_pairs"]) != OBSERVATIONS:
        raise ValueError("Worker payload does not match fixed protocol")
    if len(payload["verification_pairs"]) != 2:
        raise ValueError("Exactly two fresh verification observations are required")

    started = time.perf_counter()
    model = SimonSatModel.from_known_pairs(payload["known_pairs"], rounds=ROUNDS)
    build_seconds = time.perf_counter() - started
    remaining = CUTOFF_SECONDS - build_seconds
    if remaining <= 0:
        return {
            "solver_status": "not_started", "outcome": "worker_budget_exhausted",
            "model_build_seconds": build_seconds, "solve_wall_seconds": 0.0,
            "verification_seconds": 0.0, "worker_wall_seconds": time.perf_counter() - started,
            "recovered_master_key": None, "variables": model.variable_count,
            "cnf_clauses": model.cnf_clause_count, "xor_clauses": model.xor_clause_count,
        }

    solve_started = time.perf_counter()
    solved = model.solve((), time_limit=remaining)
    solve_seconds = time.perf_counter() - solve_started
    verification_seconds = 0.0
    recovered_key: tuple[int, int, int, int] | None = None
    if solved.satisfiable and solved.solution is not None:
        verify_started = time.perf_counter()
        recovered_key = model.master_key_from_solution(solved.solution)
        checked = verify_simon_master_key(
            recovered_key, payload["known_pairs"], payload["verification_pairs"], ROUNDS
        )
        verification_seconds = time.perf_counter() - verify_started
        outcome = "verified_recovery" if checked.verified else "verification_failure"
    elif solved.timed_out is True or solved.status in {"solver_timeout", "unsat_or_timeout"}:
        outcome = "unresolved_solver_limit"
    elif solved.status == "unsat":
        outcome = "unexpected_proven_unsat"
    else:
        outcome = "solver_unknown"

    return {
        "solver_status": solved.status,
        "outcome": outcome,
        "model_build_seconds": build_seconds,
        "solve_wall_seconds": solve_seconds,
        "verification_seconds": verification_seconds,
        "worker_wall_seconds": time.perf_counter() - started,
        "recovered_master_key": list(recovered_key) if recovered_key else None,
        "variables": model.variable_count,
        "cnf_clauses": model.cnf_clause_count,
        "xor_clauses": model.xor_clause_count,
    }


def _payload(index: int) -> tuple[dict[str, Any], tuple[int, int, int, int]]:
    instances = _instances(ROUNDS)
    if len(instances) != INSTANCE_COUNT or not 0 <= index < INSTANCE_COUNT:
        raise ValueError("Demo instance count or index does not match protocol")
    instance = instances[index]
    payload = {
        "rounds": ROUNDS,
        "known_pairs": known_observations(instance, OBSERVATIONS),
        "verification_pairs": instance.verification_pairs,
    }
    return payload, instance.master_key


def evaluate_index(index: int) -> dict[str, Any]:
    """Enforce one hard online wall limit around an independent solver worker."""
    payload, true_key = _payload(index)
    assert "master_key" not in payload and "expected_trailing_subkeys" not in payload
    started = time.perf_counter()
    outcome = "worker_exception"
    worker: dict[str, Any] = {}
    stderr = None
    try:
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker"],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=CUTOFF_SECONDS,
            check=False,
        )
        stderr = completed.stderr[-2000:] if completed.stderr else None
        if completed.returncode == 0:
            worker = json.loads(completed.stdout)
            outcome = str(worker["outcome"])
        else:
            outcome = "worker_exception"
    except subprocess.TimeoutExpired:
        outcome = "parent_wall_timeout"
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        stderr = f"{type(error).__name__}: {error}"
        outcome = "worker_exception"

    online_seconds = time.perf_counter() - started
    if outcome == "verified_recovery" and online_seconds >= CUTOFF_SECONDS:
        outcome = "late_verified_recovery"
    recovered = worker.get("recovered_master_key")
    return {
        "instance_index": index,
        "instance_id": _instances(ROUNDS)[index].instance_id,
        "online_wall_seconds": online_seconds,
        "cutoff_overshoot_seconds": max(0.0, online_seconds - CUTOFF_SECONDS),
        "outcome": outcome,
        "verified_recovery_within_cutoff": outcome == "verified_recovery",
        "truth_match_posthoc": tuple(recovered) == true_key if recovered else None,
        "worker": worker,
        "stderr": stderr,
    }


def _new_result(output: Path, resume: bool) -> dict[str, Any]:
    config = configuration()
    if resume:
        if not output.exists():
            raise FileNotFoundError(f"Cannot resume missing result: {output}")
        data = json.loads(output.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("configuration") != config:
            raise ValueError("Result configuration or source hashes changed")
        if data.get("status") not in {"IN_PROGRESS", "COMPLETE"}:
            raise ValueError("Invalid result status")
        records = data.get("records")
        if not isinstance(records, list):
            raise ValueError("Result records are malformed")
        indices = [row["instance_index"] for row in records]
        if len(set(indices)) != len(indices) or any(
            not isinstance(i, int) or i not in range(INSTANCE_COUNT) for i in indices
        ):
            raise ValueError("Duplicate or invalid instance index in result")
        if indices != list(range(len(indices))):
            raise ValueError("Result records are missing or out of order")
        required = {
            "instance_index", "instance_id", "online_wall_seconds", "outcome",
            "verified_recovery_within_cutoff", "truth_match_posthoc", "worker", "stderr",
        }
        for row in records:
            if not isinstance(row, dict) or not required.issubset(row):
                raise ValueError("Incomplete result record")
            if row["verified_recovery_within_cutoff"] != (row["outcome"] == "verified_recovery"):
                raise ValueError("Contradictory recovery fields")
        if data["status"] == "COMPLETE" and (
            len(records) != INSTANCE_COUNT
            or data.get("verified_count") != sum(
                row["verified_recovery_within_cutoff"] for row in records
            )
        ):
            raise ValueError("Completed result has incomplete or inconsistent records")
        return data
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {output}")
    try:
        import pycryptosat

        solver_version = getattr(pycryptosat, "__version__", "unknown")
    except ImportError:
        solver_version = "unavailable"
    return {
        "configuration": config,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "pycryptosat_import_version": solver_version,
            "git_head_at_start": _git_head(),
        },
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "IN_PROGRESS",
        "records": [],
    }


def run(output: Path, *, resume: bool = False) -> dict[str, Any]:
    """Run or resume all eight fixed demo controls with atomic per-key records."""
    data = _new_result(output, resume)
    if data["status"] == "COMPLETE":
        _write_checksum_sidecar(output)
        return data
    _atomic_write(output, data)
    done = {row["instance_index"] for row in data["records"]}
    for index in range(INSTANCE_COUNT):
        if index in done:
            continue
        row = evaluate_index(index)
        data["records"].append(row)
        _atomic_write(output, data)
        print(
            f"SIMON A0 {index + 1}/{INSTANCE_COUNT}: {row['outcome']} "
            f"online={row['online_wall_seconds']:.2f}s",
            flush=True,
        )
    data["status"] = "COMPLETE"
    data["finished_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    data["verified_count"] = sum(
        row["verified_recovery_within_cutoff"] for row in data["records"]
    )
    _atomic_write(output, data)
    _write_checksum_sidecar(output)
    print(f"COMPLETE {data['verified_count']}/{INSTANCE_COUNT} sha256={sha256_file(output)}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        result = worker_evaluate(json.load(sys.stdin))
        print(json.dumps(result))
    else:
        run(args.output.resolve(), resume=args.resume)


if __name__ == "__main__":
    main()
