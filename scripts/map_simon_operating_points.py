#!/usr/bin/env python3
"""Bounded, predeclared engineering map of SIMON32/64 operating points.

Evaluates 8 opened demo keys (seed 999999) across 9 and 10 rounds under matched
plaintext/ciphertext information (4 observations vs 64 observations) and
parent-enforced deadlines, measuring independent SAT-only workers and
oracle-conditioned trailing-key controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from neural_sat_speck.data.seed_policy import CLI_DEMO_SEED
from neural_sat_speck.simon.attack_instance import (
    DEFAULT_SIMON_INPUT_DIFFERENCE,
    SimonAttackInstance,
    SimonAttackInstanceConfig,
    generate_simon_attack_instances,
)
from neural_sat_speck.simon.sat_model import SimonSatModel
from neural_sat_speck.simon.verification import verify_simon_master_key

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "simon_feasibility" / "simon_operating_points_map.json"
INSTANCE_COUNT = 8
GLOBAL_CUTOFF_SECONDS = 300.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_checksum_sidecar(output: Path) -> None:
    sidecar = output.with_suffix(output.suffix + ".sha256")
    expected = f"{sha256_file(output)}  {output.name}\n"
    if sidecar.exists() and sidecar.read_text(encoding="utf-8") == expected:
        return
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=sidecar.parent,
            prefix=f".{sidecar.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(expected)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, sidecar)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def get_demo_instances(rounds: int) -> tuple[SimonAttackInstance, ...]:
    return generate_simon_attack_instances(
        SimonAttackInstanceConfig(
            instance_count=INSTANCE_COUNT,
            pair_count=32,
            rounds=rounds,
            seed=CLI_DEMO_SEED,
            split_role="demo",
            verification_pair_count=2,
            input_difference=DEFAULT_SIMON_INPUT_DIFFERENCE,
        )
    )


def worker_solve(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute one solver attempt in an isolated worker process."""
    rounds = payload["rounds"]
    known_pairs = [tuple(p) for p in payload["known_pairs"]]
    verification_pairs = [tuple(p) for p in payload["verification_pairs"]]
    assumed_subkeys = tuple(payload["assumed_subkeys"]) if payload.get("assumed_subkeys") else ()
    time_limit = float(payload.get("time_limit", GLOBAL_CUTOFF_SECONDS))

    started = time.perf_counter()
    model = SimonSatModel.from_known_pairs(known_pairs, rounds=rounds)
    build_seconds = time.perf_counter() - started
    remaining = time_limit - build_seconds

    if remaining <= 0:
        return {
            "solver_status": "not_started",
            "outcome": "budget_exhausted",
            "model_build_seconds": build_seconds,
            "solve_wall_seconds": 0.0,
            "verification_seconds": 0.0,
            "worker_wall_seconds": time.perf_counter() - started,
            "recovered_master_key": None,
            "variables": model.variable_count,
            "cnf_clauses": model.cnf_clause_count,
            "xor_clauses": model.xor_clause_count,
        }

    assumptions = (
        model.assumptions_for_trailing_round_subkeys(assumed_subkeys)
        if assumed_subkeys
        else ()
    )

    solve_started = time.perf_counter()
    solved = model.solve(assumptions, time_limit=remaining)
    solve_seconds = time.perf_counter() - solve_started

    verification_seconds = 0.0
    recovered_key = None
    if solved.satisfiable and solved.solution is not None:
        verify_started = time.perf_counter()
        recovered_key = model.master_key_from_solution(solved.solution)
        checked = verify_simon_master_key(
            recovered_key, known_pairs, verification_pairs, rounds
        )
        verification_seconds = time.perf_counter() - verify_started
        outcome = "verified_recovery" if checked.verified else "verification_failure"
    elif solved.timed_out is True or solved.status in {"solver_timeout", "unsat_or_timeout"}:
        outcome = "solver_timeout"
    elif solved.status == "unsat":
        outcome = "proven_unsat"
    else:
        outcome = f"solver_{solved.status}"

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


def evaluate_cell_key(
    rounds: int,
    observations: int,
    assumed_keys: int,
    instance_index: int,
    cutoff_seconds: float,
) -> dict[str, Any]:
    instances = get_demo_instances(rounds)
    instance = instances[instance_index]
    known = instance.known_pairs[:observations]
    assumed_subkeys = (
        instance.expected_trailing_subkeys(assumed_keys)
        if assumed_keys > 0
        else ()
    )

    payload = {
        "rounds": rounds,
        "known_pairs": known,
        "verification_pairs": instance.verification_pairs,
        "assumed_subkeys": list(assumed_subkeys),
        "time_limit": cutoff_seconds,
    }

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
            timeout=cutoff_seconds + 5.0,  # margin for parent watchdog
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
    except Exception as error:
        stderr = f"{type(error).__name__}: {error}"
        outcome = "worker_exception"

    online_seconds = time.perf_counter() - started
    if outcome == "verified_recovery" and online_seconds > cutoff_seconds:
        outcome = "late_verified_recovery"

    recovered = worker.get("recovered_master_key")
    truth_match = (
        tuple(recovered) == instance.master_key if recovered else None
    )

    return {
        "rounds": rounds,
        "observations": observations,
        "assumed_keys": assumed_keys,
        "instance_index": instance_index,
        "instance_id": instance.instance_id,
        "cutoff_seconds": cutoff_seconds,
        "online_wall_seconds": online_seconds,
        "cutoff_overshoot_seconds": max(0.0, online_seconds - cutoff_seconds),
        "outcome": outcome,
        "verified_recovery_within_cutoff": outcome == "verified_recovery",
        "truth_match_posthoc": truth_match,
        "worker": worker,
        "stderr": stderr,
    }


def run_map(output: Path, resume: bool = False) -> dict[str, Any]:
    cells = [
        # Rounds, observations, assumed_keys, cutoff_seconds, label
        (9, 4, 0, 300.0, "R9_SAT_only_N4"),
        (9, 64, 0, 300.0, "R9_SAT_only_N64"),
        (9, 4, 2, 10.0, "R9_Oracle_A2_N4"),
        (9, 64, 2, 30.0, "R9_Oracle_A2_N64"),
        (10, 4, 0, 300.0, "R10_SAT_only_N4"),
        (10, 64, 0, 300.0, "R10_SAT_only_N64"),
        (10, 4, 2, 10.0, "R10_Oracle_A2_N4"),
        (10, 64, 2, 30.0, "R10_Oracle_A2_N64"),
        (10, 64, 1, 30.0, "R10_Oracle_A1_N64"),
    ]

    data: dict[str, Any] = {
        "schema": "simon_operating_points_map.v1",
        "evidence_class": "ENGINEERING_DEMO",
        "seed": CLI_DEMO_SEED,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "git_head": _git_head(),
        },
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "IN_PROGRESS",
        "records": [],
    }

    if resume and output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("schema") == data["schema"]:
            data = existing

    done_keys = {
        (
            r["rounds"],
            r["observations"],
            r["assumed_keys"],
            r["instance_index"],
        )
        for r in data["records"]
    }

    for rounds, obs, assumed, cutoff, label in cells:
        print(f"\n--- Running Cell: {label} (R={rounds}, N={obs}, A={assumed}, cutoff={cutoff}s) ---", flush=True)
        for idx in range(INSTANCE_COUNT):
            key_tuple = (rounds, obs, assumed, idx)
            if key_tuple in done_keys:
                continue
            rec = evaluate_cell_key(rounds, obs, assumed, idx, cutoff)
            data["records"].append(rec)
            done_keys.add(key_tuple)
            _atomic_write(output, data)
            print(
                f"  [{label}] Key {idx}: {rec['outcome']} "
                f"online={rec['online_wall_seconds']:.2f}s "
                f"verified={rec['verified_recovery_within_cutoff']}",
                flush=True,
            )

    data["status"] = "COMPLETE"
    data["finished_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _atomic_write(output, data)
    _write_checksum_sidecar(output)
    print(f"\nMAP COMPLETE: {len(data['records'])} records written to {output}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        payload = json.load(sys.stdin)
        result = worker_solve(payload)
        print(json.dumps(result))
    else:
        run_map(args.output.resolve(), resume=args.resume)


if __name__ == "__main__":
    main()
