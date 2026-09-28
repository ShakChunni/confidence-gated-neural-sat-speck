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
    *,
    worker_cmd: list[str] | None = None,
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
    cmd = worker_cmd or [sys.executable, str(Path(__file__).resolve()), "--worker"]

    try:
        remaining_timeout = max(0.001, cutoff_seconds - (time.perf_counter() - started))
        completed = subprocess.run(
            cmd,
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=remaining_timeout,
            check=False,
        )
        stderr = completed.stderr[-2000:] if completed.stderr else None
        if completed.returncode == 0:
            try:
                worker = json.loads(completed.stdout)
                worker_outcome = str(worker.get("outcome", "worker_exception"))
                if worker_outcome == "verified_recovery":
                    outcome = "verified_recovery"
                elif worker_outcome == "verification_failure":
                    outcome = "verification_failure"
                elif worker_outcome in {"solver_timeout", "unsat_or_timeout"}:
                    outcome = "solver_timeout"
                elif worker_outcome == "proven_unsat":
                    outcome = "proven_unsat"
                elif worker_outcome == "budget_exhausted":
                    outcome = "budget_exhausted"
                else:
                    outcome = worker_outcome
            except json.JSONDecodeError:
                outcome = "worker_exception"
                stderr = f"JSONDecodeError: stdout={completed.stdout[:500]}"
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
        "master_key": list(instance.master_key),
        "cutoff_seconds": cutoff_seconds,
        "online_wall_seconds": online_seconds,
        "cutoff_overshoot_seconds": max(0.0, online_seconds - cutoff_seconds),
        "outcome": outcome,
        "verified_recovery_within_cutoff": outcome == "verified_recovery",
        "truth_match_posthoc": truth_match,
        "worker": worker,
        "stderr": stderr,
    }


def run_map(
    output: Path,
    resume: bool = False,
    workers: int = 1,
    profile: str = "full",
) -> dict[str, Any]:
    if workers not in (1, 2):
        raise ValueError("Workers must be 1 or 2 per AGENTS.md hardware governance")
    if profile not in ("full", "r10_serial"):
        raise ValueError("unknown operating-point profile")

    if profile == "r10_serial":
        cells = [
            (10, 4, 2, 300.0, "R10_Oracle_A2_N4"),
            (10, 64, 2, 300.0, "R10_Oracle_A2_N64"),
            (10, 4, 0, 300.0, "R10_SAT_only_N4"),
            (10, 64, 0, 300.0, "R10_SAT_only_N64"),
        ]
    else:
        cells = [
            # Rounds, observations, assumed_keys, cutoff_seconds, label
            # Fast oracle controls first (<2 minutes total)
            (9, 4, 2, 300.0, "R9_Oracle_A2_N4"),
            (9, 64, 2, 300.0, "R9_Oracle_A2_N64"),
            (10, 4, 2, 300.0, "R10_Oracle_A2_N4"),
            (10, 64, 2, 300.0, "R10_Oracle_A2_N64"),
            # Solvable unguided SAT anchor (~3-5 minutes)
            (9, 4, 0, 300.0, "R9_SAT_only_N4"),
            # Hard/timeout unguided SAT evaluations
            (10, 4, 0, 300.0, "R10_SAT_only_N4"),
            (9, 64, 0, 300.0, "R9_SAT_only_N64"),
            (10, 64, 0, 300.0, "R10_SAT_only_N64"),
        ]

    data: dict[str, Any] = {
        "schema": "simon_operating_points_map.v1",
        "evidence_class": "ENGINEERING_DEMO",
        "profile": profile,
        "cells": [list(cell) for cell in cells],
        "seed": CLI_DEMO_SEED,
        "strict_parent_deadline": True,
        "workers": workers,
        "command": " ".join(sys.argv),
        "solver_configuration": {
            "backend": "pycryptosat",
            "native_xor": True,
            "incremental_assumptions": True,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "git_head": _git_head(),
        },
        "cohort_master_keys": {
            str(r): [list(inst.master_key) for inst in get_demo_instances(r)]
            for r in (9, 10)
        },
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "IN_PROGRESS",
        "records": [],
    }

    if resume and output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        required_equal = (
            "schema", "profile", "cells", "seed", "workers",
            "strict_parent_deadline", "solver_configuration", "cohort_master_keys",
        )
        mismatched = [field for field in required_equal if existing.get(field) != data[field]]
        if existing.get("environment", {}).get("git_head") != data["environment"]["git_head"]:
            mismatched.append("environment.git_head")
        if mismatched:
            raise ValueError(f"cannot resume with changed operating-point manifest: {mismatched}")
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

    from concurrent.futures import ThreadPoolExecutor, as_completed

    for rounds, obs, assumed, cutoff, label in cells:
        print(f"\n--- Running Cell: {label} (R={rounds}, N={obs}, A={assumed}, cutoff={cutoff}s) ---", flush=True)
        pending_indices = [
            idx for idx in range(INSTANCE_COUNT)
            if (rounds, obs, assumed, idx) not in done_keys
        ]

        if not pending_indices:
            print(f"  [{label}] All {INSTANCE_COUNT} keys already completed.", flush=True)
            continue

        if workers == 1:
            for idx in pending_indices:
                key_tuple = (rounds, obs, assumed, idx)
                rec = evaluate_cell_key(rounds, obs, assumed, idx, cutoff)
                data["records"].append(rec)
                done_keys.add(key_tuple)
                _atomic_write(output, data)
                _write_checksum_sidecar(output)
                print(
                    f"  [{label}] Key {idx}: {rec['outcome']} "
                    f"online={rec['online_wall_seconds']:.2f}s "
                    f"verified={rec['verified_recovery_within_cutoff']}",
                    flush=True,
                )
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_idx = {
                    executor.submit(evaluate_cell_key, rounds, obs, assumed, idx, cutoff): idx
                    for idx in pending_indices
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    key_tuple = (rounds, obs, assumed, idx)
                    rec = future.result()
                    data["records"].append(rec)
                    done_keys.add(key_tuple)
                    _atomic_write(output, data)
                    _write_checksum_sidecar(output)
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
    parser.add_argument("--profile", choices=["full", "r10_serial"], default="full")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1, choices=[1, 2], help="Worker threads (max 2)")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        payload = json.load(sys.stdin)
        result = worker_solve(payload)
        print(json.dumps(result))
    else:
        if args.output is not None:
            output = args.output.resolve()
        elif args.profile == "r10_serial":
            output = DEFAULT_OUTPUT.parent / "simon_r10_serial_confirmation.json"
        else:
            output = DEFAULT_OUTPUT.resolve()
        run_map(output, resume=args.resume, workers=args.workers, profile=args.profile)


if __name__ == "__main__":
    main()
