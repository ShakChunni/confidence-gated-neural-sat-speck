#!/usr/bin/env python3
"""Run the predeclared, demo-only SIMON32/64 operating-window study.

The protocol is docs/SIMON_OPERATING_WINDOW_PROTOCOL.md. This script never reads
protected data and does not train or evaluate a neural ranker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time
from importlib.metadata import PackageNotFoundError, version
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
PROTOCOL = ROOT / "docs" / "SIMON_OPERATING_WINDOW_PROTOCOL.md"
DEFAULT_OUTPUT = ROOT / "results" / "simon_feasibility" / "simon_operating_window.json"
TIERS = ((8, 9, 10), (11, 12))
OBSERVATIONS = (4, 16)
ASSUMED_KEYS = (0, 1, 2)
INSTANCE_COUNT = 8
SOLVE_CUTOFF = 10.0
WRONG_CANDIDATES = 32
PROBE_WALL_CUTOFF = 180.0


def sha256_file(path: Path) -> str:
    """Return the on-disk SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unavailable"


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def configuration() -> dict[str, Any]:
    """Return the exact frozen engineering configuration."""
    return {
        "schema": "simon_operating_window.v1",
        "evidence_class": "ENGINEERING_DEMO",
        "seed": CLI_DEMO_SEED,
        "role": "demo",
        "tiers": [list(tier) for tier in TIERS],
        "observations": list(OBSERVATIONS),
        "assumed_keys": list(ASSUMED_KEYS),
        "instances_per_cell": INSTANCE_COUNT,
        "backend": "pycryptosat_threads_1",
        "solve_cutoff_seconds": SOLVE_CUTOFF,
        "wrong_candidates_per_key": WRONG_CANDIDATES,
        "wrong_probe_wall_cutoff_seconds": PROBE_WALL_CUTOFF,
        "input_difference": list(DEFAULT_SIMON_INPUT_DIFFERENCE),
        "protocol_sha256": sha256_file(PROTOCOL),
        "runner_sha256": sha256_file(Path(__file__)),
        "cipher_sha256": sha256_file(ROOT / "src/neural_sat_speck/simon/cipher.py"),
        "key_schedule_sha256": sha256_file(ROOT / "src/neural_sat_speck/simon/key_schedule.py"),
        "sat_model_sha256": sha256_file(ROOT / "src/neural_sat_speck/simon/sat_model.py"),
        "attack_instance_sha256": sha256_file(ROOT / "src/neural_sat_speck/simon/attack_instance.py"),
        "verification_sha256": sha256_file(ROOT / "src/neural_sat_speck/simon/verification.py"),
    }


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
    if sidecar.exists():
        if sidecar.read_text(encoding="utf-8") != expected:
            raise ValueError(f"Detached checksum disagrees with result: {sidecar}")
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


def _load_or_create(path: Path, resume: bool) -> dict[str, Any]:
    config = configuration()
    if resume:
        if not path.exists():
            raise FileNotFoundError(f"Cannot resume missing checkpoint: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("configuration") != config:
            raise ValueError("Checkpoint configuration or source hashes changed")
        if data.get("status") not in {"IN_PROGRESS", "COMPLETE"}:
            raise ValueError("Checkpoint has invalid status")
        if not isinstance(data.get("grid_records"), list) or not isinstance(
            data.get("probe_records"), list
        ):
            raise ValueError("Checkpoint record arrays are malformed")
        for field, key_fields in (
            ("grid_records", ("rounds", "observations", "assumed_keys", "instance_index")),
            ("probe_records", ("rounds", "observations", "assumed_keys", "instance_index")),
        ):
            keys = [tuple(row[name] for name in key_fields) for row in data[field]]
            if len(keys) != len(set(keys)):
                raise ValueError(f"Duplicate {field} entry in checkpoint")
        return data
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    return {
        "configuration": config,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pycryptosat": _package_version("pycryptosat"),
            "git_head_at_start": _git_head(),
        },
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "IN_PROGRESS",
        "grid_records": [],
        "probe_records": [],
        "selected_cell": None,
    }


def _instances(rounds: int) -> tuple[SimonAttackInstance, ...]:
    return generate_simon_attack_instances(
        SimonAttackInstanceConfig(
            instance_count=INSTANCE_COUNT,
            pair_count=max(OBSERVATIONS) // 2,
            rounds=rounds,
            seed=CLI_DEMO_SEED,
            split_role="demo",
            verification_pair_count=2,
            input_difference=DEFAULT_SIMON_INPUT_DIFFERENCE,
        )
    )


def known_observations(
    instance: SimonAttackInstance, count: int
) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    """Flatten exactly count observations (not count differential pairs)."""
    if count <= 0 or count % 2 or count > 2 * len(instance.pairs):
        raise ValueError("Observation count must be positive, even, and available")
    return tuple(instance.known_pairs[:count])


def evaluate_cell(
    instance: SimonAttackInstance, instance_index: int, observations: int,
    assumed_keys: int,
) -> dict[str, Any]:
    """Evaluate one oracle-conditioned or unconditioned SAT control."""
    known = known_observations(instance, observations)
    started = time.perf_counter()
    model: SimonSatModel | None = None
    build_seconds = 0.0
    assumption_seconds = 0.0
    solve_seconds = 0.0
    verification_seconds = 0.0
    solver_status = "not_started"
    outcome = "system_exception"
    verified = False
    exception = None
    try:
        model = SimonSatModel.from_known_pairs(known, rounds=instance.rounds)
        build_seconds = time.perf_counter() - started
        assumptions_started = time.perf_counter()
        assumptions = (
            model.assumptions_for_trailing_round_subkeys(
                instance.expected_trailing_subkeys(assumed_keys)
            ) if assumed_keys else ()
        )
        assumption_seconds = time.perf_counter() - assumptions_started
        solve_started = time.perf_counter()
        solved = model.solve(assumptions, time_limit=SOLVE_CUTOFF)
        solve_seconds = time.perf_counter() - solve_started
        solver_status = solved.status
        if solved.satisfiable and solved.solution is not None:
            verification_started = time.perf_counter()
            recovered = model.master_key_from_solution(solved.solution)
            checked = verify_simon_master_key(
                recovered, known, instance.verification_pairs, instance.rounds
            )
            verification_seconds = time.perf_counter() - verification_started
            verified = checked.verified
            outcome = (
                "verified_recovery" if verified and solve_seconds < SOLVE_CUTOFF
                else "late_verified_recovery" if verified
                else "verification_failure"
            )
        elif solved.timed_out is True or solved.status in {"solver_timeout", "unsat_or_timeout"}:
            outcome = "solver_timeout_or_unknown"
        elif solved.status == "unsat":
            outcome = "proven_unsat"
        else:
            outcome = "solver_unknown"
    except Exception as error:  # retain the failed cell instead of dropping it
        exception = f"{type(error).__name__}: {error}"
        outcome = "solver_exception" if model is not None else "model_build_exception"
    total_seconds = time.perf_counter() - started
    return {
        "instance_id": instance.instance_id,
        "instance_index": instance_index,
        "rounds": instance.rounds,
        "observations": observations,
        "differential_pairs": observations // 2,
        "assumed_keys": assumed_keys,
        "residual_depth": instance.rounds - assumed_keys,
        "variables": model.variable_count if model else None,
        "cnf_clauses": model.cnf_clause_count if model else None,
        "xor_clauses": model.xor_clause_count if model else None,
        "model_build_seconds": build_seconds,
        "assumption_preparation_seconds": assumption_seconds,
        "solve_wall_seconds": solve_seconds,
        "verification_seconds": verification_seconds,
        "total_wall_seconds": total_seconds,
        "solve_cutoff_overshoot_seconds": max(0.0, solve_seconds - SOLVE_CUTOFF),
        "solver_status": solver_status,
        "outcome": outcome,
        "verified_recovery_within_cutoff": outcome == "verified_recovery",
        "exception": exception,
    }


def wrong_subkey_tuples(
    true_tuple: tuple[int, ...], rounds: int, observations: int,
    instance_index: int,
) -> tuple[tuple[int, ...], ...]:
    """Draw reproducible distinct wrong subkey tuples from the full tuple domain."""
    width = len(true_tuple)
    if width not in (1, 2):
        raise ValueError("Wrong-candidate probe supports one or two trailing keys")
    seed = CLI_DEMO_SEED + 100_000 * rounds + 1_000 * observations + 100 * width + instance_index
    generator = random.Random(seed)
    truth = tuple(true_tuple)
    choices: set[tuple[int, ...]] = set()
    while len(choices) < WRONG_CANDIDATES:
        candidate = tuple(generator.randrange(1 << 16) for _ in range(width))
        if candidate != truth:
            choices.add(candidate)
    return tuple(sorted(choices))


def _probe_candidate(
    instance: SimonAttackInstance, known: tuple[Any, ...], candidate: tuple[int, ...],
    remaining: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    model: SimonSatModel | None = None
    status = "system_exception"
    solver_status = "not_started"
    build_seconds = 0.0
    solve_seconds = 0.0
    verified = False
    exception = None
    try:
        model = SimonSatModel.from_known_pairs(known, rounds=instance.rounds)
        build_seconds = time.perf_counter() - started
        remaining_after_build = remaining - build_seconds
        if remaining_after_build <= 0:
            status = "probe_wall_cutoff"
        else:
            assumptions = model.assumptions_for_trailing_round_subkeys(candidate)
            solve_started = time.perf_counter()
            solved = model.solve(
                assumptions, time_limit=min(SOLVE_CUTOFF, remaining_after_build)
            )
            solve_seconds = time.perf_counter() - solve_started
            solver_status = solved.status
            if solved.satisfiable and solved.solution is not None:
                key = model.master_key_from_solution(solved.solution)
                verified = verify_simon_master_key(
                    key, known, instance.verification_pairs, instance.rounds
                ).verified
                status = "verified_wrong_tuple" if verified else "sat_unverified"
            elif solved.status == "unsat":
                status = "proven_unsat"
            elif solved.timed_out is True or solved.status in {"solver_timeout", "unsat_or_timeout"}:
                status = "unresolved_timeout"
            else:
                status = "solver_unknown"
    except Exception as error:
        exception = f"{type(error).__name__}: {error}"
    return {
        "candidate": list(candidate),
        "status": status,
        "solver_status": solver_status,
        "build_seconds": build_seconds,
        "solve_wall_seconds": solve_seconds,
        "total_wall_seconds": time.perf_counter() - started,
        "verified_recovery": verified,
        "exception": exception,
    }


def probe_wrong_candidates(
    instance: SimonAttackInstance, instance_index: int, observations: int,
    assumed_keys: int,
) -> dict[str, Any]:
    """Probe fixed wrong tuples with fresh models and one wall-clock envelope."""
    if assumed_keys <= 0:
        raise ValueError("Wrong-candidate probe requires trailing-key assumptions")
    known = known_observations(instance, observations)
    true_tuple = instance.expected_trailing_subkeys(assumed_keys)
    candidates = wrong_subkey_tuples(
        true_tuple, instance.rounds, observations, instance_index
    )
    started = time.perf_counter()
    attempts: list[dict[str, Any]] = []
    for candidate in candidates:
        remaining = PROBE_WALL_CUTOFF - (time.perf_counter() - started)
        if remaining <= 0:
            break
        attempts.append(_probe_candidate(instance, known, candidate, remaining))
    wall_seconds = time.perf_counter() - started
    resolved = sum(
        row["status"] in {"proven_unsat", "sat_unverified", "verified_wrong_tuple"}
        for row in attempts
    )
    return {
        "instance_id": instance.instance_id,
        "instance_index": instance_index,
        "rounds": instance.rounds,
        "observations": observations,
        "assumed_keys": assumed_keys,
        "candidate_count": len(attempts),
        "resolved_count": resolved,
        "unresolved_count": len(attempts) - resolved,
        "verified_wrong_tuple_count": sum(
            row["status"] == "verified_wrong_tuple" for row in attempts
        ),
        "wall_seconds": wall_seconds,
        "cutoff_overshoot_seconds": max(0.0, wall_seconds - PROBE_WALL_CUTOFF),
        "passed": (
            len(attempts) == WRONG_CANDIDATES
            and resolved >= 30
            and not any(row["verified_recovery"] for row in attempts)
            and wall_seconds < PROBE_WALL_CUTOFF
        ),
        "attempts": attempts,
    }


def grid_screen(rows: list[dict[str, Any]], rounds: tuple[int, ...]) -> list[tuple[int, int, int]]:
    """Apply the predeclared paired eight-key SAT/oracle screen."""
    by_cell: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["rounds"], row["observations"], row["assumed_keys"])
        by_cell.setdefault(key, []).append(row)
    passing: list[tuple[int, int, int]] = []
    for r in rounds:
        for a in (1, 2):
            for n in OBSERVATIONS:
                base = by_cell.get((r, n, 0), [])
                conditioned = by_cell.get((r, n, a), [])
                if len(base) != INSTANCE_COUNT or len(conditioned) != INSTANCE_COUNT:
                    continue
                if len({row["instance_index"] for row in base}) != INSTANCE_COUNT:
                    raise ValueError("Duplicate or missing base instance in grid")
                if len({row["instance_index"] for row in conditioned}) != INSTANCE_COUNT:
                    raise ValueError("Duplicate or missing conditioned instance in grid")
                base_successes = sum(row["verified_recovery_within_cutoff"] for row in base)
                conditioned_successes = sum(
                    row["verified_recovery_within_cutoff"] for row in conditioned
                )
                censored_times = [
                    row["solve_wall_seconds"]
                    if row["verified_recovery_within_cutoff"]
                    else SOLVE_CUTOFF
                    for row in conditioned
                ]
                if (
                    base_successes <= 4
                    and conditioned_successes >= 7
                    and statistics.median(censored_times) <= 1.0
                ):
                    passing.append((r, n, a))
    return sorted(passing, key=lambda cell: (cell[0], cell[2], cell[1]))


def _run_tier(
    data: dict[str, Any], output: Path, rounds: tuple[int, ...]
) -> tuple[int, int, int] | None:
    existing = {
        (row["rounds"], row["observations"], row["assumed_keys"], row["instance_index"])
        for row in data["grid_records"]
    }
    for r in rounds:
        instances = _instances(r)
        for n in OBSERVATIONS:
            for a in ASSUMED_KEYS:
                for index, instance in enumerate(instances):
                    key = (r, n, a, index)
                    if key in existing:
                        continue
                    row = evaluate_cell(instance, index, n, a)
                    data["grid_records"].append(row)
                    _atomic_write(output, data)
                    existing.add(key)
                    print(
                        f"GRID R={r} N={n} A={a} key={index}: "
                        f"{row['outcome']} solve={row['solve_wall_seconds']:.3f}s",
                        flush=True,
                    )
    passing = grid_screen(data["grid_records"], rounds)
    print(f"Tier {rounds} screen-passing cells: {passing}", flush=True)
    for r, n, a in passing:
        instances = _instances(r)
        existing_probes = {
            (row["rounds"], row["observations"], row["assumed_keys"], row["instance_index"])
            for row in data["probe_records"]
        }
        for index, instance in enumerate(instances):
            if (r, n, a, index) in existing_probes:
                continue
            row = probe_wrong_candidates(instance, index, n, a)
            data["probe_records"].append(row)
            _atomic_write(output, data)
            print(
                f"PROBE R={r} N={n} A={a} key={index}: "
                f"{row['candidate_count']}/32 candidates, "
                f"{row['resolved_count']} resolved, {row['wall_seconds']:.2f}s",
                flush=True,
            )
        probes = [
            row for row in data["probe_records"]
            if (row["rounds"], row["observations"], row["assumed_keys"]) == (r, n, a)
        ]
        if len(probes) == INSTANCE_COUNT and sum(row["passed"] for row in probes) >= 7:
            return r, n, a
    return None


def run(output: Path, resume: bool = False) -> dict[str, Any]:
    """Execute the fixed staged grid and atomic checkpoint workflow."""
    data = _load_or_create(output, resume)
    if data["status"] == "COMPLETE":
        _write_checksum_sidecar(output)
        return data
    _atomic_write(output, data)
    for tier in TIERS:
        selected = _run_tier(data, output, tier)
        if selected is not None:
            data["selected_cell"] = {
                "rounds": selected[0],
                "observations": selected[1],
                "assumed_keys": selected[2],
                "residual_depth": selected[0] - selected[2],
            }
            break
    data["status"] = "COMPLETE"
    data["finished_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    data["grid_count"] = len(data["grid_records"])
    data["probe_instance_count"] = len(data["probe_records"])
    _atomic_write(output, data)
    _write_checksum_sidecar(output)
    print(f"COMPLETE selected_cell={data['selected_cell']} sha256={sha256_file(output)}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.output.resolve(), resume=args.resume)


if __name__ == "__main__":
    main()
