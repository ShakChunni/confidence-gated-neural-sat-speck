#!/usr/bin/env python3
"""R=10 unguided SAT-only 300-second control benchmark on opened demo cohort.

Governed by AGENTS.md, D-029, D-034, D-035:
- Demonstrates and measures unguided exact SAT solving on 8 opened demo keys
  (seed 999999, R=10, 64 attack pairs = 128 observations, 2 fresh verification pairs).
- Evaluates:
  1. sat_only_n4: 4 observations (2 pairs), formula-matched control
  2. sat_only_n128: 128 observations (64 pairs), information-matched all-observation control
- Parameters:
  - 300.0s time limit requested to solver (NO 45-second candidate request)
- 300.0s parent watchdog deadline for new runs; the preserved historical
  opened artifact used a 305.0s watchdog and must retain that metadata
  - Strict per-key per-method subprocess solver isolation
- Records per key and method:
  - instance_id, instance_index, master_key, method, observations, outcome,
    solver_status, model_build_wall_seconds, solve_wall_seconds,
    verification_wall_seconds, online_wall_seconds, variables, cnf_clauses,
    xor_clauses, total_clauses, peak_rss_gib, truth_match_posthoc,
    verified_recovery_within_cutoff, stderr
- Persists raw records atomically to
  results/simon_feasibility/simon_r10_sat_only_300s_controls.json
  with detached SHA-256 sidecar.
- Supports durable resume: existing partial results are loaded and already completed
  (instance_index, method) pairs are skipped.
- Scientific rigor:
  - Prior 45-second 0/8 result (simon_r10_sat_only_64pairs_128obs.json) was an advisory
    soft cutoff and does not prove the 300-second outcome; any extrapolation is withdrawn.
  - No speculative claims of zero swapping or Gaussian elimination mechanisms are made
    without direct instrumentation.
  - Peak RSS is documented honestly from getrusage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
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
DEFAULT_OUTPUT = (
    ROOT / "results" / "simon_feasibility" / "simon_r10_sat_only_300s_controls.json"
)

ROUNDS = 10
INSTANCE_COUNT = 8
PAIR_COUNT = 64
DEFAULT_TIME_LIMIT_SECONDS = 300.0
DEFAULT_WATCHDOG_SECONDS = 300.0


def sha256_file(path: Path) -> str:
    """Return on-disk SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_peak_rss_gib() -> float:
    """Measure peak RSS in GiB on macOS/Linux via getrusage."""
    # On macOS, ru_maxrss is in bytes; on Linux, ru_maxrss is in KiB.
    scale = (1024.0**3) if platform.system() == "Darwin" else (1024.0**2)
    self_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    child_rss = float(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    return max(self_rss, child_rss) / scale


def _git_head() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp = Path(handle.name)
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        temp = None
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def _write_checksum_sidecar(output: Path) -> None:
    sidecar = output.with_suffix(output.suffix + ".sha256")
    expected = f"{sha256_file(output)}  {output.name}\n"
    temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=sidecar.parent,
            prefix=f".{sidecar.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp = Path(handle.name)
            handle.write(expected)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, sidecar)
        temp = None
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def generate_demo_instances(
    instance_count: int = INSTANCE_COUNT,
) -> tuple[SimonAttackInstance, ...]:
    """Generate the 64-pair cohort of 8 opened demo keys."""
    cfg = SimonAttackInstanceConfig(
        instance_count=instance_count,
        pair_count=PAIR_COUNT,
        rounds=ROUNDS,
        seed=CLI_DEMO_SEED,
        split_role="demo",
        verification_pair_count=2,
        input_difference=DEFAULT_SIMON_INPUT_DIFFERENCE,
    )
    return generate_simon_attack_instances(cfg)


def worker_solve(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute a single unguided SAT solve within an isolated subprocess."""
    rounds = payload["rounds"]
    known_pairs = [
        ((p[0][0], p[0][1]), (p[1][0], p[1][1])) for p in payload["known_pairs"]
    ]
    verification_pairs = [
        ((p[0][0], p[0][1]), (p[1][0], p[1][1]))
        for p in payload["verification_pairs"]
    ]
    time_limit = float(payload.get("time_limit", DEFAULT_TIME_LIMIT_SECONDS))

    t_start = time.perf_counter()
    model = SimonSatModel.from_known_pairs(known_pairs, rounds=rounds)
    build_seconds = time.perf_counter() - t_start

    vars_count = model.variable_count
    cnf_count = model.cnf_clause_count
    xor_count = model.xor_clause_count
    total_clauses = cnf_count + xor_count

    t_solve_start = time.perf_counter()
    res = model.solve(time_limit=time_limit)
    solve_seconds = time.perf_counter() - t_solve_start

    verif_seconds = 0.0
    recovered_key = None
    verified = False

    if res.satisfiable and res.solution is not None:
        t_verif = time.perf_counter()
        recovered_key = model.master_key_from_solution(res.solution)
        v_res = verify_simon_master_key(
            recovered_key, known_pairs, verification_pairs, rounds=rounds
        )
        verif_seconds = time.perf_counter() - t_verif
        verified = v_res.verified
        outcome = "verified_recovery" if verified else "verification_failure"
    elif res.timed_out is True or res.status in {"solver_timeout", "unsat_or_timeout"}:
        outcome = "solver_timeout"
    elif res.status == "unsat":
        outcome = "proven_unsat"
    else:
        outcome = f"solver_{res.status}"

    worker_wall = time.perf_counter() - t_start
    peak_rss = get_peak_rss_gib()

    return {
        "outcome": outcome,
        "solver_status": res.status,
        "satisfiable": res.satisfiable,
        "timed_out": res.timed_out,
        "verified_on_fresh_pairs": verified,
        "recovered_master_key": list(recovered_key) if recovered_key else None,
        "model_build_wall_seconds": build_seconds,
        "solve_wall_seconds": solve_seconds,
        "verification_wall_seconds": verif_seconds,
        "worker_wall_seconds": worker_wall,
        "variables": vars_count,
        "cnf_clauses": cnf_count,
        "xor_clauses": xor_count,
        "total_clauses": total_clauses,
        "peak_rss_gib": peak_rss,
    }


def evaluate_control(
    instance: SimonAttackInstance,
    instance_index: int,
    method: str,
    obs_count: int,
    *,
    time_limit: float = DEFAULT_TIME_LIMIT_SECONDS,
    watchdog_seconds: float = DEFAULT_WATCHDOG_SECONDS,
    script_path: Path | None = None,
) -> dict[str, Any]:
    """Run worker under watchdog for one instance and method in isolated subprocess."""
    assert method in ("sat_only_n4", "sat_only_n128")
    known = instance.known_pairs[:obs_count]
    assert len(known) == obs_count

    payload = {
        "rounds": instance.rounds,
        "method": method,
        "obs_count": obs_count,
        "time_limit": time_limit,
        "known_pairs": [
            [list(p[0]), list(p[1])] for p in known
        ],
        "verification_pairs": [
            [list(p[0]), list(p[1])] for p in instance.verification_pairs
        ],
    }

    t0 = time.perf_counter()
    outcome = "worker_exception"
    worker_data: dict[str, Any] = {}
    stderr: str | None = None

    target_script = str((script_path or Path(__file__)).resolve())
    try:
        proc = subprocess.run(
            [sys.executable, target_script, "--worker"],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=watchdog_seconds,
            check=False,
        )
        stderr = proc.stderr[-2000:] if proc.stderr else None
        if proc.returncode == 0:
            try:
                worker_data = json.loads(proc.stdout)
                outcome = worker_data["outcome"]
            except json.JSONDecodeError as err:
                outcome = "worker_exception"
                stderr = f"JSONDecodeError: {err}; stdout: {proc.stdout[:500]}"
        else:
            outcome = "worker_exception"
            if not stderr:
                stderr = f"Worker exited with code {proc.returncode}"
    except subprocess.TimeoutExpired:
        outcome = "parent_wall_timeout"
        stderr = f"Subprocess timed out after parent watchdog ({watchdog_seconds:.1f}s)"
    except Exception as err:
        stderr = f"{type(err).__name__}: {err}"
        outcome = "worker_exception"

    online_wall = time.perf_counter() - t0
    rec_key = worker_data.get("recovered_master_key")
    truth_match = (tuple(rec_key) == instance.master_key) if rec_key else False

    return {
        "instance_id": instance.instance_id,
        "instance_index": instance_index,
        "master_key": list(instance.master_key),
        "method": method,
        "observations": obs_count,
        "differential_pairs": obs_count // 2,
        "outcome": outcome,
        "solver_status": worker_data.get("solver_status", "unknown"),
        "model_build_wall_seconds": worker_data.get("model_build_wall_seconds"),
        "solve_wall_seconds": worker_data.get("solve_wall_seconds"),
        "verification_wall_seconds": worker_data.get("verification_wall_seconds"),
        "online_wall_seconds": online_wall,
        "variables": worker_data.get("variables"),
        "cnf_clauses": worker_data.get("cnf_clauses"),
        "xor_clauses": worker_data.get("xor_clauses"),
        "total_clauses": worker_data.get("total_clauses"),
        "peak_rss_gib": worker_data.get("peak_rss_gib"),
        "truth_match_posthoc": truth_match,
        "verified_recovery_within_cutoff": (
            outcome == "verified_recovery" and online_wall <= time_limit
        ),
        "recovered_master_key": rec_key,
        "stderr": stderr,
    }


def compute_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute summary statistics for both methods."""
    summary: dict[str, Any] = {}
    for method in ("sat_only_n4", "sat_only_n128"):
        recs = [r for r in records if r["method"] == method]
        n = len(recs)
        verified = sum(1 for r in recs if r["verified_recovery_within_cutoff"])
        timeouts = sum(
            1 for r in recs if r["outcome"] in ("solver_timeout", "parent_wall_timeout")
        )
        # Historical parent-killed records used zero placeholders; these are
        # missing measurements, not instant solves or zero-memory executions.
        solves = [
            float(r["solve_wall_seconds"])
            for r in recs
            if r["outcome"] != "parent_wall_timeout"
            and r.get("solve_wall_seconds") is not None
        ]
        mem = [
            float(r["peak_rss_gib"])
            for r in recs
            if r["outcome"] != "parent_wall_timeout"
            and r.get("peak_rss_gib") is not None
        ]
        formula_source = next((r for r in recs if r.get("variables")), None)
        summary[method] = {
            "total_keys": n,
            "verified_recoveries": verified,
            "timeouts": timeouts,
            "other_outcomes": n - verified - timeouts,
            "solve_time_observed_count": len(solves),
            "solve_time_censored_count": n - len(solves),
            "min_solve_seconds": min(solves) if solves else None,
            "median_solve_seconds": statistics.median(solves) if solves else None,
            "max_solve_seconds": max(solves) if solves else None,
            "mean_solve_seconds": statistics.mean(solves) if solves else None,
            "rss_observed_count": len(mem),
            "min_peak_rss_gib": min(mem) if mem else None,
            "median_peak_rss_gib": statistics.median(mem) if mem else None,
            "max_peak_rss_gib": max(mem) if mem else None,
            "formula": {
                "variables": formula_source["variables"] if formula_source else None,
                "cnf_clauses": formula_source["cnf_clauses"] if formula_source else None,
                "xor_clauses": formula_source["xor_clauses"] if formula_source else None,
                "total_clauses": formula_source["total_clauses"] if formula_source else None,
            },
        }

    summary["scientific_notes"] = {
        "status_of_prior_45s_control": (
            "The earlier 45.0s soft-request control (simon_r10_sat_only_64pairs_128obs.json) "
            "established only that SAT-only solving on R=10 does not find a solution within "
            "an early 45.0s solver cutoff. It did NOT establish or prove the outcome under the full "
            "300.0s parent wall allowance. Any prior extrapolation of the 45.0s 0/8 result to 300.0s "
            "is formally withdrawn. This benchmark executes the full 300.0s time limit."
        ),
        "memory_and_solver_instrumentation": (
            "Peak RSS is directly recorded via getrusage (ru_maxrss) for each isolated run. "
            "No assertions of 'zero swapping' or internal Gaussian elimination mechanisms are made, "
            "as getrusage measures peak resident memory only and does not directly monitor kernel "
            "page swap events or internal solver linear algebra operations."
        ),
        "solver_behavior": (
            "pycryptosat operates under an isolated subprocess per (key, method) evaluation with "
            "a 300.0s requested solver time limit and an artifact-recorded parent watchdog. "
            "Parent-killed solve times and RSS are missing, not zero. "
            "Outcomes reflect verified master-key recovery on fresh plaintext-ciphertext pairs."
        ),
    }
    return summary


def run_benchmark(
    output_path: Path,
    *,
    resume: bool = False,
    time_limit: float = DEFAULT_TIME_LIMIT_SECONDS,
    watchdog_seconds: float = DEFAULT_WATCHDOG_SECONDS,
    max_instances: int | None = None,
    methods: list[str] | None = None,
    script_path: Path | None = None,
) -> dict[str, Any]:
    """Execute isolated SAT-only control benchmark with atomic persistence and resume."""
    if output_path.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite existing evidence: {output_path}")
    all_instances = generate_demo_instances()
    if max_instances is not None:
        instances = all_instances[:max_instances]
    else:
        instances = all_instances
    all_methods = [
        ("sat_only_n4", 4),
        ("sat_only_n128", 128),
    ]
    active_configs = [(m, obs) for m, obs in all_methods if methods is None or m in methods]
    if not active_configs:
        raise ValueError("No valid SAT-only control methods selected")

    records: list[dict[str, Any]] = []
    started_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    if resume and output_path.exists():
        sidecar = output_path.with_suffix(output_path.suffix + ".sha256")
        if not sidecar.is_file():
            raise ValueError("Resume refused: missing SHA-256 sidecar")
        expected_sidecar = f"{sha256_file(output_path)}  {output_path.name}\n"
        if sidecar.read_text(encoding="utf-8") != expected_sidecar:
            raise ValueError("Resume refused: SHA-256 sidecar mismatch")
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        expected_fields = {
            "schema": "simon_r10_sat_only_300s_controls.v1",
            "seed": CLI_DEMO_SEED,
            "rounds": ROUNDS,
            "pair_count": PAIR_COUNT,
            "instance_count": len(instances),
            "methods": [m for m, _ in active_configs],
            "time_limit_seconds": time_limit,
            "watchdog_seconds": watchdog_seconds,
        }
        for field_name, expected_value in expected_fields.items():
            if existing.get(field_name) != expected_value:
                raise ValueError(f"Resume refused: {field_name} does not match current run")
        records = existing.get("records")
        if not isinstance(records, list):
            raise ValueError("Resume refused: records must be a list")
        seen: set[tuple[int, str]] = set()
        valid_methods = {"sat_only_n4": 4, "sat_only_n128": 128}
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("Resume refused: malformed record")
            index = record.get("instance_index")
            method = record.get("method")
            if type(index) is not int or index < 0 or index >= len(instances):
                raise ValueError("Resume refused: invalid instance index")
            if method not in valid_methods or method not in {m for m, _ in active_configs}:
                raise ValueError("Resume refused: invalid method")
            key = (index, method)
            if key in seen:
                raise ValueError("Resume refused: duplicate instance/method record")
            seen.add(key)
            if (
                record.get("instance_id") != all_instances[index].instance_id
                or record.get("master_key") != list(all_instances[index].master_key)
                or record.get("observations") != valid_methods[method]
            ):
                raise ValueError("Resume refused: record cohort does not match")
        started_at_utc = existing.get("started_at_utc", started_at_utc)
        print(f"Resuming from existing artifact with {len(records)} records.")

    completed_set = {
        (r["instance_index"], r["method"])
        for r in records
        if "instance_index" in r and "method" in r
    }

    try:
        import pycryptosat
        solver_version = getattr(pycryptosat, "__version__", "unknown")
    except ImportError:
        solver_version = "unavailable"

    base_artifact: dict[str, Any] = {
        "schema": "simon_r10_sat_only_300s_controls.v1",
        "evidence_class": "OPENED_DEMO_REPLAY",
        "seed": CLI_DEMO_SEED,
        "rounds": ROUNDS,
        "instance_count": len(instances),
        "pair_count": PAIR_COUNT,
        "methods": [m for m, _ in active_configs],
        "time_limit_seconds": time_limit,
        "watchdog_seconds": watchdog_seconds,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "pycryptosat_version": solver_version,
            "git_head": _git_head(),
        },
        "started_at_utc": started_at_utc,
        "status": "IN_PROGRESS",
        "records": records,
    }
    _atomic_write(output_path, base_artifact)
    _write_checksum_sidecar(output_path)

    print(f"Starting SAT-only R=10 300s benchmark on {len(instances)} opened demo keys...")
    print(f"Configurations: {[c[0] for c in active_configs]}")
    print(f"Time limit: {time_limit:.1f}s, Watchdog: {watchdog_seconds:.1f}s\n")

    for idx, inst in enumerate(instances):
        for method, obs_count in active_configs:
            if (idx, method) in completed_set:
                print(f"[{idx+1}/{len(instances)}] {method} already completed, skipping.")
                continue

            print(
                f"[{idx+1}/{len(instances)}] Running {method} (obs={obs_count})...",
                end="",
                flush=True,
            )
            rec = evaluate_control(
                inst,
                idx,
                method,
                obs_count,
                time_limit=time_limit,
                watchdog_seconds=watchdog_seconds,
                script_path=script_path,
            )
            records.append(rec)
            completed_set.add((idx, method))

            print(
                f" -> {rec['outcome']} (solve={rec['solve_wall_seconds']}s, "
                f"online={rec['online_wall_seconds']:.2f}s, "
                f"peak_rss={rec['peak_rss_gib']}GiB, "
                f"verified={rec['verified_recovery_within_cutoff']})",
                flush=True,
            )

            base_artifact["records"] = records
            _atomic_write(output_path, base_artifact)
            _write_checksum_sidecar(output_path)

    base_artifact["status"] = "COMPLETE"
    base_artifact["finished_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    base_artifact["records"] = records
    base_artifact["summary"] = compute_summary(records)
    _atomic_write(output_path, base_artifact)
    _write_checksum_sidecar(output_path)

    print(f"\nBenchmark finished successfully. Artifact saved to: {output_path}")
    print(f"SHA-256: {sha256_file(output_path)}")
    return base_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to output JSON artifact")
    parser.add_argument("--resume", action="store_true", help="Resume from existing artifact if present")
    parser.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT_SECONDS, help="Solver time limit in seconds (default: 300.0)")
    parser.add_argument("--watchdog-seconds", type=float, default=DEFAULT_WATCHDOG_SECONDS, help="Parent watchdog timeout in seconds (default: 300.0)")
    parser.add_argument("--max-instances", type=int, default=None, help="Limit number of demo instances to evaluate (default: all 8)")
    parser.add_argument("--methods", nargs="+", choices=["sat_only_n4", "sat_only_n128"], default=None, help="Subset of methods to run (default: both)")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        payload = json.load(sys.stdin)
        result = worker_solve(payload)
        print(json.dumps(result))
    else:
        run_benchmark(
            args.output.resolve(),
            resume=args.resume,
            time_limit=args.time_limit,
            watchdog_seconds=args.watchdog_seconds,
            max_instances=args.max_instances,
            methods=args.methods,
        )


if __name__ == "__main__":
    main()
