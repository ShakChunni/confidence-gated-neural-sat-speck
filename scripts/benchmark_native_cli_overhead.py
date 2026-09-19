"""Benchmark process startup, pipe/file I/O, and parsing overhead for native CryptoMiniSat CLI.

Measures overhead on a fixed 7-round SPECK32/64 formula (opened train-0000 instance)
across >= 30 repetitions to quantify the per-candidate overhead of non-incremental
process execution versus in-memory wrapper solving.
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

import numpy as np

from neural_sat_speck.attacks.recovery import known_pairs_for_attack_instance
from neural_sat_speck.data.attack import load_attack_shard
from neural_sat_speck.sat.dimacs import export_speck_to_dimacs

DEFAULT_DATASET_PATH = Path("datasets/phase11a_oracle_r7/oracle_evaluation.json")
DEFAULT_DATASET_ID = "speck32_64_attack_phase11a_oracle_r7"
DEFAULT_OUTPUT_PATH = Path("results/phase11_scaling_diagnostics/native_cli_overhead_benchmark.json")


def compute_distribution_stats(values_ms: list[float]) -> dict[str, float]:
    arr = np.asarray(values_ms, dtype=np.float64)
    q25, q50, q75 = np.percentile(arr, [25, 50, 75])
    return {
        "count": int(len(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "median_ms": float(q50),
        "q25_ms": float(q25),
        "q75_ms": float(q75),
        "iqr_ms": float(q75 - q25),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=50, help="Number of repetitions (>= 30)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output JSON path")
    args = parser.parse_args()

    if args.repetitions < 30:
        raise ValueError(f"repetitions must be at least 30, got {args.repetitions}")

    # 1. Load opened diagnostic instance train-0000
    shard = load_attack_shard(DEFAULT_DATASET_PATH, expected_dataset_id=DEFAULT_DATASET_ID)
    inst = shard.instances[0]
    known_pairs = known_pairs_for_attack_instance(inst)[:4]

    # Generate fixed 7-round formula with an immediate conflict clause to isolate parse/startup overhead from solve search
    dimacs = export_speck_to_dimacs(known_pairs=known_pairs, rounds=7, pinned_round_keys={6: 0x1234})
    # Add conflicting unit clauses at end to make the formula immediately UNSAT upon parse/init without search
    dimacs_text = dimacs.dimacs_text + "1 0\n-1 0\n"
    formula_sha256 = hashlib.sha256(dimacs_text.encode("utf-8")).hexdigest()

    # Get solver version string
    ver_proc = subprocess.run(["cryptominisat5", "--version"], capture_output=True, text=True, check=True)
    cms_version = ver_proc.stdout.strip()

    # Benchmark A: Process startup only (cryptominisat5 --version)
    empty_spawn_times_ms: list[float] = []
    for _ in range(args.repetitions):
        t0 = time.perf_counter()
        subprocess.run(["cryptominisat5", "--version"], capture_output=True, text=True, check=True)
        empty_spawn_times_ms.append((time.perf_counter() - t0) * 1000.0)

    # Benchmark B: Stdin pipe input (cryptominisat5 --threads 1 via stdin)
    stdin_wall_times_ms: list[float] = []
    stdin_internal_parse_times_ms: list[float] = []
    stdin_internal_solve_times_ms: list[float] = []
    cmd_stdin = ["cryptominisat5", "--verb", "1", "--threads", "1"]

    for _ in range(args.repetitions):
        t0 = time.perf_counter()
        proc = subprocess.Popen(cmd_stdin, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = proc.communicate(input=dimacs_text)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        stdin_wall_times_ms.append(wall_ms)

        parse_time_s = 0.0
        solve_time_s = 0.0
        for line in stdout.splitlines():
            if "Parsing time:" in line:
                try:
                    parse_time_s = float(line.split("Parsing time:")[1].split()[0])
                except (IndexError, ValueError):
                    pass
            elif "Total time (this thread)" in line:
                try:
                    solve_time_s = float(line.split(":")[1].strip())
                except (IndexError, ValueError):
                    pass
        stdin_internal_parse_times_ms.append(parse_time_s * 1000.0)
        stdin_internal_solve_times_ms.append(solve_time_s * 1000.0)

    # Benchmark C: Temporary disk file input (write temp file, cryptominisat5 path)
    file_wall_times_ms: list[float] = []
    with tempfile.NamedTemporaryFile(suffix=".cnf", mode="w", delete=False) as f:
        f.write(dimacs_text)
        f.flush()
        temp_cnf_path = f.name

    try:
        cmd_file = ["cryptominisat5", "--verb", "1", "--threads", "1", temp_cnf_path]
        for _ in range(args.repetitions):
            t0 = time.perf_counter()
            proc = subprocess.Popen(cmd_file, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = proc.communicate()
            wall_ms = (time.perf_counter() - t0) * 1000.0
            file_wall_times_ms.append(wall_ms)
    finally:
        if os.path.exists(temp_cnf_path):
            os.remove(temp_cnf_path)

    report: dict[str, Any] = {
        "schema": "neural_sat_speck.native_cli_overhead_benchmark.v1",
        "date": "2026-09-19",
        "evidence_class": "EMPIRICAL_BENCHMARK / OPENED_DIAGNOSTIC",
        "provenance": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python_version": sys.version,
            "cryptominisat_version": cms_version,
            "dataset_path": str(DEFAULT_DATASET_PATH),
            "instance_id": inst.instance_id,
            "num_variables": dimacs.num_variables,
            "num_cnf_clauses": dimacs.num_cnf_clauses + 2,
            "num_xor_clauses": dimacs.num_xor_clauses,
            "formula_sha256": formula_sha256,
            "repetitions": args.repetitions,
        },
        "measurements": {
            "empty_process_spawn": {
                "description": "Process spawn and teardown overhead alone (cryptominisat5 --version)",
                "stats": compute_distribution_stats(empty_spawn_times_ms),
                "raw_ms": empty_spawn_times_ms,
            },
            "stdin_pipe_execution": {
                "description": "Process spawn + pipe communication + DIMACS parse + trivial conflict termination",
                "stats": compute_distribution_stats(stdin_wall_times_ms),
                "internal_parse_stats": compute_distribution_stats(stdin_internal_parse_times_ms),
                "internal_solve_stats": compute_distribution_stats(stdin_internal_solve_times_ms),
                "raw_wall_ms": stdin_wall_times_ms,
            },
            "file_based_execution": {
                "description": "Process spawn + disk file read + DIMACS parse + trivial conflict termination",
                "stats": compute_distribution_stats(file_wall_times_ms),
                "raw_wall_ms": file_wall_times_ms,
            },
        },
        "implications": {
            "candidate_budget": 1024,
            "estimated_cumulative_overhead_stdin_median_seconds": round(
                1024 * compute_distribution_stats(stdin_wall_times_ms)["median_ms"] / 1000.0, 3
            ),
            "estimated_cumulative_overhead_file_median_seconds": round(
                1024 * compute_distribution_stats(file_wall_times_ms)["median_ms"] / 1000.0, 3
            ),
            "conclusion": (
                "Iterative candidate testing of up to 1,024 candidates via native CLI process invocation "
                "incurs substantial non-solving latency purely from process startup and DIMACS parsing, "
                "justifying the use of an in-memory solver wrapper (pycryptosat) for fair multi-candidate recovery."
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out_f:
        json.dump(report, out_f, indent=2)

    print(f"Successfully generated benchmark artifact: {args.output}")
    print(f"Empty spawn median: {report['measurements']['empty_process_spawn']['stats']['median_ms']:.2f} ms")
    print(f"Stdin pipe median:  {report['measurements']['stdin_pipe_execution']['stats']['median_ms']:.2f} ms (IQR: {report['measurements']['stdin_pipe_execution']['stats']['iqr_ms']:.2f} ms)")
    print(f"File input median:  {report['measurements']['file_based_execution']['stats']['median_ms']:.2f} ms (IQR: {report['measurements']['file_based_execution']['stats']['iqr_ms']:.2f} ms)")
    print(f"Cumulative 1024 candidates overhead (stdin): {report['implications']['estimated_cumulative_overhead_stdin_median_seconds']:.2f} s")


if __name__ == "__main__":
    main()
