#!/usr/bin/env python3
"""Disentangle the CryptoMiniSat native XOR Gaussian elimination solver confound.

Investigates whether the 3/3 verified seven-round recoveries under the binary CLI
are causally driven by:
1. Native XOR Gaussian elimination (--maxmatrixcols 2500 --autodisablegauss 0);
2. Multithreaded portfolio search (--threads 4); or
3. An interaction between both.

Evaluates a clean 2x2 factorial matrix on opened Seed 300001 instances 0, 1, and 2
with true k6 pinned and 4 known pairs (exact Phase 11A condition).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from neural_sat_speck.attacks.recovery import (  # noqa: E402
    independent_verification_pairs,
    known_pairs_for_attack_instance,
    verify_master_key,
)
from neural_sat_speck.data.attack import load_attack_shard  # noqa: E402
from neural_sat_speck.sat.dimacs import (  # noqa: E402
    decode_master_key_from_dimacs_solution,
    export_speck_to_dimacs,
)
from neural_sat_speck.speck.key_schedule import expand_key  # noqa: E402

DATASET_PATH = (
    REPOSITORY_ROOT / "datasets" / "phase11a_oracle_r7" / "oracle_evaluation.json"
)
DATASET_ID = "speck32_64_attack_phase11a_oracle_r7"
RESULTS_DIR = REPOSITORY_ROOT / "results" / "phase11_scaling_diagnostics"

CONFIGURATIONS = {
    "T1_GaussOff": {
        "description": "Single thread, Gauss disabled (default baseline)",
        "threads": 1,
        "maxmatrixcols": 1000,
        "autodisablegauss": 1,
    },
    "T1_GaussOn": {
        "description": "Single thread, Gauss enabled (isolated Gauss effect)",
        "threads": 1,
        "maxmatrixcols": 2500,
        "autodisablegauss": 0,
    },
    "T4_GaussOff": {
        "description": "Four threads, Gauss disabled (isolated threads effect)",
        "threads": 4,
        "maxmatrixcols": 1000,
        "autodisablegauss": 1,
    },
    "T4_GaussOn": {
        "description": "Four threads, Gauss enabled (full combined configuration)",
        "threads": 4,
        "maxmatrixcols": 2500,
        "autodisablegauss": 0,
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=int, default=3, help="Number of instances to evaluate (default 3)")
    parser.add_argument("--start", type=int, default=0, help="Starting instance index (default 0)")
    parser.add_argument("--maxtime", type=int, default=120, help="Per-run timeout in wall seconds (default 120)")
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "solver_confound_ablation.json",
        help="Path to output JSON result file",
    )
    return parser.parse_args(argv)


def run_solver_run(
    dimacs_text: str,
    master_key_bits: Any,
    known_pairs: list[Any],
    verification_pairs: list[Any],
    threads: int,
    maxmatrixcols: int,
    autodisablegauss: int,
    maxtime: int,
) -> dict[str, Any]:
    # cryptominisat5 --maxtime is CPU-seconds; scale by thread count for constant wall-clock limit
    cpu_limit = maxtime * threads
    cmd = [
        "cryptominisat5",
        "--verb", "1",
        "--threads", str(threads),
        "--maxmatrixcols", str(maxmatrixcols),
        "--autodisablegauss", str(autodisablegauss),
        "--maxtime", str(cpu_limit),
        "--zero-exit-status",
    ]

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        input=dimacs_text,
        text=True,
        capture_output=True,
    )
    wall_seconds = time.perf_counter() - t0
    stdout = proc.stdout

    # Determine status
    if "s SATISFIABLE" in stdout:
        status = "sat"
    elif "s UNSATISFIABLE" in stdout:
        status = "unsat"
    else:
        status = "timeout"

    # Check whether Gauss matrix was used or disabled
    # e.g. "Using 1 matrices" or "UNused matrix ... -> set usage to NO"
    matrix_used = False
    match_used = re.search(r"Using\s+([0-9]+)\s+matrices", stdout)
    if match_used and int(match_used.group(1)) > 0:
        matrix_used = True

    # Parse reported CPU time from solver if available
    # e.g. "Total time (CPU/wall): 45.21 / 12.34"
    reported_cpu_seconds = None
    match_time = re.search(r"Total time \(CPU/wall\):\s*([0-9.]+)\s*/", stdout)
    if match_time:
        reported_cpu_seconds = float(match_time.group(1))

    verified_recovery = False
    recovered_key = None

    if status == "sat":
        decoded = decode_master_key_from_dimacs_solution(stdout, master_key_bits)
        if decoded is not None:
            recovered_key = list(decoded)
            verif = verify_master_key(
                decoded,
                known_pairs=known_pairs,
                verification_pairs=verification_pairs,
                rounds=7,
            )
            verified_recovery = verif.verified

    return {
        "status": status,
        "wall_seconds": round(wall_seconds, 3),
        "reported_cpu_seconds": reported_cpu_seconds,
        "matrix_used": matrix_used,
        "verified_recovery": verified_recovery,
        "recovered_master_key": recovered_key,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    shard = load_attack_shard(DATASET_PATH, expected_dataset_id=DATASET_ID)
    instances = shard.instances[args.start : args.start + args.instances]
    if not instances:
        raise SystemExit("No instances found for evaluation")

    print(
        f"=== DISENTANGLING SOLVER CONFOUND (2x2 Factorial Matrix) ===\n"
        f"Instances: {len(instances)} (indices {args.start}..{args.start + len(instances) - 1})\n"
        f"Timeout: {args.maxtime}s per configuration\n"
        f"Configurations: {list(CONFIGURATIONS.keys())}\n"
    )

    results_by_instance: list[dict[str, Any]] = []

    for idx, inst in enumerate(instances):
        inst_idx = args.start + idx
        true_k6 = expand_key(inst.master_key, rounds=7)[-1]
        all_known = known_pairs_for_attack_instance(inst)
        known_4_pairs = list(all_known[:4])
        verif_pairs = list(
            independent_verification_pairs(
                inst.master_key,
                rounds=7,
                pair_count=2,
                seed=999999,
            )
        )

        print(f"[{inst_idx:02d}] Building DIMACS for {inst.instance_id} (true k6={true_k6:#06x})...")
        model = export_speck_to_dimacs(
            known_4_pairs,
            rounds=7,
            pinned_round_keys={6: true_k6},
        )
        print(f"     Model: {model.num_variables} vars, {model.num_cnf_clauses} CNF, {model.num_xor_clauses} XOR")

        inst_record: dict[str, Any] = {
            "instance_index": inst_idx,
            "instance_id": inst.instance_id,
            "true_master_key": list(inst.master_key),
            "true_k6": true_k6,
            "runs": {},
        }

        for cfg_name, cfg in CONFIGURATIONS.items():
            print(f"     Running {cfg_name} ({cfg['description']})...", end="", flush=True)
            run_res = run_solver_run(
                dimacs_text=model.dimacs_text,
                master_key_bits=model.master_key_bits,
                known_pairs=known_4_pairs,
                verification_pairs=verif_pairs,
                threads=cfg["threads"],
                maxmatrixcols=cfg["maxmatrixcols"],
                autodisablegauss=cfg["autodisablegauss"],
                maxtime=args.maxtime,
            )
            inst_record["runs"][cfg_name] = run_res
            print(
                f" -> {run_res['status'].upper()} in {run_res['wall_seconds']}s wall "
                f"(CPU: {run_res['reported_cpu_seconds']}s, Gauss: {run_res['matrix_used']}, "
                f"Verified: {run_res['verified_recovery']})"
            )

        results_by_instance.append(inst_record)

    # Summary table across configurations
    summary: dict[str, Any] = {}
    for cfg_name in CONFIGURATIONS:
        solved_count = sum(
            1 for rec in results_by_instance if rec["runs"][cfg_name]["verified_recovery"]
        )
        times = [
            rec["runs"][cfg_name]["wall_seconds"]
            for rec in results_by_instance
            if rec["runs"][cfg_name]["status"] == "sat"
        ]
        summary[cfg_name] = {
            "verified_recoveries": f"{solved_count}/{len(instances)}",
            "solve_rate": round(solved_count / len(instances), 4),
            "solved_wall_times": times,
        }

    output_payload = {
        "experiment": "disentangle_solver_confound_2x2",
        "timestamp": datetime.now(UTC).isoformat(),
        "dataset_path": str(DATASET_PATH),
        "instances_evaluated": len(instances),
        "maxtime": args.maxtime,
        "configurations": CONFIGURATIONS,
        "summary": summary,
        "instance_records": results_by_instance,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\n[✓] Results successfully written to {args.output}")
    print("\nSummary:")
    for cfg_name, s in summary.items():
        print(f"  {cfg_name:<12}: {s['verified_recoveries']} verified solves | Times: {s['solved_wall_times']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
