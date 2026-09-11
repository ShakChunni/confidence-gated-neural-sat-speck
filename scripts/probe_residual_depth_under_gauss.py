#!/usr/bin/env python3
"""Probe the Residual-Depth Law under CryptoMiniSat Gaussian Elimination.

Investigates whether enabling native XOR Gaussian elimination shifts the
exact-solver solvability frontier from residual depth <= 5 to residual depth <= 6.

Evaluates test points:
- Residual depth 6:
  * 6 rounds, 0 keys assumed (unguided SAT-only)
  * 7 rounds, 1 key assumed (k6) [Phase 11A setup]
  * 8 rounds, 2 keys assumed (k6, k7)
- Residual depth 7:
  * 7 rounds, 0 keys assumed (unguided SAT-only)
  * 8 rounds, 1 key assumed (k7)

Evaluates on opened Seed 300001 instance 0 (master key 0x971a, 0xc1de, 0x87ed, 0x422c).
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

PROBE_POINTS = [
    {"rounds": 6, "trailing_keys": 0, "residual_depth": 6, "label": "6R_0keys_res6"},
    {"rounds": 7, "trailing_keys": 1, "residual_depth": 6, "label": "7R_1key_res6"},
    {"rounds": 8, "trailing_keys": 2, "residual_depth": 6, "label": "8R_2keys_res6"},
    {"rounds": 7, "trailing_keys": 0, "residual_depth": 7, "label": "7R_0keys_res7"},
    {"rounds": 8, "trailing_keys": 1, "residual_depth": 7, "label": "8R_1key_res7"},
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-idx", type=int, default=0, help="Instance index to probe (default 0)")
    parser.add_argument("--maxtime", type=int, default=120, help="Wall timeout in seconds (default 120)")
    parser.add_argument("--threads", type=int, default=1, help="Thread count (default 1)")
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "residual_depth_under_gauss_probe.json",
        help="Path to output JSON file",
    )
    return parser.parse_args(argv)


def run_probe(
    known_pairs: list[Any],
    verification_pairs: list[Any],
    rounds: int,
    pinned_keys: dict[int, int],
    threads: int,
    maxtime_wall: int,
    master_key: tuple[int, int, int, int],
) -> dict[str, Any]:
    model = export_speck_to_dimacs(
        known_pairs,
        rounds=rounds,
        pinned_round_keys=pinned_keys,
    )

    cpu_limit = maxtime_wall * threads
    cmd = [
        "cryptominisat5",
        "--verb", "1",
        "--threads", str(threads),
        "--maxmatrixcols", "2500",
        "--autodisablegauss", "0",
        "--maxtime", str(cpu_limit),
        "--zero-exit-status",
    ]

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        input=model.dimacs_text,
        text=True,
        capture_output=True,
    )
    wall_seconds = time.perf_counter() - t0
    stdout = proc.stdout

    if "s SATISFIABLE" in stdout:
        status = "sat"
    elif "s UNSATISFIABLE" in stdout:
        status = "unsat"
    else:
        status = "timeout"

    matrix_used = False
    match_used = re.search(r"Using\s+([0-9]+)\s+matrices", stdout)
    if match_used and int(match_used.group(1)) > 0:
        matrix_used = True

    verified = False
    recovered_key = None
    if status == "sat":
        decoded = decode_master_key_from_dimacs_solution(stdout, model.master_key_bits)
        if decoded is not None:
            recovered_key = list(decoded)
            verif = verify_master_key(
                decoded,
                known_pairs=known_pairs,
                verification_pairs=verification_pairs,
                rounds=rounds,
            )
            verified = verif.verified

    return {
        "status": status,
        "wall_seconds": round(wall_seconds, 3),
        "matrix_used": matrix_used,
        "verified": verified,
        "recovered_key": recovered_key,
        "model_variables": model.num_variables,
        "model_cnf": model.num_cnf_clauses,
        "model_xor": model.num_xor_clauses,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    shard = load_attack_shard(DATASET_PATH, expected_dataset_id=DATASET_ID)
    inst = shard.instances[args.instance_idx]

    print(
        f"=== PROBING RESIDUAL-DEPTH LAW UNDER GAUSS (Instance {args.instance_idx}) ===\n"
        f"Instance ID: {inst.instance_id}, Master Key: {[hex(k) for k in inst.master_key]}\n"
        f"Threads: {args.threads}, Wall Timeout: {args.maxtime}s\n"
        f"Gauss: ENABLED (--maxmatrixcols 2500 --autodisablegauss 0)\n"
    )

    probe_records: list[dict[str, Any]] = []

    for pt in PROBE_POINTS:
        r = pt["rounds"]
        num_keys = pt["trailing_keys"]
        res_depth = pt["residual_depth"]
        label = pt["label"]

        all_round_keys = expand_key(inst.master_key, rounds=r)
        pinned = {}
        if num_keys > 0:
            for k_idx in range(r - num_keys, r):
                pinned[k_idx] = all_round_keys[k_idx]

        # Generate fresh observations for the specific round count r
        from neural_sat_speck.speck.cipher import encrypt_block

        known_pairs = [
            (p.plaintext0, encrypt_block(p.plaintext0, inst.master_key, rounds=r))
            for p in inst.pairs[:4]
        ]
        verif_pairs = list(
            independent_verification_pairs(
                inst.master_key,
                rounds=r,
                pair_count=2,
                seed=999999,
            )
        )

        print(
            f"Probing {label:<16}: Rounds={r}, Pinned={num_keys} keys {list(pinned.keys())}, "
            f"Residual={res_depth}... ",
            end="",
            flush=True,
        )

        res = run_probe(
            known_pairs=known_pairs,
            verification_pairs=verif_pairs,
            rounds=r,
            pinned_keys=pinned,
            threads=args.threads,
            maxtime_wall=args.maxtime,
            master_key=inst.master_key,
        )
        rec = {**pt, **res}
        probe_records.append(rec)

        print(
            f"-> {res['status'].upper()} in {res['wall_seconds']}s wall "
            f"(Gauss: {res['matrix_used']}, Verified: {res['verified']})"
        )

    summary_payload = {
        "experiment": "residual_depth_under_gauss_probe",
        "timestamp": datetime.now(UTC).isoformat(),
        "instance_index": args.instance_idx,
        "instance_id": inst.instance_id,
        "threads": args.threads,
        "maxtime_wall": args.maxtime,
        "records": probe_records,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print(f"\n[✓] Results saved to {args.output}")
    print("\nSummary Table:")
    print(f"{'Label':<18} | {'Rounds':<6} | {'Keys':<4} | {'Residual':<8} | {'Status':<8} | {'Time':<8} | {'Verified'}")
    print("-" * 75)
    for rec in probe_records:
        print(
            f"{rec['label']:<18} | {rec['rounds']:<6} | {rec['trailing_keys']:<4} | "
            f"{rec['residual_depth']:<8} | {rec['status']:<8} | {rec['wall_seconds']:<6.2f}s | {rec['verified']}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
