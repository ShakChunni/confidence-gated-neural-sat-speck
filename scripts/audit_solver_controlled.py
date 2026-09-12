#!/usr/bin/env python3
"""Package P2: Controlled Solver Backend & Observable Configuration Audit.

Executes a controlled 3-way ablation holding instance, true subkey assumptions,
formula, machine, and thread count (1 core) fixed across opened Seed 300001
instances (0, 1, 2):

Treatments compared (observable configurations):
1. 1_pycryptosat_default_policy:
   Python pycryptosat wrapper under default parameters (threads=1, XOR Gauss unengaged
   by default, CPU-second time limit). Adjacent execution-path diagnostic.
2. 2_native_default_matrix_limit:
   Native cryptominisat5 CLI with default matrix column limit (--maxmatrixcols 1000
   --autodisablegauss 1 --threads 1 --random 0 --verb 1). Consumes exported DIMACS.
3. 3_native_raised_matrix_limit_autodisable_off:
   Native cryptominisat5 CLI with raised matrix column limit (--maxmatrixcols 2500
   --autodisablegauss 0 --threads 1 --random 0 --verb 1). Consumes exported DIMACS.

Formula Structure:
- Unpinned 7-round 4-pair base model: 2,368 variables, 3,350 CNF clauses, 1,632 XOR clauses.
- Pinned true k6 model: 2,368 variables, 3,366 CNF clauses (3,350 base + 16 unit clauses), 1,632 XOR clauses.
- Formulas differ across instances because plaintexts, ciphertexts, and true subkeys differ.
  Controls are paired within each instance, not globally identical across instances.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import pycryptosat

from neural_sat_speck.attacks.recovery import (
    independent_verification_pairs,
    known_pairs_for_attack_instance,
    verify_master_key,
)
from neural_sat_speck.data.attack import load_attack_shard
from neural_sat_speck.sat.dimacs import (
    decode_master_key_from_dimacs_solution,
    export_speck_to_dimacs,
)
from neural_sat_speck.sat.encoding import WordBits
from neural_sat_speck.sat.model import SpeckSatModel

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = (
    REPOSITORY_ROOT / "datasets" / "phase11a_oracle_r7" / "oracle_evaluation.json"
)
DATASET_ID = "speck32_64_attack_phase11a_oracle_r7"
OUTPUT_PATH = (
    REPOSITORY_ROOT
    / "results"
    / "phase11_scaling_diagnostics"
    / "solver_controlled_audit_v3.json"
)

DEPENDENCY_PATHS = [
    REPOSITORY_ROOT / "src" / "neural_sat_speck" / "sat" / "model.py",
    REPOSITORY_ROOT / "src" / "neural_sat_speck" / "sat" / "dimacs.py",
    REPOSITORY_ROOT / "src" / "neural_sat_speck" / "sat" / "encoding.py",
    REPOSITORY_ROOT / "src" / "neural_sat_speck" / "attacks" / "recovery.py",
    REPOSITORY_ROOT / "src" / "neural_sat_speck" / "data" / "attack.py",
]

TREATMENTS = {
    "1_pycryptosat_default_policy": {
        "backend": "pycryptosat",
        "threads": 1,
        "description": "Python pycryptosat wrapper (threads=1, default policy, adjacent execution-path diagnostic)",
    },
    "2_native_default_matrix_limit": {
        "backend": "cryptominisat5_binary",
        "threads": 1,
        "maxmatrixcols": 1000,
        "autodisablegauss": 1,
        "description": "Native cryptominisat5 CLI (threads=1, default maxmatrixcols 1000, autodisablegauss 1)",
    },
    "3_native_raised_matrix_limit_autodisable_off": {
        "backend": "cryptominisat5_binary",
        "threads": 1,
        "maxmatrixcols": 2500,
        "autodisablegauss": 0,
        "description": "Native cryptominisat5 CLI (threads=1, raised maxmatrixcols 2500, autodisablegauss 0)",
    },
}


def file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_state() -> dict[str, Any]:
    """Capture repository commit, dirty status, and diff digest."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=REPOSITORY_ROOT
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, cwd=REPOSITORY_ROOT
        ).strip()
        is_dirty = len(status) > 0
        diff_sha256 = (
            hashlib.sha256(
                subprocess.check_output(
                    ["git", "diff", "HEAD"], text=True, cwd=REPOSITORY_ROOT
                ).encode("utf-8")
            ).hexdigest()
            if is_dirty
            else None
        )
        return {
            "commit": commit,
            "is_dirty": is_dirty,
            "diff_sha256": diff_sha256,
        }
    except Exception as e:
        return {"commit": None, "is_dirty": None, "error": str(e)}


def get_cms_version() -> str:
    """Retrieve CryptoMiniSat CLI version string."""
    try:
        proc = subprocess.run(
            ["cryptominisat5", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in proc.stdout.splitlines():
            if "CryptoMiniSat version" in line:
                return line.strip()
        return proc.stdout.strip()
    except Exception as e:
        return f"unknown ({e})"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instances",
        type=int,
        default=3,
        help="Number of instances to evaluate (default 3)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Wall-clock timeout per run in seconds (default 60.0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Output JSON path (default: results/phase11_scaling_diagnostics/solver_controlled_audit_v3.json)",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow overwriting existing output artifact",
    )
    return parser.parse_args(argv)


def run_pycryptosat_wrapper(
    known_pairs: Any,
    verification_pairs: Any,
    master_key: tuple[int, int, int, int],
    true_k6: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Run exact solver using pycryptosat wrapper under strict isolation."""
    model = SpeckSatModel.from_known_pairs(known_pairs, rounds=7)

    t0 = time.perf_counter()
    res = model.solve_final_round_subkey(true_k6, time_limit=timeout_seconds)
    elapsed = time.perf_counter() - t0

    verified = False
    recovered_key = None
    if res.satisfiable and res.solution is not None:
        recovered_key = model.master_key_from_solution(res.solution)
        verif_res = verify_master_key(
            recovered_key, known_pairs, verification_pairs, rounds=7
        )
        verified = verif_res.verified

    if res.timed_out:
        exit_classification = "solver_limit_termination"
    elif res.satisfiable:
        exit_classification = "sat_and_verified" if verified else "verification_failure"
    elif res.status == "unsat":
        exit_classification = "unsat"
    else:
        exit_classification = "solver_limit_termination" if elapsed >= timeout_seconds else "indeterminate"

    return {
        "status": res.status,
        "exit_classification": exit_classification,
        "is_sat": res.satisfiable,
        "verified": verified,
        "recovered_key": list(recovered_key) if recovered_key else None,
        "key_match": recovered_key == master_key if recovered_key else False,
        "wall_seconds_raw": elapsed,
        "wall_seconds": round(elapsed, 3),
        "command_or_api": "SpeckSatModel.solve_final_round_subkey (pycryptosat.Solver)",
        "returncode": None,
        "stdout": None,
        "stderr": None,
        "solver_diagnostics": {
            "solver_backend": "pycryptosat",
            "threads": 1,
            "matrix_use_logged": False,
            "note": "pycryptosat wrapper does not expose native XOR Gaussian matrix configuration.",
        },
    }


def parse_cryptominisat_log(stdout: str) -> dict[str, Any]:
    """Parse CryptoMiniSat verbose log for matrix construction and usage evidence."""
    matrix_detected = False
    matrix_dimensions = None
    matrix_count_used = 0
    unused_reason = None
    autodisable_observed = False
    solver_version = None

    for line in stdout.splitlines():
        line_s = line.strip()
        if "CryptoMiniSat version" in line_s:
            solver_version = line_s
        if "Good   matrix" in line_s or "Good matrix" in line_s:
            matrix_detected = True
        if "UNused matrix" in line_s:
            matrix_detected = True
            parts = line_s.split()
            if "x" in parts:
                idx = parts.index("x")
                matrix_dimensions = f"{parts[idx - 1]} x {parts[idx + 1]}"
        if "Too many columns in matrix" in line_s:
            unused_reason = line_s
        if "density:" in line_s and matrix_dimensions is None:
            parts = line_s.split()
            if "x" in parts:
                idx = parts.index("x")
                matrix_dimensions = f"{parts[idx - 1]} x {parts[idx + 1]}"
        if "Using 1 matrices" in line_s:
            matrix_count_used = 1
        elif "Using 0 matrices" in line_s:
            matrix_count_used = 0
        if "autodisable" in line_s.lower() or "auto-disable" in line_s.lower():
            autodisable_observed = True

    return {
        "solver_version": solver_version,
        "matrix_detected": matrix_detected,
        "matrix_dimensions": matrix_dimensions,
        "matrix_used": matrix_count_used > 0,
        "matrix_count_used": matrix_count_used,
        "unused_reason": unused_reason,
        "autodisable_observed": autodisable_observed,
        "random_seed": 0,
    }


def run_native_binary(
    dimacs_text: str,
    master_key_bits: tuple[WordBits, ...],
    known_pairs: Any,
    verification_pairs: Any,
    master_key: tuple[int, int, int, int],
    maxmatrixcols: int,
    autodisablegauss: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Run exact solver using native cryptominisat5 executable with verbose logging."""
    cmd = [
        "cryptominisat5",
        "--verb", "1",
        "--random", "0",
        "--threads", "1",
        "--maxmatrixcols", str(maxmatrixcols),
        "--autodisablegauss", str(autodisablegauss),
        "--maxtime", str(int(timeout_seconds)),
    ]

    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    timed_out = False
    try:
        stdout, stderr = proc.communicate(input=dimacs_text, timeout=timeout_seconds + 5.0)
        elapsed = time.perf_counter() - t0
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        elapsed = time.perf_counter() - t0
        timed_out = True

    is_sat = "s SATISFIABLE" in stdout
    is_unsat = "s UNSATISFIABLE" in stdout
    is_unknown = "s UNKNOWN" in stdout
    verified = False
    recovered_key = None

    if timed_out:
        exit_classification = "parent_process_timeout"
        status = "timeout"
    elif proc.returncode == 15:
        exit_classification = "solver_limit_termination"
        status = "timeout"
    elif is_sat:
        decoded = decode_master_key_from_dimacs_solution(stdout, master_key_bits)
        if decoded is not None:
            recovered_key = decoded
            verif_res = verify_master_key(
                recovered_key, known_pairs, verification_pairs, rounds=7
            )
            verified = verif_res.verified
            if verified:
                exit_classification = "sat_and_verified"
                status = "sat_and_verified"
            else:
                exit_classification = "verification_failure"
                status = "sat_verification_failed"
        else:
            exit_classification = "parse_failure"
            status = "parse_error"
    elif is_unsat or proc.returncode == 20:
        exit_classification = "unsat"
        status = "unsat"
    elif is_unknown or proc.returncode == 0:
        exit_classification = "indeterminate"
        status = "indeterminate"
    elif proc.returncode not in (10, 20, 15, 0):
        exit_classification = "process_failure"
        status = "process_error"
    else:
        exit_classification = "indeterminate"
        status = "indeterminate"

    solver_diag = parse_cryptominisat_log(stdout)

    return {
        "status": status,
        "exit_classification": exit_classification,
        "is_sat": is_sat,
        "verified": verified,
        "recovered_key": list(recovered_key) if recovered_key else None,
        "key_match": recovered_key == master_key if recovered_key else False,
        "wall_seconds_raw": elapsed,
        "wall_seconds": round(elapsed, 3),
        "command_or_api": " ".join(cmd),
        "returncode": proc.returncode,
        "solver_diagnostics": solver_diag,
        "stdout": stdout,
        "stderr": stderr,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = args.output.resolve()

    if output_path.exists() and not args.allow_overwrite:
        print(f"[!] Refusing to overwrite existing file: {output_path}")
        print("    Pass --allow-overwrite if intentional.")
        return 1

    print("=== CONTROLLED SOLVER & CONFIGURATION AUDIT (PACKAGE P2) ===")
    print(f"Instances: {args.instances} | Timeout: {args.timeout}s per run | Threads: 1")
    print(f"Output: {output_path}\n")

    shard = load_attack_shard(DATASET_PATH, expected_dataset_id=DATASET_ID)
    results_by_treatment: dict[str, list[dict[str, Any]]] = {t: [] for t in TREATMENTS}
    formula_metadata: dict[str, dict[str, Any]] = {}

    for idx in range(min(args.instances, len(shard.instances))):
        inst = shard.instances[idx]
        all_known = known_pairs_for_attack_instance(inst)
        known_pairs = all_known[:4]
        verif_pairs = independent_verification_pairs(
            inst.master_key,
            rounds=7,
            pair_count=2,
            seed=999999,
        )
        true_k6 = inst.final_round_subkey

        print(f"--- Instance {idx} ({inst.instance_id}): True k6={true_k6:#06x} ---")

        # Export unpinned base and pinned DIMACS
        dimacs_unpinned = export_speck_to_dimacs(known_pairs=known_pairs, rounds=7)
        dimacs_pinned = export_speck_to_dimacs(
            known_pairs=known_pairs,
            rounds=7,
            pinned_round_keys={6: true_k6},
        )
        formula_sha256 = hashlib.sha256(dimacs_pinned.dimacs_text.encode("utf-8")).hexdigest()
        formula_metadata[inst.instance_id] = {
            "instance_index": idx,
            "true_k6": true_k6,
            "true_k6_hex": f"{true_k6:#06x}",
            "num_variables": dimacs_pinned.num_variables,
            "unpinned_base_cnf_clauses": dimacs_unpinned.num_cnf_clauses,
            "pinned_k6_unit_clauses": 16,
            "num_cnf_clauses": dimacs_pinned.num_cnf_clauses,
            "num_xor_clauses": dimacs_pinned.num_xor_clauses,
            "formula_sha256": formula_sha256,
        }

        for treat_id, treat_cfg in TREATMENTS.items():
            if treat_cfg["backend"] == "pycryptosat":
                res = run_pycryptosat_wrapper(
                    known_pairs=known_pairs,
                    verification_pairs=verif_pairs,
                    master_key=inst.master_key,
                    true_k6=true_k6,
                    timeout_seconds=args.timeout,
                )
            else:
                res = run_native_binary(
                    dimacs_text=dimacs_pinned.dimacs_text,
                    master_key_bits=dimacs_pinned.master_key_bits,
                    known_pairs=known_pairs,
                    verification_pairs=verif_pairs,
                    master_key=inst.master_key,
                    maxmatrixcols=treat_cfg["maxmatrixcols"],
                    autodisablegauss=treat_cfg["autodisablegauss"],
                    timeout_seconds=args.timeout,
                )

            entry = {
                "instance_index": idx,
                "instance_id": inst.instance_id,
                **res,
            }
            results_by_treatment[treat_id].append(entry)
            matrix_note = ""
            if "solver_diagnostics" in res and "matrix_used" in res["solver_diagnostics"]:
                diag = res["solver_diagnostics"]
                matrix_note = f" | Matrix: {'USED (' + str(diag.get('matrix_dimensions')) + ')' if diag.get('matrix_used') else 'NOT USED'}"
            print(
                f"  [{treat_id:<44}] Status: {res['exit_classification']:<25} | "
                f"Verified: {str(res['verified']):<5} | Time: {res['wall_seconds']:<6.3f}s{matrix_note}"
            )
        print()

    # Summary statistics with exact median calculation via statistics.median
    summary_by_treatment: dict[str, Any] = {}
    for treat_id, runs in results_by_treatment.items():
        solves = sum(r["verified"] for r in runs)
        timeouts = sum(
            r["exit_classification"] in ("solver_limit_termination", "parent_process_timeout")
            for r in runs
        )
        solved_times_raw = [r["wall_seconds_raw"] for r in runs if r["verified"]]
        all_times_raw = [r["wall_seconds_raw"] for r in runs]
        summary_by_treatment[treat_id] = {
            "description": TREATMENTS[treat_id]["description"],
            "total_evaluated": len(runs),
            "verified_solves": solves,
            "solve_rate": solves / len(runs),
            "timeouts": timeouts,
            "mean_wall_seconds": statistics.mean(all_times_raw),
            "median_wall_seconds": statistics.median(all_times_raw),
            "solved_mean_seconds": statistics.mean(solved_times_raw) if solved_times_raw else None,
            "solved_median_seconds": statistics.median(solved_times_raw) if solved_times_raw else None,
            "all_times_raw": all_times_raw,
            "solved_times_raw": solved_times_raw,
        }

    dep_hashes = {str(p.relative_to(REPOSITORY_ROOT)): file_sha256(p) for p in DEPENDENCY_PATHS if p.exists()}
    git_state = get_git_state()

    audit_document = {
        "schema": "neural_sat_speck.solver_controlled_audit.v3",
        "date": "2026-09-12",
        "evidence_class": "ENGINEERING / DIAGNOSTIC",
        "methodology_notes": [
            "Diagnostic evidence evaluated on 3 opened Seed 300001 instances.",
            "Unpinned 7-round base formula contains 3,350 CNF clauses; pinning the 16 bits of k6 appends 16 unit clauses, yielding 3,366 CNF clauses.",
            "Because plaintext/ciphertext pairs and true key values differ across instances, formula text and formula SHA-256 hashes differ per instance. Controls are paired within each instance, not globally identical.",
            "Native default versus raised-limit runs consume the exact same pinned DIMACS within each instance.",
            "The pycryptosat wrapper run uses a separately constructed API formula and assumption interface; therefore the wrapper comparison is an adjacent execution-path diagnostic, not an exact same-representation treatment.",
            "Timeout mechanisms differ (pycryptosat CPU-second interrupt inside C++ vs native CLI wall-clock --maxtime cap plus parent-process timeout); not a strict identical common wall-clock experiment.",
            "Timeouts at 60s wall-clock are right-censored; wrapper overhead cannot be measured or ruled out from timeout data alone.",
        ],
        "provenance": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "pycryptosat_version": getattr(pycryptosat, "__version__", "unknown"),
            "cryptominisat_cli_version": get_cms_version(),
            "dataset_path": str(DATASET_PATH),
            "dataset_sha256": file_sha256(DATASET_PATH),
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": file_sha256(Path(__file__).resolve()),
            "dependency_sha256": dep_hashes,
            "git_state": git_state,
        },
        "configuration": {
            "instances_evaluated": args.instances,
            "timeout_seconds": args.timeout,
            "threads": 1,
        },
        "formula_metadata": formula_metadata,
        "summary": summary_by_treatment,
        "runs": results_by_treatment,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(f".tmp.{os.getpid()}")
    try:
        temp_path.write_text(json.dumps(audit_document, indent=2), encoding="utf-8")
        temp_path.replace(output_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise

    out_hash = file_sha256(output_path)

    print(f"[✓] Solver audit complete! Results saved to {output_path}")
    print(f"[✓] SHA-256: {out_hash}\n")
    for t_id, s in summary_by_treatment.items():
        print(
            f"  {t_id}: {s['verified_solves']}/{s['total_evaluated']} solves "
            f"({s['solve_rate']*100:.1f}%), {s['timeouts']} timeouts, "
            f"solved median: {s['solved_median_seconds']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
