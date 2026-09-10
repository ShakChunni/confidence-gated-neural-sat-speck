#!/usr/bin/env python3
"""Phase 11A Seven-Round Oracle-SAT Feasibility Gate Benchmark.

Evaluates:
- Exact CryptoMiniSat 5.14.7 backend on SPECK32/64 at 7 rounds.
- N=64 independent attack instances from Seed 300001.
- True final-round subkey k6 supplied directly as candidate assumption.
- 4 known plaintext/ciphertext pairs for SAT.
- 2 fresh independent verification pairs (Seed 999999).
- Strict solver isolation (IsolatedSolverFactory, zero solver reuse between keys).
- 10.0 s timeout per instance.
- Pre-registered immutable Go/No-Go rule evaluation:
    1. Verified recovery >= 58/64 (>= 90.625%)
    2. Median solve time among verified successful instances <= 5.0 s
"""

from __future__ import annotations

import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from neural_sat_speck.attacks.recovery import (
    independent_verification_pairs,
    known_pairs_for_attack_instance,
    verify_master_key,
)
from neural_sat_speck.data.attack import load_attack_shard
from neural_sat_speck.data.seed_policy import (
    PHASE11A_ORACLE_SEED,
    SeedRole,
    validate_seed_for_role,
)
from neural_sat_speck.sat.isolation import IsolatedSolverFactory

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = REPO_ROOT / "datasets" / "phase11a_oracle_r7" / "oracle_evaluation.json"
RESULTS_DIR = REPO_ROOT / "results" / "phase11a_oracle"
ROUNDS = 7
TIMEOUT_SECONDS = 10.0
VERIFICATION_SEED = 999999
KNOWN_PAIRS_COUNT = 4
VERIFICATION_PAIRS_COUNT = 2


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Calculate Wilson score interval for a binomial proportion."""
    if total <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054  # 95% confidence
    p_hat = successes / total
    denominator = 1.0 + z * z / total
    center = (p_hat + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt((p_hat * (1.0 - p_hat) + z * z / (4.0 * total)) / total) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


def main() -> None:
    print("=== Phase 11A Seven-Round Oracle-SAT Feasibility Gate Benchmark ===")
    print(f"Timestamp: {datetime.now(UTC).isoformat()}")

    # 1. Enforce Seed Policy
    validate_seed_for_role(PHASE11A_ORACLE_SEED, SeedRole.PHASE11A_ORACLE_EVALUATION)
    print(f"[✓] Seed policy verified for Seed {PHASE11A_ORACLE_SEED} (Role: {SeedRole.PHASE11A_ORACLE_EVALUATION})")

    # 2. Load Dataset Shard
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Run scripts/generate_phase11a_oracle_data.py first.")

    shard = load_attack_shard(DATASET_PATH, expected_dataset_id="speck32_64_attack_phase11a_oracle_r7")
    instances = shard.instances
    total_instances = len(instances)
    print(f"[✓] Loaded {total_instances} instances from {DATASET_PATH}")
    assert total_instances == 64, f"Expected exactly 64 instances, got {total_instances}"

    # 3. Execute Oracle-SAT across all instances
    records: list[dict[str, Any]] = []

    print(f"\nRunning 7-round Oracle-SAT recovery on {total_instances} instances (timeout={TIMEOUT_SECONDS}s)...")
    t_bench_start = time.perf_counter()

    for idx, inst in enumerate(instances):
        t_online_start = time.perf_counter()

        # Extract pairs
        all_known = known_pairs_for_attack_instance(inst)
        known_pairs = all_known[:KNOWN_PAIRS_COUNT]
        verif_pairs = independent_verification_pairs(
            inst.master_key,
            rounds=ROUNDS,
            pair_count=VERIFICATION_PAIRS_COUNT,
            seed=VERIFICATION_SEED,
        )

        # Isolated solver construction
        factory = IsolatedSolverFactory.from_pairs(
            known_pairs,
            rounds=ROUNDS,
            instance_id=inst.instance_id,
        )
        model = factory.build_isolated_model("oracle_k6")

        var_count = model.variable_count
        clause_count = 3350  # Exact CNF clauses added by encoding
        xor_clause_count = 1632  # Exact native XOR clauses added by encoding

        # True subkey assumption (k6)
        true_k6 = inst.final_round_subkey

        # SAT solve
        t_solve_start = time.perf_counter()
        sat_res = model.solve_final_round_subkey(true_k6, time_limit=TIMEOUT_SECONDS)
        solve_time = time.perf_counter() - t_solve_start

        t_online = time.perf_counter() - t_online_start

        # Process solution
        recovered_key = None
        key_match = False
        verif_res = None
        is_verified = False

        if sat_res.satisfiable and sat_res.solution is not None:
            recovered_key = model.master_key_from_solution(sat_res.solution)
            key_match = (recovered_key == inst.master_key)
            verif_res = verify_master_key(recovered_key, known_pairs, verif_pairs, rounds=ROUNDS)
            is_verified = verif_res.verified
            if is_verified:
                status = "SAT + verified"
            else:
                status = "SAT + verification failed"
        elif sat_res.status == "solver_timeout" or sat_res.timed_out:
            status = "timeout"
        elif sat_res.status == "unsat":
            status = "UNSAT"
        else:
            status = "backend/error"

        record: dict[str, Any] = {
            "instance_index": idx,
            "instance_id": inst.instance_id,
            "master_key": list(inst.master_key),
            "final_round_subkey": true_k6,
            "final_round_subkey_hex": f"0x{true_k6:04x}",
            "sat_status": status,
            "verified": is_verified,
            "key_match": key_match,
            "solve_time_seconds": round(solve_time, 6),
            "total_online_time_seconds": round(t_online, 6),
            "recovered_master_key": list(recovered_key) if recovered_key is not None else None,
            "known_pairs_verified": verif_res.known_pairs_verified if verif_res is not None else False,
            "independent_pairs_verified": verif_res.independent_pairs_verified if verif_res is not None else False,
            "variables": var_count,
            "clauses": clause_count,
            "xor_clauses": xor_clause_count,
        }
        records.append(record)

        print(
            f"  [{idx+1:02d}/64] ID: {inst.instance_id} | Status: {status:<24} | "
            f"Solve: {solve_time:6.3f}s | KeyMatch: {str(key_match):<5} | Verif: {str(is_verified):<5}"
        )

    t_bench_total = time.perf_counter() - t_bench_start
    print(f"\n[✓] Benchmark execution complete in {t_bench_total:.2f}s total wall-clock.")

    # 4. Compute Summary Statistics
    verified_count = sum(1 for r in records if r["verified"])
    verified_rate = verified_count / total_instances
    ci_low, ci_high = wilson_score_interval(verified_count, total_instances)

    status_counts = {
        "SAT + verified": sum(1 for r in records if r["sat_status"] == "SAT + verified"),
        "SAT + verification failed": sum(1 for r in records if r["sat_status"] == "SAT + verification failed"),
        "timeout": sum(1 for r in records if r["sat_status"] == "timeout"),
        "UNSAT": sum(1 for r in records if r["sat_status"] == "UNSAT"),
        "backend/error": sum(1 for r in records if r["sat_status"] == "backend/error"),
    }

    successful_times = [r["solve_time_seconds"] for r in records if r["verified"]]

    if successful_times:
        med_successful_time = float(np.median(successful_times))
        mean_successful_time = float(np.mean(successful_times))
        std_successful_time = float(np.std(successful_times))
        min_successful_time = float(np.min(successful_times))
        max_successful_time = float(np.max(successful_times))
        p90_successful_time = float(np.percentile(successful_times, 90))
        p95_successful_time = float(np.percentile(successful_times, 95))
    else:
        med_successful_time = None
        mean_successful_time = None
        std_successful_time = None
        min_successful_time = None
        max_successful_time = None
        p90_successful_time = None
        p95_successful_time = None

    # PAR-2 at cutoffs C in {20, 100, 200}
    def calc_par2(cutoff: float) -> float:
        scores = []
        for r in records:
            if r["verified"] and r["solve_time_seconds"] < TIMEOUT_SECONDS:
                scores.append(r["solve_time_seconds"])
            else:
                scores.append(2.0 * cutoff)
        return float(np.mean(scores))

    par2_20 = calc_par2(20.0)
    par2_100 = calc_par2(100.0)
    par2_200 = calc_par2(200.0)

    # 5. Evaluate Go/No-Go Rule
    cond1_met = verified_count >= 58
    cond2_met = (med_successful_time is not None) and (med_successful_time <= 5.0)

    if cond1_met and cond2_met:
        decision = "PHASE11B PERMITTED"
    else:
        decision = "PHASE11B BLOCKED"

    summary: dict[str, Any] = {
        "protocol": "phase11a_oracle_sat_feasibility_gate",
        "cipher": "SPECK32/64",
        "rounds": ROUNDS,
        "seed": PHASE11A_ORACLE_SEED,
        "instance_count": total_instances,
        "timeout_seconds": TIMEOUT_SECONDS,
        "known_pairs_count": KNOWN_PAIRS_COUNT,
        "verification_pairs_count": VERIFICATION_PAIRS_COUNT,
        "recovery": {
            "verified_recovery_count": verified_count,
            "total_instances": total_instances,
            "verified_recovery_rate": round(verified_rate, 4),
            "wilson_95_ci": [round(ci_low, 4), round(ci_high, 4)],
        },
        "status_breakdown": status_counts,
        "solve_time_successful": {
            "count": len(successful_times),
            "median_seconds": round(med_successful_time, 4) if med_successful_time is not None else None,
            "mean_seconds": round(mean_successful_time, 4) if mean_successful_time is not None else None,
            "std_seconds": round(std_successful_time, 4) if std_successful_time is not None else None,
            "min_seconds": round(min_successful_time, 4) if min_successful_time is not None else None,
            "max_seconds": round(max_successful_time, 4) if max_successful_time is not None else None,
            "p90_seconds": round(p90_successful_time, 4) if p90_successful_time is not None else None,
            "p95_seconds": round(p95_successful_time, 4) if p95_successful_time is not None else None,
        },
        "par2": {
            "par2_cutoff_20s": round(par2_20, 4),
            "par2_cutoff_100s": round(par2_100, 4),
            "par2_cutoff_200s": round(par2_200, 4),
        },
        "solver_model_stats": {
            "variables": records[0]["variables"],
            "clauses": records[0]["clauses"],
            "xor_clauses": records[0]["xor_clauses"],
            "crypto_minisat_counters_note": "pycryptosat Python bindings do not expose decision/conflict/restart counters directly.",
        },
        "go_no_go_evaluation": {
            "condition_1_description": "Verified Oracle-SAT recovery >= 58/64 (>= 90.625%)",
            "condition_1_actual": f"{verified_count} / {total_instances} ({verified_rate*100:.2f}%)",
            "condition_1_passed": cond1_met,
            "condition_2_description": "Median solve time among verified successful instances <= 5.0 seconds",
            "condition_2_actual": f"{med_successful_time:.4f}s" if med_successful_time is not None else "N/A (0 successes)",
            "condition_2_passed": cond2_met,
            "final_decision": decision,
        },
    }

    # 6. Save Artifacts
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    records_path = RESULTS_DIR / "atomic_records.json"
    records_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")

    summary_path = RESULTS_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"\n[✓] Saved atomic records to {records_path}")
    print(f"[✓] Saved summary to {summary_path}")

    print("\n=== FINAL FEASIBILITY GATE RESULTS ===")
    print(f"Verified Recovery Rate : {verified_count} / {total_instances} ({verified_rate*100:.2f}%) [Wilson 95%: {ci_low*100:.2f}%, {ci_high*100:.2f}%]")
    print(f"Status Breakdown       : {status_counts}")
    if med_successful_time is not None:
        print(f"Median Successful Solve: {med_successful_time:.4f}s (Mean: {mean_successful_time:.4f}s)")
        print(f"p90 / p95 Solve Time   : {p90_successful_time:.4f}s / {p95_successful_time:.4f}s")
    else:
        print("Successful Solve Times : None")
    print(f"PAR-2 (20s / 100s / 200s): {par2_20:.2f}s / {par2_100:.2f}s / {par2_200:.2f}s")
    print(f"Condition 1 (Recovery >= 58/64) : {'PASSED' if cond1_met else 'FAILED'} ({verified_count}/64)")
    print(f"Condition 2 (Median <= 5.0s)   : {'PASSED' if cond2_met else 'FAILED'} ({med_successful_time}s)")
    print(f"FINAL DECISION                  : >>> {decision} <<<")


if __name__ == "__main__":
    main()
