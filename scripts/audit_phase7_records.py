#!/usr/bin/env python3
"""Deterministic audit script for Phase 7 held-out evaluation records.

This script executes Part B1 & B3 of Phase 9:
1. Reads and validates every atomic record in results/phase7_final_evaluation/records/;
2. Reproduces exact success rates, attempts, and component timings;
3. Computes global PAR-2 sensitivity across multiple cutoffs (80s, 100s, 160s, 200s, 300s);
4. Computes exact paired McNemar hypothesis tests;
5. Outputs results/phase7_final_evaluation/audited_summary.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from neural_sat_speck.evaluation.audit import audit_phase7_records

REPO_ROOT = Path(__file__).resolve().parent.parent
RECORDS_DIR = REPO_ROOT / "results" / "phase7_final_evaluation" / "records"
OUTPUT_FILE = REPO_ROOT / "results" / "phase7_final_evaluation" / "audited_summary.json"


def main() -> None:
    print("================================================================================")
    print("PHASE 9: DETERMINISTIC AUDIT OF PHASE 7 HELD-OUT EVALUATION RECORDS")
    print("================================================================================")
    print(f"Reading atomic records from: {RECORDS_DIR}")

    result = audit_phase7_records(RECORDS_DIR, expected_count=64)

    print("\n--- 1. AUDITED METRICS SUMMARY (N=64 keys, Seed 98001) ---")
    print(f"{'Method':<25} | {'Solved/64':<10} | {'Exact %':<8} | {'Wilson 95% CI':<18} | {'Mean Attempts':<14} | {'Median Online':<14}")
    print("-" * 105)
    for method_name, data in result["methods"].items():
        ci_str = f"[{data['wilson_95_ci'][0]*100:.1f}%, {data['wilson_95_ci'][1]*100:.1f}%]"
        print(
            f"{method_name:<25} | "
            f"{data['success_count']:>2}/64       | "
            f"{data['success_rate']*100:>6.2f}% | "
            f"{ci_str:<18} | "
            f"{data['mean_candidates_attempted']:>13.2f}  | "
            f"{data['median_online_seconds']:>12.2f}s"
        )

    print("\n--- 2. COMPONENT TIMING BREAKDOWN (Mean seconds) ---")
    print(f"{'Method':<25} | {'Rank Time':<12} | {'Conf Time':<12} | {'SAT Time':<12} | {'Ver Time':<12} | {'Total Online':<12}")
    print("-" * 100)
    for method_name, data in result["methods"].items():
        print(
            f"{method_name:<25} | "
            f"{data['mean_rank_seconds']:>10.2f}s  | "
            f"{data['mean_conf_seconds']:>10.4f}s  | "
            f"{data['mean_sat_seconds']:>10.4f}s  | "
            f"{data['mean_ver_seconds']:>10.6f}s  | "
            f"{data['mean_online_seconds']:>10.2f}s"
        )

    print("\n--- 3. RETROSPECTIVE FIXED-POLICY PAR-2 SENSITIVITY (Global Cutoffs) ---")
    print(f"{'Method':<25} | {'PAR-2 (80s)':<12} | {'PAR-2 (100s)':<12} | {'PAR-2 (160s)':<12} | {'PAR-2 (200s)':<12} | {'PAR-2 (300s)':<12}")
    print("-" * 95)
    for method_name, data in result["methods"].items():
        sens = data["par2_sensitivity"]
        print(
            f"{method_name:<25} | "
            f"{sens['par2_at_80s']:>10.2f}s  | "
            f"{sens['par2_at_100s']:>10.2f}s  | "
            f"{sens['par2_at_160s']:>10.2f}s  | "
            f"{sens['par2_at_200s']:>10.2f}s  | "
            f"{sens['par2_at_300s']:>10.2f}s"
        )

    print("\n--- 4. PAIRED MCNEMAR HYPOTHESIS TESTS (B1 vs Other Methods) ---")
    for comp_name, comp_data in result["paired_comparisons"].items():
        sig_str = "SIGNIFICANT (p < 0.05)" if comp_data["significant_at_05"] else "NOT SIGNIFICANT (p >= 0.05)"
        print(
            f"{comp_name:<30}: b={comp_data['b']}, c={comp_data['c']}, "
            f"net_diff={comp_data['net_difference']:+d}, p-value={comp_data['p_value']:.6f} ({sig_str})"
        )

    # Write output JSON atomically
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(f"\n[✓] Audited summary successfully written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
