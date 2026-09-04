#!/usr/bin/env python3
"""Phase 10G — Non-Destructive Confidence-Gated Augmentation Cascade.

Investigates:
1. Preserving B1 top-16 exactly.
2. Appending R1 candidates (ranks 17-32) up to cumulative budget k<=32 when G1 escalates.
3. SAT testing B1 top-16 FIRST, so false alarms on B1 successes exit early at zero extra SAT cost.
4. Asymmetric cost-sensitive threshold selection.
5. Comparisons against B1 direct, verification-triggered B1->R1, verification-triggered B1->B0,
   and previous destructive eager G1->R1.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "results" / "phase10_development"
MASTER_TABLE_PATH = OUTPUT_DIR / "master_audit_table_192.json"
CACHE_DIR = OUTPUT_DIR / "ranking_cache"
B0_CACHE_DIR = OUTPUT_DIR / "b0_fallback_cache"


def main() -> None:
    print("=== Phase 10G: Non-Destructive Confidence-Gated Augmentation Cascade ===")

    master_records = json.loads(MASTER_TABLE_PATH.read_text(encoding="utf-8"))
    N = len(master_records)

    # Component timing constants (measured from Phase 10B audit)
    T_B1 = 5.85          # mean B1 ranking seconds
    T_G1 = 0.0013        # mean G1 feature + inference seconds
    T_R1_RERANK = 0.150  # mean R1 reranking seconds for M=32
    T_SAT_PER_CAND = 0.48  # median SAT solver seconds per candidate
    T_VERIF = 0.0002     # independent verification seconds
    PAR2_PENALTY = 20.0  # standard 2x timeout penalty

    # Load ranking cache to get full candidate orders
    ranking_records = {}
    for f in sorted(CACHE_DIR.glob("instance_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        ranking_records[d["instance_id"]] = d

    # Load B0 fallback cache
    b0_records = {}
    for f in sorted(B0_CACHE_DIR.glob("instance_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        b0_records[d["instance_id"]] = d

    # Extract arrays
    y_fail = np.array([1 if not r["B1_top16_success"] else 0 for r in master_records], dtype=np.int32)
    b1_ranks = np.array([r["B1_true_rank"] for r in master_records], dtype=np.int32)
    g1_scores = np.array([r["G1_failure_score"] for r in master_records], dtype=np.float64)

    # R1 M=32 augmentation candidates: ranks 17 to 32
    # If true key is in ranks 17-32, R1 augmentation contains the true key!
    # In M=32, the true key is in top-32 for 14 of the 17 failures!
    in_top32 = np.array([r["B1_true_rank"] <= 32 for r in master_records], dtype=bool)

    # 1. Baseline 1: B1 Direct (k=16)
    b1_recovered = np.sum(b1_ranks <= 16)
    b1_sat_times = []
    for r in b1_ranks:
        if r <= 16:
            b1_sat_times.append(r * T_SAT_PER_CAND)
        else:
            b1_sat_times.append(16 * T_SAT_PER_CAND)
    b1_online_times = T_B1 + np.array(b1_sat_times) + T_VERIF
    b1_par2 = np.where(b1_ranks <= 16, b1_online_times, b1_online_times + PAR2_PENALTY)

    print(f"Baseline A (B1 Direct k=16): {b1_recovered}/{N} ({b1_recovered/N*100:.2f}%) | "
          f"Mean Online: {np.mean(b1_online_times):.2f}s | PAR-2: {np.mean(b1_par2):.2f}s")

    # 2. Baseline 2: Destructive Eager G1->R1 (k=16, previous negative comparator at theta=0.0892)
    dest_recovered = sum(1 for r in master_records if r["cascade_C_success"])
    print(f"Destructive G1->R1 (k=16, theta=0.0892): {dest_recovered}/{N} ({dest_recovered/N*100:.2f}%) [Negative Comparator]")

    # 3. Baseline 3: Verification-Triggered Lazy B1 -> R1 (k<=32)
    # Stage 1 tests B1 top-16. If fails verification, Stage 2 tests R1 candidates (ranks 17-32).
    lazy_r1_recovered = 0
    lazy_r1_online_times = []
    lazy_r1_cands_tested = []
    for idx in range(N):
        r = b1_ranks[idx]
        if r <= 16:
            lazy_r1_recovered += 1
            t_sat = r * T_SAT_PER_CAND
            lazy_r1_cands_tested.append(r)
            lazy_r1_online_times.append(T_B1 + t_sat + T_VERIF)
        else:
            # Stage 1 failed (16 cands). Trigger R1 reranking of shortlist M=32.
            # R1 tests ranks 17-32 (16 cands). If true key in top-32:
            if in_top32[idx]:
                lazy_r1_recovered += 1
                r_tail = r - 16
                t_sat = (16 + r_tail) * T_SAT_PER_CAND
                lazy_r1_cands_tested.append(16 + r_tail)
            else:
                t_sat = 32 * T_SAT_PER_CAND
                lazy_r1_cands_tested.append(32)
            lazy_r1_online_times.append(T_B1 + 16 * T_SAT_PER_CAND + T_R1_RERANK + (t_sat - 16 * T_SAT_PER_CAND) + T_VERIF)

    lazy_r1_par2 = np.where(
        [b1_ranks[i] <= 32 for i in range(N)],
        lazy_r1_online_times,
        np.array(lazy_r1_online_times) + PAR2_PENALTY
    )
    print(f"Baseline D (Verification-Triggered Lazy B1->R1, k<=32): {lazy_r1_recovered}/{N} ({lazy_r1_recovered/N*100:.2f}%) | "
          f"Mean Online: {np.mean(lazy_r1_online_times):.2f}s | PAR-2: {np.mean(lazy_r1_par2):.2f}s")

    # 4. Baseline 4: Verification-Triggered Lazy B1 -> B0 (k<=32)
    lazy_b0_recovered = sum(1 for r in master_records if r["cascade_D_success"])
    print(f"Baseline E (Verification-Triggered Lazy B1->B0, k<=32): {lazy_b0_recovered}/{N} ({lazy_b0_recovered/N*100:.2f}%)")

    # 5. Non-Destructive Confidence-Gated Augmentation Cascade: Sweep Across Thresholds
    print("\n--- Non-Destructive Augmentation: Threshold Sweep ---")
    print(f"{'Theta':<7} | {'Coverage':<9} | {'Escalated':<10} | {'False-Trust':<11} | {'False-Alarm':<11} | {'Recovered':<10} | {'Recovery%':<9} | {'Mean Time':<10} | {'PAR-2':<8}")
    print("-" * 105)

    thresholds = np.linspace(0.075, 0.100, 26)
    sweep_results = []

    best_theta_par2 = None
    best_par2_val = 9999.0
    best_theta_constrained = None
    best_cost_val = 9999.0

    for th in thresholds:
        n_accepted = 0
        n_escalated = 0
        n_false_trust = 0
        n_false_alarm = 0
        n_recovered = 0
        online_times = []
        par2_scores = []
        cands_tested_list = []

        for idx in range(N):
            p_fail = g1_scores[idx]
            r = b1_ranks[idx]
            is_success_b1 = bool(r <= 16)

            if p_fail < th:
                # Route: B1 direct (k=16)
                n_accepted += 1
                if is_success_b1:
                    n_recovered += 1
                    t_sat = r * T_SAT_PER_CAND
                    c_tested = r
                else:
                    # False Trust!
                    n_false_trust += 1
                    t_sat = 16 * T_SAT_PER_CAND
                    c_tested = 16
                t_online = T_B1 + T_G1 + t_sat + T_VERIF
                par2 = t_online if is_success_b1 else t_online + PAR2_PENALTY
            else:
                # Route: Non-Destructive Augmentation (k<=32)
                # Protected B1 top-16 tested FIRST!
                n_escalated += 1
                if is_success_b1:
                    # False Alarm! But non-destructive: B1 candidate r <= 16 solves first!
                    n_false_alarm += 1
                    n_recovered += 1
                    t_sat = r * T_SAT_PER_CAND
                    c_tested = r
                    # Extra cost is ONLY the R1 rerank time! SAT does NOT test tail!
                    t_online = T_B1 + T_G1 + T_R1_RERANK + t_sat + T_VERIF
                    par2 = t_online
                else:
                    # True Alarm! Tests 16 B1 candidates, then tests tail from R1!
                    if in_top32[idx]:
                        n_recovered += 1
                        r_tail = r - 16
                        t_sat = (16 + r_tail) * T_SAT_PER_CAND
                        c_tested = 16 + r_tail
                        t_online = T_B1 + T_G1 + T_R1_RERANK + t_sat + T_VERIF
                        par2 = t_online
                    else:
                        # Outside top-32: tests all 32 candidates and fails
                        t_sat = 32 * T_SAT_PER_CAND
                        c_tested = 32
                        t_online = T_B1 + T_G1 + T_R1_RERANK + t_sat + T_VERIF
                        par2 = t_online + PAR2_PENALTY

            online_times.append(t_online)
            par2_scores.append(par2)
            cands_tested_list.append(c_tested)

        coverage = n_accepted / N
        ft_rate = n_false_trust / n_accepted if n_accepted > 0 else 0.0
        fa_rate = n_false_alarm / (N - np.sum(y_fail))
        mean_t = float(np.mean(online_times))
        mean_par2 = float(np.mean(par2_scores))

        res_entry = {
            "theta": float(th),
            "coverage": coverage,
            "escalated_count": n_escalated,
            "accepted_count": n_accepted,
            "false_trust_count": n_false_trust,
            "false_trust_rate": ft_rate,
            "false_alarm_count": n_false_alarm,
            "false_alarm_rate": fa_rate,
            "recovered_count": n_recovered,
            "recovery_rate": n_recovered / N,
            "mean_online_seconds": mean_t,
            "par2_score": mean_par2,
            "mean_candidates_tested": float(np.mean(cands_tested_list)),
        }
        sweep_results.append(res_entry)

        if mean_par2 < best_par2_val:
            best_par2_val = mean_par2
            best_theta_par2 = float(th)

        # Cost-sensitive objective: Minimum recovery >= 96.5% with lowest expected online time
        if n_recovered / N >= 0.965 and mean_t < best_cost_val:
            best_cost_val = mean_t
            best_theta_constrained = float(th)

        print(f"{th:<7.4f} | {coverage*100:<8.1f}% | {n_escalated:<10d} | {ft_rate*100:<10.2f}% | {fa_rate*100:<10.2f}% | {n_recovered:<10d} | {n_recovered/N*100:<8.2f}% | {mean_t:<9.2f}s | {mean_par2:<8.2f}s")

    print("\n--- Decision Rule Optimization Summary ---")
    print(f"Optimal Theta under Minimum PAR-2: theta* = {best_theta_par2:.4f} (PAR-2 = {best_par2_val:.2f}s)")
    print(f"Optimal Theta under Constrained Recovery (>=96.5%): theta* = {best_theta_constrained:.4f} (Mean Time = {best_cost_val:.2f}s)")

    # Detailed performance at the frozen Phase 10 threshold theta = 0.0892
    th_frozen = 0.0892
    f_entry = next((e for e in sweep_results if abs(e["theta"] - th_frozen) < 0.001), sweep_results[len(sweep_results)//2])
    print(f"\n=== Non-Destructive Augmentation at Frozen Theta ({f_entry['theta']:.4f}) ===")
    print(f"Recovered Keys:         {f_entry['recovered_count']}/192 ({f_entry['recovery_rate']*100:.2f}%) [Net Gain vs B1: +{f_entry['recovered_count'] - b1_recovered}]")
    print(f"Coverage (B1 Direct):   {f_entry['accepted_count']}/192 ({f_entry['coverage']*100:.2f}%)")
    print(f"Escalated (Augment R1): {f_entry['escalated_count']}/192 ({f_entry['escalated_count']/N*100:.2f}%)")
    print(f"False-Trust Count:      {f_entry['false_trust_count']} (Observed False-Trust Rate: {f_entry['false_trust_rate']*100:.2f}%)")
    print(f"False-Alarm Count:      {f_entry['false_alarm_count']} ({f_entry['false_alarm_rate']*100:.2f}% of successes)")
    print(f"Mean Online Seconds:    {f_entry['mean_online_seconds']:.2f}s (vs B1 Direct: {np.mean(b1_online_times):.2f}s, Delta: +{f_entry['mean_online_seconds'] - np.mean(b1_online_times):.2f}s)")
    print(f"PAR-2 Score:            {f_entry['par2_score']:.2f}s (vs B1 Direct: {np.mean(b1_par2):.2f}s, Delta: {f_entry['par2_score'] - np.mean(b1_par2):.2f}s)")

    # Save Phase 10G results
    out_file = OUTPUT_DIR / "phase10g_augmentation_sweep.json"
    out_file.write_text(json.dumps(sweep_results, indent=2) + "\n", encoding="utf-8")
    print(f"\n[✓] Saved Phase 10G sweep results to {out_file}")


if __name__ == "__main__":
    main()
