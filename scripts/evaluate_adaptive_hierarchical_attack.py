#!/usr/bin/env python3
"""Evaluate Adaptive and Cascade Multi-Round Attacks (Phase 11G).

Compares four operational recovery policies on 7-round SPECK32/64:
1. Static Narrow Beam: M1=8, M2=32
2. Static Wide Beam: M1=32, M2=32
3. Verification-Triggered Cascade: M1=8 -> M1=32 (with stage caching & candidate deduplication)
4. G2-Gated Adaptive Routing: M1=8 if G2_risk <= tau, else M1=32 (both direct and with fallback)

Evaluated on the 16 opened Seed 300001 instances under strict solver isolation
and independent fresh-pair cryptographic verification.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from neural_sat_speck.attacks.hierarchical_multi_round import (
    HierarchicalStage,
    run_hierarchical_multi_round_attack,
)
from neural_sat_speck.attacks.recovery import (
    independent_verification_pairs,
    known_pairs_for_attack_instance,
)
from neural_sat_speck.data.attack import load_attack_shard
from neural_sat_speck.evaluation.metrics import wilson_score_interval
from neural_sat_speck.neural.group_model import PairGroupRanker

DATASET_PATH = (
    REPOSITORY_ROOT / "datasets" / "phase11a_oracle_r7" / "oracle_evaluation.json"
)
DATASET_ID = "speck32_64_attack_phase11a_oracle_r7"
STAGE1_MODEL_PATH = (
    REPOSITORY_ROOT / "models" / "speck32_64_phase11b_prime_stage1_r6_b1_mean.pt"
)
STAGE2_MODEL_PATH = (
    REPOSITORY_ROOT / "models" / "speck32_64_phase6a_b1_mean_g64.pt"
)
G2_SUMMARY_PATH = (
    REPOSITORY_ROOT / "results" / "phase11_scaling_diagnostics" / "g2_stage1_transfer_summary.json"
)
STATIC_WIDE_CONSOLIDATED_PATH = (
    REPOSITORY_ROOT / "results" / "phase11_scaling_diagnostics" / "hierarchical_eval_16_instances_consolidated.json"
)
RESULTS_DIR = REPOSITORY_ROOT / "results" / "phase11_scaling_diagnostics"
ROUNDS = 7
KNOWN_PAIRS = 4
VERIFICATION_PAIRS = 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=int, default=16, help="Instances to evaluate (default 16)")
    parser.add_argument("--start", type=int, default=0, help="Starting instance index (default 0)")
    parser.add_argument("--timeout", type=float, default=10.0, help="Per-tuple solver timeout (default 10s)")
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "adaptive_cascade_comparison.json",
        help="Output JSON path",
    )
    return parser.parse_args(argv)


def load_pytorch_ranker(path: Path, device: torch.device) -> PairGroupRanker:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    variant = payload.get("variant", "b1_mean")
    model = PairGroupRanker(variant=variant)
    state = payload["state_dict"] if "state_dict" in payload else payload
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"=== EVALUATING ADAPTIVE & CASCADE MULTI-ROUND ATTACKS (PHASE 11G) ===")
    print(f"Cohort: {args.instances} instances ({args.start}..{args.start + args.instances - 1}) on {device}\n")

    # Load Models
    stage1_model = load_pytorch_ranker(STAGE1_MODEL_PATH, device)
    stage2_model = load_pytorch_ranker(STAGE2_MODEL_PATH, device)

    # Load G2 Probs
    g2_map: dict[str, float] = {}
    if G2_SUMMARY_PATH.exists():
        with open(G2_SUMMARY_PATH) as f:
            g2_data = json.load(f)
            for r in g2_data.get("records", []):
                g2_map[r["instance_id"]] = r["g2_prob"]

    # Load Static Wide Consolidated Results
    static_wide_map: dict[str, dict[str, Any]] = {}
    if STATIC_WIDE_CONSOLIDATED_PATH.exists():
        with open(STATIC_WIDE_CONSOLIDATED_PATH) as f:
            sw_data = json.load(f)
            for r in sw_data.get("records", []):
                static_wide_map[r["instance_id"]] = r

    shard = load_attack_shard(DATASET_PATH, expected_dataset_id=DATASET_ID)
    instances = shard.instances[args.start : args.start + args.instances]

    instance_records: list[dict[str, Any]] = []

    for offset, inst in enumerate(instances):
        inst_idx = args.start + offset
        known = known_pairs_for_attack_instance(inst)[:KNOWN_PAIRS]
        verification = independent_verification_pairs(
            inst.master_key,
            rounds=ROUNDS,
            pair_count=VERIFICATION_PAIRS,
            seed=999999,
        )

        g2_prob = g2_map.get(inst.instance_id, 0.0)
        sw_rec = static_wide_map.get(inst.instance_id, {})
        static_wide_success = sw_rec.get("success", False)
        static_wide_wall = sw_rec.get("wall_seconds", 112.0)
        static_wide_tuples = sw_rec.get("candidates_attempted", 1024)

        # ----------------------------------------------------
        # CASCASE EVALUATION: STAGE 1 WITH M1=8 FIRST
        # ----------------------------------------------------
        stage_cache: dict[tuple[int, ...], Any] = {}
        t0 = time.perf_counter()
        stages_8 = (
            HierarchicalStage(round_index=ROUNDS - 1, model=stage1_model, beam_width=8),
            HierarchicalStage(
                round_index=ROUNDS - 2,
                model=stage2_model,
                beam_width=32,
                screened=True,
                screen_pairs=16,
                screen_top_k=512,
            ),
        )
        res_8 = run_hierarchical_multi_round_attack(
            inst,
            rounds=ROUNDS,
            stages=stages_8,
            known_pairs=known,
            verification_pairs=verification,
            solver_timeout=args.timeout,
            device=device,
            stage_cache=stage_cache,
        )
        t_8 = time.perf_counter() - t0

        if res_8.success:
            cascade_success = True
            cascade_time = t_8
            cascade_stage = "M1_8"
            cascade_tuples = res_8.candidates_attempted
        else:
            # Fallback to M1=32 with Stage Caching and Deduplication
            t1 = time.perf_counter()
            stages_32 = (
                HierarchicalStage(round_index=ROUNDS - 1, model=stage1_model, beam_width=32),
                HierarchicalStage(
                    round_index=ROUNDS - 2,
                    model=stage2_model,
                    beam_width=32,
                    screened=True,
                    screen_pairs=16,
                    screen_top_k=512,
                ),
            )
            res_32 = run_hierarchical_multi_round_attack(
                inst,
                rounds=ROUNDS,
                stages=stages_32,
                known_pairs=known,
                verification_pairs=verification,
                solver_timeout=args.timeout,
                device=device,
                stage_cache=stage_cache,
                excluded_paths=set(res_8.attempted_paths),
            )
            t_32 = time.perf_counter() - t1
            cascade_success = res_32.success
            cascade_time = t_8 + t_32  # cumulative cost of cascade
            cascade_stage = "M1_32_fallback"
            cascade_tuples = res_8.candidates_attempted + res_32.candidates_attempted

        rec = {
            "instance_index": inst_idx,
            "instance_id": inst.instance_id,
            "g2_prob": g2_prob,
            # Static Narrow (M1=8)
            "static_8_success": res_8.success,
            "static_8_wall_seconds": round(t_8, 2),
            "static_8_tuples": res_8.candidates_attempted,
            # Static Wide (M1=32)
            "static_32_success": static_wide_success,
            "static_32_wall_seconds": round(static_wide_wall, 2),
            "static_32_tuples": static_wide_tuples,
            # Cascade (M1=8 -> M1=32)
            "cascade_success": cascade_success,
            "cascade_stage": cascade_stage,
            "cascade_tuples": cascade_tuples,
            "cascade_wall_seconds": round(cascade_time, 2),
            "stage1_rank": res_8.stages[0].true_round_key_rank,
            "stage2_rank": res_8.stages[1].true_round_key_rank if len(res_8.stages) > 1 else None,
        }
        instance_records.append(rec)

        status_str = "RECOVERED (M1=8)" if cascade_stage == "M1_8" else (
            "RECOVERED (Cascade Fallback)" if cascade_success else "EXHAUSTED"
        )
        print(
            f"[{inst_idx:02d}] {inst.instance_id:<11} | G2 Risk: {g2_prob:.4f} | "
            f"k6 Rank: {rec['stage1_rank']:3d} | "
            f"Status: {status_str:<28} | M1=8 Time: {t_8:5.1f}s | "
            f"Cascade Time: {cascade_time:5.1f}s"
        )
        with open(args.output, "w") as f:
            json.dump({"partial_records": instance_records}, f, indent=2)

    # -------------------------------------------------------------------------
    # SUMMARY ANALYSIS ACROSS ALL POLICIES
    # -------------------------------------------------------------------------
    n = len(instance_records)

    # 1. Static Narrow (M1=8)
    succ_8 = sum(r["static_8_success"] for r in instance_records)
    times_8 = [r["static_8_wall_seconds"] for r in instance_records]
    tuples_8 = [r["static_8_tuples"] for r in instance_records]
    ci_8 = wilson_score_interval(succ_8, n)

    # 2. Static Wide (M1=32)
    succ_32 = sum(r["static_32_success"] for r in instance_records)
    times_32 = [r["static_32_wall_seconds"] for r in instance_records]
    tuples_32 = [r["static_32_tuples"] for r in instance_records]
    ci_32 = wilson_score_interval(succ_32, n)

    # 3. Verification-Triggered Cascade (M1=8 -> M1=32)
    succ_casc = sum(r["cascade_success"] for r in instance_records)
    times_casc = [r["cascade_wall_seconds"] for r in instance_records]
    tuples_casc = [r["cascade_tuples"] for r in instance_records]
    ci_casc = wilson_score_interval(succ_casc, n)

    # 4. G2-Gated Policies across Tau Thresholds
    tau_grid = [0.001, 0.005, 0.01, 0.05, 0.10, 0.20, 0.50]
    g2_direct_eval: dict[str, Any] = {}
    g2_cascade_eval: dict[str, Any] = {}

    for tau in tau_grid:
        d_succ, d_times, d_tuples = 0, [], []
        c_succ, c_times, c_tuples = 0, [], []

        for r in instance_records:
            is_low_risk = r["g2_prob"] <= tau
            if is_low_risk:
                d_succ += int(r["static_8_success"])
                d_times.append(r["static_8_wall_seconds"])
                d_tuples.append(r["static_8_tuples"])

                c_succ += int(r["cascade_success"])
                c_times.append(r["cascade_wall_seconds"])
                c_tuples.append(r["cascade_tuples"])
            else:
                d_succ += int(r["static_32_success"])
                d_times.append(r["static_32_wall_seconds"])
                d_tuples.append(r["static_32_tuples"])

                c_succ += int(r["static_32_success"])
                c_times.append(r["static_32_wall_seconds"])
                c_tuples.append(r["static_32_tuples"])

        low_d, high_d = wilson_score_interval(d_succ, n)
        low_c, high_c = wilson_score_interval(c_succ, n)

        g2_direct_eval[f"tau_{tau}"] = {
            "tau": tau,
            "recoveries": d_succ,
            "recovery_rate": round(d_succ / n, 4),
            "wilson_95_ci": [round(low_d, 4), round(high_d, 4)],
            "median_wall_seconds": round(float(np.median(d_times)), 2),
            "mean_wall_seconds": round(float(np.mean(d_times)), 2),
            "mean_tuples_attempted": round(float(np.mean(d_tuples)), 1),
        }
        g2_cascade_eval[f"tau_{tau}"] = {
            "tau": tau,
            "recoveries": c_succ,
            "recovery_rate": round(c_succ / n, 4),
            "wilson_95_ci": [round(low_c, 4), round(high_c, 4)],
            "median_wall_seconds": round(float(np.median(c_times)), 2),
            "mean_wall_seconds": round(float(np.mean(c_times)), 2),
            "mean_tuples_attempted": round(float(np.mean(c_tuples)), 1),
        }

    summary = {
        "experiment": "phase11g_adaptive_cascade_multi_round_evaluation",
        "timestamp": datetime.now(UTC).isoformat(),
        "total_instances": n,
        "static_narrow_m1_8": {
            "recoveries": succ_8,
            "recovery_rate": round(succ_8 / n, 4),
            "wilson_95_ci": [round(ci_8[0], 4), round(ci_8[1], 4)],
            "median_wall_seconds": round(float(np.median(times_8)), 2),
            "mean_wall_seconds": round(float(np.mean(times_8)), 2),
            "mean_tuples_attempted": round(float(np.mean(tuples_8)), 1),
        },
        "static_wide_m1_32": {
            "recoveries": succ_32,
            "recovery_rate": round(succ_32 / n, 4),
            "wilson_95_ci": [round(ci_32[0], 4), round(ci_32[1], 4)],
            "median_wall_seconds": round(float(np.median(times_32)), 2),
            "mean_wall_seconds": round(float(np.mean(times_32)), 2),
            "mean_tuples_attempted": round(float(np.mean(tuples_32)), 1),
        },
        "verification_triggered_cascade": {
            "recoveries": succ_casc,
            "recovery_rate": round(succ_casc / n, 4),
            "wilson_95_ci": [round(ci_casc[0], 4), round(ci_casc[1], 4)],
            "median_wall_seconds": round(float(np.median(times_casc)), 2),
            "mean_wall_seconds": round(float(np.mean(times_casc)), 2),
            "total_wall_seconds": round(float(sum(times_casc)), 2),
            "mean_tuples_attempted": round(float(np.mean(tuples_casc)), 1),
        },
        "g2_gated_direct": g2_direct_eval,
        "g2_gated_cascade": g2_cascade_eval,
        "records": instance_records,
    }

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[✓] Results written to {args.output}")
    print("\n" + "=" * 78)
    print("POLICY COMPARISON SUMMARY (N=16)")
    print("=" * 78)
    print(f"1. Static Narrow (M1=8)        : {succ_8:2d}/{n} ({succ_8/n*100:5.1f}%) | Median Wall: {np.median(times_8):5.1f}s | Mean Tuples: {np.mean(tuples_8):5.1f}")
    print(f"2. Static Wide (M1=32)         : {succ_32:2d}/{n} ({succ_32/n*100:5.1f}%) | Median Wall: {np.median(times_32):5.1f}s | Mean Tuples: {np.mean(tuples_32):5.1f}")
    print(f"3. Verification Cascade (8->32): {succ_casc:2d}/{n} ({succ_casc/n*100:5.1f}%) | Median Wall: {np.median(times_casc):5.1f}s | Mean Tuples: {np.mean(tuples_casc):5.1f}")
    best_tau = "tau_0.01"
    g2_c = g2_cascade_eval[best_tau]
    print(f"4. G2-Adaptive Cascade (tau=0.01): {g2_c['recoveries']:2d}/{n} ({g2_c['recovery_rate']*100:5.1f}%) | Median Wall: {g2_c['median_wall_seconds']:5.1f}s | Mean Tuples: {g2_c['mean_tuples_attempted']:5.1f}")
    print("=" * 78)

    return 0


if __name__ == "__main__":
    sys.exit(main())
