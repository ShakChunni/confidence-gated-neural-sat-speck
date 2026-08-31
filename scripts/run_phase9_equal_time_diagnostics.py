#!/usr/bin/env python3
"""Run retrospective equal-wall-clock SAT-only and Random diagnostics on historical seed 98001.

This script executes Phase 9 Parts B4 & B5:
1. Runs unguided SAT-only under extended time budgets (80s, 160s);
2. Runs random candidate order incremental SAT under the same extended time budgets (80s, 160s);
3. Records atomic results per instance and summarizes success rates, attempts, and times.

NOTE: This is a retrospective fairness diagnostic on opened historical data ONLY.
It is NOT used for method selection, threshold tuning, or final confirmation claims.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
from pathlib import Path
from time import perf_counter
from typing import Any

from neural_sat_speck.attacks.recovery import (
    known_pairs_for_attack_instance,
    verify_master_key,
)
from neural_sat_speck.data.attack import AttackInstance, load_attack_shard
from neural_sat_speck.evaluation.metrics import (
    calculate_global_par2,
    wilson_score_interval,
)
from neural_sat_speck.sat.model import SpeckSatModel
from neural_sat_speck.speck.key_schedule import expand_key

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiments" / "phase9_equal_time_diagnostics.json"


def _evaluate_instance(
    instance: AttackInstance,
    index: int,
    rounds: int = 6,
    max_timeout: float = 160.0,
    known_pairs_count: int = 4,
    verification_pairs_count: int = 2,
    random_seed_offset: int = 500000,
) -> dict[str, Any]:
    known_pairs_all = known_pairs_for_attack_instance(instance)
    known_sat_pairs = known_pairs_all[:known_pairs_count]
    ver_pairs = known_pairs_all[known_pairs_count : known_pairs_count + verification_pairs_count]

    true_key = instance.master_key
    true_final_subkey = expand_key(true_key, rounds)[-1]

    # 1. SAT-Only Equal-Time Diagnostic (Run up to max_timeout = 160.0s)
    sat_model = SpeckSatModel.from_known_pairs(known_sat_pairs, rounds)
    t_start = perf_counter()
    sat_res = sat_model.solve(time_limit=max_timeout)
    sat_elapsed = perf_counter() - t_start

    sat_success = False
    sat_status = "solver_timeout"
    if sat_res.satisfiable and sat_res.solution is not None:
        rec_key = sat_model.master_key_from_solution(sat_res.solution)
        v = verify_master_key(rec_key, known_sat_pairs, ver_pairs, rounds)
        if v.verified:
            sat_success = True
            sat_status = "recovered"
        else:
            sat_status = "sat_unverified"
    elif sat_res.status == "unsat":
        sat_status = "unsat"

    sat_only_data = {
        "t_online": float(sat_elapsed),
        "status": sat_status,
        "success": sat_success,
        "success_at_80s": sat_success and sat_elapsed <= 80.0,
        "success_at_160s": sat_success and sat_elapsed <= 160.0,
    }

    # 2. Random-Order Incremental SAT Diagnostic (Run up to max_timeout = 160.0s)
    # Generate deterministic random candidate permutation
    rng = random.Random(index + random_seed_offset)
    candidates = list(range(65536))
    rng.shuffle(candidates)

    rand_sat_model = SpeckSatModel.from_known_pairs(known_sat_pairs, rounds)
    rand_start = perf_counter()
    rand_success = False
    rand_winning_cand = None
    rand_attempts = 0
    rand_status = "budget_or_time_exhausted"
    t_80s_record: dict[str, Any] | None = None

    for cand_pos, cand in enumerate(candidates, start=1):
        elapsed_now = perf_counter() - rand_start
        if elapsed_now >= max_timeout:
            break

        # Capture state at 80s boundary if not yet captured
        if elapsed_now > 80.0 and t_80s_record is None:
            t_80s_record = {
                "candidates_attempted": rand_attempts,
                "t_online": float(elapsed_now),
                "success": rand_success,
                "status": rand_status if rand_success else "time_limit_reached",
                "winning_candidate": rand_winning_cand,
            }

        rand_attempts = cand_pos
        time_rem = max(0.01, min(10.0, max_timeout - elapsed_now))
        # Test candidate assumption
        sub_res = rand_sat_model.solve_final_round_subkey(cand, time_limit=time_rem)

        if sub_res.satisfiable and sub_res.solution is not None:
            rec_key = rand_sat_model.master_key_from_solution(sub_res.solution)
            v = verify_master_key(rec_key, known_sat_pairs, ver_pairs, rounds)
            if v.verified:
                rand_success = True
                rand_winning_cand = cand
                rand_status = "recovered"
                break

    rand_total_elapsed = perf_counter() - rand_start
    if t_80s_record is None:
        t_80s_record = {
            "candidates_attempted": rand_attempts,
            "t_online": float(min(80.0, rand_total_elapsed)),
            "success": rand_success and rand_total_elapsed <= 80.0,
            "status": rand_status if (rand_success and rand_total_elapsed <= 80.0) else "time_limit_reached",
            "winning_candidate": rand_winning_cand if (rand_success and rand_total_elapsed <= 80.0) else None,
        }

    random_data = {
        "t_online": float(rand_total_elapsed),
        "candidates_attempted": rand_attempts,
        "status": rand_status,
        "success": rand_success,
        "winning_candidate": rand_winning_cand,
        "success_at_80s": rand_success and rand_total_elapsed <= 80.0,
        "success_at_160s": rand_success and rand_total_elapsed <= 160.0,
        "snapshot_at_80s": t_80s_record,
    }

    return {
        "schema": "neural_sat_speck.phase9_equal_time_instance_result.v1",
        "instance_index": index,
        "instance_id": instance.instance_id,
        "true_final_round_subkey": true_final_subkey,
        "sat_only": sat_only_data,
        "random_incremental_sat": random_data,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 9 Equal-Time SAT Diagnostics")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--smoke", action="store_true", help="Run 2 instances for smoke test")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    manifest_path = REPO_ROOT / config["dataset"]["manifest_path"]
    shard_path = manifest_path.parent / "test.json"
    collection = load_attack_shard(shard_path)
    instances = collection.instances
    rounds = int(config["dataset"]["rounds"])

    if args.smoke:
        instances = instances[:2]
        print(f"[SMOKE MODE] Running {len(instances)} instances...")
    else:
        print(f"Running Phase 9 Equal-Time Diagnostics on {len(instances)} instances using {args.workers} workers...")

    records_dir = REPO_ROOT / config["outputs"]["records_dir"]
    records_dir.mkdir(parents=True, exist_ok=True)

    max_timeout = float(config["random_strategy"]["max_time_seconds"])

    results: list[dict[str, Any]] = []

    t_diag_start = perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _evaluate_instance,
                inst,
                idx,
                rounds=rounds,
                max_timeout=max_timeout,
            ): idx
            for idx, inst in enumerate(instances)
        }

        for fut in concurrent.futures.as_completed(futures):
            idx = futures[fut]
            try:
                res = fut.result()
                results.append(res)
                # Write atomic record
                rec_path = records_dir / f"instance_{idx:04d}.json"
                with open(rec_path, "w", encoding="utf-8") as f:
                    json.dump(res, f, indent=2, sort_keys=True)
                print(
                    f"[{len(results)}/{len(instances)}] Instance {idx:04d}: "
                    f"SAT-only {'SOLVED' if res['sat_only']['success'] else 'TIMEOUT'} ({res['sat_only']['t_online']:.1f}s), "
                    f"Random {'SOLVED' if res['random_incremental_sat']['success'] else 'EXHAUSTED'} ({res['random_incremental_sat']['t_online']:.1f}s, {res['random_incremental_sat']['candidates_attempted']} attempts)"
                )
            except Exception as exc:
                print(f"[-] Error processing instance {idx:04d}: {exc}")
                raise exc

    results.sort(key=lambda r: r["instance_index"])
    total_diag_time = perf_counter() - t_diag_start
    print(f"\nCompleted {len(results)} instances in {total_diag_time:.1f}s ({total_diag_time/60.0:.2f} min).")

    # Summarize metrics at 80s and 160s
    n = len(results)
    sat_succ_80 = sum(1 for r in results if r["sat_only"]["success_at_80s"])
    sat_succ_160 = sum(1 for r in results if r["sat_only"]["success_at_160s"])
    rand_succ_80 = sum(1 for r in results if r["random_incremental_sat"]["success_at_80s"])
    rand_succ_160 = sum(1 for r in results if r["random_incremental_sat"]["success_at_160s"])

    rand_attempts_160 = [r["random_incremental_sat"]["candidates_attempted"] for r in results]
    mean_rand_attempts = sum(rand_attempts_160) / n if n > 0 else 0.0

    summary = {
        "schema": "neural_sat_speck.phase9_equal_time_summary.v1",
        "phase": 9,
        "study": "phase9_equal_time_diagnostics",
        "instance_count": n,
        "total_wall_seconds": total_diag_time,
        "sat_only": {
            "at_80s": {
                "solved_count": sat_succ_80,
                "success_rate": sat_succ_80 / n if n > 0 else 0.0,
                "wilson_95_ci": list(wilson_score_interval(sat_succ_80, n)),
                "par2_seconds": calculate_global_par2(
                    [r["sat_only"]["t_online"] for r in results],
                    [r["sat_only"]["success_at_80s"] for r in results],
                    global_cutoff=80.0,
                ),
            },
            "at_160s": {
                "solved_count": sat_succ_160,
                "success_rate": sat_succ_160 / n if n > 0 else 0.0,
                "wilson_95_ci": list(wilson_score_interval(sat_succ_160, n)),
                "par2_seconds": calculate_global_par2(
                    [r["sat_only"]["t_online"] for r in results],
                    [r["sat_only"]["success_at_160s"] for r in results],
                    global_cutoff=160.0,
                ),
            },
        },
        "random_incremental_sat": {
            "mean_candidates_attempted_at_160s": mean_rand_attempts,
            "at_80s": {
                "solved_count": rand_succ_80,
                "success_rate": rand_succ_80 / n if n > 0 else 0.0,
                "wilson_95_ci": list(wilson_score_interval(rand_succ_80, n)),
                "par2_seconds": calculate_global_par2(
                    [r["random_incremental_sat"]["t_online"] for r in results],
                    [r["random_incremental_sat"]["success_at_80s"] for r in results],
                    global_cutoff=80.0,
                ),
            },
            "at_160s": {
                "solved_count": rand_succ_160,
                "success_rate": rand_succ_160 / n if n > 0 else 0.0,
                "wilson_95_ci": list(wilson_score_interval(rand_succ_160, n)),
                "par2_seconds": calculate_global_par2(
                    [r["random_incremental_sat"]["t_online"] for r in results],
                    [r["random_incremental_sat"]["success_at_160s"] for r in results],
                    global_cutoff=160.0,
                ),
            },
        },
    }

    summary_path = REPO_ROOT / config["outputs"]["summary_path"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("\n================================================================================")
    print("PHASE 9 EQUAL-TIME DIAGNOSTIC SUMMARY")
    print("================================================================================")
    print(f"SAT-Only at 80s:  {sat_succ_80}/{n} ({sat_succ_80/n*100:.2f}%), PAR-2 (80s):  {summary['sat_only']['at_80s']['par2_seconds']:.2f}s")
    print(f"SAT-Only at 160s: {sat_succ_160}/{n} ({sat_succ_160/n*100:.2f}%), PAR-2 (160s): {summary['sat_only']['at_160s']['par2_seconds']:.2f}s")
    print(f"Random at 80s:    {rand_succ_80}/{n} ({rand_succ_80/n*100:.2f}%), PAR-2 (80s):  {summary['random_incremental_sat']['at_80s']['par2_seconds']:.2f}s")
    print(f"Random at 160s:   {rand_succ_160}/{n} ({rand_succ_160/n*100:.2f}%), PAR-2 (160s): {summary['random_incremental_sat']['at_160s']['par2_seconds']:.2f}s")
    print(f"Random mean candidates attempted within 160s: {mean_rand_attempts:.1f} candidates")
    print(f"[✓] Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
