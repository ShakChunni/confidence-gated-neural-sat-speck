#!/usr/bin/env python3
"""Run prospective Phase 10 Verification-Triggered Cascaded Fallback (VTC) validation.

Memory-safe, resource-bounded execution:
- Default: single worker (sequential, peak RAM < 800 MB).
- Explicit garbage collection (gc.collect()) per instance.
- Atomic per-instance resume from disk (zero duplicate work).
- Unbuffered real-time stdout logging.

Evaluates on fresh validation seed 97001 (N=64 independent master keys):
1. Independent Baseline (B0) + SAT (k=16)
2. Pair-Aware Deep Sets (B1) + SAT (k=16)
3. Eager Dual-Ranking Diversity Fallback + SAT (k=32)
4. Verification-Triggered Cascaded Fallback (VTC-B1-B0) + SAT (k_max=32)

Invariant: Every compared method receives a newly constructed, independent
SpeckSatModel and distinct pycryptosat.Solver instance (cross-method solver isolation).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neural_sat_speck.attacks.diversity_fallback import compose_diversity_fallback
from neural_sat_speck.attacks.group_ranking import (
    evaluate_pair_group_subkey_ranking,
    rank_pair_group_subkeys,
)
from neural_sat_speck.attacks.ranking import (
    evaluate_subkey_ranking,
    rank_final_round_subkeys,
)
from neural_sat_speck.attacks.recovery import (
    independent_verification_pairs,
    known_pairs_for_attack_instance,
    run_candidate_recovery,
)
from neural_sat_speck.attacks.vtc import run_vtc_attack
from neural_sat_speck.data.attack import (
    AttackInstance,
    AttackInstanceConfig,
    generate_attack_instances,
    load_attack_shard,
    write_attack_shard,
)
from neural_sat_speck.evaluation.metrics import (
    calculate_global_par2,
    exact_mcnemar_test,
    wilson_score_interval,
)
from neural_sat_speck.neural.group_model import PairGroupRanker
from neural_sat_speck.neural.model import GohrResNet
from neural_sat_speck.neural.numpy_inference import (
    NumpyInferenceModule,
    NumpyRankerInference,
)
from neural_sat_speck.sat.model import SpeckSatModel

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiments" / "phase10_vtc_validation_r6.json"
MANIFEST_SCHEMA = "neural_sat_speck.dataset_manifest.v1"
RESULT_SCHEMA = "neural_sat_speck.phase10_instance_result.v1"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _git_commit() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def generate_phase10_dataset(config: Mapping[str, Any]) -> Path:
    dataset_cfg = config["dataset"]
    manifest_path = REPO_ROOT / dataset_cfg["manifest_path"]
    output_dir = REPO_ROOT / dataset_cfg["output_dir"]

    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        val_json = output_dir / "validation.json"
        if val_json.exists():
            return manifest_path

    print(f"Generating Phase 10 validation dataset (seed {dataset_cfg['seed']})...", flush=True)
    input_diff = tuple(dataset_cfg["input_difference"])
    instance_config = AttackInstanceConfig(
        instance_count=int(dataset_cfg["instance_count"]),
        pair_count=int(dataset_cfg["pair_count"]),
        rounds=int(dataset_cfg["rounds"]),
        seed=int(dataset_cfg["seed"]),
        split_role="validation",
        input_difference=(int(input_diff[0]), int(input_diff[1])),
    )
    collection = generate_attack_instances(
        instance_config,
        generated_at=str(dataset_cfg.get("generated_at", "2026-09-01T00:00:00+08:00")),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    val_json = output_dir / "validation.json"
    write_attack_shard(collection, val_json)

    manifest = {
        "dataset_id": dataset_cfg["dataset_id"],
        "schema": MANIFEST_SCHEMA,
        "phase": 10,
        "validation_only": True,
        "held_out_test_generated": False,
        "rounds": instance_config.rounds,
        "pair_count": instance_config.pair_count,
        "instance_count": instance_config.instance_count,
        "seed": instance_config.seed,
        "input_difference": list(instance_config.input_difference),
        "splits": {
            "validation": {
                "split_role": "validation",
                "instance_count": instance_config.instance_count,
                "pair_count": instance_config.pair_count,
                "seed": instance_config.seed,
                "path": "validation.json",
                "file_sha256": _sha256_file(val_json),
            }
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Generated Phase 10 validation dataset in {output_dir}", flush=True)
    return manifest_path


def evaluate_phase10_single_instance(
    instance: AttackInstance,
    index: int,
    b0_infer: NumpyInferenceModule,
    b1_infer: NumpyInferenceModule,
    config: Mapping[str, Any],
    manifest_sha256: str,
    config_sha256: str,
) -> dict[str, Any]:
    solver_cfg = config["solver"]
    rounds = int(config["dataset"]["rounds"])
    timeout = float(solver_cfg["solver_timeout_seconds"])

    # 1. Observations
    all_pairs = known_pairs_for_attack_instance(instance)
    known_pairs = all_pairs[: int(solver_cfg["known_pairs_used"])]
    verif_seed = int(config["dataset"]["seed"]) + 10000 + index
    verif_pairs = independent_verification_pairs(
        instance.master_key,
        rounds=rounds,
        pair_count=int(solver_cfg["independent_verification_pairs_count"]),
        seed=verif_seed,
    )

    # 2. Pre-run rankings once per instance to obtain exact rankings & timings
    b0_ranking = rank_final_round_subkeys(instance, b0_infer, candidate_batch_size=4096)
    t_rank_b0 = b0_ranking.total_seconds
    b0_order = b0_ranking.candidate_order
    b0_eval = evaluate_subkey_ranking(b0_ranking, instance.final_round_subkey)

    b1_ranking = rank_pair_group_subkeys(instance, b1_infer, candidate_batch_size=4096)
    t_rank_b1 = b1_ranking.total_seconds
    b1_order = b1_ranking.candidate_order
    b1_eval = evaluate_pair_group_subkey_ranking(b1_ranking, instance.final_round_subkey)

    # 3. Method 1: b0_sat (Fresh Solver)
    m_b0 = SpeckSatModel.from_known_pairs(known_pairs, rounds=rounds)
    rec_b0 = run_candidate_recovery(
        m_b0,
        b0_order,
        known_pairs,
        verif_pairs,
        candidate_budget=16,
        solver_timeout=timeout,
        strategy="b0_sat",
    )
    t_online_b0 = t_rank_b0 + rec_b0.solver_seconds + rec_b0.verification_seconds
    b0_data = {
        "success": rec_b0.success,
        "status": rec_b0.status,
        "exact_true_rank": b0_eval["true_rank"],
        "candidates_attempted": rec_b0.candidates_attempted,
        "winning_candidate": rec_b0.winning_candidate,
        "winning_candidate_position": rec_b0.winning_candidate_position,
        "t_rank": t_rank_b0,
        "t_sat": rec_b0.solver_seconds,
        "t_ver": rec_b0.verification_seconds,
        "t_online": t_online_b0,
    }

    # 4. Method 2: b1_sat (Fresh Solver)
    m_b1 = SpeckSatModel.from_known_pairs(known_pairs, rounds=rounds)
    rec_b1 = run_candidate_recovery(
        m_b1,
        b1_order,
        known_pairs,
        verif_pairs,
        candidate_budget=16,
        solver_timeout=timeout,
        strategy="b1_sat",
    )
    t_online_b1 = t_rank_b1 + rec_b1.solver_seconds + rec_b1.verification_seconds
    b1_data = {
        "success": rec_b1.success,
        "status": rec_b1.status,
        "exact_true_rank": b1_eval["true_rank"],
        "candidates_attempted": rec_b1.candidates_attempted,
        "winning_candidate": rec_b1.winning_candidate,
        "winning_candidate_position": rec_b1.winning_candidate_position,
        "t_rank": t_rank_b1,
        "t_sat": rec_b1.solver_seconds,
        "t_ver": rec_b1.verification_seconds,
        "t_online": t_online_b1,
    }

    # 5. Method 3: eager_b1_b0_fallback_sat (Fresh Solver)
    m_eager = SpeckSatModel.from_known_pairs(known_pairs, rounds=rounds)
    eager_order = compose_diversity_fallback(b1_order, b0_order)
    rec_eager = run_candidate_recovery(
        m_eager,
        eager_order,
        known_pairs,
        verif_pairs,
        candidate_budget=32,
        solver_timeout=timeout,
        strategy="eager_b1_b0_fallback_sat",
    )
    t_online_eager = (
        t_rank_b1 + t_rank_b0 + rec_eager.solver_seconds + rec_eager.verification_seconds
    )
    eager_data = {
        "success": rec_eager.success,
        "status": rec_eager.status,
        "candidates_attempted": rec_eager.candidates_attempted,
        "winning_candidate": rec_eager.winning_candidate,
        "winning_candidate_position": rec_eager.winning_candidate_position,
        "candidates_sequence": list(eager_order[: rec_eager.candidates_attempted]),
        "t_rank_b1": t_rank_b1,
        "t_rank_b0": t_rank_b0,
        "t_rank_total": t_rank_b1 + t_rank_b0,
        "t_sat": rec_eager.solver_seconds,
        "t_ver": rec_eager.verification_seconds,
        "t_online": t_online_eager,
    }

    # 6. Method 4: vtc_b1_b0_sat (Fresh Solver, Lazy B0)
    vtc_result = run_vtc_attack(
        instance,
        b1_infer=b1_infer,
        b0_infer=b0_infer,
        known_pairs=known_pairs,
        verification_pairs=verif_pairs,
        rounds=rounds,
        k1_budget=16,
        b0_allowance=16,
        kmax_budget=32,
        solver_timeout=timeout,
        precomputed_b1_ranking=b1_ranking,
        precomputed_b0_ranking=b0_ranking,
        strategy="vtc_b1_b0_sat",
    )
    vtc_data = vtc_result.to_dict()

    # Solver isolation assertions
    models = [m_b0, m_b1, m_eager]
    model_ids = {id(m) for m in models}
    solver_ids = {id(m.solver) for m in models}
    if len(model_ids) != 3 or len(solver_ids) != 3:
        raise RuntimeError("Solver isolation violation: shared model or solver detected!")

    # Explicit memory cleanup
    del m_b0, m_b1, m_eager, b0_ranking, b1_ranking, vtc_result
    gc.collect()

    return {
        "schema": RESULT_SCHEMA,
        "instance_id": instance.instance_id,
        "instance_index": index,
        "master_key": list(instance.master_key),
        "true_final_round_subkey": instance.final_round_subkey,
        "provenance": {
            "config_sha256": config_sha256,
            "manifest_sha256": manifest_sha256,
            "git_commit": _git_commit(),
            "platform": platform.platform(),
            "python_version": sys.version,
        },
        "methods": {
            "b0_sat": b0_data,
            "b1_sat": b1_data,
            "eager_b1_b0_fallback_sat": eager_data,
            "vtc_b1_b0_sat": vtc_data,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 10 VTC Validation")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--smoke", action="store_true", help="Run 2 instances for smoke check")
    parser.add_argument("--workers", type=int, default=1, help="Worker count (default 1 for strict low-memory execution)")
    args = parser.parse_args()

    config_path = args.config
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    if int(config["dataset"]["seed"]) != 97001:
        raise ValueError("Phase 10 MUST use validation seed 97001 only!")

    manifest_path = generate_phase10_dataset(config)
    manifest_sha256 = _sha256_file(manifest_path)
    config_sha256 = _sha256_file(config_path)

    shard_path = manifest_path.parent / "validation.json"
    collection = load_attack_shard(shard_path)
    instances = collection.instances

    if args.smoke:
        instances = instances[:2]
        print(f"[SMOKE MODE] Running {len(instances)} instances...", flush=True)
    else:
        print(f"Running Phase 10 Validation on {len(instances)} instances (workers={args.workers}, low-memory profile)...", flush=True)

    models_cfg = config["models"]
    b0_path = REPO_ROOT / models_cfg["b0"]["checkpoint_path"]
    b1_path = REPO_ROOT / models_cfg["b1"]["checkpoint_path"]

    print("Loading frozen neural models (B0 and B1)...", flush=True)
    b0_torch = GohrResNet()
    b0_torch.load_state_dict(torch.load(b0_path, weights_only=False)["state_dict"])
    b0_torch.eval()
    b0_infer = NumpyInferenceModule(NumpyRankerInference(b0_torch))

    b1_data = torch.load(b1_path, weights_only=False)
    b1_torch = PairGroupRanker(variant=b1_data["variant"])
    b1_torch.load_state_dict(b1_data["state_dict"])
    b1_torch.eval()
    b1_infer = NumpyInferenceModule(NumpyRankerInference(b1_torch))

    records_dir = REPO_ROOT / config["outputs"]["records_dir"]
    records_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    t_start = time.perf_counter()

    for idx, inst in enumerate(instances):
        rec_path = records_dir / f"instance_{idx:04d}.json"
        if rec_path.exists() and not args.smoke:
            with open(rec_path, encoding="utf-8") as f:
                res = json.load(f)
            results.append(res)
            print(f"[{len(results)}/{len(instances)}] Loaded existing record for instance {idx:04d}", flush=True)
            continue

        res = evaluate_phase10_single_instance(
            inst,
            idx,
            b0_infer,
            b1_infer,
            config,
            manifest_sha256,
            config_sha256,
        )
        results.append(res)

        temp_path = rec_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(res, indent=2))
        temp_path.replace(rec_path)

        vtc_info = res["methods"]["vtc_b1_b0_sat"]
        eager_info = res["methods"]["eager_b1_b0_fallback_sat"]
        print(
            f"[{len(results)}/{len(instances)}] Instance {idx:04d}: "
            f"B1 {'SOLVED' if res['methods']['b1_sat']['success'] else 'FAIL'}, "
            f"B0 {'SOLVED' if res['methods']['b0_sat']['success'] else 'FAIL'}, "
            f"Eager {'SOLVED' if eager_info['success'] else 'FAIL'} ({eager_info['t_online']:.1f}s), "
            f"VTC {'SOLVED' if vtc_info['success'] else 'FAIL'} (Stage: {vtc_info['winning_stage'] or 'none'}, B0-invoked: {vtc_info['b0_invoked']}, {vtc_info['timing_seconds']['t_online']:.1f}s)",
            flush=True,
        )
        gc.collect()

    results.sort(key=lambda r: r["instance_index"])
    total_wall_seconds = time.perf_counter() - t_start
    print(f"\nPhase 10 Execution Complete: {len(results)} instances in {total_wall_seconds:.1f}s ({total_wall_seconds/60.0:.2f} min).", flush=True)

    if args.smoke:
        print("[SMOKE MODE COMPLETE] Exiting without overwriting summary.", flush=True)
        return

    # Compute comprehensive audit summary
    n = len(results)
    method_names = ["b0_sat", "b1_sat", "eager_b1_b0_fallback_sat", "vtc_b1_b0_sat"]
    methods_summary: dict[str, Any] = {}

    for m in method_names:
        solved = sum(1 for r in results if r["methods"][m]["success"])
        rate = solved / n if n > 0 else 0.0
        ci = wilson_score_interval(solved, n)
        attempts = [r["methods"][m]["candidates_attempted"] for r in results]
        times = [r["methods"][m]["timing_seconds"]["t_online"] if "timing_seconds" in r["methods"][m] else r["methods"][m]["t_online"] for r in results]
        methods_summary[m] = {
            "solved_count": solved,
            "success_rate": rate,
            "wilson_95_ci": list(ci),
            "candidates_attempted": {
                "mean": float(np.mean(attempts)),
                "median": float(np.median(attempts)),
                "iqr": float(np.percentile(attempts, 75) - np.percentile(attempts, 25)),
            },
            "online_seconds": {
                "mean": float(np.mean(times)),
                "median": float(np.median(times)),
                "iqr": float(np.percentile(times, 75) - np.percentile(times, 25)),
            },
            "par2_seconds_200s": calculate_global_par2(
                times,
                [r["methods"][m]["success"] for r in results],
                global_cutoff=200.0,
            ),
        }

    # Paired McNemar Comparisons
    def _paired_stats(m_a: str, m_b: str) -> dict[str, Any]:
        succ_a = [r["methods"][m_a]["success"] for r in results]
        succ_b = [r["methods"][m_b]["success"] for r in results]
        n11 = sum(1 for a, b in zip(succ_a, succ_b) if a and b)
        n10 = sum(1 for a, b in zip(succ_a, succ_b) if a and not b)
        n01 = sum(1 for a, b in zip(succ_a, succ_b) if not a and b)
        n00 = sum(1 for a, b in zip(succ_a, succ_b) if not a and not b)
        mcnemar = exact_mcnemar_test(succ_a, succ_b)
        return {
            "table_2x2": {
                "both_solved_11": n11,
                f"{m_a}_only_10": n10,
                f"{m_b}_only_01": n01,
                "both_failed_00": n00,
            },
            "discordant_pairs": mcnemar["discordant_pairs"],
            "net_difference": mcnemar["net_difference"],
            "exact_p_value": mcnemar["exact_p_value"],
        }

    paired_comparisons = {
        "b1_vs_b0": _paired_stats("b1_sat", "b0_sat"),
        "vtc_vs_b1": _paired_stats("vtc_b1_b0_sat", "b1_sat"),
        "vtc_vs_eager": _paired_stats("vtc_b1_b0_sat", "eager_b1_b0_fallback_sat"),
        "vtc_vs_b0": _paired_stats("vtc_b1_b0_sat", "b0_sat"),
    }

    # VTC Escalation Analysis
    vtc_records = [r["methods"]["vtc_b1_b0_sat"] for r in results]
    escalated_count = sum(1 for v in vtc_records if v["b0_invoked"])
    escalated_rate = escalated_count / n if n > 0 else 0.0
    rescued_count = sum(1 for v in vtc_records if v["b0_invoked"] and v["success"])
    rescue_rate_conditional = rescued_count / escalated_count if escalated_count > 0 else 0.0

    escalation_analysis = {
        "escalated_count": escalated_count,
        "escalation_rate": escalated_rate,
        "escalation_wilson_95_ci": list(wilson_score_interval(escalated_count, n)),
        "rescued_count": rescued_count,
        "rescue_rate_conditional": rescue_rate_conditional,
        "rescue_wilson_95_ci": list(wilson_score_interval(rescued_count, escalated_count)) if escalated_count > 0 else None,
    }

    # Paired Wall-Clock Delta Analysis: delta = T_online(VTC) - T_online(Eager)
    deltas = [
        r["methods"]["vtc_b1_b0_sat"]["timing_seconds"]["t_online"] - r["methods"]["eager_b1_b0_fallback_sat"]["t_online"]
        for r in results
    ]
    vtc_faster_count = sum(1 for d in deltas if d < 0)

    paired_runtime_analysis = {
        "mean_delta_seconds": float(np.mean(deltas)),
        "median_delta_seconds": float(np.median(deltas)),
        "iqr_delta_seconds": float(np.percentile(deltas, 75) - np.percentile(deltas, 25)),
        "vtc_faster_count": vtc_faster_count,
        "vtc_faster_proportion": vtc_faster_count / n if n > 0 else 0.0,
    }

    # Candidate-Set Equivalence Analysis
    jaccard_list = []
    for r in results:
        eager_cand_set = set(r["methods"]["eager_b1_b0_fallback_sat"]["candidates_sequence"])
        vtc_cand_set = set(r["methods"]["vtc_b1_b0_sat"]["cumulative_candidates_sequence"])
        union = eager_cand_set | vtc_cand_set
        inter = eager_cand_set & vtc_cand_set
        jaccard = len(inter) / len(union) if len(union) > 0 else 1.0
        jaccard_list.append(jaccard)

    candidate_set_analysis = {
        "mean_candidate_set_jaccard": float(np.mean(jaccard_list)),
        "min_candidate_set_jaccard": float(np.min(jaccard_list)),
        "max_candidate_set_jaccard": float(np.max(jaccard_list)),
    }

    summary = {
        "schema": "neural_sat_speck.phase10_vtc_summary.v1",
        "phase": 10,
        "study": "phase10_vtc_validation",
        "dataset_seed": 97001,
        "instance_count": n,
        "total_execution_wall_seconds": total_wall_seconds,
        "methods": methods_summary,
        "paired_comparisons": paired_comparisons,
        "vtc_escalation": escalation_analysis,
        "paired_runtime_delta": paired_runtime_analysis,
        "candidate_set_analysis": candidate_set_analysis,
    }

    summary_path = REPO_ROOT / config["outputs"]["summary_path"]
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary written to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
