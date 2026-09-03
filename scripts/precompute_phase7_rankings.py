#!/usr/bin/env python3
"""Precompute and cache B0/B1 neural rankings for Phase 7 held-out instances.

This script isolates the neural inference computation from SAT solving, allowing
deterministic, parallel caching of complete 65,536-candidate rankings across the 64
held-out test instances (seed 98001). Cached rankings are stored in
results/phase7_final_evaluation/ranking_cache/instance_{index:04d}.json.
"""

# ruff: noqa: E402
from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import json
import multiprocessing as mp
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

torch.set_num_threads(1)

from neural_sat_speck.attacks.diversity_fallback import (
    choose_phase6c_route,
    compose_diversity_fallback,
)
from neural_sat_speck.attacks.group_ranking import (
    evaluate_pair_group_subkey_ranking,
    rank_pair_group_subkeys,
)
from neural_sat_speck.attacks.ranking import (
    evaluate_subkey_ranking,
    rank_final_round_subkeys,
)
from neural_sat_speck.confidence.calibration import (
    PlattCalibrator,
    top8_score_separation,
)
from neural_sat_speck.data.attack import AttackInstance, load_attack_shard
from neural_sat_speck.neural.group_model import PairGroupRanker
from neural_sat_speck.neural.model import GohrResNet
from neural_sat_speck.neural.numpy_inference import (
    NumpyInferenceModule,
    NumpyRankerInference,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiments" / "phase7_final_evaluation.json"
DATASET_PATH = REPO_ROOT / "datasets" / "phase7_held_out_r6" / "test.json"
CACHE_DIR = REPO_ROOT / "results" / "phase7_final_evaluation" / "ranking_cache"


def _write_atomic(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as tmp:
        json.dump(data, tmp, indent=2)
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def _process_instance(args: tuple[int, AttackInstance, Path, Path]) -> None:
    index, inst, b0_path, b1_path = args
    cache_file = CACHE_DIR / f"instance_{index:04d}.json"
    if cache_file.is_file():
        print(f"[{index:02d}/64] Already cached: {cache_file.name}", flush=True)
        return

    started = time.perf_counter()
    torch.set_num_threads(1)
    print(f"[{index:02d}/64] Starting neural ranking for test-{index:04d}...", flush=True)

    # Load worker-local models
    b0_torch = GohrResNet()
    b0_torch.load_state_dict(torch.load(b0_path, weights_only=False)["state_dict"])
    b0_torch.eval()
    b0_infer = NumpyInferenceModule(NumpyRankerInference(b0_torch))

    b1_data = torch.load(b1_path, weights_only=False)
    b1_torch = PairGroupRanker(variant=b1_data["variant"])
    b1_torch.load_state_dict(b1_data["state_dict"])
    b1_torch.eval()
    b1_infer = NumpyInferenceModule(NumpyRankerInference(b1_torch))

    # Calibrator setup
    calibrator = PlattCalibrator(
        feature_mean=4.12053918838501,
        feature_std=0.5482019782066345,
        intercept=0.5789959917731415,
        slope=0.07917109093865415,
        l2_lambda=1.0,
        fit_count=64,
        positive_count=41,
        iterations=10,
        gradient_tolerance=1e-10,
        step_tolerance=1e-12,
    )

    # B0 ranking
    t0 = time.perf_counter()
    b0_ranking = rank_final_round_subkeys(inst, b0_infer, candidate_batch_size=1024)
    t_rank_b0 = time.perf_counter() - t0
    b0_order = list(b0_ranking.candidate_order)
    b0_eval = evaluate_subkey_ranking(b0_ranking, inst.final_round_subkey)

    # B1 ranking
    t0 = time.perf_counter()
    b1_ranking = rank_pair_group_subkeys(inst, b1_infer, candidate_batch_size=1024)
    t_rank_b1 = time.perf_counter() - t0
    b1_order = list(b1_ranking.candidate_order)
    b1_eval = evaluate_pair_group_subkey_ranking(b1_ranking, inst.final_round_subkey)

    # Confidence calculation
    t0 = time.perf_counter()
    b1_dict = {
        "ranking": {
            "candidate_values": list(b1_ranking.candidate_values),
            "candidate_scores": list(b1_ranking.candidate_scores),
            "candidate_order": b1_order,
        }
    }
    conf_feature = top8_score_separation(b1_dict)
    prob = calibrator.predict(conf_feature)
    t_conf = time.perf_counter() - t0

    # Decision route & Fallback composition
    route = choose_phase6c_route(prob, 0.5)
    if route == "trusted_top8":
        gated_order = b1_order[:8]
        gated_budget = 8
        t_gated_rank = t_rank_b1
    else:
        gated_order = list(compose_diversity_fallback(tuple(b1_order), tuple(b0_order)))
        gated_budget = 32
        t_gated_rank = t_rank_b1 + t_rank_b0

    record = {
        "instance_index": index,
        "instance_id": inst.instance_id,
        "true_final_round_subkey": inst.final_round_subkey,
        "b0_order": b0_order,
        "t_rank_b0": t_rank_b0,
        "b0_eval": b0_eval,
        "b1_order": b1_order,
        "t_rank_b1": t_rank_b1,
        "b1_eval": b1_eval,
        "conf_feature": conf_feature,
        "prob": prob,
        "route": route,
        "gated_order": gated_order,
        "gated_budget": gated_budget,
        "t_gated_rank": t_gated_rank,
        "t_conf": t_conf,
    }

    _write_atomic(cache_file, record)
    elapsed = time.perf_counter() - started
    print(
        f"[{index:02d}/64] ✓ Completed test-{index:04d} in {elapsed:.1f}s "
        f"(B0 true rank={b0_eval['true_rank']}, B1 true rank={b1_eval['true_rank']})",
        flush=True,
    )
    import gc
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute Phase 7 Neural Rankings")
    parser.add_argument("--workers", type=int, default=2, help="Parallel worker count")
    parser.add_argument("--max-instances", type=int, default=64, help="Maximum instances to process")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    shard = load_attack_shard(DATASET_PATH)
    instances = shard.instances[: args.max_instances]

    b0_path = REPO_ROOT / "models" / "speck32_64_phase3_r5_distinguisher.pt"
    b1_path = REPO_ROOT / "models" / "speck32_64_phase6a_b1_mean_g64.pt"

    tasks = [
        (idx, inst, b0_path, b1_path)
        for idx, inst in enumerate(instances)
    ]

    print(f"Precomputing rankings for {len(tasks)} instances using {args.workers} workers...")
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers) as pool:
        pool.map(_process_instance, tasks)

    print("\n[✓] All Phase 7 rankings successfully precomputed and cached.")


if __name__ == "__main__":
    main()
