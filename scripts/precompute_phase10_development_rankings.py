#!/usr/bin/env python3
"""Precompute B1 multi-subset rankings and extract G0/G1 features for Phase 10 Development.

This script executes rank_pair_group_multi_subsets on all 192 Phase 10 development
instances using the frozen B1 model on MPS/CPU with embedding reuse.
For each instance:
- Ranks full 65,536 candidate domain on 16, 32, and 64 pairs.
- Computes true-subkey rank in B1.
- Computes failure target y_fail = 1 iff true-subkey rank > 16.
- Extracts frozen 7 G1 features and G0 score-margin baseline.
- Saves atomic JSON records in results/phase10_development/ranking_cache/.
"""

from __future__ import annotations

import dataclasses
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from neural_sat_speck.attacks.group_ranking import rank_pair_group_multi_subsets
from neural_sat_speck.confidence.stability import (
    extract_g1_features_from_rankings,
    score_curvature,
)
from neural_sat_speck.data.attack import load_attack_shard
from neural_sat_speck.neural.group_model import PairGroupRanker

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = REPO_ROOT / "datasets" / "phase10_development_r6" / "development.json"
B1_MODEL_PATH = REPO_ROOT / "models" / "speck32_64_phase6a_b1_mean_g64.pt"
CACHE_DIR = REPO_ROOT / "results" / "phase10_development" / "ranking_cache"
SUMMARY_PATH = REPO_ROOT / "results" / "phase10_development" / "ranking_summary.json"


def main() -> None:
    print("=== Precomputing Phase 10 Development Rankings (N=192) ===")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Dataset
    print(f"Loading development dataset from {DATASET_PATH}...")
    instance_set = load_attack_shard(
        DATASET_PATH,
        expected_dataset_id="speck32_64_attack_phase10_development_r6",
    )
    instances = instance_set.instances
    print(f"Loaded {len(instances)} attack instances.")

    # 2. Setup Device and Load Model
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using compute device: {device}")

    b1_ckpt = torch.load(B1_MODEL_PATH, map_location="cpu", weights_only=False)
    model = PairGroupRanker(variant=b1_ckpt["variant"])
    model.load_state_dict(b1_ckpt["state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded frozen B1 model from {B1_MODEL_PATH}")

    t_start_all = time.perf_counter()
    summary_records: list[dict[str, Any]] = []

    reverse_order = "--reverse" in sys.argv
    items = list(reversed(list(enumerate(instances)))) if reverse_order else list(enumerate(instances))
    print(f"Processing order: {'REVERSE (191 -> 0)' if reverse_order else 'FORWARD (0 -> 191)'}")

    for idx, inst in items:
        cache_file = CACHE_DIR / f"instance_{idx:04d}.json"
        if cache_file.exists():
            try:
                rec = json.loads(cache_file.read_text(encoding="utf-8"))
                summary_records.append(rec)
                print(f"[{idx+1:03d}/{len(instances)}] Loaded from cache: {inst.instance_id} (true rank {rec['b1_true_rank']})")
                continue
            except Exception:
                pass

        t0 = time.perf_counter()
        multi_subsets = rank_pair_group_multi_subsets(
            inst,
            model,
            subset_sizes=(16, 32, 64),
            candidate_batch_size=1024,
            device=device,
        )
        elapsed = time.perf_counter() - t0

        r64 = multi_subsets[64]
        r32 = multi_subsets[32]
        r16 = multi_subsets[16]

        # Extract true subkey rank in B1 (64 pairs)
        true_key = inst.final_round_subkey
        true_rank_b1 = r64.candidate_order.index(true_key) + 1
        y_fail = 1 if true_rank_b1 > 16 else 0

        # Extract G1 features
        g1_feats = extract_g1_features_from_rankings(r64, r32, r16)

        # Extract G0 feature: score curvature
        g0_val = score_curvature(r64.candidate_scores, top_k=16)

        rec = {
            "index": idx,
            "instance_id": inst.instance_id,
            "true_subkey": f"0x{true_key:04x}",
            "b1_true_rank": true_rank_b1,
            "y_fail": y_fail,
            "b1_top16_containment": bool(true_rank_b1 <= 16),
            "b1_top32_containment": bool(true_rank_b1 <= 32),
            "b1_top64_containment": bool(true_rank_b1 <= 64),
            "b1_top128_containment": bool(true_rank_b1 <= 128),
            "ranking_seconds": elapsed,
            "g0_curvature": g0_val,
            "g1_features": dataclasses.asdict(g1_feats),
            "b1_candidate_order_top128": list(r64.candidate_order[:128]),
            "b1_candidate_order_top16": list(r64.candidate_order[:16]),
        }

        # Write atomic record
        cache_file.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
        summary_records.append(rec)

        print(
            f"[{idx+1:03d}/{len(instances)}] {inst.instance_id}: "
            f"B1 rank={true_rank_b1:4d} (y_fail={y_fail}) in {elapsed:.2f}s | "
            f"Jaccard={g1_feats.top16_jaccard_32_64:.3f}"
        )

        # Per-instance memory cleanup
        if idx % 10 == 0:
            gc.collect()

    total_time = time.perf_counter() - t_start_all
    failure_count = sum(r["y_fail"] for r in summary_records)
    print(f"\nAll 192 rankings complete in {total_time:.1f}s ({total_time/60:.1f}m)!")
    print(f"Total B1 top-16 failures: {failure_count} / {len(summary_records)} ({failure_count/len(summary_records)*100:.1f}%)")

    # Shortlist availability ceilings
    c32 = sum(1 for r in summary_records if r["y_fail"] == 1 and r["b1_top32_containment"])
    c64 = sum(1 for r in summary_records if r["y_fail"] == 1 and r["b1_top64_containment"])
    c128 = sum(1 for r in summary_records if r["y_fail"] == 1 and r["b1_top128_containment"])

    print(f"B1 Failures in top-32:  {c32}/{failure_count} ({c32/failure_count*100:.1f}%)")
    print(f"B1 Failures in top-64:  {c64}/{failure_count} ({c64/failure_count*100:.1f}%)")
    print(f"B1 Failures in top-128: {c128}/{failure_count} ({c128/failure_count*100:.1f}%)")

    summary = {
        "dataset": str(DATASET_PATH.relative_to(REPO_ROOT)),
        "instances_total": len(summary_records),
        "failures_total": failure_count,
        "failure_prevalence": failure_count / len(summary_records),
        "total_ranking_seconds": total_time,
        "availability_ceiling_m32": c32 / failure_count if failure_count > 0 else 0.0,
        "availability_ceiling_m64": c64 / failure_count if failure_count > 0 else 0.0,
        "availability_ceiling_m128": c128 / failure_count if failure_count > 0 else 0.0,
        "records": summary_records,
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved ranking summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
