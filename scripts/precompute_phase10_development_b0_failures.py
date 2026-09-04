#!/usr/bin/env python3
"""Precompute B0 rankings specifically for Phase 10 development failure instances.

Under the lazy fallback protocol, B0 ranking is ONLY computed if primary SAT verification fails.
In the development set of 192 keys, exactly 17 instances fail B1 top-16 containment.
This script computes B0 rankings for those 17 instances using the frozen B0 model on MPS,
caching results in results/phase10_development/b0_fallback_cache/instance_{index:04d}.json.
Supports --reverse for parallel 2-worker execution.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import torch

from neural_sat_speck.attacks.ranking import rank_final_round_subkeys
from neural_sat_speck.data.attack import load_attack_shard
from neural_sat_speck.neural.model import GohrResNet

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = REPO_ROOT / "datasets" / "phase10_development_r6" / "development.json"
B0_MODEL_PATH = REPO_ROOT / "models" / "speck32_64_phase3_r5_distinguisher.pt"
CACHE_DIR = REPO_ROOT / "results" / "phase10_development" / "b0_fallback_cache"
SUMMARY_PATH = REPO_ROOT / "results" / "phase10_development" / "ranking_summary.json"


def main() -> None:
    print("=== Precomputing B0 Rankings for Failure Instances (N=17) ===")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    failure_indices = [r["index"] for r in summary["records"] if r["y_fail"] == 1]
    print(f"Targeting {len(failure_indices)} failure instances: {failure_indices}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using compute device: {device}")

    b0_torch = GohrResNet()
    ckpt = torch.load(B0_MODEL_PATH, map_location="cpu", weights_only=False)
    b0_torch.load_state_dict(ckpt["state_dict"])
    b0_torch.to(device)
    b0_torch.eval()

    instances = load_attack_shard(
        DATASET_PATH,
        expected_dataset_id="speck32_64_attack_phase10_development_r6",
    ).instances

    items = list(reversed(failure_indices)) if "--reverse" in sys.argv else failure_indices
    order_label = "REVERSE" if "--reverse" in sys.argv else "FORWARD"
    print(f"Running order: {order_label}")

    for idx in items:
        cache_file = CACHE_DIR / f"instance_{idx:04d}.json"
        if cache_file.exists():
            print(f"Index {idx:3d} already cached, skipping.")
            continue

        inst = instances[idx]
        t0 = time.perf_counter()
        ranking = rank_final_round_subkeys(inst, b0_torch, candidate_batch_size=1024, device=device)
        elapsed = time.perf_counter() - t0

        true_key = inst.final_round_subkey
        true_rank = ranking.candidate_order.index(true_key) + 1

        rec = {
            "index": idx,
            "instance_id": inst.instance_id,
            "true_subkey": f"0x{true_key:04x}",
            "b0_true_rank": true_rank,
            "b0_top16_containment": bool(true_rank <= 16),
            "b0_top32_containment": bool(true_rank <= 32),
            "ranking_seconds": elapsed,
            "b0_candidate_order_top64": list(ranking.candidate_order[:64]),
        }
        cache_file.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
        print(f"[{order_label}] Index {idx:3d} ({inst.instance_id}): B0 rank={true_rank:4d} in {elapsed:.1f}s")
        gc.collect()

    print("B0 failure precomputation completed.")


if __name__ == "__main__":
    main()
