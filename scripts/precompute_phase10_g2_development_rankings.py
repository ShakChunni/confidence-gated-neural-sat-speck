#!/usr/bin/env python3
"""Precompute B1 Rankings, G2 Features, and Frozen G1 Features for Phase 10-G2 Development.

Strictly adheres to:
- Candidate batch size = 1024 (MPS numerical invariance).
- Seed 200030 for deterministic random balanced partition masks (CORRECTION 6).
- Shift-invariant D1..D7 features (CORRECTION 5).
- Frozen G1-v1 evaluation using exact artifact weights (CORRECTION 1).
- Atomic per-instance JSON caching in results/phase10_g2_development/ranking_cache/.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from neural_sat_speck.attacks.group_ranking import (
    PairGroupRanker,
    vectorized_final_round_features,
)
from neural_sat_speck.confidence.g2_features import (
    DeterministicPartitionMasks,
    extract_g2_feature_vector,
)
from neural_sat_speck.confidence.stability import extract_g1_features_from_rankings
from neural_sat_speck.data.attack import load_attack_shard

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = REPO_ROOT / "datasets" / "phase10_g2_development_r6" / "development.json"
B1_MODEL_PATH = REPO_ROOT / "models" / "speck32_64_phase6a_b1_mean_g64.pt"
FROZEN_G1_PATH = REPO_ROOT / "models" / "phase10_promotional_g1_estimator.json"
CACHE_DIR = REPO_ROOT / "results" / "phase10_g2_development" / "ranking_cache"


def load_b1_model(device: torch.device) -> PairGroupRanker:
    """Load the frozen Phase 6A B1 ranker."""
    b1_ckpt = torch.load(B1_MODEL_PATH, map_location="cpu", weights_only=False)
    model = PairGroupRanker(variant=b1_ckpt["variant"])
    model.load_state_dict(b1_ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model


def load_frozen_g1_estimator() -> dict:
    with open(FROZEN_G1_PATH, encoding="utf-8") as f:
        return json.load(f)


def predict_frozen_g1(g1_feature_dict: dict[str, float], g1_artifact: dict) -> float:
    """Evaluate frozen G1-v1 on feature dictionary without refitting."""
    feat_names = g1_artifact["feature_names"]
    means = np.array(g1_artifact["mean"], dtype=np.float64)
    stds = np.array(g1_artifact["std"], dtype=np.float64)
    weights = np.array(g1_artifact["weights"], dtype=np.float64)
    bias = float(g1_artifact["bias"])

    x = np.array([g1_feature_dict[name] for name in feat_names], dtype=np.float64)
    x_norm = (x - means) / (stds + 1e-12)
    z = float(np.dot(x_norm, weights) + bias)
    z = max(-20.0, min(20.0, z))
    return float(1.0 / (1.0 + math.exp(-z)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute B1 rankings and features for G2 development")
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}, candidate batch size: {args.batch_size}")

    assert args.batch_size == 1024, "Candidate batch size must be 1024 for numerical invariance"

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset
    print(f"Loading dataset from {DATASET_PATH}...")
    dataset = load_attack_shard(
        DATASET_PATH,
        expected_dataset_id="speck32_64_attack_phase10_g2_development_r6",
    )
    all_instances = dataset.instances
    end_idx = args.end if args.end is not None else len(all_instances)
    if args.limit:
        end_idx = min(end_idx, args.start + args.limit)
    instances = [(i, all_instances[i]) for i in range(args.start, end_idx)]
    print(f"Targeting {len(instances)} instances (range {args.start} to {end_idx}).")

    # 2. Load models
    print(f"Loading B1 model from {B1_MODEL_PATH}...")
    model = load_b1_model(device)
    frozen_g1 = load_frozen_g1_estimator()

    # 3. Setup partition masks
    masks = DeterministicPartitionMasks.generate(seed=200030)
    idx_s16_0 = torch.tensor(masks.s16_0, device=device, dtype=torch.long)
    idx_s16_1 = torch.tensor(masks.s16_1, device=device, dtype=torch.long)
    idx_s16_2 = torch.tensor(masks.s16_2, device=device, dtype=torch.long)
    idx_s16_3 = torch.tensor(masks.s16_3, device=device, dtype=torch.long)
    idx_s32_0 = torch.tensor(masks.s32_0, device=device, dtype=torch.long)
    idx_s32_1 = torch.tensor(masks.s32_1, device=device, dtype=torch.long)

    candidates = list(range(65536))
    num_candidates = len(candidates)

    print("Starting precomputation...")
    start_all = time.perf_counter()

    for loop_i, (idx, inst) in enumerate(instances):
        cache_path = CACHE_DIR / f"instance_{idx:04d}.json"
        if cache_path.exists():
            try:
                with open(cache_path, encoding="utf-8") as f:
                    rec = json.load(f)
                if "g2_features" in rec and "D1_top1_robust_margin" in rec["g2_features"] and "g1_features" in rec:
                    continue
            except Exception:
                pass

        t0 = time.perf_counter()

        # Score arrays: 9 views
        # 0: s16_0, 1: s16_1, 2: s16_2, 3: s16_3, 4: s32_0, 5: s32_1, 6: s64
        # 7: g1_16 (0:16), 8: g1_32 (0:32)
        scores_all = np.zeros((9, num_candidates), dtype=np.float32)

        with torch.no_grad():
            for b_start in range(0, num_candidates, args.batch_size):
                b_end = min(b_start + args.batch_size, num_candidates)
                c_batch = candidates[b_start:b_end]

                feat_arr = vectorized_final_round_features(inst.pairs, c_batch)
                feat_tensor = torch.from_numpy(feat_arr).to(device)

                # Shared embedding: (B, 64, 64)
                emb = model.pair_embeddings(feat_tensor)
                b_sz = emb.shape[0]

                # Pool 9 views
                pool_0 = emb.index_select(1, idx_s16_0).mean(dim=1)
                pool_1 = emb.index_select(1, idx_s16_1).mean(dim=1)
                pool_2 = emb.index_select(1, idx_s16_2).mean(dim=1)
                pool_3 = emb.index_select(1, idx_s16_3).mean(dim=1)
                pool_4 = emb.index_select(1, idx_s32_0).mean(dim=1)
                pool_5 = emb.index_select(1, idx_s32_1).mean(dim=1)
                pool_6 = emb.mean(dim=1)
                pool_7 = emb[:, :16, :].mean(dim=1)
                pool_8 = emb[:, :32, :].mean(dim=1)

                stacked = torch.stack(
                    [pool_0, pool_1, pool_2, pool_3, pool_4, pool_5, pool_6, pool_7, pool_8],
                    dim=1,
                )  # (B, 9, 64)

                logits = model.head(stacked.view(b_sz * 9, 64)).view(b_sz, 9).transpose(0, 1)
                scores_all[:, b_start:b_end] = logits.cpu().numpy()

        # Sort view 6 (full 64 pairs)
        s64 = scores_all[6]
        order_64 = sorted(candidates, key=lambda c: (-s64[c], c))

        # True rank
        true_key = inst.final_round_subkey
        true_rank = order_64.index(true_key) + 1
        y_fail = 1 if true_rank > 16 else 0

        # Severity bin: 0 (<=8), 1 (9..16), 2 (17..32), 3 (>32)
        if true_rank <= 8:
            y_ordinal = 0
        elif true_rank <= 16:
            y_ordinal = 1
        elif true_rank <= 32:
            y_ordinal = 2
        else:
            y_ordinal = 3

        # Extract G2 features (views 0..6)
        scores_7 = scores_all[:7]
        g2_feats = extract_g2_feature_vector(scores_7)

        # Extract G1 features (views 6, 8, 7)
        s32_g1 = scores_all[8]
        s16_g1 = scores_all[7]
        order_32_g1 = sorted(candidates, key=lambda c: (-s32_g1[c], c))
        order_16_g1 = sorted(candidates, key=lambda c: (-s16_g1[c], c))

        from types import SimpleNamespace
        r64_obj = SimpleNamespace(
            candidate_order=tuple(order_64),
            candidate_scores=s64,
        )
        r32_obj = SimpleNamespace(
            candidate_order=tuple(order_32_g1),
        )
        r16_obj = SimpleNamespace(
            candidate_order=tuple(order_16_g1),
        )

        g1_feats_obj = extract_g1_features_from_rankings(r64_obj, r32_obj, r16_obj)
        g1_feats = {
            "top1_top2_margin": g1_feats_obj.top1_top2_margin,
            "top1_top16_margin": g1_feats_obj.top1_top16_margin,
            "normalized_top16_entropy": g1_feats_obj.normalized_top16_entropy,
            "top16_jaccard_16_32": g1_feats_obj.top16_jaccard_16_32,
            "top16_jaccard_32_64": g1_feats_obj.top16_jaccard_32_64,
            "top1_rank_variance": g1_feats_obj.top1_rank_variance,
            "dispersion_normalized_margin": g1_feats_obj.dispersion_normalized_margin,
        }
        g1_pred_prob = predict_frozen_g1(g1_feats, frozen_g1)

        elapsed = time.perf_counter() - t0

        record = {
            "index": idx,
            "instance_id": inst.instance_id,
            "true_subkey": f"0x{true_key:04x}",
            "b1_true_rank": true_rank,
            "y_fail": y_fail,
            "y_ordinal": y_ordinal,
            "elapsed_seconds": elapsed,
            "g2_features": g2_feats,
            "g1_features": g1_feats,
            "frozen_g1_prob": g1_pred_prob,
        }

        # Write atomically
        temp_path = cache_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        temp_path.replace(cache_path)

        if (loop_i + 1) % 10 == 0 or loop_i == 0 or loop_i == len(instances) - 1:
            total_elapsed = time.perf_counter() - start_all
            avg_per_inst = total_elapsed / (loop_i + 1)
            rem_inst = len(instances) - (loop_i + 1)
            est_rem = rem_inst * avg_per_inst
            print(
                f"[{loop_i + 1}/{len(instances)} (idx {idx})] Rank: {true_rank:5d} | "
                f"Fail: {y_fail} | G1 prob: {g1_pred_prob:.4f} | "
                f"Inst time: {elapsed:.2f}s | Est remaining: {est_rem / 60:.1f}m"
            )

    print("Precomputation finished successfully!")


if __name__ == "__main__":
    main()
