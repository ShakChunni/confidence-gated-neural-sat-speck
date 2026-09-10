#!/usr/bin/env python3
"""Phase 10-G2 Prospective Validation Pipeline (N=256, Seed 201001).

Strictly follows the frozen prospective manifest:
- Dataset: Seed 201001, N=256, 6 rounds, 64 pairs/instance.
- Frozen B1 Ranker: models/speck32_64_phase6a_b1_mean_g64.pt (batch size 1024).
- Frozen G1-v1 Baseline: models/phase10_promotional_g1_estimator.json (SHA256: 432f997f...).
- Promoted G2-v1 Model: models/phase10_promotional_g2_estimator.json (SHA256: 14c258ab...).
- Partition mask seed: 200030.
- Statistical inference:
  * 100,000-replicate label permutation test for H0: ROC-AUC <= 0.5 (seed 200050).
  * 100,000-replicate paired score-swap permutation test for H0: Delta AUC <= 0 (seed 200050).
  * 10,000-replicate stratified bootstrap for 95% CIs (seed 200040).
- Operating thresholds: theta = 0.0758 (<= 2% budget) and theta = 0.3119 (<= 5% budget).
- Zero post-hoc model retraining, zero parameter tuning, zero threshold adjustments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
MANIFEST_PATH = REPO_ROOT / "configs" / "experiments" / "phase10_g2_prospective_manifest.json"
DATASET_PATH = REPO_ROOT / "datasets" / "phase10_g2_prospective_r6" / "validation.json"
DATASET_MANIFEST_PATH = REPO_ROOT / "datasets" / "phase10_g2_prospective_r6" / "manifest.json"
B1_MODEL_PATH = REPO_ROOT / "models" / "speck32_64_phase6a_b1_mean_g64.pt"
FROZEN_G1_PATH = REPO_ROOT / "models" / "phase10_promotional_g1_estimator.json"
PROMOTED_G2_PATH = REPO_ROOT / "models" / "phase10_promotional_g2_estimator.json"

OUTPUT_DIR = REPO_ROOT / "results" / "phase10_g2_prospective"
CACHE_DIR = OUTPUT_DIR / "ranking_cache"

BOOTSTRAP_SEED = 200040
PERMUTATION_SEED = 200050


# ==============================================================================
# Helper & Inference Functions
# ==============================================================================

def wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z = 1.95996 if confidence == 0.95 else 2.57583
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def rankdata_midranks(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    order = np.argsort(a)
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    unique_vals, inverse = np.unique(a, return_inverse=True)
    if len(unique_vals) < len(a):
        tie_ranks = np.zeros(len(unique_vals), dtype=np.float64)
        for i in range(len(unique_vals)):
            tie_ranks[i] = np.mean(ranks[inverse == i])
        return tie_ranks[inverse]
    return ranks


def compute_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    n1 = int(np.sum(y_true))
    n0 = len(y_true) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    unique_scores, inverse = np.unique(scores, return_inverse=True)
    if len(unique_scores) < len(scores):
        tie_ranks = np.zeros_like(unique_scores, dtype=np.float64)
        for i in range(len(unique_scores)):
            tie_ranks[i] = np.mean(ranks[scores == unique_scores[i]])
        ranks = tie_ranks[inverse]
    sum_r1 = float(np.sum(ranks[y_true == 1]))
    return float((sum_r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def compute_pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    desc_idx = np.argsort(scores)[::-1]
    y_sorted = y_true[desc_idx]
    n_pos = int(np.sum(y_true))
    if n_pos == 0:
        return 0.0
    tps = np.cumsum(y_sorted)
    precisions = tps / np.arange(1, len(y_true) + 1)
    recalls = tps / n_pos
    recalls_diff = np.diff(np.r_[0, recalls])
    return float(np.sum(precisions * recalls_diff))


def compute_brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    return float(np.mean((probs - y_true) ** 2))


def compute_ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10, equal_freq: bool = False) -> float:
    N = len(y_true)
    if equal_freq:
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.quantile(probs, quantiles)
        bin_edges[0] -= 1e-7
        bin_edges[-1] += 1e-7
    else:
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_edges[0] -= 1e-7
        bin_edges[-1] += 1e-7

    ece = 0.0
    for i in range(n_bins):
        idx = np.where((probs > bin_edges[i]) & (probs <= bin_edges[i + 1]))[0]
        if len(idx) > 0:
            bin_acc = float(np.mean(y_true[idx]))
            bin_conf = float(np.mean(probs[idx]))
            ece += (len(idx) / N) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_aurc_and_risks(y_true: np.ndarray, risk_scores: np.ndarray) -> tuple[float, np.ndarray]:
    order = np.argsort(risk_scores)
    y_sorted = y_true[order]
    N = len(y_true)
    risks = np.cumsum(y_sorted) / np.arange(1, N + 1)
    aurc = float(np.mean(risks))
    return aurc, risks


def run_stratified_bootstrap_eval(
    y_true: np.ndarray,
    scores_g2: np.ndarray,
    scores_g1: np.ndarray,
    n_reps: int = 10000,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    rng = np.random.RandomState(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    boot_auc_g2 = np.empty(n_reps)
    boot_auc_g1 = np.empty(n_reps)
    boot_pr_g2 = np.empty(n_reps)
    boot_pr_g1 = np.empty(n_reps)

    for b in range(n_reps):
        b_pos = rng.choice(pos_idx, size=n_pos, replace=True)
        b_neg = rng.choice(neg_idx, size=n_neg, replace=True)
        b_idx = np.concatenate([b_pos, b_neg])

        yb = y_true[b_idx]
        sg2 = scores_g2[b_idx]
        sg1 = scores_g1[b_idx]

        boot_auc_g2[b] = compute_roc_auc(yb, sg2)
        boot_auc_g1[b] = compute_roc_auc(yb, sg1)
        boot_pr_g2[b] = compute_pr_auc(yb, sg2)
        boot_pr_g1[b] = compute_pr_auc(yb, sg1)

    boot_delta_auc = boot_auc_g2 - boot_auc_g1
    boot_delta_pr = boot_pr_g2 - boot_pr_g1

    return {
        "auc_g2_ci": [float(np.percentile(boot_auc_g2, 2.5)), float(np.percentile(boot_auc_g2, 97.5))],
        "auc_g1_ci": [float(np.percentile(boot_auc_g1, 2.5)), float(np.percentile(boot_auc_g1, 97.5))],
        "delta_auc_ci": [float(np.percentile(boot_delta_auc, 2.5)), float(np.percentile(boot_delta_auc, 97.5))],
        "pr_g2_ci": [float(np.percentile(boot_pr_g2, 2.5)), float(np.percentile(boot_pr_g2, 97.5))],
        "pr_g1_ci": [float(np.percentile(boot_pr_g1, 2.5)), float(np.percentile(boot_pr_g1, 97.5))],
        "delta_pr_ci": [float(np.percentile(boot_delta_pr, 2.5)), float(np.percentile(boot_delta_pr, 97.5))],
    }


def run_label_permutation_test(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_perms: int = 100000,
    seed: int = PERMUTATION_SEED,
) -> float:
    """One-sided permutation test for H0: ROC-AUC <= 0.5."""
    obs_auc = compute_roc_auc(y_true, scores)
    rng = np.random.RandomState(seed)
    y_copy = y_true.copy()
    count = 0
    for _ in range(n_perms):
        rng.shuffle(y_copy)
        perm_auc = compute_roc_auc(y_copy, scores)
        if perm_auc >= obs_auc - 1e-12:
            count += 1
    return float(count / n_perms)


def run_paired_score_swap_test(
    y_true: np.ndarray,
    scores_cand: np.ndarray,
    scores_base: np.ndarray,
    n_perms: int = 100000,
    seed: int = PERMUTATION_SEED,
) -> tuple[float, float]:
    """Two-sided paired score-swap permutation test for H0: Delta AUC = 0."""
    N = len(y_true)
    n1 = int(np.sum(y_true))
    n0 = N - n1
    denom = float(n1 * n0)
    if denom == 0.0:
        return 0.0, 1.0
    sub_c = float(n1 * (n1 + 1) / 2.0)
    y_bool = (y_true == 1)

    def _auc(s: np.ndarray) -> float:
        order = np.argsort(s)
        ranks = np.empty(N, dtype=np.float64)
        ranks[order] = np.arange(1, N + 1)
        return (np.sum(ranks[y_bool]) - sub_c) / denom

    obs_delta = _auc(scores_cand) - _auc(scores_base)
    rng = np.random.RandomState(seed)

    count = 0
    block_size = 2000
    for start in range(0, n_perms, block_size):
        curr_b = min(block_size, n_perms - start)
        swaps = rng.randint(0, 2, size=(curr_b, N))
        for i in range(curr_b):
            sw = swaps[i]
            s_c = np.where(sw, scores_base, scores_cand)
            s_b = np.where(sw, scores_cand, scores_base)
            d = _auc(s_c) - _auc(s_b)
            if abs(d) >= abs(obs_delta) - 1e-12:
                count += 1

    return float(obs_delta), float(count / n_perms)


def predict_frozen_g1(g1_feature_dict: dict[str, float], g1_artifact: dict) -> float:
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


def predict_promoted_g2(g2_feature_dict: dict[str, float], g2_artifact: dict) -> float:
    feat_names = g2_artifact["feature_names"]
    means = np.array(g2_artifact["mean"], dtype=np.float64)
    stds = np.array(g2_artifact["std"], dtype=np.float64)
    weights = np.array(g2_artifact["weights"], dtype=np.float64)
    platt_a = float(g2_artifact["platt_a"])
    platt_b = float(g2_artifact["platt_b"])

    x = np.array([g2_feature_dict[name] for name in feat_names], dtype=np.float64)
    x_norm = (x - means) / (stds + 1e-12)
    raw_score = float(np.dot(x_norm, weights))
    z = platt_a * raw_score + platt_b
    z = max(-20.0, min(20.0, z))
    return float(1.0 / (1.0 + math.exp(-z)))


# ==============================================================================
# Main Pipeline
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 10-G2 Prospective Validation Pipeline")
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    print("================================================================================")
    print("=== Phase 10-G2 Prospective Validation Pipeline (N=256, Seed=201001) ===")
    print("================================================================================")
    print(f"Device: {device}, Candidate batch size: {args.batch_size}")
    assert args.batch_size == 1024, "Candidate batch size must be 1024 for numerical invariance"

    # 1. Verify Frozen Model Artifacts
    print("\n--- 1. VERIFYING FROZEN MODEL ARTIFACTS ---")
    with open(FROZEN_G1_PATH, encoding="utf-8") as f:
        g1_artifact = json.load(f)
    with open(FROZEN_G1_PATH, "rb") as f:
        g1_sha256 = hashlib.sha256(f.read()).hexdigest()
    assert g1_sha256 == "432f997f56d7c6fbcfe5b9c0e9dff4918ab73dd265c4d514bf0c56c30ff8872c", f"G1 hash mismatch: {g1_sha256}"
    print(f"[✓] Frozen G1-v1 SHA256 verified: {g1_sha256}")

    with open(PROMOTED_G2_PATH, encoding="utf-8") as f:
        g2_artifact = json.load(f)
    with open(PROMOTED_G2_PATH, "rb") as f:
        g2_sha256 = hashlib.sha256(f.read()).hexdigest()
    assert g2_sha256 == "14c258abc664bc90390aa31f1ad2b3a153db3539954ae3668dad04a18ed388f6", f"G2 hash mismatch: {g2_sha256}"
    print(f"[✓] Promoted G2-v1 SHA256 verified: {g2_sha256}")

    with open(B1_MODEL_PATH, "rb") as f:
        b1_sha256 = hashlib.sha256(f.read()).hexdigest()
    assert b1_sha256 == "a93d29a76a521b6f3fd0f1dff943df9f74de3857169b9d735c9aa478817954d5", f"B1 hash mismatch: {b1_sha256}"
    print(f"[✓] Frozen B1 Ranker SHA256 verified: {b1_sha256}")

    # 2. Load Dataset Shard
    print("\n--- 2. LOADING PROSPECTIVE VALIDATION DATASET ---")
    dataset = load_attack_shard(
        DATASET_PATH,
        expected_dataset_id="speck32_64_attack_phase10_g2_prospective_r6",
    )
    instances = dataset.instances
    if args.limit:
        instances = instances[:args.limit]
    N = len(instances)
    if not args.limit:
        assert N == 256, f"Expected 256 validation instances, found {N}"
    print(f"[✓] Successfully loaded {N} prospective validation attack instances from {DATASET_PATH}.")

    # 3. Setup Models & Masks
    b1_ckpt = torch.load(B1_MODEL_PATH, map_location="cpu", weights_only=False)
    b1_model = PairGroupRanker(variant=b1_ckpt["variant"])
    b1_model.load_state_dict(b1_ckpt["state_dict"])
    b1_model.to(device)
    b1_model.eval()

    masks = DeterministicPartitionMasks.generate(seed=200030)
    idx_s16_0 = torch.tensor(masks.s16_0, device=device, dtype=torch.long)
    idx_s16_1 = torch.tensor(masks.s16_1, device=device, dtype=torch.long)
    idx_s16_2 = torch.tensor(masks.s16_2, device=device, dtype=torch.long)
    idx_s16_3 = torch.tensor(masks.s16_3, device=device, dtype=torch.long)
    idx_s32_0 = torch.tensor(masks.s32_0, device=device, dtype=torch.long)
    idx_s32_1 = torch.tensor(masks.s32_1, device=device, dtype=torch.long)

    candidates = list(range(65536))
    num_candidates = len(candidates)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # 4. Precompute Rankings & Features
    print(f"\n--- 3. EXECUTING PROSPECTIVE INFERENCE & FEATURE EXTRACTION (N={N}) ---")
    records = []
    start_all = time.perf_counter()

    for idx, inst in enumerate(instances):
        cache_path = CACHE_DIR / f"instance_{idx:04d}.json"
        if cache_path.exists():
            try:
                with open(cache_path, encoding="utf-8") as f:
                    rec = json.load(f)
                if "g2_features" in rec and "g2_prob" in rec and "g1_prob" in rec:
                    records.append(rec)
                    continue
            except Exception:
                pass

        t0 = time.perf_counter()
        scores_all = np.zeros((9, num_candidates), dtype=np.float32)

        with torch.no_grad():
            for b_start in range(0, num_candidates, args.batch_size):
                b_end = min(b_start + args.batch_size, num_candidates)
                c_batch = candidates[b_start:b_end]

                feat_arr = vectorized_final_round_features(inst.pairs, c_batch)
                feat_tensor = torch.from_numpy(feat_arr).to(device)

                emb = b1_model.pair_embeddings(feat_tensor)
                b_sz = emb.shape[0]

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
                )

                logits = b1_model.head(stacked.view(b_sz * 9, 64)).view(b_sz, 9).transpose(0, 1)
                scores_all[:, b_start:b_end] = logits.cpu().numpy()

        s64 = scores_all[6]
        order_64 = sorted(candidates, key=lambda c: (-s64[c], c))

        true_key = inst.final_round_subkey
        true_rank = order_64.index(true_key) + 1
        y_fail = 1 if true_rank > 16 else 0

        # G2 Features (views 0..6)
        scores_7 = scores_all[:7]
        g2_feats = extract_g2_feature_vector(scores_7)
        g2_prob = predict_promoted_g2(g2_feats, g2_artifact)

        # G1 Features (views 6, 8, 7)
        s32_g1 = scores_all[8]
        s16_g1 = scores_all[7]
        order_32_g1 = sorted(candidates, key=lambda c: (-s32_g1[c], c))
        order_16_g1 = sorted(candidates, key=lambda c: (-s16_g1[c], c))

        r64_obj = SimpleNamespace(candidate_order=tuple(order_64), candidate_scores=s64)
        r32_obj = SimpleNamespace(candidate_order=tuple(order_32_g1))
        r16_obj = SimpleNamespace(candidate_order=tuple(order_16_g1))

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
        g1_prob = predict_frozen_g1(g1_feats, g1_artifact)

        elapsed = time.perf_counter() - t0

        record = {
            "index": idx,
            "instance_id": inst.instance_id,
            "true_subkey": f"0x{true_key:04x}",
            "b1_true_rank": true_rank,
            "y_fail": y_fail,
            "elapsed_seconds": elapsed,
            "g2_features": g2_feats,
            "g1_features": g1_feats,
            "g2_prob": g2_prob,
            "g1_prob": g1_prob,
        }

        temp_path = cache_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        temp_path.replace(cache_path)
        records.append(record)

        if (idx + 1) % 10 == 0 or idx == 0 or idx == N - 1:
            tot_el = time.perf_counter() - start_all
            avg_t = tot_el / (idx + 1)
            est_rem = (N - (idx + 1)) * avg_t
            print(
                f"[{idx + 1:3d}/{N}] Rank: {true_rank:5d} | Fail: {y_fail} | "
                f"G1 prob: {g1_prob:.4f} | G2 prob: {g2_prob:.4f} | "
                f"Inst time: {elapsed:.2f}s | Est remaining: {est_rem / 60:.1f}m"
            )

    print(f"\n[✓] Completed feature extraction for all {N} instances in {(time.perf_counter() - start_all)/60:.2f} minutes.")

    # 5. Extract Arrays
    y_fail_arr = np.array([r["y_fail"] for r in records], dtype=np.int64)
    b1_ranks_arr = np.array([r["b1_true_rank"] for r in records], dtype=np.int64)
    g1_probs_arr = np.array([r["g1_prob"] for r in records], dtype=np.float64)
    g2_probs_arr = np.array([r["g2_prob"] for r in records], dtype=np.float64)

    n_fail = int(np.sum(y_fail_arr))
    n_succ = N - n_fail
    fail_rate = n_fail / N

    print("\n--- 4. PROSPECTIVE VALIDATION LABEL & RANK DISTRIBUTION ---")
    print(f"Sample Size (N)             : {N}")
    print(f"Success Count (rank <= 16)  : {n_succ} ({n_succ / N * 100:.2f}%)")
    print(f"Failure Count (rank > 16)   : {n_fail} ({fail_rate * 100:.2f}%)")
    w_low, w_high = wilson_ci(n_fail, N)
    print(f"Observed Failure Rate       : {fail_rate * 100:.2f}% [Wilson 95% CI: {w_low * 100:.2f}%, {w_high * 100:.2f}%]")

    q1, med, q3 = np.percentile(b1_ranks_arr, [25, 50, 75])
    print(f"B1 Subkey Rank Summary      : Min: {np.min(b1_ranks_arr)}, Q1: {q1:.1f}, Med: {med:.1f}, Mean: {np.mean(b1_ranks_arr):.2f}, Q3: {q3:.1f}, Max: {np.max(b1_ranks_arr)}")
    print(f"Rank = 1                    : {np.sum(b1_ranks_arr == 1)} ({np.sum(b1_ranks_arr == 1)/N*100:.2f}%)")
    print(f"Rank <= 16                  : {np.sum(b1_ranks_arr <= 16)} ({np.sum(b1_ranks_arr <= 16)/N*100:.2f}%)")
    print(f"Rank 17..32 (Rescuable tail): {np.sum((b1_ranks_arr >= 17) & (b1_ranks_arr <= 32))} ({np.sum((b1_ranks_arr >= 17) & (b1_ranks_arr <= 32))/N*100:.2f}%)")
    print(f"Rank > 32 (Deep tail)       : {np.sum(b1_ranks_arr > 32)} ({np.sum(b1_ranks_arr > 32)/N*100:.2f}%)")

    # 6. Primary Statistical Inference
    print("\n--- 5. PRIMARY STATISTICAL INFERENCE (N=256) ---")
    # ROC-AUC
    auc_g1 = compute_roc_auc(y_fail_arr, g1_probs_arr)
    auc_g2 = compute_roc_auc(y_fail_arr, g2_probs_arr)
    delta_auc = auc_g2 - auc_g1

    # PR-AUC
    pr_g1 = compute_pr_auc(y_fail_arr, g1_probs_arr)
    pr_g2 = compute_pr_auc(y_fail_arr, g2_probs_arr)
    delta_pr = pr_g2 - pr_g1

    # Brier & ECE
    brier_g1 = compute_brier_score(y_fail_arr, g1_probs_arr)
    brier_g2 = compute_brier_score(y_fail_arr, g2_probs_arr)
    ece_ew_g1 = compute_ece(y_fail_arr, g1_probs_arr, n_bins=10, equal_freq=False)
    ece_ew_g2 = compute_ece(y_fail_arr, g2_probs_arr, n_bins=10, equal_freq=False)
    ece_ef_g1 = compute_ece(y_fail_arr, g1_probs_arr, n_bins=10, equal_freq=True)
    ece_ef_g2 = compute_ece(y_fail_arr, g2_probs_arr, n_bins=10, equal_freq=True)

    # AURC
    aurc_g1, _ = compute_aurc_and_risks(y_fail_arr, g1_probs_arr)
    aurc_g2, _ = compute_aurc_and_risks(y_fail_arr, g2_probs_arr)

    # Bootstrap CIs (10,000 reps)
    print("Running 10,000-replicate stratified bootstrap...")
    boot_res = run_stratified_bootstrap_eval(
        y_fail_arr, g2_probs_arr, g1_probs_arr, n_reps=10000, seed=BOOTSTRAP_SEED
    )

    # Permutation Tests (100,000 reps)
    print("Running 100,000-replicate permutation tests...")
    p_perm_g1_above_half = run_label_permutation_test(y_fail_arr, g1_probs_arr, n_perms=100000, seed=PERMUTATION_SEED)
    p_perm_g2_above_half = run_label_permutation_test(y_fail_arr, g2_probs_arr, n_perms=100000, seed=PERMUTATION_SEED + 1)
    obs_swap_delta, p_perm_swap = run_paired_score_swap_test(
        y_fail_arr, g2_probs_arr, g1_probs_arr, n_perms=100000, seed=PERMUTATION_SEED + 2
    )

    print("\n--- RESULTS SUMMARY TABLE ---")
    print(f"{'Metric':<28} | {'Frozen G1-v1':<24} | {'Promoted G2-v1':<24} | {'Difference (G2 - G1)':<24}")
    print("-" * 105)
    print(
        f"{'ROC-AUC':<28} | {auc_g1:.4f} [{boot_res['auc_g1_ci'][0]:.4f}, {boot_res['auc_g1_ci'][1]:.4f}] | "
        f"{auc_g2:.4f} [{boot_res['auc_g2_ci'][0]:.4f}, {boot_res['auc_g2_ci'][1]:.4f}] | "
        f"{delta_auc:+.4f} [{boot_res['delta_auc_ci'][0]:+.4f}, {boot_res['delta_auc_ci'][1]:+.4f}]"
    )
    print(
        f"{'PR-AUC':<28} | {pr_g1:.4f} [{boot_res['pr_g1_ci'][0]:.4f}, {boot_res['pr_g1_ci'][1]:.4f}] | "
        f"{pr_g2:.4f} [{boot_res['pr_g2_ci'][0]:.4f}, {boot_res['pr_g2_ci'][1]:.4f}] | "
        f"{delta_pr:+.4f} [{boot_res['delta_pr_ci'][0]:+.4f}, {boot_res['delta_pr_ci'][1]:+.4f}]"
    )
    print(f"{'One-sided Perm p (AUC > 0.5)':<28} | {p_perm_g1_above_half:.5f}                  | {p_perm_g2_above_half:.5f}                  | —")
    print(f"{'Paired Permutation p (swap)':<28} | —                        | —                        | p = {p_perm_swap:.5f}")
    print(f"{'Brier Score':<28} | {brier_g1:.4f}                   | {brier_g2:.4f}                   | {brier_g2 - brier_g1:+.4f}")
    print(f"{'ECE (Equal-Width)':<28} | {ece_ew_g1:.4f}                   | {ece_ew_g2:.4f}                   | {ece_ew_g2 - ece_ew_g1:+.4f}")
    print(f"{'ECE (Equal-Freq)':<28} | {ece_ef_g1:.4f}                   | {ece_ef_g2:.4f}                   | {ece_ef_g2 - ece_ef_g1:+.4f}")
    print(f"{'AURC':<28} | {aurc_g1:.4f}                   | {aurc_g2:.4f}                   | {aurc_g2 - aurc_g1:+.4f}")

    # 7. Operating Threshold Evaluation
    print("\n--- 6. PROSPECTIVELY FROZEN OPERATING THRESHOLDS EVALUATION ---")
    thresholds = [
        ("Theta 1 (Conservative Budget <= 2%)", 0.0758),
        ("Theta 2 (Balanced Budget <= 5%)", 0.3119),
        ("Phase 10 Baseline Theta (G1)", 0.0890),
    ]

    thresh_results = {}
    for label, th in thresholds:
        print(f"\n{label} [theta = {th:.4f}]:")
        for m_name, probs in [("Frozen G1-v1", g1_probs_arr), ("Promoted G2-v1", g2_probs_arr)]:
            trusted_mask = (probs < th)
            n_trusted = int(np.sum(trusted_mask))
            n_escalated = N - n_trusted
            cov = n_trusted / N

            # False trust = actual failure that was trusted (risk)
            ft = int(np.sum((probs < th) & (y_fail_arr == 1)))
            risk = ft / n_trusted if n_trusted > 0 else 0.0
            r_low, r_high = wilson_ci(ft, n_trusted)

            # Failure recall = fraction of actual failures escalated
            esc_fails = int(np.sum((probs >= th) & (y_fail_arr == 1)))
            recall = esc_fails / n_fail if n_fail > 0 else 0.0

            key_res = f"{m_name}_{th:.4f}"
            thresh_results[key_res] = {
                "threshold": th,
                "model": m_name,
                "coverage_count": n_trusted,
                "coverage_pct": cov * 100.0,
                "false_trust_count": ft,
                "false_trust_risk": risk * 100.0,
                "risk_wilson_ci": [r_low * 100.0, r_high * 100.0],
                "escalated_count": n_escalated,
                "escalation_pct": n_escalated / N * 100.0,
                "failure_escalation_recall": recall * 100.0,
            }

            print(
                f"  {m_name:<16s} | Trusted Coverage: {n_trusted:3d}/{N} ({cov*100:5.2f}%) | "
                f"False Trust: {ft:2d} (Risk: {risk*100:5.2f}% [{r_low*100:4.2f}%, {r_high*100:4.2f}%]) | "
                f"Escalated: {n_escalated:3d} | Failure Recall: {recall*100:5.2f}% ({esc_fails}/{n_fail})"
            )

    # 8. Selective Routing Table
    print("\n--- 7. PROSPECTIVE SELECTIVE ROUTING TABLE ---")
    coverages = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    print(f"{'Coverage':8s} | {'Metric':16s} | {'Frozen G1-v1':16s} | {'Promoted G2-v1':16s}")
    print("-" * 65)
    routing_data = {"Frozen_G1": {}, "Promoted_G2": {}}
    for cov in coverages:
        k = max(1, int(round(cov * N)))
        row_risk = []
        row_ft = []
        row_prec = []
        for m_key, probs in [("Frozen_G1", g1_probs_arr), ("Promoted_G2", g2_probs_arr)]:
            order = np.argsort(probs)
            trusted_y = y_fail_arr[order[:k]]
            ft = int(np.sum(trusted_y))
            risk = ft / k
            prec = 1.0 - risk
            routing_data[m_key][cov] = {"risk": risk, "false_trust": ft, "precision": prec, "k": k}
            row_risk.append(f"{risk*100:6.2f}%")
            row_ft.append(f"{ft:6d}")
            row_prec.append(f"{prec*100:6.2f}%")

        print(f"{cov*100:5.0f}%   | {'Risk (Fail Rate)':16s} | " + " | ".join(f"{v:>16s}" for v in row_risk))
        print(f"{'':8s} | {'False Trust':16s} | " + " | ".join(f"{v:>16s}" for v in row_ft))
        print(f"{'':8s} | {'Precision':16s} | " + " | ".join(f"{v:>16s}" for v in row_prec))

    # 9. Save Atomic Artifacts
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_manifest_path": str(DATASET_MANIFEST_PATH),
        "sample_size": N,
        "n_fail": n_fail,
        "n_success": n_succ,
        "observed_failure_rate": fail_rate,
        "failure_rate_wilson_ci": [w_low, w_high],
        "b1_rank_summary": {
            "min": int(np.min(b1_ranks_arr)),
            "q1": float(q1),
            "median": float(med),
            "mean": float(np.mean(b1_ranks_arr)),
            "q3": float(q3),
            "max": int(np.max(b1_ranks_arr)),
            "count_rank_1": int(np.sum(b1_ranks_arr == 1)),
            "count_rank_le_16": int(np.sum(b1_ranks_arr <= 16)),
            "count_rank_17_32": int(np.sum((b1_ranks_arr >= 17) & (b1_ranks_arr <= 32))),
            "count_rank_gt_32": int(np.sum(b1_ranks_arr > 32)),
        },
        "statistical_results": {
            "g1_v1": {
                "roc_auc": auc_g1,
                "roc_auc_ci": boot_res["auc_g1_ci"],
                "pr_auc": pr_g1,
                "pr_auc_ci": boot_res["pr_g1_ci"],
                "p_perm_auc_gt_half": p_perm_g1_above_half,
                "brier": brier_g1,
                "ece_equal_width": ece_ew_g1,
                "ece_equal_freq": ece_ef_g1,
                "aurc": aurc_g1,
            },
            "g2_v1": {
                "roc_auc": auc_g2,
                "roc_auc_ci": boot_res["auc_g2_ci"],
                "pr_auc": pr_g2,
                "pr_auc_ci": boot_res["pr_g2_ci"],
                "p_perm_auc_gt_half": p_perm_g2_above_half,
                "brier": brier_g2,
                "ece_equal_width": ece_ew_g2,
                "ece_equal_freq": ece_ef_g2,
                "aurc": aurc_g2,
            },
            "paired_comparison": {
                "delta_roc_auc": delta_auc,
                "delta_roc_auc_ci": boot_res["delta_auc_ci"],
                "delta_pr_auc": delta_pr,
                "delta_pr_auc_ci": boot_res["delta_pr_ci"],
                "p_perm_score_swap": p_perm_swap,
            },
        },
        "threshold_evaluations": thresh_results,
        "selective_routing": routing_data,
    }

    report_path = OUTPUT_DIR / "phase10_g2_prospective_evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)
    print(f"\n[✓] Saved prospective evaluation report to {report_path}")

    preds_path = OUTPUT_DIR / "prospective_predictions.json"
    preds_dict = {
        "instance_ids": [r["instance_id"] for r in records],
        "true_subkeys": [r["true_subkey"] for r in records],
        "b1_true_ranks": [int(r["b1_true_rank"]) for r in records],
        "y_fail": [int(y) for y in y_fail_arr],
        "g1_probs": [float(p) for p in g1_probs_arr],
        "g2_probs": [float(p) for p in g2_probs_arr],
    }
    with open(preds_path, "w", encoding="utf-8") as f:
        json.dump(preds_dict, f, indent=2)
    print(f"[✓] Saved prospective predictions to {preds_path}")


if __name__ == "__main__":
    main()
