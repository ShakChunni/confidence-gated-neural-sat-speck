#!/usr/bin/env python3
"""Evaluate G2 reliability estimator transfer to Phase 11B-prime Stage-1 ranking.

Applies the frozen, prospectively-validated G2-v1 reliability estimator
(trained on 6-round B1 top-16 failures) to 7-round Stage-1 (k6) ranking
distributions from opened Seed 300001 instances.

Measures:
1. G2 ROC-AUC and PR-AUC for predicting Stage-1 top-16 failure (y_fail = 1[rank_k6 > 16]);
2. G2 ROC-AUC and PR-AUC at alternative beam thresholds (top-8, top-32);
3. Baseline comparison against scalar margin G0 (D3: top1 - top2 gap);
4. Spearman rank correlation between G2 failure risk and true subkey rank.

Diagnostics only: writes to results/phase11_scaling_diagnostics/ and uses
the opened Seed 300001 family.
"""

from __future__ import annotations

import argparse
import json
import math
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

from neural_sat_speck.confidence.g2_features import (  # noqa: E402
    DeterministicPartitionMasks,
    extract_g2_feature_vector,
)
from neural_sat_speck.data.attack import load_attack_shard  # noqa: E402
from neural_sat_speck.neural.group_model import PairGroupRanker  # noqa: E402
from neural_sat_speck.neural.numpy_inference import (  # noqa: E402
    vectorized_final_round_features,
)
from neural_sat_speck.speck.key_schedule import expand_key  # noqa: E402

DATASET_PATH = (
    REPOSITORY_ROOT / "datasets" / "phase11a_oracle_r7" / "oracle_evaluation.json"
)
DATASET_ID = "speck32_64_attack_phase11a_oracle_r7"
STAGE1_MODEL_PATH = (
    REPOSITORY_ROOT / "models" / "speck32_64_phase11b_prime_stage1_r6_b1_mean.pt"
)
G2_ESTIMATOR_PATH = (
    REPOSITORY_ROOT / "models" / "phase10_promotional_g2_estimator.json"
)
RESULTS_DIR = REPOSITORY_ROOT / "results" / "phase11_scaling_diagnostics"
CACHE_DIR = RESULTS_DIR / "stage1_g2_transfer_cache"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=int, default=16, help="Number of instances to evaluate")
    parser.add_argument("--start", type=int, default=0, help="Starting instance offset")
    parser.add_argument("--batch-size", type=int, default=1024, help="Candidate batch size (default 1024)")
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "g2_stage1_transfer_summary.json",
        help="Summary JSON output path",
    )
    return parser.parse_args(argv)


def predict_g2_failure_prob(
    g2_features: dict[str, float],
    g2_artifact: dict[str, Any],
) -> float:
    """Compute monotonic failure probability from 14-feature vector using frozen G2 estimator."""
    feat_names = g2_artifact["feature_names"]
    means = np.array(g2_artifact["mean"], dtype=np.float64)
    stds = np.array(g2_artifact["std"], dtype=np.float64)
    weights = np.array(g2_artifact["weights"], dtype=np.float64)
    platt_a = float(g2_artifact["platt_a"])
    platt_b = float(g2_artifact["platt_b"])

    x = np.array([g2_features[name] for name in feat_names], dtype=np.float64)
    x_norm = (x - means) / (stds + 1e-12)
    raw_score = float(np.dot(x_norm, weights))
    z = platt_a * raw_score + platt_b
    z = max(-20.0, min(20.0, z))
    return float(1.0 / (1.0 + math.exp(-z)))


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC-AUC with deterministic mid-rank tie resolution."""
    pos_count = int(np.sum(y_true == 1))
    neg_count = int(np.sum(y_true == 0))
    if pos_count == 0 or neg_count == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)

    # Tie correction
    unique_scores, inv, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    for idx, count in enumerate(counts):
        if count > 1:
            match_mask = (inv == idx)
            ranks[match_mask] = np.mean(ranks[match_mask])

    pos_rank_sum = float(np.sum(ranks[y_true == 1]))
    auc = (pos_rank_sum - pos_count * (pos_count + 1) / 2.0) / (pos_count * neg_count)
    return float(auc)


def compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute PR-AUC (Average Precision) for failure prediction."""
    pos_count = int(np.sum(y_true == 1))
    if pos_count == 0:
        return float("nan")

    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / (tp + fp)
    recall = tp / pos_count

    # Integrate step-wise precision-recall curve
    # PR-AUC = sum (R_k - R_{k-1}) * P_k
    recall_prev = np.concatenate(([0.0], recall[:-1]))
    pr_auc = float(np.sum((recall - recall_prev) * precision))
    return pr_auc


def compute_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman rank correlation between x and y."""
    n = len(x)
    if n < 2:
        return float("nan")
    rank_x = np.argsort(np.argsort(x)).astype(np.float64)
    rank_y = np.argsort(np.argsort(y)).astype(np.float64)
    d = rank_x - rank_y
    rho = 1.0 - (6.0 * float(np.sum(d**2))) / (n * (n**2 - 1))
    return float(rho)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Load G2 Estimator
    with open(G2_ESTIMATOR_PATH, encoding="utf-8") as f:
        g2_artifact = json.load(f)

    # Load Attack Instances
    shard = load_attack_shard(DATASET_PATH, expected_dataset_id=DATASET_ID)
    instances = shard.instances[args.start : args.start + args.instances]
    if not instances:
        raise SystemExit("No instances selected for evaluation")

    # Load Stage-1 Model
    device = torch.device(args.device)
    print(f"Loading Stage-1 model on {device}...")
    ckpt = torch.load(STAGE1_MODEL_PATH, map_location="cpu", weights_only=False)
    b1_model = PairGroupRanker(variant=ckpt["variant"])
    b1_model.load_state_dict(ckpt["state_dict"])
    b1_model.to(device)
    b1_model.eval()

    # Predeclared deterministic partition masks (seed 200030)
    masks = DeterministicPartitionMasks.generate(seed=200030)
    idx_s16_0 = torch.tensor(masks.s16_0, device=device, dtype=torch.long)
    idx_s16_1 = torch.tensor(masks.s16_1, device=device, dtype=torch.long)
    idx_s16_2 = torch.tensor(masks.s16_2, device=device, dtype=torch.long)
    idx_s16_3 = torch.tensor(masks.s16_3, device=device, dtype=torch.long)
    idx_s32_0 = torch.tensor(masks.s32_0, device=device, dtype=torch.long)
    idx_s32_1 = torch.tensor(masks.s32_1, device=device, dtype=torch.long)

    candidates = list(range(65536))
    num_candidates = len(candidates)

    print(
        f"=== EVALUATING G2 FAILURE TRANSFER ON PHASE 11B' STAGE-1 ===\n"
        f"Instances: {len(instances)} (indices {args.start}..{args.start + len(instances) - 1})\n"
        f"Device: {device}, Batch size: {args.batch_size}\n"
    )

    records: list[dict[str, Any]] = []

    for offset, inst in enumerate(instances):
        inst_idx = args.start + offset
        cache_path = CACHE_DIR / f"instance_{inst_idx:04d}.json"

        if cache_path.exists():
            try:
                with open(cache_path, encoding="utf-8") as f:
                    rec = json.load(f)
                records.append(rec)
                print(
                    f"[{inst_idx:02d}] {inst.instance_id} (CACHED): "
                    f"true_rank={rec['true_rank']:>5} G2_prob={rec['g2_prob']:.4f} "
                    f"y_fail16={rec['y_fail_16']}"
                )
                continue
            except Exception:
                pass

        t0 = time.perf_counter()
        scores_all = np.zeros((7, num_candidates), dtype=np.float32)

        with torch.no_grad():
            for b_start in range(0, num_candidates, args.batch_size):
                b_end = min(b_start + args.batch_size, num_candidates)
                c_batch = candidates[b_start:b_end]

                feat_arr = vectorized_final_round_features(inst.pairs, c_batch)
                feat_tensor = torch.from_numpy(feat_arr).to(device)

                emb = b1_model.pair_embeddings(feat_tensor)
                b_sz = emb.shape[0]

                # 7 partition views
                p0 = emb.index_select(1, idx_s16_0).mean(dim=1)
                p1 = emb.index_select(1, idx_s16_1).mean(dim=1)
                p2 = emb.index_select(1, idx_s16_2).mean(dim=1)
                p3 = emb.index_select(1, idx_s16_3).mean(dim=1)
                p4 = emb.index_select(1, idx_s32_0).mean(dim=1)
                p5 = emb.index_select(1, idx_s32_1).mean(dim=1)
                p6 = emb.mean(dim=1)

                stacked = torch.stack([p0, p1, p2, p3, p4, p5, p6], dim=1)
                logits = b1_model.head(stacked.view(b_sz * 7, 64)).view(b_sz, 7).transpose(0, 1)
                scores_all[:, b_start:b_end] = logits.cpu().numpy()

        elapsed = time.perf_counter() - t0

        # Full 64-pair scores are view 6
        s64 = scores_all[6]
        order_64 = sorted(candidates, key=lambda c: (-s64[c], c))

        true_k6 = expand_key(inst.master_key, rounds=7)[-1]
        true_rank = order_64.index(true_k6) + 1

        g2_feats = extract_g2_feature_vector(scores_all)
        g2_prob = predict_g2_failure_prob(g2_feats, g2_artifact)

        # Baseline G0 scalar margin: D3 (top1 - top2 gap). Invert sign so higher = higher failure risk
        d3_gap = g2_feats["D3_top1_top2_gap"]
        g0_risk = -d3_gap

        rec = {
            "index": inst_idx,
            "instance_id": inst.instance_id,
            "true_final_round_subkey": true_k6,
            "true_rank": true_rank,
            "y_fail_8": int(true_rank > 8),
            "y_fail_16": int(true_rank > 16),
            "y_fail_32": int(true_rank > 32),
            "g2_prob": round(g2_prob, 6),
            "g0_risk": round(g0_risk, 6),
            "g2_features": {k: round(v, 6) for k, v in g2_feats.items()},
            "elapsed_seconds": round(elapsed, 2),
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)

        records.append(rec)
        print(
            f"[{inst_idx:02d}] {inst.instance_id}: true_rank={true_rank:>5} "
            f"G2_prob={g2_prob:.4f} (y_fail16={rec['y_fail_16']}) [{elapsed:5.1f}s]"
        )

    # Compute Aggregate Metrics
    ranks = np.array([r["true_rank"] for r in records])
    g2_probs = np.array([r["g2_prob"] for r in records])
    g0_risks = np.array([r["g0_risk"] for r in records])

    y16 = np.array([r["y_fail_16"] for r in records])
    y8 = np.array([r["y_fail_8"] for r in records])
    y32 = np.array([r["y_fail_32"] for r in records])

    spearman_g2 = compute_spearman(g2_probs, ranks)
    spearman_g0 = compute_spearman(g0_risks, ranks)

    metrics: dict[str, Any] = {
        "n_instances": len(records),
        "rank_distribution": {
            "median": float(np.median(ranks)),
            "mean": float(np.mean(ranks)),
            "min": int(np.min(ranks)),
            "max": int(np.max(ranks)),
            "top_8_containment": int(np.sum(ranks <= 8)),
            "top_16_containment": int(np.sum(ranks <= 16)),
            "top_32_containment": int(np.sum(ranks <= 32)),
        },
        "spearman_rank_correlation": {
            "g2_prob_vs_rank": round(spearman_g2, 4),
            "g0_margin_vs_rank": round(spearman_g0, 4),
        },
        "top16_failure_prediction": {
            "positives": int(np.sum(y16 == 1)),
            "negatives": int(np.sum(y16 == 0)),
            "g2_roc_auc": round(compute_auc(y16, g2_probs), 4),
            "g2_pr_auc": round(compute_pr_auc(y16, g2_probs), 4),
            "g0_roc_auc": round(compute_auc(y16, g0_risks), 4),
            "g0_pr_auc": round(compute_pr_auc(y16, g0_risks), 4),
        },
        "top8_failure_prediction": {
            "positives": int(np.sum(y8 == 1)),
            "negatives": int(np.sum(y8 == 0)),
            "g2_roc_auc": round(compute_auc(y8, g2_probs), 4),
            "g2_pr_auc": round(compute_pr_auc(y8, g2_probs), 4),
        },
        "top32_failure_prediction": {
            "positives": int(np.sum(y32 == 1)),
            "negatives": int(np.sum(y32 == 0)),
            "g2_roc_auc": round(compute_auc(y32, g2_probs), 4),
            "g2_pr_auc": round(compute_pr_auc(y32, g2_probs), 4),
        },
    }

    summary_payload = {
        "experiment": "phase11b_prime_g2_transfer_evaluation",
        "timestamp": datetime.now(UTC).isoformat(),
        "stage1_model": str(STAGE1_MODEL_PATH),
        "g2_estimator": str(G2_ESTIMATOR_PATH),
        "metrics": metrics,
        "records": records,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print("\n=================== G2 TRANSFER SUMMARY ===================")
    print(f"Evaluated Instances: {len(records)}")
    print(f"Top-16 Failures (Positive Class): {metrics['top16_failure_prediction']['positives']}/{len(records)}")
    print(f"G2-v1 ROC-AUC (Top-16): {metrics['top16_failure_prediction']['g2_roc_auc']}")
    print(f"G2-v1 PR-AUC  (Top-16): {metrics['top16_failure_prediction']['g2_pr_auc']}")
    print(f"G0    ROC-AUC (Top-16): {metrics['top16_failure_prediction']['g0_roc_auc']}")
    print(f"Spearman(G2_prob, true_rank): {metrics['spearman_rank_correlation']['g2_prob_vs_rank']}")
    print(f"Saved full summary to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
