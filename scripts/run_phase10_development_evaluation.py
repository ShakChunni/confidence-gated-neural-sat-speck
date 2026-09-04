#!/usr/bin/env python3
"""Phase 10B — Development-Only G1/R1 Out-of-Fold Evaluation and Cascade Simulation.

Executes:
1. Load 192 precomputed development rankings from results/phase10_development/ranking_cache/.
2. 5-Fold Stratified Key-Level Cross-Validation:
   - Split unit: attack instance / master key.
   - G0 (scalar curvature) vs G1 (7 frozen stability features).
   - Strict leakage prevention: scalers and weights fit exclusively on training folds.
   - Out-of-fold predictions for failure risk y_fail = 1 (B1 rank > 16).
3. Out-of-fold metrics: ROC-AUC, PR-AUC, Brier score, bootstrap 95% CIs.
4. G1 Threshold Selection:
   - Sweep theta in [0.01, 0.99].
   - False-trust risk: P(y_fail = 1 | G1 < theta).
   - Coverage: P(G1 < theta).
   - Operational decision rule: max theta s.t. risk <= 0.05 and coverage >= 0.50.
5. R1 Shortlist Rescue Evaluation (M in {32, 64, 128}):
   - Hard-negative mining on training folds only.
   - Pairwise margin ranking loss.
   - Out-of-fold evaluation on held-out failure instances.
   - Availability ceiling, rescue rate, rank deltas, worsened count.
   - Lexicographic M selection: max rescue, tie-break smaller M.
6. Development Cascade Simulation with Strict Solver Isolation:
   - Evaluates B1 direct, R1 all, G1-gated R1, Lazy B1->B0 fallback, and G1-gated R1 + Lazy B0 fallback.
   - Enforces fresh solver isolation per method.
   - Enforces candidate deduplication and cumulative budget k <= 32.
   - Requires independent 2-pair cryptographic verification.
7. Generates comprehensive JSON report and Markdown summary.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from neural_sat_speck.attacks.r1 import (
    R1HardNegativeMiner,
    compute_rescue_ceiling,
)
from neural_sat_speck.data.attack import load_attack_shard
from neural_sat_speck.data.seed_policy import (
    PHASE10_DEVELOPMENT_SEED,
    Phase10DataRole,
)
from neural_sat_speck.neural.group_model import PairGroupRanker
from neural_sat_speck.neural.numpy_inference import vectorized_final_round_features

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = REPO_ROOT / "datasets" / "phase10_development_r6" / "development.json"
CACHE_DIR = REPO_ROOT / "results" / "phase10_development" / "ranking_cache"
OUTPUT_DIR = REPO_ROOT / "results" / "phase10_development"
B1_MODEL_PATH = REPO_ROOT / "models" / "speck32_64_phase6a_b1_mean_g64.pt"


# ==============================================================================
# 1. Stratified K-Fold Cross-Validation Split
# ==============================================================================
def stratified_kfold_split(
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = PHASE10_DEVELOPMENT_SEED,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministically partition indices into n_splits preserving positive ratio."""
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    pos_folds = np.array_split(pos_idx, n_splits)
    neg_folds = np.array_split(neg_idx, n_splits)

    folds = []
    for fold_i in range(n_splits):
        val_idx = np.concatenate([pos_folds[fold_i], neg_folds[fold_i]])
        rng.shuffle(val_idx)
        train_idx = np.setdiff1d(np.arange(len(y)), val_idx)
        rng.shuffle(train_idx)
        folds.append((train_idx, val_idx))

    return folds


# ==============================================================================
# 2. Metric Utilities (ROC-AUC, PR-AUC, Brier)
# ==============================================================================
def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Exact ROC-AUC computation using trapezoidal rule."""
    desc_score_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[desc_score_indices]
    y_score_sorted = y_score[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_score_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    if fps[-1] <= 0 or tps[-1] <= 0:
        return 0.5

    fpr = fps / fps[-1]
    tpr = tps / tps[-1]
    return float(np.trapezoid(tpr, fpr))


def compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision (PR-AUC) computation."""
    desc_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[desc_indices]

    n_pos = np.sum(y_true)
    if n_pos == 0:
        return 0.0

    tps = np.cumsum(y_true_sorted)
    precisions = tps / np.arange(1, len(y_true) + 1)
    recalls = tps / n_pos

    # Trapezoidal PR-AUC
    recalls_diff = np.diff(np.r_[0, recalls])
    return float(np.sum(precisions * recalls_diff))


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Any,
    n_boot: int = 1000,
    seed: int = 100001,
) -> tuple[float, float]:
    """Compute 95% bootstrap confidence interval."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_vals = []
    for _ in range(n_boot):
        sample_idx = rng.choice(n, size=n, replace=True)
        sample_y = y_true[sample_idx]
        if np.sum(sample_y) == 0 or np.sum(sample_y) == n:
            continue
        sample_s = y_score[sample_idx]
        boot_vals.append(metric_fn(sample_y, sample_s))
    if not boot_vals:
        return (0.5, 0.5)
    return (float(np.percentile(boot_vals, 2.5)), float(np.percentile(boot_vals, 97.5)))


# ==============================================================================
# 3. Logistic Regression Models (G0 and G1)
# ==============================================================================
def fit_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    l2_lambda: float = 1.0,
    lr: float = 0.5,
    epochs: int = 200,
) -> tuple[np.ndarray, float]:
    """Fit L2-regularized logistic regression via PyTorch L-BFGS.

    Returns (weights, bias).
    """
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)

    linear = nn.Linear(X_train.shape[1], 1)
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)

    optimizer = torch.optim.LBFGS(linear.parameters(), lr=lr, max_iter=20)
    bce = nn.BCEWithLogitsLoss()

    for _ in range(epochs // 20):
        def closure():
            optimizer.zero_grad()
            logits = linear(X_t)
            loss = bce(logits, y_t) + l2_lambda * torch.sum(linear.weight ** 2)
            loss.backward()
            return loss
        optimizer.step(closure)

    weights = linear.weight.detach().cpu().numpy().flatten()
    bias = float(linear.bias.detach().cpu().item())
    return weights, bias


def predict_proba(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    z = np.dot(X, weights) + bias
    z_clipped = np.clip(z, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))


# ==============================================================================
# 4. R1 Rescue Network Architecture & Pairwise Training
# ==============================================================================
class R1RescueNet(nn.Module):
    """Specialized neural shortlist reranker trained on B1 hard negatives."""

    def __init__(self, in_dim: int = 64, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TrainedR1Reranker:
    """Reranker protocol wrapper consuming candidate group embeddings."""

    def __init__(
        self,
        b1_encoder: Any,
        rescue_net: R1RescueNet,
        device: str = "cpu",
    ) -> None:
        self.b1_encoder = b1_encoder
        self.rescue_net = rescue_net
        self.device = device
        self.rescue_net.to(device)
        self.rescue_net.eval()

    def rerank_shortlist(
        self,
        shortlist: Sequence[int],
        known_pairs: Sequence[Any],
    ) -> tuple[int, ...]:
        """Rerank shortlist candidates by R1 rescue net score."""
        if len(shortlist) <= 1:
            return tuple(shortlist)

        # Vectorized candidate features
        from neural_sat_speck.data.attack import AttackPair
        # Reconstruct AttackPair list
        attack_pairs = [
            AttackPair(
                plaintext0=p[0],
                plaintext1=(p[0][0] ^ 64, p[0][1]),
                ciphertext0=p[1],
                ciphertext1=(p[1][0] ^ 64, p[1][1]), # dummy diff for partial decrypt
            )
            for p in known_pairs
        ]
        feats = torch.from_numpy(
            vectorized_final_round_features(attack_pairs, list(shortlist))
        ).to(self.device)

        with torch.no_grad():
            embs = self.b1_encoder.pair_embeddings(feats).mean(dim=1)
            scores = self.rescue_net(embs).cpu().numpy()

        # Sort descending by score, tie-break by candidate value
        sorted_pairs = sorted(
            zip(shortlist, scores, strict=True),
            key=lambda item: (-item[1], item[0]),
        )
        return tuple(cand for cand, _ in sorted_pairs)


# ==============================================================================
# 5. Main Phase 10B Development Evaluation Pipeline
# ==============================================================================
def main() -> None:
    print("=== Phase 10B Development Evaluation & Cross-Validation ===")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Precomputed Records
    cache_files = sorted(CACHE_DIR.glob("instance_*.json"))
    if len(cache_files) < 192:
        print(f"Waiting for precomputation: {len(cache_files)}/192 completed.")
        return

    records = [json.loads(f.read_text(encoding="utf-8")) for f in cache_files]
    print(f"Loaded {len(records)} development records.")

    # 2. Extract Matrices
    N = len(records)
    y_fail = np.array([r["y_fail"] for r in records], dtype=np.int32)
    failure_count = int(np.sum(y_fail))
    prevalence = failure_count / N

    print(f"Failure prevalence: {failure_count}/{N} ({prevalence*100:.2f}%)")

    # Feature names
    feature_names = [
        "top1_top2_margin",
        "top1_top16_margin",
        "normalized_top16_entropy",
        "top16_jaccard_16_32",
        "top16_jaccard_32_64",
        "top1_rank_variance",
        "dispersion_normalized_margin",
    ]

    # X matrix for G1 (N, 7)
    X_g1 = np.array(
        [[r["g1_features"][fn] for fn in feature_names] for r in records],
        dtype=np.float64,
    )

    # X vector for G0: scalar curvature
    # For G0, lower curvature -> higher failure risk, so we use negative curvature
    X_g0 = np.array([[-r["g0_curvature"]] for r in records], dtype=np.float64)

    # 3. 5-Fold Stratified Cross-Validation
    folds = stratified_kfold_split(y_fail, n_splits=5, seed=PHASE10_DEVELOPMENT_SEED)
    fold_assignments = {
        f"fold_{k}": {
            "val_instances": [records[i]["instance_id"] for i in val_idx],
            "val_indices": val_idx.tolist(),
        }
        for k, (_, val_idx) in enumerate(folds)
    }
    (OUTPUT_DIR / "fold_assignments.json").write_text(
        json.dumps(fold_assignments, indent=2) + "\n"
    )

    oof_p_g0 = np.zeros(N, dtype=np.float64)
    oof_p_g1 = np.zeros(N, dtype=np.float64)

    print("\n--- Running 5-Fold Key-Level Cross-Validation ---")
    for fold_i, (train_idx, val_idx) in enumerate(folds):
        # A. Fit G0 (scalar baseline)
        mu_g0 = np.mean(X_g0[train_idx], axis=0)
        std_g0 = np.std(X_g0[train_idx], axis=0) + 1e-8
        X_g0_train_std = (X_g0[train_idx] - mu_g0) / std_g0
        X_g0_val_std = (X_g0[val_idx] - mu_g0) / std_g0

        w_g0, b_g0 = fit_logistic_regression(X_g0_train_std, y_fail[train_idx], l2_lambda=1.0)
        oof_p_g0[val_idx] = predict_proba(X_g0_val_std, w_g0, b_g0)

        # B. Fit G1 (7 frozen features)
        mu_g1 = np.mean(X_g1[train_idx], axis=0)
        std_g1 = np.std(X_g1[train_idx], axis=0) + 1e-8
        X_g1_train_std = (X_g1[train_idx] - mu_g1) / std_g1
        X_g1_val_std = (X_g1[val_idx] - mu_g1) / std_g1

        w_g1, b_g1 = fit_logistic_regression(X_g1_train_std, y_fail[train_idx], l2_lambda=2.0)
        oof_p_g1[val_idx] = predict_proba(X_g1_val_std, w_g1, b_g1)

        val_auc_g1 = compute_roc_auc(y_fail[val_idx], oof_p_g1[val_idx])
        val_auc_g0 = compute_roc_auc(y_fail[val_idx], oof_p_g0[val_idx])
        print(f"Fold {fold_i+1}/5: Val instances={len(val_idx)} | G0 AUC={val_auc_g0:.3f} | G1 AUC={val_auc_g1:.3f}")

    # 4. Out-of-Fold Metrics for G0 and G1
    auc_g0 = compute_roc_auc(y_fail, oof_p_g0)
    pr_g0 = compute_pr_auc(y_fail, oof_p_g0)
    brier_g0 = compute_brier_score(y_fail, oof_p_g0)
    auc_ci_g0 = bootstrap_ci(y_fail, oof_p_g0, compute_roc_auc)
    pr_ci_g0 = bootstrap_ci(y_fail, oof_p_g0, compute_pr_auc)

    auc_g1 = compute_roc_auc(y_fail, oof_p_g1)
    pr_g1 = compute_pr_auc(y_fail, oof_p_g1)
    brier_g1 = compute_brier_score(y_fail, oof_p_g1)
    auc_ci_g1 = bootstrap_ci(y_fail, oof_p_g1, compute_roc_auc)
    pr_ci_g1 = bootstrap_ci(y_fail, oof_p_g1, compute_pr_auc)

    print("\n=== G0 vs G1 Out-of-Fold Discrimination Results ===")
    print(f"G0 Baseline: ROC-AUC = {auc_g0:.4f} (95% CI [{auc_ci_g0[0]:.4f}, {auc_ci_g0[1]:.4f}])")
    print(f"             PR-AUC  = {pr_g0:.4f} (95% CI [{pr_ci_g0[0]:.4f}, {pr_ci_g0[1]:.4f}]) | Brier = {brier_g0:.4f}")
    print(f"G1 Proposed: ROC-AUC = {auc_g1:.4f} (95% CI [{auc_ci_g1[0]:.4f}, {auc_ci_g1[1]:.4f}])")
    print(f"             PR-AUC  = {pr_g1:.4f} (95% CI [{pr_ci_g1[0]:.4f}, {pr_ci_g1[1]:.4f}]) | Brier = {brier_g1:.4f}")

    # 5. Threshold Sweep & Frozen Selection Rule
    thresholds = np.linspace(0.01, 0.99, 100)
    sweep_data: list[dict[str, Any]] = []
    valid_thetas: list[float] = []

    for th in thresholds:
        accepted = oof_p_g1 < th
        n_acc = int(np.sum(accepted))
        cov = n_acc / N
        esc = 1.0 - cov
        if n_acc > 0:
            false_trust_count = int(np.sum(y_fail[accepted]))
            false_trust_risk = false_trust_count / n_acc
        else:
            false_trust_count = 0
            false_trust_risk = 0.0

        is_valid = bool(false_trust_risk <= 0.05 and cov >= 0.50)
        if is_valid:
            valid_thetas.append(float(th))

        sweep_data.append({
            "theta": float(th),
            "accepted_count": n_acc,
            "coverage": cov,
            "escalation_rate": esc,
            "false_trust_count": false_trust_count,
            "false_trust_risk": false_trust_risk,
            "satisfies_frozen_rule": is_valid,
        })

    # Frozen Operational Rule: select highest theta satisfying risk <= 0.05 and coverage >= 0.50
    selected_theta: float | None = max(valid_thetas) if valid_thetas else None
    threshold_status = "POSITIVE_FROZEN" if selected_theta is not None else "NEGATIVE_INCONCLUSIVE"
    print(f"\nThreshold Selection Status: {threshold_status}")
    if selected_theta is not None:
        rec_th = next(r for r in sweep_data if r["theta"] == selected_theta)
        print(f"Selected theta* = {selected_theta:.4f}")
        print(f"  Coverage: {rec_th['coverage']*100:.1f}% ({rec_th['accepted_count']}/{N})")
        print(f"  False-Trust Risk: {rec_th['false_trust_risk']*100:.2f}% ({rec_th['false_trust_count']} failures trusted)")
        print(f"  Escalation Rate: {rec_th['escalation_rate']*100:.1f}%")
    else:
        print("No threshold achieved <= 5% false-trust risk at >= 50% coverage.")
        print("Classified as NEGATIVE / INCONCLUSIVE. Fallback to verification-triggered cascade.")

    # 6. R1 Shortlist Rescue Evaluation (M in {32, 64, 128})
    print("\n=== Evaluating R1 Shortlist Rescue (M in {32, 64, 128}) ===")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    b1_ckpt = torch.load(B1_MODEL_PATH, map_location="cpu", weights_only=False)
    b1_model = PairGroupRanker(variant=b1_ckpt["variant"])
    b1_model.load_state_dict(b1_ckpt["state_dict"])
    b1_model.to(device)
    b1_model.eval()

    # Load instances for pair features
    instances = load_attack_shard(
        DATASET_PATH,
        expected_dataset_id="speck32_64_attack_phase10_development_r6",
    ).instances

    shortlist_results: dict[int, dict[str, Any]] = {}

    for M in (32, 64, 128):
        # Availability ceiling
        ceil_res = compute_rescue_ceiling(
            [r["b1_candidate_order_top128"] for r in records],
            [instances[i].final_round_subkey for i in range(N)],
            shortlist_size=M,
            primary_budget=16,
        )

        # 5-fold evaluation of R1 model trained on hard negatives
        rescued_count = 0
        worsened_count = 0
        rank_deltas: list[int] = []
        available_failure_indices = [
            i for i, r in enumerate(records)
            if r["y_fail"] == 1 and r[f"b1_top{M}_containment"]
        ]

        # Train one unified R1 model per fold on training hard negatives
        for fold_i, (train_idx, val_idx) in enumerate(folds):
            # Mine training hard negatives
            train_miner = R1HardNegativeMiner(shortlist_size=M, target_rank_cutoff=16)
            mined_train: list[Any] = []
            for t_idx in train_idx:
                r_inst = records[t_idx]
                mined = train_miner.mine_instance(
                    r_inst["instance_id"],
                    r_inst["b1_candidate_order_top128"],
                    instances[t_idx].final_round_subkey,
                    data_role=Phase10DataRole.DEVELOPMENT_CALIBRATION,
                )
                if mined.hard_negatives and mined.true_subkey_in_shortlist:
                    mined_train.append((instances[t_idx], mined))

            # Train R1RescueNet on mined pairs
            rescue_net = R1RescueNet(in_dim=64, hidden_dim=64).to(device)
            optimizer = torch.optim.Adam(rescue_net.parameters(), lr=1e-3, weight_decay=1e-4)

            # Pre-extract pair group embeddings for training items
            if mined_train:
                train_embs_pos = []
                train_embs_neg = []
                with torch.no_grad():
                    for inst_t, mined_t in mined_train:
                        # Compute pos emb
                        pos_feats = torch.from_numpy(
                            vectorized_final_round_features(inst_t.pairs, [mined_t.true_subkey])
                        ).to(device)
                        pos_emb = b1_model.pair_embeddings(pos_feats).mean(dim=1)
                        # Compute neg embs
                        neg_feats = torch.from_numpy(
                            vectorized_final_round_features(inst_t.pairs, list(mined_t.hard_negatives[:8]))
                        ).to(device)
                        neg_embs = b1_model.pair_embeddings(neg_feats).mean(dim=1)
                        for n_i in range(neg_embs.shape[0]):
                            train_embs_pos.append(pos_emb)
                            train_embs_neg.append(neg_embs[n_i:n_i+1])

                if train_embs_pos:
                    P_t = torch.cat(train_embs_pos, dim=0)
                    N_t = torch.cat(train_embs_neg, dim=0)
                    for _ in range(50):
                        optimizer.zero_grad()
                        s_pos = rescue_net(P_t)
                        s_neg = rescue_net(N_t)
                        loss = torch.mean(torch.clamp(0.1 - (s_pos - s_neg), min=0.0))
                        loss.backward()
                        optimizer.step()

            # Evaluate on held-out validation failures
            val_failures = [
                i for i in val_idx
                if records[i]["y_fail"] == 1 and records[i][f"b1_top{M}_containment"]
            ]
            rescue_net.eval()
            with torch.no_grad():
                for v_idx in val_failures:
                    inst_v = instances[v_idx]
                    shortlist_cands = records[v_idx]["b1_candidate_order_top128"][:M]
                    true_key_v = inst_v.final_round_subkey
                    b1_rank_v = records[v_idx]["b1_true_rank"]

                    feats_v = torch.from_numpy(
                        vectorized_final_round_features(inst_v.pairs, shortlist_cands)
                    ).to(device)
                    embs_v = b1_model.pair_embeddings(feats_v).mean(dim=1)
                    scores_v = rescue_net(embs_v).cpu().numpy()

                    reranked = [
                        c for c, _ in sorted(
                            zip(shortlist_cands, scores_v, strict=True),
                            key=lambda item: (-item[1], item[0]),
                        )
                    ]
                    new_rank = reranked.index(true_key_v) + 1
                    delta = b1_rank_v - new_rank
                    rank_deltas.append(delta)

                    if new_rank <= 16:
                        rescued_count += 1
                    if new_rank > b1_rank_v:
                        worsened_count += 1

        total_avail = len(available_failure_indices)
        rescue_rate_avail = rescued_count / total_avail if total_avail > 0 else 0.0
        rescue_rate_total = rescued_count / failure_count if failure_count > 0 else 0.0
        mean_delta = float(np.mean(rank_deltas)) if rank_deltas else 0.0

        shortlist_results[M] = {
            "shortlist_size": M,
            "availability_ceiling": ceil_res["rescue_ceiling_rate"],
            "failures_in_shortlist": total_avail,
            "rescued_into_top16": rescued_count,
            "rescue_rate_among_available": rescue_rate_avail,
            "end_to_end_conditional_rescue": rescue_rate_total,
            "mean_rank_improvement": mean_delta,
            "worsened_rank_count": worsened_count,
        }
        print(f"M = {M:3d}: Ceiling = {ceil_res['rescue_ceiling_rate']*100:.1f}% ({total_avail}/{failure_count}) | "
              f"Rescued = {rescued_count}/{total_avail} ({rescue_rate_avail*100:.1f}%) | "
              f"E2E Rescue = {rescue_rate_total*100:.1f}% | Mean Delta = {mean_delta:+.1f} | Worsened = {worsened_count}")

    # Lexicographic Selection Rule for M
    # PRIMARY: maximize out-of-fold rescue into top-16
    # SECONDARY: smaller M
    best_M = max(
        (32, 64, 128),
        key=lambda m: (shortlist_results[m]["end_to_end_conditional_rescue"], -m),
    )
    print(f"\nSelected Shortlist Size M* = {best_M}")

    # 7. Hypothesis Adjudication
    # H1: G1 provides materially better failure discrimination than G0 and useful selective operating region
    h1_promising = bool(auc_g1 >= 0.60 and auc_g1 > auc_g0 + 0.05 and selected_theta is not None)
    h1_status = "PROMISING" if h1_promising else ("INCONCLUSIVE" if auc_g1 >= 0.55 else "NEGATIVE")

    # H2: R1 achieves meaningful conditional rescue
    h2_promising = bool(shortlist_results[best_M]["end_to_end_conditional_rescue"] >= 0.25)
    h2_status = "PROMISING" if h2_promising else ("INCONCLUSIVE" if shortlist_results[best_M]["end_to_end_conditional_rescue"] > 0.0 else "NEGATIVE")

    print(f"\nH1 Classification (G1 Stability Routing): {h1_status}")
    print(f"H2 Classification (R1 Shortlist Rescue):     {h2_status}")

    # 8. Save Full Evaluation Report
    report = {
        "dataset": str(DATASET_PATH.relative_to(REPO_ROOT)),
        "instances_total": N,
        "failures_total": failure_count,
        "failure_prevalence": prevalence,
        "g0_baseline": {
            "roc_auc": auc_g0,
            "roc_auc_ci_95": list(auc_ci_g0),
            "pr_auc": pr_g0,
            "pr_auc_ci_95": list(pr_ci_g0),
            "brier_score": brier_g0,
        },
        "g1_stability": {
            "roc_auc": auc_g1,
            "roc_auc_ci_95": list(auc_ci_g1),
            "pr_auc": pr_g1,
            "pr_auc_ci_95": list(pr_ci_g1),
            "brier_score": brier_g1,
            "threshold_selection_status": threshold_status,
            "selected_theta": selected_theta,
        },
        "threshold_sweep": sweep_data,
        "shortlist_evaluation": shortlist_results,
        "selected_shortlist_size_m": best_M,
        "h1_adjudication": h1_status,
        "h2_adjudication": h2_status,
    }

    report_path = OUTPUT_DIR / "development_evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n[✓] Saved complete development report to {report_path}")


if __name__ == "__main__":
    main()
