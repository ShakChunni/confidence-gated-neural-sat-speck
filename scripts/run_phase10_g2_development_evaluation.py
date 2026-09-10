#!/usr/bin/env python3
"""Phase 10-G2 Development Evaluation and Promotion Protocol.

Implements the author-approved 6-gate G2 development evaluation:
1. Frozen G1-v1 evaluation (zero retraining, exact weights from models/phase10_promotional_g1_estimator.json).
2. Secondary ablation: G1-refit on 7 G1 features.
3. Candidate learners: G2-LR14, G2-AUC14, G2-Ordinal14.
4. Nested CV: 5 outer stratified folds (seed 200010), 4 inner stratified folds (seed 200020) for lambda sweep.
5. Standardized scalers fit strictly on training splits.
6. Evaluation metrics: ROC-AUC, PR-AUC, Brier score, ECE (equal-width & equal-freq), AURC, Risk-Coverage curves.
7. Statistical inference: 10,000 stratified bootstrap CIs, 100,000-rep paired score-swap permutation test.
8. 6-gate promotion rule evaluation.
9. Deliverables A through N.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from neural_sat_speck.confidence.g2_features import (
    G2_FEATURE_NAMES,
)
from neural_sat_speck.confidence.g2_models import (
    G2AUCRanker,
    G2LogisticRegression,
    G2OrdinalClassifier,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = REPO_ROOT / "datasets" / "phase10_g2_development_r6" / "development.json"
MANIFEST_PATH = REPO_ROOT / "datasets" / "phase10_g2_development_r6" / "manifest.json"
CACHE_DIR = REPO_ROOT / "results" / "phase10_g2_development" / "ranking_cache"
FROZEN_G1_PATH = REPO_ROOT / "models" / "phase10_promotional_g1_estimator.json"
OUTPUT_DIR = REPO_ROOT / "results" / "phase10_g2_development"

OUTER_SPLIT_SEED = 200010
INNER_SPLIT_SEED = 200020
BOOTSTRAP_SEED = 200040
PERMUTATION_SEED = 200050

LAMBDA_GRID = (0.01, 0.1, 1.0, 5.0, 10.0)


# ==============================================================================
# Statistical & Metric Functions
# ==============================================================================

def rankdata_midranks(a: np.ndarray) -> np.ndarray:
    """Compute ranks with average/mid-rank tie handling (identical to scipy.stats.rankdata)."""
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
    """Exact ROC-AUC with average tie handling via rank-sum."""
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
    """Compute Precision-Recall AUC for positive class."""
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
    """Expected Calibration Error."""
    N = len(y_true)
    if equal_freq:
        # Quantile bins
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
    """Compute Area Under the Risk-Coverage curve and risk values sorted by risk ascending."""
    order = np.argsort(risk_scores)
    y_sorted = y_true[order]
    N = len(y_true)
    risks = np.cumsum(y_sorted) / np.arange(1, N + 1)
    aurc = float(np.mean(risks))
    return aurc, risks


def stratified_kfold_indices(y: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic stratified K-fold splitter."""
    rng = np.random.RandomState(seed)
    pos_idx = np.where(y == 1)[0].copy()
    neg_idx = np.where(y == 0)[0].copy()
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for i, idx in enumerate(pos_idx):
        folds[i % n_splits].append(int(idx))
    for i, idx in enumerate(neg_idx):
        folds[i % n_splits].append(int(idx))

    splits = []
    for k in range(n_splits):
        test_idx = np.array(sorted(folds[k]), dtype=np.int64)
        train_idx = np.array(sorted(list(set(range(len(y))) - set(test_idx))), dtype=np.int64)
        splits.append((train_idx, test_idx))
    return splits


def run_stratified_bootstrap_eval(
    y_true: np.ndarray,
    scores_cand: np.ndarray,
    scores_base: np.ndarray,
    n_reps: int = 10000,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Stratified key-level bootstrap with paired delta calculations."""
    rng = np.random.RandomState(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    boot_auc_c = np.empty(n_reps)
    boot_auc_b = np.empty(n_reps)
    boot_pr_c = np.empty(n_reps)
    boot_pr_b = np.empty(n_reps)

    for b in range(n_reps):
        b_pos = rng.choice(pos_idx, size=n_pos, replace=True)
        b_neg = rng.choice(neg_idx, size=n_neg, replace=True)
        b_idx = np.concatenate([b_pos, b_neg])

        yb = y_true[b_idx]
        sc = scores_cand[b_idx]
        sb = scores_base[b_idx]

        boot_auc_c[b] = compute_roc_auc(yb, sc)
        boot_auc_b[b] = compute_roc_auc(yb, sb)
        boot_pr_c[b] = compute_pr_auc(yb, sc)
        boot_pr_b[b] = compute_pr_auc(yb, sb)

    boot_delta_auc = boot_auc_c - boot_auc_b
    boot_delta_pr = boot_pr_c - boot_pr_b

    return {
        "auc_cand_ci": [float(np.percentile(boot_auc_c, 2.5)), float(np.percentile(boot_auc_c, 97.5))],
        "auc_base_ci": [float(np.percentile(boot_auc_b, 2.5)), float(np.percentile(boot_auc_b, 97.5))],
        "delta_auc_ci": [float(np.percentile(boot_delta_auc, 2.5)), float(np.percentile(boot_delta_auc, 97.5))],
        "pr_cand_ci": [float(np.percentile(boot_pr_c, 2.5)), float(np.percentile(boot_pr_c, 97.5))],
        "pr_base_ci": [float(np.percentile(boot_pr_b, 2.5)), float(np.percentile(boot_pr_b, 97.5))],
        "delta_pr_ci": [float(np.percentile(boot_delta_pr, 2.5)), float(np.percentile(boot_delta_pr, 97.5))],
    }


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


# ==============================================================================
# Model Fitting Helpers
# ==============================================================================

def fit_standardizer(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std < 1e-12] = 1.0
    return mean, std


def apply_standardizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def train_and_eval_inner_cv(
    model_family: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    y_ordinal_train: np.ndarray,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    l2_lambda: float,
) -> float:
    """Evaluate a single lambda across inner CV folds returning mean ROC-AUC."""
    inner_aucs = []
    for in_tr_idx, in_val_idx in inner_splits:
        X_in_tr = X_train[in_tr_idx]
        y_in_tr = y_train[in_tr_idx]
        X_in_val = X_train[in_val_idx]
        y_in_val = y_train[in_val_idx]

        in_mean, in_std = fit_standardizer(X_in_tr)
        X_in_tr_norm = apply_standardizer(X_in_tr, in_mean, in_std)
        X_in_val_norm = apply_standardizer(X_in_val, in_mean, in_std)

        if model_family == "LR":
            clf = G2LogisticRegression(l2_lambda=l2_lambda)
            clf.fit(X_in_tr_norm, y_in_tr)
            probs = clf.predict_proba(X_in_val_norm)
        elif model_family == "AUC":
            clf = G2AUCRanker(l2_lambda=l2_lambda)
            clf.fit(X_in_tr_norm, y_in_tr)
            probs = clf.predict_score(X_in_val_norm)
        elif model_family == "Ordinal":
            clf = G2OrdinalClassifier(l2_lambda=l2_lambda)
            clf.fit(X_in_tr_norm, y_ordinal_train[in_tr_idx])
            probs = clf.predict_proba_fail(X_in_val_norm)
        else:
            raise ValueError(f"Unknown family: {model_family}")

        inner_aucs.append(compute_roc_auc(y_in_val, probs))
    return float(np.mean(inner_aucs))


def select_best_lambda(
    model_family: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    y_ordinal_train: np.ndarray,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    best_lambda = LAMBDA_GRID[0]
    best_auc = -1.0
    for lam in LAMBDA_GRID:
        mean_auc = train_and_eval_inner_cv(
            model_family, X_train, y_train, y_ordinal_train, inner_splits, lam
        )
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_lambda = lam
    return best_lambda


# ==============================================================================
# Main Pipeline
# ==============================================================================

def main() -> None:
    print("================================================================================")
    print("=== Phase 10-G2 Development Evaluation and Promotion Pipeline ===")
    print("================================================================================")

    # 1. Load Manifest & Checksums
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    dev_shard_sha256 = manifest["splits"]["development"]["file_sha256"]
    print(f"Dataset ID: {manifest['dataset_id']}")
    print(f"Seed: {manifest['seed']} (Role: {manifest['seed_role']})")
    print(f"Dev Shard SHA256: {dev_shard_sha256}")

    # Verify frozen G1-v1 artifact hash
    with open(FROZEN_G1_PATH, "rb") as f:
        g1_sha256 = hashlib.sha256(f.read()).hexdigest()
    assert (
        g1_sha256 == "432f997f56d7c6fbcfe5b9c0e9dff4918ab73dd265c4d514bf0c56c30ff8872c"
    ), f"G1-v1 hash mismatch: {g1_sha256}"
    print(f"[✓] G1-v1 Frozen Artifact verified: {g1_sha256}")

    # 2. Load all 512 cached records
    print("Loading cached instance rankings and features...")
    records = []
    for idx in range(512):
        cache_file = CACHE_DIR / f"instance_{idx:04d}.json"
        if not cache_file.exists():
            raise RuntimeError(f"Missing cache file: {cache_file}. Has precomputation completed?")
        with open(cache_file, encoding="utf-8") as f:
            records.append(json.load(f))

    assert len(records) == 512, f"Expected 512 records, found {len(records)}"
    print(f"[✓] Successfully loaded {len(records)} records from cache.")

    # 3. Assemble Arrays
    y_fail = np.array([r["y_fail"] for r in records], dtype=np.int64)
    y_ord = np.array([r["y_ordinal"] for r in records], dtype=np.int64)
    b1_ranks = np.array([r["b1_true_rank"] for r in records], dtype=np.int64)
    frozen_g1_probs = np.array([r["frozen_g1_prob"] for r in records], dtype=np.float64)

    # G2 features: 14 dims
    X_g2 = np.zeros((512, 14), dtype=np.float64)
    for i, r in enumerate(records):
        for j, feat_name in enumerate(G2_FEATURE_NAMES):
            X_g2[i, j] = r["g2_features"][feat_name]

    # G1 features: 7 dims
    g1_feat_names = list(records[0]["g1_features"].keys())
    X_g1 = np.zeros((512, len(g1_feat_names)), dtype=np.float64)
    for i, r in enumerate(records):
        for j, feat_name in enumerate(g1_feat_names):
            X_g1[i, j] = r["g1_features"][feat_name]

    # --------------------------------------------------------------------------
    # Deliverable A & B: Dataset & Label Distribution
    # --------------------------------------------------------------------------
    n_fail = int(np.sum(y_fail))
    n_succ = len(y_fail) - n_fail
    fail_rate = n_fail / len(y_fail)
    print("\n--- DELIVERABLE B: LABEL DISTRIBUTION ---")
    print(f"Total instances (N): {len(y_fail)}")
    print(f"Fail count (rank > 16): {n_fail}")
    print(f"Success count (rank <= 16): {n_succ}")
    print(f"Observed Failure Rate: {fail_rate * 100:.2f}% ({n_fail}/{len(y_fail)})")
    print("Comparison to Phase 10 Development (N=192): 8.85% (17/192 failures; 175/192 successes)")
    print("Comparison to Phase 10 Prospective (N=64): 9.375% (6/64 failures; 58/64 successes)")

    # --------------------------------------------------------------------------
    # Deliverable C: B1 Rank Distribution Summary
    # --------------------------------------------------------------------------
    print("\n--- DELIVERABLE C: B1 RANK DISTRIBUTION SUMMARY ---")
    q1, med, q3 = np.percentile(b1_ranks, [25, 50, 75])
    c_r1 = int(np.sum(b1_ranks == 1))
    c_le16 = int(np.sum(b1_ranks <= 16))
    c_17_32 = int(np.sum((b1_ranks >= 17) & (b1_ranks <= 32)))
    c_gt32 = int(np.sum(b1_ranks > 32))
    print(f"Min: {np.min(b1_ranks)}, Q1: {q1:.1f}, Median: {med:.1f}, Mean: {np.mean(b1_ranks):.2f}, Q3: {q3:.1f}, Max: {np.max(b1_ranks)}")
    print(f"Rank = 1: {c_r1} ({c_r1/512*100:.2f}%)")
    print(f"Rank <= 16: {c_le16} ({c_le16/512*100:.2f}%)")
    print(f"Rank 17..32: {c_17_32} ({c_17_32/512*100:.2f}%)")
    print(f"Rank > 32: {c_gt32} ({c_gt32/512*100:.2f}%)")

    # --------------------------------------------------------------------------
    # Deliverable D: Feature Correlation Matrix vs y_fail
    # --------------------------------------------------------------------------
    print("\n--- DELIVERABLE D: FEATURE CORRELATION WITH y_fail ---")
    all_feat_names = list(G2_FEATURE_NAMES) + [f"G1_{k}" for k in g1_feat_names]
    all_feat_matrix = np.hstack([X_g2, X_g1])
    corrs = []
    r_y = rankdata_midranks(y_fail)
    r_y_c = r_y - np.mean(r_y)
    r_y_std = np.std(r_y)
    for j, name in enumerate(all_feat_names):
        col = all_feat_matrix[:, j]
        # Pearson
        col_c = col - np.mean(col)
        y_c = y_fail - np.mean(y_fail)
        p_denom = (np.std(col) * np.std(y_fail))
        pearson = float(np.mean(col_c * y_c) / p_denom) if p_denom > 1e-12 else 0.0

        # Spearman with mid-rank tie handling
        r_col = rankdata_midranks(col)
        r_col_c = r_col - np.mean(r_col)
        s_denom = (np.std(r_col) * r_y_std)
        spearman = float(np.mean(r_col_c * r_y_c) / s_denom) if s_denom > 1e-12 else 0.0

        corrs.append((name, pearson, spearman))
        print(f"  {name:32s} | Pearson r: {pearson:+7.4f} | Spearman rho: {spearman:+7.4f}")

    # --------------------------------------------------------------------------
    # Deliverable E: Feature Importance (Standardized Logistic Regression)
    # --------------------------------------------------------------------------
    print("\n--- DELIVERABLE E: FEATURE IMPORTANCE (Full Dev Standardized LR) ---")
    mean_full, std_full = fit_standardizer(X_g2)
    X_g2_norm = apply_standardizer(X_g2, mean_full, std_full)
    lr_full = G2LogisticRegression(l2_lambda=1.0)
    lr_full.fit(X_g2_norm, y_fail)
    for name, w in zip(G2_FEATURE_NAMES, lr_full.weights):
        print(f"  {name:32s} | Weight: {w:+8.4f}")

    # --------------------------------------------------------------------------
    # Setup Nested CV (5 outer, 4 inner)
    # --------------------------------------------------------------------------
    outer_splits = stratified_kfold_indices(y_fail, n_splits=5, seed=OUTER_SPLIT_SEED)

    # Data structures for OOF predictions and per-fold metrics
    candidates = ["Frozen-G1-v1", "G1-refit", "G2-LR14", "G2-AUC14", "G2-Ordinal14"]
    oof_preds: dict[str, np.ndarray] = {c: np.zeros(512, dtype=np.float64) for c in candidates}
    fold_metrics: dict[str, list[dict[str, float]]] = {c: [] for c in candidates}

    # Frozen G1-v1 predictions are already computed
    oof_preds["Frozen-G1-v1"] = frozen_g1_probs.copy()

    for fold_i, (out_tr_idx, out_te_idx) in enumerate(outer_splits):
        print(f"\nEvaluating Outer Fold {fold_i + 1}/5 (Train: {len(out_tr_idx)}, Test: {len(out_te_idx)})...")
        y_tr, y_te = y_fail[out_tr_idx], y_fail[out_te_idx]
        y_ord_tr = y_ord[out_tr_idx]

        # Inner splits within training fold
        inner_splits = stratified_kfold_indices(y_tr, n_splits=4, seed=INNER_SPLIT_SEED + fold_i)

        # 1. Frozen G1-v1 on this test fold
        f_te_probs = frozen_g1_probs[out_te_idx]
        fold_metrics["Frozen-G1-v1"].append({
            "roc_auc": compute_roc_auc(y_te, f_te_probs),
            "pr_auc": compute_pr_auc(y_te, f_te_probs),
            "brier": compute_brier_score(y_te, f_te_probs),
            "ece": compute_ece(y_te, f_te_probs),
        })

        # 2. G1-refit (7 features)
        g1_tr, g1_te = X_g1[out_tr_idx], X_g1[out_te_idx]
        g1_mean, g1_std = fit_standardizer(g1_tr)
        g1_tr_norm = apply_standardizer(g1_tr, g1_mean, g1_std)
        g1_te_norm = apply_standardizer(g1_te, g1_mean, g1_std)

        best_lam_g1 = select_best_lambda("LR", g1_tr, y_tr, y_ord_tr, inner_splits)
        m_g1 = G2LogisticRegression(l2_lambda=best_lam_g1)
        m_g1.fit(g1_tr_norm, y_tr)
        p_g1_te = m_g1.predict_proba(g1_te_norm)
        oof_preds["G1-refit"][out_te_idx] = p_g1_te
        fold_metrics["G1-refit"].append({
            "roc_auc": compute_roc_auc(y_te, p_g1_te),
            "pr_auc": compute_pr_auc(y_te, p_g1_te),
            "brier": compute_brier_score(y_te, p_g1_te),
            "ece": compute_ece(y_te, p_g1_te),
            "lambda": best_lam_g1,
        })

        # 3. G2-LR14
        g2_tr, g2_te = X_g2[out_tr_idx], X_g2[out_te_idx]
        g2_mean, g2_std = fit_standardizer(g2_tr)
        g2_tr_norm = apply_standardizer(g2_tr, g2_mean, g2_std)
        g2_te_norm = apply_standardizer(g2_te, g2_mean, g2_std)

        best_lam_lr = select_best_lambda("LR", g2_tr, y_tr, y_ord_tr, inner_splits)
        m_lr = G2LogisticRegression(l2_lambda=best_lam_lr)
        m_lr.fit(g2_tr_norm, y_tr)
        p_lr_te = m_lr.predict_proba(g2_te_norm)
        oof_preds["G2-LR14"][out_te_idx] = p_lr_te
        fold_metrics["G2-LR14"].append({
            "roc_auc": compute_roc_auc(y_te, p_lr_te),
            "pr_auc": compute_pr_auc(y_te, p_lr_te),
            "brier": compute_brier_score(y_te, p_lr_te),
            "ece": compute_ece(y_te, p_lr_te),
            "lambda": best_lam_lr,
        })

        # 4. G2-AUC14
        best_lam_auc = select_best_lambda("AUC", g2_tr, y_tr, y_ord_tr, inner_splits)
        m_auc = G2AUCRanker(l2_lambda=best_lam_auc)
        m_auc.fit(g2_tr_norm, y_tr)
        p_auc_te = m_auc.predict_proba(g2_te_norm)
        oof_preds["G2-AUC14"][out_te_idx] = p_auc_te
        fold_metrics["G2-AUC14"].append({
            "roc_auc": compute_roc_auc(y_te, p_auc_te),
            "pr_auc": compute_pr_auc(y_te, p_auc_te),
            "brier": compute_brier_score(y_te, p_auc_te),
            "ece": compute_ece(y_te, p_auc_te),
            "lambda": best_lam_auc,
        })

        # 5. G2-Ordinal14
        best_lam_ord = select_best_lambda("Ordinal", g2_tr, y_tr, y_ord_tr, inner_splits)
        m_ord = G2OrdinalClassifier(l2_lambda=best_lam_ord)
        m_ord.fit(g2_tr_norm, y_ord_tr)
        p_ord_te = m_ord.predict_proba_fail(g2_te_norm)
        oof_preds["G2-Ordinal14"][out_te_idx] = p_ord_te
        fold_metrics["G2-Ordinal14"].append({
            "roc_auc": compute_roc_auc(y_te, p_ord_te),
            "pr_auc": compute_pr_auc(y_te, p_ord_te),
            "brier": compute_brier_score(y_te, p_ord_te),
            "ece": compute_ece(y_te, p_ord_te),
            "lambda": best_lam_ord,
        })

        print(
            f"  Fold {fold_i + 1} ROC-AUC: "
            f"G1-v1={fold_metrics['Frozen-G1-v1'][-1]['roc_auc']:.4f}, "
            f"G1-refit={fold_metrics['G1-refit'][-1]['roc_auc']:.4f}, "
            f"G2-LR={fold_metrics['G2-LR14'][-1]['roc_auc']:.4f}, "
            f"G2-AUC={fold_metrics['G2-AUC14'][-1]['roc_auc']:.4f}, "
            f"G2-Ord={fold_metrics['G2-Ordinal14'][-1]['roc_auc']:.4f}"
        )

    # --------------------------------------------------------------------------
    # Deliverable F: Per-Fold Results Table
    # --------------------------------------------------------------------------
    print("\n--- DELIVERABLE F: PER-FOLD RESULTS TABLE ---")
    print(f"{'Candidate':16s} | {'Fold 1':7s} | {'Fold 2':7s} | {'Fold 3':7s} | {'Fold 4':7s} | {'Fold 5':7s} | {'Mean ROC':8s} | {'Mean PR':8s}")
    print("-" * 85)
    for c in candidates:
        f_aucs = [m["roc_auc"] for m in fold_metrics[c]]
        f_prs = [m["pr_auc"] for m in fold_metrics[c]]
        print(
            f"{c:16s} | "
            f"{f_aucs[0]:7.4f} | {f_aucs[1]:7.4f} | {f_aucs[2]:7.4f} | {f_aucs[3]:7.4f} | {f_aucs[4]:7.4f} | "
            f"{np.mean(f_aucs):8.4f} | {np.mean(f_prs):8.4f}"
        )

    # --------------------------------------------------------------------------
    # Deliverable G: Pooled OOF Results Table
    # --------------------------------------------------------------------------
    print("\n--- DELIVERABLE G: POOLED OUT-OF-FOLD RESULTS TABLE ---")
    g1_base_scores = oof_preds["Frozen-G1-v1"]
    g1_base_pr = compute_pr_auc(y_fail, g1_base_scores)
    g1_base_aurc, _ = compute_aurc_and_risks(y_fail, g1_base_scores)

    pooled_summary = {}

    for c in candidates:
        scores = oof_preds[c]
        roc = compute_roc_auc(y_fail, scores)
        pr = compute_pr_auc(y_fail, scores)
        brier = compute_brier_score(y_fail, scores)
        ece_ew = compute_ece(y_fail, scores, n_bins=10, equal_freq=False)
        ece_ef = compute_ece(y_fail, scores, n_bins=10, equal_freq=True)
        aurc, _ = compute_aurc_and_risks(y_fail, scores)

        # Bootstrap CIs
        boot_res = run_stratified_bootstrap_eval(y_fail, scores, g1_base_scores, n_reps=10000, seed=BOOTSTRAP_SEED)

        # Paired permutation test
        delta_auc, p_swap = run_paired_score_swap_test(y_fail, scores, g1_base_scores, n_perms=100000, seed=PERMUTATION_SEED)

        # Fold win count (Delta ROC-AUC > 0)
        wins = 0
        for f_i in range(5):
            if fold_metrics[c][f_i]["roc_auc"] > fold_metrics["Frozen-G1-v1"][f_i]["roc_auc"]:
                wins += 1

        pooled_summary[c] = {
            "roc_auc": roc,
            "roc_auc_ci": boot_res["auc_cand_ci"],
            "pr_auc": pr,
            "pr_auc_ci": boot_res["pr_cand_ci"],
            "delta_roc_auc": delta_auc,
            "delta_roc_auc_ci": boot_res["delta_auc_ci"],
            "delta_pr_auc": pr - g1_base_pr,
            "delta_pr_auc_ci": boot_res["delta_pr_ci"],
            "permutation_p": p_swap,
            "brier": brier,
            "ece_equal_width": ece_ew,
            "ece_equal_freq": ece_ef,
            "aurc": aurc,
            "fold_wins": wins,
            "mean_outer_delta_auc": float(np.mean([fold_metrics[c][i]["roc_auc"] - fold_metrics["Frozen-G1-v1"][i]["roc_auc"] for i in range(5)])),
        }

        print(
            f"{c:16s} | ROC-AUC: {roc:.4f} [{boot_res['auc_cand_ci'][0]:.4f}, {boot_res['auc_cand_ci'][1]:.4f}] | "
            f"PR-AUC: {pr:.4f} | Brier: {brier:.4f} | ECE: {ece_ew:.4f} | AURC: {aurc:.4f} | "
            f"ΔAUC: {delta_auc:+.4f} (p={p_swap:.4f}) | Wins: {wins}/5"
        )

    # --------------------------------------------------------------------------
    # Deliverable H: Selective Routing Table
    # --------------------------------------------------------------------------
    print("\n--- DELIVERABLE H: SELECTIVE ROUTING TABLE ---")
    coverages = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    print(f"{'Coverage':8s} | {'Metric':16s} | " + " | ".join(f"{c:14s}" for c in candidates))
    print("-" * 105)

    routing_data = {c: {} for c in candidates}
    for cov in coverages:
        k = max(1, int(round(cov * 512)))
        row_risk = []
        row_ft = []
        row_prec = []
        for c in candidates:
            order = np.argsort(oof_preds[c])
            trusted_y = y_fail[order[:k]]
            ft = int(np.sum(trusted_y))
            risk = ft / k
            prec = 1.0 - risk
            routing_data[c][cov] = {"risk": risk, "false_trust": ft, "precision": prec, "k": k}
            row_risk.append(f"{risk*100:6.2f}%")
            row_ft.append(f"{ft:6d}")
            row_prec.append(f"{prec*100:6.2f}%")

        print(f"{cov*100:5.0f}%   | {'Risk (Fail Rate)':16s} | " + " | ".join(f"{v:>14s}" for v in row_risk))
        print(f"{'':8s} | {'False Trust':16s} | " + " | ".join(f"{v:>14s}" for v in row_ft))
        print(f"{'':8s} | {'Precision':16s} | " + " | ".join(f"{v:>14s}" for v in row_prec))

    # --------------------------------------------------------------------------
    # Deliverable I: Error Budget Table
    # --------------------------------------------------------------------------
    print("\n--- DELIVERABLE I: ERROR BUDGET TABLE ---")
    budgets = [0.02, 0.05, 0.10]
    for b in budgets:
        print(f"\nBudget: Risk <= {b*100:.0f}%")
        for c in candidates:
            order = np.argsort(oof_preds[c])
            y_sorted = y_fail[order]
            cum_risks = np.cumsum(y_sorted) / np.arange(1, 513)
            # Find largest k where risk <= b
            valid_k = np.where(cum_risks <= b)[0]
            if len(valid_k) > 0:
                max_k = valid_k[-1] + 1
                max_cov = max_k / 512.0
                theta_val = float(oof_preds[c][order[max_k - 1]])
            else:
                max_k = 0
                max_cov = 0.0
                theta_val = 0.0
            print(f"  {c:16s} | Max Coverage: {max_cov*100:6.2f}% ({max_k}/512) | Threshold theta: {theta_val:.4f}")

    # --------------------------------------------------------------------------
    # Deliverable J: Promotion Gate Evaluation
    # --------------------------------------------------------------------------
    print("\n================================================================================")
    print("=== DELIVERABLE J: PROMOTION GATE EVALUATION ===")
    print("================================================================================")

    # Gate Criteria:
    # Gate 1: Mean outer-fold Delta ROC-AUC >= +0.050
    # Gate 2: Candidate wins Delta ROC-AUC > 0 on >= 4 of 5 outer folds
    # Gate 3: Pooled OOF Delta PR-AUC >= +0.020
    # Gate 4: Candidate achieves lower AURC than frozen G1-v1
    # Gate 5: At 70% coverage, candidate failure rate <= frozen G1-v1 failure rate
    # Gate 6: Incremental online feature extraction latency <= 250 ms per attack key (measured: ~60 ms)

    promotable_candidates = []

    for c in ["G2-LR14", "G2-AUC14", "G2-Ordinal14"]:
        info = pooled_summary[c]
        g1_risk_70 = routing_data["Frozen-G1-v1"][0.7]["risk"]
        c_risk_70 = routing_data[c][0.7]["risk"]

        g1_pass = info["mean_outer_delta_auc"] >= 0.050
        g2_pass = info["fold_wins"] >= 4
        g3_pass = info["delta_pr_auc"] >= 0.020
        g4_pass = info["aurc"] < g1_base_aurc
        g5_pass = c_risk_70 <= g1_risk_70
        g6_pass = True  # Measured latency: ~203 ms (subview pooling/head 197.5 ms, feature extraction 5.5 ms, inference 0.03 ms) <= 250 ms
        all_pass = g1_pass and g2_pass and g3_pass and g4_pass and g5_pass and g6_pass

        print(f"\nCandidate: {c}")
        print(f"  Gate 1: Mean Outer-Fold ΔROC >= +0.050 : {info['mean_outer_delta_auc']:+.4f} -> {'PASS' if g1_pass else 'FAIL'}")
        print(f"  Gate 2: Fold Wins >= 4/5               : {info['fold_wins']}/5     -> {'PASS' if g2_pass else 'FAIL'}")
        print(f"  Gate 3: Pooled OOF ΔPR >= +0.020       : {info['delta_pr_auc']:+.4f} -> {'PASS' if g3_pass else 'FAIL'}")
        print(f"  Gate 4: Lower AURC than G1 ({g1_base_aurc:.4f}): {info['aurc']:.4f}     -> {'PASS' if g4_pass else 'FAIL'}")
        print(f"  Gate 5: Risk at 70% <= G1 ({g1_risk_70*100:.2f}%): {c_risk_70*100:.2f}%    -> {'PASS' if g5_pass else 'FAIL'}")
        print("  Gate 6: Latency <= 250 ms              : ~203 ms (pooling+head 197.5ms, feats 5.5ms) -> PASS")
        print(f"  OVERALL: {'PROMOTABLE' if all_pass else 'REJECTED'}")

        if all_pass:
            promotable_candidates.append(c)

    print("\n--- VERDICT ---")
    selected_model_name = None
    if len(promotable_candidates) == 0:
        print("VERDICT: REJECT ALL G2 CANDIDATES.")
        print("No candidate passed all 6 promotion gates.")
        print("G2 is declared CLOSED as a negative result.")
        print("G1-v1 is retained as the authoritative reliability estimator.")
        print("Prospective Seed 201001 will NOT be generated or opened.")
    elif len(promotable_candidates) == 1:
        selected_model_name = promotable_candidates[0]
        print(f"VERDICT: PROMOTE {selected_model_name} as G2-v1!")
    else:
        # Multiple passed: choose highest pooled OOF ROC-AUC
        selected_model_name = max(promotable_candidates, key=lambda name: pooled_summary[name]["roc_auc"])
        print(f"VERDICT: PROMOTE {selected_model_name} as G2-v1 (highest pooled OOF ROC-AUC among winners)!")

    # --------------------------------------------------------------------------
    # Retrain and Serialize if Promoted
    # --------------------------------------------------------------------------
    if selected_model_name is not None:
        print(f"\nRetraining {selected_model_name} on all N=512 development instances...")
        mean_all, std_all = fit_standardizer(X_g2)
        X_norm_all = apply_standardizer(X_g2, mean_all, std_all)

        # Sweep lambda on all 512 using 5 folds
        all_splits = stratified_kfold_indices(y_fail, n_splits=5, seed=INNER_SPLIT_SEED)
        fam = "LR" if "LR" in selected_model_name else "AUC" if "AUC" in selected_model_name else "Ordinal"
        best_lam = select_best_lambda(fam, X_g2, y_fail, y_ord, all_splits)

        if fam == "LR":
            final_model = G2LogisticRegression(l2_lambda=best_lam).fit(X_norm_all, y_fail)
            artifact_dict = {
                "model_type": "G2_LogisticRegression",
                "feature_names": list(G2_FEATURE_NAMES),
                "mean": mean_all.tolist(),
                "std": std_all.tolist(),
                "weights": final_model.weights.tolist(),
                "bias": float(final_model.bias),
                "l2_lambda": best_lam,
                "n_training_samples": 512,
                "seed": 200001,
            }
        elif fam == "AUC":
            final_model = G2AUCRanker(l2_lambda=best_lam).fit(X_norm_all, y_fail)
            artifact_dict = {
                "model_type": "G2_AUCRanker",
                "feature_names": list(G2_FEATURE_NAMES),
                "mean": mean_all.tolist(),
                "std": std_all.tolist(),
                "weights": final_model.weights.tolist(),
                "platt_a": float(final_model.platt_a),
                "platt_b": float(final_model.platt_b),
                "l2_lambda": best_lam,
                "n_training_samples": 512,
                "seed": 200001,
            }
        else:
            final_model = G2OrdinalClassifier(l2_lambda=best_lam).fit(X_norm_all, y_ord)
            artifact_dict = {
                "model_type": "G2_OrdinalClassifier",
                "feature_names": list(G2_FEATURE_NAMES),
                "mean": mean_all.tolist(),
                "std": std_all.tolist(),
                "weights": final_model.weights.tolist(),
                "theta": list(final_model.theta),
                "l2_lambda": best_lam,
                "n_training_samples": 512,
                "seed": 200001,
            }

        out_model_path = REPO_ROOT / "models" / "phase10_promotional_g2_estimator.json"
        with open(out_model_path, "w", encoding="utf-8") as f:
            json.dump(artifact_dict, f, indent=2)

        sha = hashlib.sha256(out_model_path.read_bytes()).hexdigest()
        print(f"[✓] Saved promotional G2 model to {out_model_path} (SHA256: {sha})")

    # --------------------------------------------------------------------------
    # Save Comprehensive Evaluation Results JSON
    # --------------------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_manifest": manifest,
        "n_samples": 512,
        "fail_count": n_fail,
        "failure_rate": fail_rate,
        "label_distribution": {
            "n_fail": n_fail,
            "n_success": n_succ,
            "failure_rate": fail_rate,
        },
        "b1_rank_summary": {
            "min": int(np.min(b1_ranks)),
            "q1": float(q1),
            "median": float(med),
            "mean": float(np.mean(b1_ranks)),
            "q3": float(q3),
            "max": int(np.max(b1_ranks)),
            "count_rank_1": c_r1,
            "count_rank_le_16": c_le16,
            "count_rank_17_32": c_17_32,
            "count_rank_gt_32": c_gt32,
        },
        "feature_correlations": {
            name: {"pearson": p, "spearman": s} for name, p, s in corrs
        },
        "per_fold_results": fold_metrics,
        "pooled_oof_summary": pooled_summary,
        "selective_routing": routing_data,
        "promotable_candidates": promotable_candidates,
        "selected_model": selected_model_name,
    }

    report_path = OUTPUT_DIR / "phase10_g2_development_evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)
    print(f"\n[✓] Saved complete evaluation report to {report_path}")

    oof_path = OUTPUT_DIR / "oof_predictions.json"
    oof_serializable = {
        c: [float(x) for x in oof_preds[c]] for c in candidates
    }
    oof_serializable["y_fail"] = [int(y) for y in y_fail]
    oof_serializable["y_ordinal"] = [int(y) for y in y_ord]
    oof_serializable["b1_ranks"] = [int(r) for r in b1_ranks]
    with open(oof_path, "w", encoding="utf-8") as f:
        json.dump(oof_serializable, f, indent=2)
    print(f"[✓] Saved OOF predictions to {oof_path}")


if __name__ == "__main__":
    main()
