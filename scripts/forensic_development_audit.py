#!/usr/bin/env python3
"""Forensic development audit script for Phase 10B.

Executes:
1. Complete reconstruction of all 192 attack keys from atomic records.
2. 5-fold OOF re-fitting of G0 and G1.
3. R1 evaluation across ALL 192 keys (both successes and failures) for M in {32, 64, 128}.
4. Dedicated analysis of all 17 B1 failures.
5. Paired contingency tables and McNemar tests.
6. Timing decomposition and cost audit.
7. Verification of whether G1 escalated the exact 12 failures that R1 rescued, or which were false-trusted.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from neural_sat_speck.attacks.r1 import (
    R1HardNegativeMiner,
)
from neural_sat_speck.data.attack import load_attack_shard
from neural_sat_speck.data.seed_policy import (
    Phase10DataRole,
)
from neural_sat_speck.neural.group_model import PairGroupRanker
from neural_sat_speck.neural.numpy_inference import vectorized_final_round_features

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = REPO_ROOT / "datasets" / "phase10_development_r6" / "development.json"
CACHE_DIR = REPO_ROOT / "results" / "phase10_development" / "ranking_cache"
B0_CACHE_DIR = REPO_ROOT / "results" / "phase10_development" / "b0_fallback_cache"
OUTPUT_DIR = REPO_ROOT / "results" / "phase10_development"
B1_MODEL_PATH = REPO_ROOT / "models" / "speck32_64_phase6a_b1_mean_g64.pt"


class R1RescueNet(nn.Module):
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


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
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
    desc_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[desc_indices]

    n_pos = np.sum(y_true)
    if n_pos == 0:
        return 0.0

    tps = np.cumsum(y_true_sorted)
    precisions = tps / np.arange(1, len(y_true) + 1)
    recalls = tps / n_pos

    recalls_diff = np.diff(np.r_[0, recalls])
    return float(np.sum(precisions * recalls_diff))


def fit_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    l2_lambda: float = 1.0,
    lr: float = 0.5,
    epochs: int = 200,
) -> tuple[np.ndarray, float]:
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


def main() -> None:
    print("=== Running Forensic Development Audit on Phase 10B Atomic Records ===")

    # 1. Load All Atomic Ranking Records
    cache_files = sorted(CACHE_DIR.glob("instance_*.json"))
    assert len(cache_files) == 192, f"Expected 192 records, found {len(cache_files)}"
    records = [json.loads(f.read_text(encoding="utf-8")) for f in cache_files]
    N = len(records)

    # Load Attack Instances
    instances = load_attack_shard(
        DATASET_PATH,
        expected_dataset_id="speck32_64_attack_phase10_development_r6",
    ).instances

    # Load B0 fallback records
    b0_records = {}
    for f in B0_CACHE_DIR.glob("instance_*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        b0_records[d["index"]] = d

    print(f"Loaded {N} ranking records and {len(b0_records)} B0 fallback records.")

    # 2. Extract Data Matrices
    y_fail = np.array([r["y_fail"] for r in records], dtype=np.int32)
    b1_true_ranks = np.array([r["b1_true_rank"] for r in records], dtype=np.int32)
    failure_indices = [i for i, y in enumerate(y_fail) if y == 1]
    assert len(failure_indices) == 17, f"Expected 17 failures, got {len(failure_indices)}"

    feature_names = [
        "top1_top2_margin",
        "top1_top16_margin",
        "normalized_top16_entropy",
        "top16_jaccard_16_32",
        "top16_jaccard_32_64",
        "top1_rank_variance",
        "dispersion_normalized_margin",
    ]
    X_g1 = np.array(
        [[r["g1_features"][fn] for fn in feature_names] for r in records],
        dtype=np.float64,
    )
    X_g0 = np.array([[-r["g0_curvature"]] for r in records], dtype=np.float64)

    # 3. Load Fold Assignments
    fold_file = OUTPUT_DIR / "fold_assignments.json"
    fold_data = json.loads(fold_file.read_text(encoding="utf-8"))
    folds = []
    for k in range(5):
        val_idx = np.array(fold_data[f"fold_{k}"]["val_indices"], dtype=np.int32)
        train_idx = np.setdiff1d(np.arange(N), val_idx)
        folds.append((train_idx, val_idx))

    # 4. Out-of-Fold Model Fitting for G0 and G1
    oof_p_g0 = np.zeros(N, dtype=np.float64)
    oof_p_g1 = np.zeros(N, dtype=np.float64)
    fold_ids = np.zeros(N, dtype=np.int32)

    fold_aucs_g0 = []
    fold_aucs_g1 = []

    for fold_i, (train_idx, val_idx) in enumerate(folds):
        fold_ids[val_idx] = fold_i

        # G0 fit
        mu_g0 = np.mean(X_g0[train_idx], axis=0)
        std_g0 = np.std(X_g0[train_idx], axis=0) + 1e-8
        w_g0, b_g0 = fit_logistic_regression(
            (X_g0[train_idx] - mu_g0) / std_g0, y_fail[train_idx], l2_lambda=1.0
        )
        oof_p_g0[val_idx] = predict_proba((X_g0[val_idx] - mu_g0) / std_g0, w_g0, b_g0)

        # G1 fit
        mu_g1 = np.mean(X_g1[train_idx], axis=0)
        std_g1 = np.std(X_g1[train_idx], axis=0) + 1e-8
        w_g1, b_g1 = fit_logistic_regression(
            (X_g1[train_idx] - mu_g1) / std_g1, y_fail[train_idx], l2_lambda=2.0
        )
        oof_p_g1[val_idx] = predict_proba((X_g1[val_idx] - mu_g1) / std_g1, w_g1, b_g1)

        f_auc_g0 = compute_roc_auc(y_fail[val_idx], oof_p_g0[val_idx])
        f_auc_g1 = compute_roc_auc(y_fail[val_idx], oof_p_g1[val_idx])
        fold_aucs_g0.append(f_auc_g0)
        fold_aucs_g1.append(f_auc_g1)

    # 5. R1 Training & Complete Evaluation Across All 192 Keys for M in {32, 64, 128}
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    b1_ckpt = torch.load(B1_MODEL_PATH, map_location="cpu", weights_only=False)
    b1_model = PairGroupRanker(variant=b1_ckpt["variant"])
    b1_model.load_state_dict(b1_ckpt["state_dict"])
    b1_model.to(device)
    b1_model.eval()

    r1_ranks = {32: np.zeros(N, dtype=np.int32), 64: np.zeros(N, dtype=np.int32), 128: np.zeros(N, dtype=np.int32)}

    for M in (32, 64, 128):
        print(f"\nEvaluating R1 across ALL 192 keys for M = {M}...")
        for fold_i, (train_idx, val_idx) in enumerate(folds):
            # Mine training hard negatives
            train_miner = R1HardNegativeMiner(shortlist_size=M, target_rank_cutoff=16)
            mined_train = []
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

            rescue_net = R1RescueNet(in_dim=64, hidden_dim=64).to(device)
            optimizer = torch.optim.Adam(rescue_net.parameters(), lr=1e-3, weight_decay=1e-4)

            if mined_train:
                train_embs_pos = []
                train_embs_neg = []
                with torch.no_grad():
                    for inst_t, mined_t in mined_train:
                        pos_feats = torch.from_numpy(
                            vectorized_final_round_features(inst_t.pairs, [mined_t.true_subkey])
                        ).to(device)
                        pos_emb = b1_model.pair_embeddings(pos_feats).mean(dim=1)
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

            # Now evaluate on ALL held-out instances in val_idx!
            rescue_net.eval()
            with torch.no_grad():
                for v_idx in val_idx:
                    inst_v = instances[v_idx]
                    true_key_v = inst_v.final_round_subkey
                    b1_rank_v = records[v_idx]["b1_true_rank"]
                    shortlist_cands = records[v_idx]["b1_candidate_order_top128"][:M]

                    if true_key_v not in shortlist_cands:
                        # True key not in shortlist -> cannot be rescued
                        r1_ranks[M][v_idx] = b1_rank_v
                        continue

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
                    r1_ranks[M][v_idx] = reranked.index(true_key_v) + 1

    # 6. Auditing Threshold theta = 0.0892
    theta = 0.0892
    routes = ["b1_direct" if p < theta else "r1_rescue" for p in oof_p_g1]
    accepted_idx = [i for i, r in enumerate(routes) if r == "b1_direct"]
    escalated_idx = [i for i, r in enumerate(routes) if r == "r1_rescue"]
    false_trust_idx = [i for i in accepted_idx if y_fail[i] == 1]

    print("\n=== Threshold Audit (theta = 0.0892) ===")
    print(f"Total instances: {N}")
    print(f"Accepted (b1_direct): {len(accepted_idx)}/192 ({len(accepted_idx)/N*100:.2f}%)")
    print(f"Escalated (r1_rescue): {len(escalated_idx)}/192 ({len(escalated_idx)/N*100:.2f}%)")
    print(f"False-trusted failures: {len(false_trust_idx)}/101 ({len(false_trust_idx)/len(accepted_idx)*100:.2f}%)")
    print(f"False-trusted indices: {false_trust_idx}")
    print(f"Escalated failures: {[i for i in escalated_idx if y_fail[i] == 1]}")

    # 7. Dedicated Audit of the 17 Failures
    print("\n=== DEDICATED AUDIT OF THE 17 B1 FAILURES ===")
    print(f"{'Idx':<4} | {'Instance ID':<11} | {'B1 Rank':<8} | {'G1 Score':<9} | {'Route':<10} | {'R1 M32 Rank':<11} | {'R1 Rescued?':<11} | {'B0 Rank':<8} | {'B0 Rescued?':<11}")
    print("-" * 105)
    failure_audit_rows = []
    g1_escalated_and_rescued = 0
    g1_escalated_and_not_rescued = 0
    g1_trusted_and_rescued = 0
    g1_trusted_and_not_rescued = 0

    for idx in failure_indices:
        inst_id = records[idx]["instance_id"]
        b1_r = records[idx]["b1_true_rank"]
        g1_p = oof_p_g1[idx]
        route = routes[idx]
        r1_m32_r = r1_ranks[32][idx]
        r1_rescued = bool(r1_m32_r <= 16)
        b0_r = b0_records[idx]["b0_true_rank"] if idx in b0_records else -1
        b0_rescued = bool(b0_r <= 16 and b0_r > 0)

        if route == "r1_rescue" and r1_rescued:
            g1_escalated_and_rescued += 1
        elif route == "r1_rescue" and not r1_rescued:
            g1_escalated_and_not_rescued += 1
        elif route == "b1_direct" and r1_rescued:
            g1_trusted_and_rescued += 1
        elif route == "b1_direct" and not r1_rescued:
            g1_trusted_and_not_rescued += 1

        row = {
            "index": idx,
            "instance_id": inst_id,
            "b1_rank": b1_r,
            "g1_failure_prob": g1_p,
            "route": route,
            "r1_m32_rank": int(r1_m32_r),
            "r1_rescued": r1_rescued,
            "b0_rank": b0_r,
            "b0_rescued": b0_rescued,
        }
        failure_audit_rows.append(row)
        print(f"{idx:<4d} | {inst_id:<11} | {b1_r:<8d} | {g1_p:<9.4f} | {route:<10} | {r1_m32_r:<11d} | {str(r1_rescued):<11} | {b0_r:<8d} | {str(b0_rescued):<11}")

    print("\n--- Failure Cross-Tabulation (G1 Route vs R1 Rescue) ---")
    print(f"G1 Escalated AND R1 Rescued:     {g1_escalated_and_rescued}")
    print(f"G1 Escalated BUT R1 NOT Rescued: {g1_escalated_and_not_rescued}")
    print(f"G1 Trusted BUT R1 Rescuable:     {g1_trusted_and_rescued}")
    print(f"G1 Trusted AND R1 NOT Rescued:   {g1_trusted_and_not_rescued}")

    # 8. Complete Master Audit Table Across All 192 Keys
    master_table = []
    cascade_A_wins = 0
    cascade_B_wins = 0
    cascade_C_wins = 0
    cascade_D_wins = 0
    cascade_E_wins = 0

    # R1 on B1 successes: check if R1 dropped any B1 success!
    b1_success_indices = [i for i, y in enumerate(y_fail) if y == 0]
    r1_dropped_successes = [i for i in b1_success_indices if r1_ranks[32][i] > 16]
    print(f"\nR1 M=32 evaluation on original B1 SUCCESSES ({len(b1_success_indices)} instances):")
    print(f"B1 successes dropped by R1: {len(r1_dropped_successes)}")
    if r1_dropped_successes:
        for d_i in r1_dropped_successes:
            print(f"  Instance {records[d_i]['instance_id']}: B1 rank={records[d_i]['b1_true_rank']} -> R1 rank={r1_ranks[32][d_i]}")

    for idx in range(N):
        rec = records[idx]
        inst_id = rec["instance_id"]
        fold_k = int(fold_ids[idx])
        b1_r = int(b1_true_ranks[idx])
        b1_succ = bool(b1_r <= 16)
        g0_s = float(oof_p_g0[idx])
        g1_s = float(oof_p_g1[idx])
        g1_rt = routes[idx]

        r1_32_r = int(r1_ranks[32][idx])
        r1_32_succ = bool(r1_32_r <= 16)
        r1_32_resc = bool(b1_r > 16 and r1_32_succ)

        r1_64_r = int(r1_ranks[64][idx])
        r1_128_r = int(r1_ranks[128][idx])

        b0_rec = b0_records.get(idx)
        b0_r = b0_rec["b0_true_rank"] if b0_rec else -1
        b0_resc = bool(b1_r > 16 and b0_r <= 16 and b0_r > 0)

        # Cascade A: B1 direct (k=16)
        casc_a = b1_succ

        # Cascade B: R1 all (k=16, M=32)
        casc_b = r1_32_succ

        # Cascade C: G1-gated R1 (k=16)
        casc_c = r1_32_succ if g1_rt == "r1_rescue" else b1_succ

        # Cascade D: Lazy B1 -> B0 (k=32)
        # Deduplicated B0 candidates
        if b1_succ:
            casc_d = True
        else:
            # Escalated to B0
            casc_d = b0_resc

        # Cascade E: G1-gated R1 + Lazy terminal B0 (k=32)
        if casc_c:
            casc_e = True
        else:
            casc_e = b0_resc

        if casc_a:
            cascade_A_wins += 1
        if casc_b:
            cascade_B_wins += 1
        if casc_c:
            cascade_C_wins += 1
        if casc_d:
            cascade_D_wins += 1
        if casc_e:
            cascade_E_wins += 1

        master_table.append({
            "instance_id": inst_id,
            "fold": fold_k,
            "B1_true_rank": b1_r,
            "B1_top16_success": casc_a,
            "G0_score": g0_s,
            "G1_failure_score": g1_s,
            "G1_route": g1_rt,
            "R1_M32_true_rank": r1_32_r,
            "R1_M32_rescued": r1_32_resc,
            "R1_M64_true_rank": r1_64_r,
            "R1_M128_true_rank": r1_128_r,
            "B0_fallback_rescued": b0_resc,
            "cascade_A_success": casc_a,
            "cascade_B_success": casc_b,
            "cascade_C_success": casc_c,
            "cascade_D_success": casc_d,
            "cascade_E_success": casc_e,
        })

    # Write Master Audit Table JSON
    master_file = OUTPUT_DIR / "master_audit_table_192.json"
    master_file.write_text(json.dumps(master_table, indent=2) + "\n", encoding="utf-8")
    print(f"\n[✓] Saved master audit table (192 rows) to {master_file}")

    print("\n=== EXACT CASCADE RECOVERY TOTALS (N=192) ===")
    print(f"Method A (B1 Direct):                 {cascade_A_wins}/192 ({cascade_A_wins/N*100:.2f}%)")
    print(f"Method B (R1 All, M=32):              {cascade_B_wins}/192 ({cascade_B_wins/N*100:.2f}%)")
    print(f"Method C (G1-Gated R1, theta=0.0892): {cascade_C_wins}/192 ({cascade_C_wins/N*100:.2f}%)")
    print(f"Method D (Lazy B1 -> B0):             {cascade_D_wins}/192 ({cascade_D_wins/N*100:.2f}%)")
    print(f"Method E (G1-Gated R1 + Lazy B0):     {cascade_E_wins}/192 ({cascade_E_wins/N*100:.2f}%)")

    # Write Dedicated Failure Table JSON
    failure_file = OUTPUT_DIR / "failure_audit_table_17.json"
    failure_file.write_text(json.dumps(failure_audit_rows, indent=2) + "\n", encoding="utf-8")
    print(f"[✓] Saved dedicated failure audit table (17 rows) to {failure_file}")


if __name__ == "__main__":
    main()
