#!/usr/bin/env python3
"""Phase 10 Prospective Validation — AUTHOR APPROVED.

Executes the single locked Phase 10 prospective validation experiment (N=64, Seed 101001)
with zero post-hoc tuning, strictly adhering to the frozen prospective protocol:

1. Validates seed 101001 for role PROSPECTIVE_VALIDATION and confirms rejection of quarantined seeds.
2. Generates the official N=64 Phase 10 validation family (6 rounds, 64 pairs/instance).
   Verifies deterministic replay before writing shard and manifest.
3. Evaluates frozen B1 model on MPS/CPU with embedding reuse across subsets (16, 32, 64 pairs).
4. Extracts frozen 7-dimensional G1 stability features and G0 scalar feature (-curvature).
5. Evaluates frozen promotional G0 and G1 estimators (from Phase 10B freeze).
6. Executes isolated CryptoMiniSat key recovery on fresh models via IsolatedSolverFactory:
   - Method 1: B1 (kmax=16)
   - Method 2: B1 Sequential-32 (kmax=32)
   - Method 3: G1-Adaptive B1 (theta=0.0890, kmax in {16, 32})
   - Method 4: R1 Tail Scheduling (kmax=32, frozen negative ablation)
7. Uses independent 2-pair verification with seed 999999 for all reported recoveries.
8. Performs exact prospective statistical inference:
   - One-sided H0: AUC <= 0.5 test as label-permutation test (100,000 permutations, seed 20261010).
   - 10,000 stratified key-level bootstrap replicates for ROC-AUC, PR-AUC, Delta AUC, Delta PR-AUC.
   - Brier scores vs sample prevalence reference.
   - Exact paired McNemar tests for recovery differences.
   - Exact Wilson 95% CIs for recovery and false-trust rates.
   - Frozen threshold metrics at theta* = 0.0890.
   - Method-specific online wall-clock accounting:
     * B1 baselines do NOT pay G1 feature/inference cost (T_G1 = 0).
     * G1-Adaptive B1 pays T_B1 + T_G1 + T_SAT + T_verif.
     * R1 Tail Scheduling pays T_B1 + T_G1 + T_R1 (if tail reranked) + T_SAT + T_verif.
   - Standard common-cutoff PAR-2 at cutoffs C in {20s, 100s, 200s}.
9. Exports complete per-instance atomic JSON records and summary results.
"""

from __future__ import annotations

import dataclasses
import gc
import hashlib
import json
import math
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from neural_sat_speck.attacks.group_ranking import rank_pair_group_multi_subsets
from neural_sat_speck.attacks.recovery import (
    independent_verification_pairs,
    known_pairs_for_attack_instance,
    run_candidate_recovery,
)
from neural_sat_speck.confidence.stability import (
    extract_g1_features_from_rankings,
    score_curvature,
)
from neural_sat_speck.data.attack import (
    AttackInstanceConfig,
    generate_attack_instances,
    write_attack_shard,
)
from neural_sat_speck.data.seed_policy import (
    HISTORICAL_PHASE7_SEED,
    QUARANTINED_PHASE10_SEED,
    RETIRED_BLIND_SEED,
    SeedRole,
    validate_seed_for_role,
)
from neural_sat_speck.neural.group_model import PairGroupRanker
from neural_sat_speck.neural.numpy_inference import vectorized_final_round_features
from neural_sat_speck.sat.isolation import IsolatedSolverFactory

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiments" / "phase10_prospective_manifest.json"
DATASET_DIR = REPO_ROOT / "datasets" / "phase10_validation_r6"
OUTPUT_DIR = REPO_ROOT / "results" / "phase10_validation"

VALIDATION_SEED = 101001
INSTANCE_COUNT = 64
ROUNDS = 6
PAIR_COUNT = 64
FROZEN_THETA = 0.0890
VERIFICATION_SEED = 999999
PERM_TEST_SEED = 20261010
PERM_TEST_ITERATIONS = 100000
BOOTSTRAP_SEED = 20261011
BOOTSTRAP_ITERATIONS = 10000


class R1RescueNet(nn.Module):
    """Frozen R1 rescue architecture from Phase 10B."""

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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Compute Wilson score interval for binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    z = 1.95996 if confidence == 0.95 else 2.57583
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def exact_mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar test using binomial distribution."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    prob = sum(math.comb(n, i) * (0.5**n) for i in range(k + 1))
    return float(min(1.0, 2.0 * prob))


def compute_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute exact ROC-AUC with average tie handling via rank-sum."""
    n1 = int(np.sum(y_true))
    n0 = len(y_true) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    unique_scores, inverse = np.unique(scores, return_inverse=True)
    if len(unique_scores) < len(scores):
        tie_ranks = np.zeros_like(unique_scores, dtype=float)
        for i in range(len(unique_scores)):
            tie_ranks[i] = np.mean(ranks[scores == unique_scores[i]])
        ranks = tie_ranks[inverse]
    sum_r1 = np.sum(ranks[y_true == 1])
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


def run_label_permutation_test(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_perms: int = 100000,
    seed: int = 20261010,
) -> tuple[float, float, float]:
    """Fast vectorized label-permutation test for H0: AUC <= 0.5."""
    n1 = int(np.sum(y_true))
    N = len(y_true)
    n0 = N - n1
    if n1 == 0 or n0 == 0:
        return 0.5, 1.0, 0.0

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    unique_scores, inverse = np.unique(scores, return_inverse=True)
    if len(unique_scores) < len(scores):
        tie_ranks = np.zeros_like(unique_scores, dtype=float)
        for i in range(len(unique_scores)):
            tie_ranks[i] = np.mean(ranks[scores == unique_scores[i]])
        ranks = tie_ranks[inverse]

    obs_sum_r1 = np.sum(ranks[y_true == 1])
    obs_auc = float((obs_sum_r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0))

    t0 = time.perf_counter()
    rng = np.random.RandomState(seed)
    noise = rng.rand(n_perms, N)
    perm_pos_idx = np.argpartition(noise, n1, axis=1)[:, :n1]
    perm_sum_r1 = np.take(ranks, perm_pos_idx).sum(axis=1)
    perm_aucs = (perm_sum_r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)

    p_val = float(np.mean(perm_aucs >= obs_auc))
    elapsed = time.perf_counter() - t0
    return obs_auc, p_val, elapsed


def run_stratified_bootstrap(
    y_true: np.ndarray,
    scores_g0: np.ndarray,
    scores_g1: np.ndarray,
    n_reps: int = 10000,
    seed: int = 20261011,
) -> dict[str, Any]:
    """Stratified key-level bootstrap resampling positive and negative classes separately."""
    rng = np.random.RandomState(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    boot_auc_g0 = np.empty(n_reps)
    boot_auc_g1 = np.empty(n_reps)
    boot_pr_g0 = np.empty(n_reps)
    boot_pr_g1 = np.empty(n_reps)

    for b in range(n_reps):
        b_pos = rng.choice(pos_idx, size=n_pos, replace=True)
        b_neg = rng.choice(neg_idx, size=n_neg, replace=True)
        b_idx = np.concatenate([b_pos, b_neg])

        y_b = y_true[b_idx]
        s0_b = scores_g0[b_idx]
        s1_b = scores_g1[b_idx]

        boot_auc_g0[b] = compute_roc_auc(y_b, s0_b)
        boot_auc_g1[b] = compute_roc_auc(y_b, s1_b)
        boot_pr_g0[b] = compute_pr_auc(y_b, s0_b)
        boot_pr_g1[b] = compute_pr_auc(y_b, s1_b)

    boot_delta_auc = boot_auc_g1 - boot_auc_g0
    boot_delta_pr = boot_pr_g1 - boot_pr_g0

    return {
        "auc_g0_ci": [float(np.percentile(boot_auc_g0, 2.5)), float(np.percentile(boot_auc_g0, 97.5))],
        "auc_g1_ci": [float(np.percentile(boot_auc_g1, 2.5)), float(np.percentile(boot_auc_g1, 97.5))],
        "delta_auc_ci": [float(np.percentile(boot_delta_auc, 2.5)), float(np.percentile(boot_delta_auc, 97.5))],
        "pr_auc_g0_ci": [float(np.percentile(boot_pr_g0, 2.5)), float(np.percentile(boot_pr_g0, 97.5))],
        "pr_auc_g1_ci": [float(np.percentile(boot_pr_g1, 2.5)), float(np.percentile(boot_pr_g1, 97.5))],
        "pr_g0_ci": [float(np.percentile(boot_pr_g0, 2.5)), float(np.percentile(boot_pr_g0, 97.5))],
        "pr_g1_ci": [float(np.percentile(boot_pr_g1, 2.5)), float(np.percentile(boot_pr_g1, 97.5))],
        "delta_pr_auc_ci": [float(np.percentile(boot_delta_pr, 2.5)), float(np.percentile(boot_delta_pr, 97.5))],
    }


def main() -> None:
    print("==================================================================")
    print("PHASE 10 PROSPECTIVE VALIDATION EXECUTION (N=64, SEED 101001)")
    print("==================================================================\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Verify Seed Governance and Reject Quarantined Seeds
    print("--- 1. SEED GOVERNANCE & PREFLIGHT AUDIT ---")
    validate_seed_for_role(VALIDATION_SEED, SeedRole.PROSPECTIVE_VALIDATION)
    print(f"[✓] Seed {VALIDATION_SEED} validated for SeedRole.PROSPECTIVE_VALIDATION.")

    for q_seed in [HISTORICAL_PHASE7_SEED, QUARANTINED_PHASE10_SEED, RETIRED_BLIND_SEED]:
        try:
            validate_seed_for_role(q_seed, SeedRole.PROSPECTIVE_VALIDATION)
            raise RuntimeError(f"Quarantine breach! Seed {q_seed} was not rejected.")
        except ValueError:
            print(f"[✓] Quarantined seed {q_seed} successfully rejected.")

    # 2. Verify Manifest and Artifact Hashes
    manifest_data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    for model_key, model_info in manifest_data["artifact_hashes"].items():
        m_path = REPO_ROOT / model_info["path"]
        actual_hash = sha256_file(m_path)
        expected_hash = model_info["sha256"]
        assert actual_hash == expected_hash, f"Hash mismatch for {model_key}: {actual_hash} != {expected_hash}"
        print(f"[✓] Verified hash for {model_key}: {actual_hash[:16]}...")

    # 3. Generate Official Prospective Validation Family (N=64, seed 101001)
    print("\n--- 2. DETERMINISTIC DATASET GENERATION ---")
    git_commit = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
        .decode("utf-8")
        .strip()
    )
    dataset_cfg = AttackInstanceConfig(
        instance_count=INSTANCE_COUNT,
        pair_count=PAIR_COUNT,
        rounds=ROUNDS,
        seed=VALIDATION_SEED,
        split_role="validation",
        input_difference=(64, 0),
    )

    print(f"Generating official validation instances (N={INSTANCE_COUNT}, Seed={VALIDATION_SEED})...")
    instances_a = generate_attack_instances(
        dataset_cfg,
        generated_at="2026-09-10T00:00:00+08:00",
        repository_commit=git_commit,
    )
    instances_b = generate_attack_instances(
        dataset_cfg,
        generated_at="2026-09-10T00:00:00+08:00",
        repository_commit=git_commit,
    )
    assert instances_a.metadata["data_checksum"] == instances_b.metadata["data_checksum"], "Deterministic replay failed!"
    print(f"[✓] Deterministic replay verified: Checksum={instances_a.metadata['data_checksum']}")

    shard_path = DATASET_DIR / "validation.json"
    shard_record = write_attack_shard(
        instances_a,
        shard_path,
        dataset_id="speck32_64_attack_phase10_validation_r6",
    )
    shard_record["split_role"] = "prospective_validation"

    dataset_manifest = {
        "dataset_id": "speck32_64_attack_phase10_validation_r6",
        "data_role": "prospective_validation",
        "phase": 10,
        "rounds": ROUNDS,
        "pair_count": PAIR_COUNT,
        "instance_count": INSTANCE_COUNT,
        "seed": VALIDATION_SEED,
        "seed_role": SeedRole.PROSPECTIVE_VALIDATION,
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
        "input_difference": [64, 0],
        "splits": {
            "validation": shard_record,
        },
    }
    manifest_file = DATASET_DIR / "manifest.json"
    manifest_file.write_text(json.dumps(dataset_manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[✓] Saved official validation dataset to {shard_path} (SHA256: {shard_record['file_sha256']})")

    # 4. Load Frozen Models
    print("\n--- 3. LOADING FROZEN PROMOTIONAL MODELS ---")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Compute device for neural inference: {device}")

    b1_path = REPO_ROOT / manifest_data["artifact_hashes"]["b1_base_ranker"]["path"]
    b1_ckpt = torch.load(b1_path, map_location="cpu", weights_only=False)
    b1_model = PairGroupRanker(variant=b1_ckpt["variant"])
    b1_model.load_state_dict(b1_ckpt["state_dict"])
    b1_model.to(device)
    b1_model.eval()
    print(f"[✓] B1 base model loaded from {b1_path}")

    g0_path = REPO_ROOT / manifest_data["artifact_hashes"]["g0_scalar_baseline"]["path"]
    g0_data = json.loads(g0_path.read_text(encoding="utf-8"))
    print(f"[✓] G0 scalar model loaded from {g0_path}")

    g1_path = REPO_ROOT / manifest_data["artifact_hashes"]["g1_stability_estimator"]["path"]
    g1_data = json.loads(g1_path.read_text(encoding="utf-8"))
    g1_mu = np.array(g1_data["mean"], dtype=np.float64)
    g1_std = np.array(g1_data["std"], dtype=np.float64)
    g1_w = np.array(g1_data["weights"], dtype=np.float64)
    g1_b = float(g1_data["bias"])
    g1_features_list = g1_data["feature_names"]
    print(f"[✓] G1 stability model loaded from {g1_path}")

    r1_path = REPO_ROOT / manifest_data["artifact_hashes"]["r1_ablation_model"]["path"]
    r1_ckpt = torch.load(r1_path, map_location="cpu", weights_only=False)
    r1_net = R1RescueNet(in_dim=64, hidden_dim=64)
    r1_net.load_state_dict(r1_ckpt["state_dict"])
    r1_net.to(device)
    r1_net.eval()
    print(f"[✓] R1 ablation model loaded from {r1_path}")

    # 5. Execute Full Pipeline: Neural Ranking, Confidence, SAT, and Verification
    print("\n--- 4. EXECUTING PROSPECTIVE EXPERIMENTAL EVALUATION ---")
    instances = instances_a.instances

    methods = [
        "1_b1_k16",
        "2_b1_seq32",
        "3_g1_adaptive_b1",
        "4_r1_tail_ablation",
    ]

    method_results: dict[str, dict[str, Any]] = {
        m: {
            "recovered": 0,
            "candidates_attempted": [],
            "solver_seconds": [],
            "verification_seconds": [],
            "online_seconds": [],
            "outcomes": [],
        }
        for m in methods
    }

    CACHE_DIR = OUTPUT_DIR / "instance_cache"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    atomic_records: list[dict[str, Any]] = []

    t_start_all = time.perf_counter()

    for idx, inst in enumerate(instances):
        inst_start = time.perf_counter()
        cache_file = CACHE_DIR / f"instance_{idx:04d}.json"
        if cache_file.exists():
            try:
                rec = json.loads(cache_file.read_text(encoding="utf-8"))
                atomic_records.append(rec)
                for m_key, rec_key in [
                    ("1_b1_k16", "recovery_m1_b1_k16"),
                    ("2_b1_seq32", "recovery_m2_b1_seq32"),
                    ("3_g1_adaptive_b1", "recovery_m3_g1_adaptive_b1"),
                    ("4_r1_tail_ablation", "recovery_m4_r1_tail_ablation"),
                ]:
                    m_res = rec[rec_key]
                    method_results[m_key]["recovered"] += int(m_res["success"])
                    method_results[m_key]["candidates_attempted"].append(m_res["candidates_attempted"])
                    method_results[m_key]["solver_seconds"].append(m_res["solver_seconds"])
                    method_results[m_key]["verification_seconds"].append(m_res["verification_seconds"])
                    method_results[m_key]["online_seconds"].append(m_res["t_online"])
                    method_results[m_key]["outcomes"].append(m_res["status"])
                print(
                    f"[{idx+1:02d}/64] {inst.instance_id}: Loaded from cache (B1 Rank={rec['b1_true_rank']}, y_fail={rec['y_fail']})"
                )
                continue
            except Exception:
                pass

        # Step 4a: B1 multi-subset ranking (16, 32, 64 pairs)
        t_rank_start = time.perf_counter()
        multi_subsets = rank_pair_group_multi_subsets(
            inst,
            b1_model,
            subset_sizes=(16, 32, 64),
            candidate_batch_size=1024,
            device=device,
        )
        t_ranking = time.perf_counter() - t_rank_start

        r64 = multi_subsets[64]
        r32 = multi_subsets[32]
        r16 = multi_subsets[16]

        true_key = inst.final_round_subkey
        b1_candidate_order = tuple(r64.candidate_order)
        true_rank_b1 = b1_candidate_order.index(true_key) + 1
        y_fail = 1 if true_rank_b1 > 16 else 0
        in_top16 = bool(true_rank_b1 <= 16)
        in_top32 = bool(true_rank_b1 <= 32)

        # Step 4b: Extract G1 features and G0 feature
        t_g1_start = time.perf_counter()
        g1_feats = extract_g1_features_from_rankings(r64, r32, r16)
        g0_curv = score_curvature(r64.candidate_scores, top_k=16)

        # Evaluate G0
        g0_neg_curv = -g0_curv
        z_g0 = ((g0_neg_curv - g0_data["mean"]) / g0_data["std"]) * g0_data["weight"] + g0_data["bias"]
        p_g0 = float(1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, z_g0)))))

        # Evaluate G1
        x_g1_vec = np.array([getattr(g1_feats, fn) for fn in g1_features_list], dtype=np.float64)
        z_g1 = np.dot((x_g1_vec - g1_mu) / g1_std, g1_w) + g1_b
        p_g1 = float(1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, z_g1)))))
        t_g1 = time.perf_counter() - t_g1_start

        # G1 escalation decision at frozen threshold theta* = 0.0890
        g1_escalated = bool(p_g1 >= FROZEN_THETA)
        g1_budget = 32 if g1_escalated else 16

        # Step 4c: Prepare constraint pairs and independent verification pairs
        all_known = known_pairs_for_attack_instance(inst)
        known_pairs = all_known[:4]
        verif_pairs = independent_verification_pairs(
            inst.master_key, rounds=ROUNDS, pair_count=2, seed=VERIFICATION_SEED
        )

        # Step 4d: Prepare candidate orders
        b1_top16 = b1_candidate_order[:16]
        b1_top32 = b1_candidate_order[:32]
        plain_tail = b1_candidate_order[16:32]

        # R1 tail reranking (M=32, reranking ranks 17..32)
        t_r1 = 0.0
        with torch.no_grad():
            t_r1_start = time.perf_counter()
            r1_feats = torch.from_numpy(
                vectorized_final_round_features(inst.pairs[:64], list(plain_tail))
            ).to(device)
            r1_embs = b1_model.pair_embeddings(r1_feats).mean(dim=1)
            r1_scores = r1_net(r1_embs).cpu().numpy()
            r1_sorted_tail = tuple(c for c, _ in sorted(zip(plain_tail, r1_scores), key=lambda x: -x[1]))
            t_r1 = time.perf_counter() - t_r1_start

        r1_candidate_order = b1_top16 + r1_sorted_tail

        # Step 4e: Isolated solver execution across methods
        factory = IsolatedSolverFactory.from_pairs(known_pairs, rounds=ROUNDS, instance_id=inst.instance_id)

        # --- Method 1: B1 (kmax=16) ---
        m1 = factory.build_isolated_model("b1_k16")
        res1 = run_candidate_recovery(
            m1,
            b1_top16,
            known_pairs,
            verif_pairs,
            candidate_budget=16,
            solver_timeout=10.0,
            strategy="b1_k16",
        )
        t_online_m1 = t_ranking + 0.0 + res1.solver_seconds + res1.verification_seconds
        method_results["1_b1_k16"]["recovered"] += int(res1.success)
        method_results["1_b1_k16"]["candidates_attempted"].append(res1.candidates_attempted)
        method_results["1_b1_k16"]["solver_seconds"].append(res1.solver_seconds)
        method_results["1_b1_k16"]["verification_seconds"].append(res1.verification_seconds)
        method_results["1_b1_k16"]["online_seconds"].append(t_online_m1)
        method_results["1_b1_k16"]["outcomes"].append(res1.status)

        # --- Method 2: B1 Sequential-32 (kmax=32) ---
        m2 = factory.build_isolated_model("b1_seq32")
        res2 = run_candidate_recovery(
            m2,
            b1_top32,
            known_pairs,
            verif_pairs,
            candidate_budget=32,
            solver_timeout=10.0,
            strategy="b1_seq32",
        )
        t_online_m2 = t_ranking + 0.0 + res2.solver_seconds + res2.verification_seconds
        method_results["2_b1_seq32"]["recovered"] += int(res2.success)
        method_results["2_b1_seq32"]["candidates_attempted"].append(res2.candidates_attempted)
        method_results["2_b1_seq32"]["solver_seconds"].append(res2.solver_seconds)
        method_results["2_b1_seq32"]["verification_seconds"].append(res2.verification_seconds)
        method_results["2_b1_seq32"]["online_seconds"].append(t_online_m2)
        method_results["2_b1_seq32"]["outcomes"].append(res2.status)

        # --- Method 3: G1-Adaptive B1 (theta=0.0890, kmax in {16, 32}) ---
        m3 = factory.build_isolated_model("g1_adaptive_b1")
        res3 = run_candidate_recovery(
            m3,
            b1_top32[:g1_budget],
            known_pairs,
            verif_pairs,
            candidate_budget=g1_budget,
            solver_timeout=10.0,
            strategy="g1_adaptive_b1",
        )
        t_online_m3 = t_ranking + t_g1 + res3.solver_seconds + res3.verification_seconds
        method_results["3_g1_adaptive_b1"]["recovered"] += int(res3.success)
        method_results["3_g1_adaptive_b1"]["candidates_attempted"].append(res3.candidates_attempted)
        method_results["3_g1_adaptive_b1"]["solver_seconds"].append(res3.solver_seconds)
        method_results["3_g1_adaptive_b1"]["verification_seconds"].append(res3.verification_seconds)
        method_results["3_g1_adaptive_b1"]["online_seconds"].append(t_online_m3)
        method_results["3_g1_adaptive_b1"]["outcomes"].append(res3.status)

        # --- Method 4: R1 Tail Scheduling (kmax=32, ablation) ---
        m4 = factory.build_isolated_model("r1_tail_ablation")
        res4 = run_candidate_recovery(
            m4,
            r1_candidate_order,
            known_pairs,
            verif_pairs,
            candidate_budget=32,
            solver_timeout=10.0,
            strategy="r1_tail_ablation",
        )
        t_r1_applied = t_r1 if res4.candidates_attempted > 16 else 0.0
        t_online_m4 = t_ranking + t_g1 + t_r1_applied + res4.solver_seconds + res4.verification_seconds
        method_results["4_r1_tail_ablation"]["recovered"] += int(res4.success)
        method_results["4_r1_tail_ablation"]["candidates_attempted"].append(res4.candidates_attempted)
        method_results["4_r1_tail_ablation"]["solver_seconds"].append(res4.solver_seconds)
        method_results["4_r1_tail_ablation"]["verification_seconds"].append(res4.verification_seconds)
        method_results["4_r1_tail_ablation"]["online_seconds"].append(t_online_m4)
        method_results["4_r1_tail_ablation"]["outcomes"].append(res4.status)

        # Verify solver isolation across all models
        factory.verify_isolation({
            "b1_k16": m1,
            "b1_seq32": m2,
            "g1_adaptive_b1": m3,
            "r1_tail_ablation": m4,
        })

        rec = {
            "index": idx,
            "instance_id": inst.instance_id,
            "true_subkey": f"0x{true_key:04x}",
            "b1_true_rank": true_rank_b1,
            "y_fail": y_fail,
            "in_top16": in_top16,
            "in_top32": in_top32,
            "g0_curvature": float(g0_curv),
            "p_g0_fail": p_g0,
            "p_g1_fail": p_g1,
            "g1_escalated": g1_escalated,
            "g1_allocated_budget": g1_budget,
            "g1_features": dataclasses.asdict(g1_feats),
            "timing_components": {
                "t_ranking": t_ranking,
                "t_g1": t_g1,
                "t_r1": t_r1,
            },
            "recovery_m1_b1_k16": {
                "success": res1.success,
                "status": res1.status,
                "candidates_attempted": res1.candidates_attempted,
                "solver_seconds": res1.solver_seconds,
                "verification_seconds": res1.verification_seconds,
                "t_online": t_online_m1,
            },
            "recovery_m2_b1_seq32": {
                "success": res2.success,
                "status": res2.status,
                "candidates_attempted": res2.candidates_attempted,
                "solver_seconds": res2.solver_seconds,
                "verification_seconds": res2.verification_seconds,
                "t_online": t_online_m2,
            },
            "recovery_m3_g1_adaptive_b1": {
                "success": res3.success,
                "status": res3.status,
                "candidates_attempted": res3.candidates_attempted,
                "solver_seconds": res3.solver_seconds,
                "verification_seconds": res3.verification_seconds,
                "t_online": t_online_m3,
            },
            "recovery_m4_r1_tail_ablation": {
                "success": res4.success,
                "status": res4.status,
                "candidates_attempted": res4.candidates_attempted,
                "solver_seconds": res4.solver_seconds,
                "verification_seconds": res4.verification_seconds,
                "t_online": t_online_m4,
            },
        }
        cache_file.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
        atomic_records.append(rec)

        inst_elapsed = time.perf_counter() - inst_start
        print(
            f"[{idx+1:02d}/64] {inst.instance_id}: B1 Rank={true_rank_b1:4d} (y_fail={y_fail}) | "
            f"p_G1={p_g1:.4f} (Escalated={g1_escalated}) | "
            f"Rec: M1={int(res1.success)} M2={int(res2.success)} M3={int(res3.success)} M4={int(res4.success)} | "
            f"Time={inst_elapsed:.2f}s",
            flush=True,
        )

        if idx % 8 == 0:
            gc.collect()

    t_total_experiment = time.perf_counter() - t_start_all
    print(f"\n[✓] All 64 instances completed in {t_total_experiment:.1f}s ({t_total_experiment/60:.2f}m)!\n")

    # 6. Statistical Inference and Metrics Evaluation
    print("--- 5. STATISTICAL INFERENCE & HYPOTHESIS TESTS ---")
    y_fail_all = np.array([r["y_fail"] for r in atomic_records], dtype=np.int32)
    scores_g0_all = np.array([r["p_g0_fail"] for r in atomic_records], dtype=np.float64)
    scores_g1_all = np.array([r["p_g1_fail"] for r in atomic_records], dtype=np.float64)

    n_failures = int(np.sum(y_fail_all))
    failure_rate = n_failures / INSTANCE_COUNT
    print(f"B1 Top-16 Failures (y_fail=1): {n_failures} / 64 ({failure_rate*100:.2f}%)")

    # Reliability Metrics: ROC-AUC and PR-AUC
    auc_g0 = compute_roc_auc(y_fail_all, scores_g0_all)
    auc_g1 = compute_roc_auc(y_fail_all, scores_g1_all)
    delta_auc = auc_g1 - auc_g0

    pr_g0 = compute_pr_auc(y_fail_all, scores_g0_all)
    pr_g1 = compute_pr_auc(y_fail_all, scores_g1_all)
    delta_pr = pr_g1 - pr_g0

    # Brier Scores
    brier_g0 = float(np.mean((scores_g0_all - y_fail_all) ** 2))
    brier_g1 = float(np.mean((scores_g1_all - y_fail_all) ** 2))
    brier_prev = float(failure_rate * (1.0 - failure_rate))

    # Stratified Bootstrap (10,000 replicates)
    print("Running 10,000 stratified key-level bootstrap replicates...")
    boot_res = run_stratified_bootstrap(
        y_fail_all, scores_g0_all, scores_g1_all, n_reps=BOOTSTRAP_ITERATIONS, seed=BOOTSTRAP_SEED
    )

    # Label-Permutation Tests (100,000 permutations)
    print("Running 100,000 label-permutation test for G1 and G0...")
    _, p_val_g1_perm, perm_elapsed_g1 = run_label_permutation_test(
        y_fail_all, scores_g1_all, n_perms=PERM_TEST_ITERATIONS, seed=PERM_TEST_SEED
    )
    _, p_val_g0_perm, perm_elapsed_g0 = run_label_permutation_test(
        y_fail_all, scores_g0_all, n_perms=PERM_TEST_ITERATIONS, seed=PERM_TEST_SEED + 1
    )

    print(f"G0 Scalar Baseline: ROC-AUC = {auc_g0:.4f} (95% CI {boot_res['auc_g0_ci']}) | PR-AUC = {pr_g0:.4f} (95% CI {boot_res['pr_auc_g0_ci']}) | Permutation p = {p_val_g0_perm:.5f}")
    print(f"G1 Stability Model: ROC-AUC = {auc_g1:.4f} (95% CI {boot_res['auc_g1_ci']}) | PR-AUC = {pr_g1:.4f} (95% CI {boot_res['pr_auc_g1_ci']}) | Permutation p = {p_val_g1_perm:.5f}")
    print(f"Delta AUC (G1 - G0): {delta_auc:+.4f} (95% CI {boot_res['delta_auc_ci']})")
    print(f"Delta PR-AUC (G1 - G0): {delta_pr:+.4f} (95% CI {boot_res['delta_pr_auc_ci']})")
    print(f"Brier Scores: G0 = {brier_g0:.4f}, G1 = {brier_g1:.4f}, Prevalence Baseline = {brier_prev:.4f}")

    # Frozen Threshold Metrics at theta* = 0.0890
    trusted_mask = scores_g1_all < FROZEN_THETA
    escalated_mask = scores_g1_all >= FROZEN_THETA
    n_trusted = int(np.sum(trusted_mask))
    n_escalated = int(np.sum(escalated_mask))

    trusted_coverage = n_trusted / INSTANCE_COUNT
    escalation_freq = n_escalated / INSTANCE_COUNT

    false_trust_count = int(np.sum(trusted_mask & (y_fail_all == 1)))
    false_trust_rate = false_trust_count / n_trusted if n_trusted > 0 else 0.0
    ft_ci_low, ft_ci_high = wilson_ci(false_trust_count, n_trusted)

    escalated_failures = int(np.sum(escalated_mask & (y_fail_all == 1)))
    failure_recall = escalated_failures / n_failures if n_failures > 0 else 1.0

    print(f"\n--- FROZEN THRESHOLD METRICS (theta* = {FROZEN_THETA}) ---")
    print(f"Trusted/Direct Coverage : {n_trusted}/64 ({trusted_coverage*100:.2f}%)")
    print(f"Escalation Frequency    : {n_escalated}/64 ({escalation_freq*100:.2f}%)")
    print(f"False-Trust Count       : {false_trust_count} (Rate: {false_trust_rate*100:.2f}%, 95% Wilson CI [{ft_ci_low*100:.2f}%, {ft_ci_high*100:.2f}%])")
    print(f"Failure-Escalation Recall: {escalated_failures}/{n_failures} ({failure_recall*100:.2f}%)")

    # 7. Operational Recovery & PAR-2 Evaluation
    print("\n--- 6. OPERATIONAL RECOVERY & PAR-2 RESULTS ---")
    cutoffs = [20.0, 100.0, 200.0]

    par2_results: dict[str, dict[str, float]] = {m: {} for m in methods}
    summary_methods: dict[str, Any] = {}

    labels = {
        "1_b1_k16": "1. B1 (kmax=16)",
        "2_b1_seq32": "2. B1 Sequential-32 (kmax=32)",
        "3_g1_adaptive_b1": "3. G1-Adaptive B1 (theta=0.0890, kmax in {16,32})",
        "4_r1_tail_ablation": "4. R1 Tail Scheduling (kmax=32) [Ablation]",
    }

    header = f"{'Method':<46} | {'Recovery':<18} | {'Mean Cands':<10} | {'Mean SAT':<9} | {'T_online':<9} | {'PAR2 (20s)':<10} | {'PAR2 (100s)':<11} | {'PAR2 (200s)':<11}"
    print(header)
    print("-" * len(header))

    for m_key in methods:
        m_data = method_results[m_key]
        n_rec = m_data["recovered"]
        pct_rec = n_rec / INSTANCE_COUNT * 100.0
        ci_low, ci_high = wilson_ci(n_rec, INSTANCE_COUNT)
        mean_cands = float(np.mean(m_data["candidates_attempted"]))
        mean_sat = float(np.mean(m_data["solver_seconds"]))
        mean_ver = float(np.mean(m_data["verification_seconds"]))
        mean_online = float(np.mean(m_data["online_seconds"]))

        scores_by_c = {}
        for C in cutoffs:
            scores = []
            for i in range(INSTANCE_COUNT):
                t_on = m_data["online_seconds"][i]
                succ = (m_data["outcomes"][i] == "recovered")
                if succ and t_on <= C:
                    scores.append(t_on)
                else:
                    scores.append(2.0 * C)
            par2_val = float(np.mean(scores))
            scores_by_c[f"par2_{int(C)}s"] = par2_val
            par2_results[m_key][f"par2_{int(C)}s"] = par2_val

        summary_methods[m_key] = {
            "label": labels[m_key],
            "recovered": n_rec,
            "success_rate": pct_rec,
            "wilson_95_ci": [ci_low, ci_high],
            "mean_candidates_tested": mean_cands,
            "mean_solver_seconds": mean_sat,
            "mean_verification_seconds": mean_ver,
            "mean_online_seconds": mean_online,
            "par2": scores_by_c,
        }

        print(
            f"{labels[m_key]:<46} | {n_rec:>2d}/64 ({pct_rec:>5.2f}%) | "
            f"{mean_cands:>10.2f} | {mean_sat:>8.3f}s | {mean_online:>8.2f}s | "
            f"{scores_by_c['par2_20s']:>9.2f}s | {scores_by_c['par2_100s']:>10.2f}s | {scores_by_c['par2_200s']:>10.2f}s"
        )

    # 8. Paired Contingency Tables and McNemar Tests
    print("\n--- 7. PAIRED RECOVERY COMPARISONS & MCNEMAR TESTS ---")
    succ_m1 = np.array([r == "recovered" for r in method_results["1_b1_k16"]["outcomes"]], dtype=bool)
    succ_m2 = np.array([r == "recovered" for r in method_results["2_b1_seq32"]["outcomes"]], dtype=bool)
    succ_m3 = np.array([r == "recovered" for r in method_results["3_g1_adaptive_b1"]["outcomes"]], dtype=bool)
    succ_m4 = np.array([r == "recovered" for r in method_results["4_r1_tail_ablation"]["outcomes"]], dtype=bool)

    # M1 vs M2
    b_12 = int(np.sum(succ_m1 & ~succ_m2))
    c_12 = int(np.sum(~succ_m1 & succ_m2))
    p_mcnemar_12 = exact_mcnemar(b_12, c_12)
    print(f"B1(16) vs B1 Seq-32      : Discordant pairs (b={b_12}, c={c_12}) -> Exact McNemar p = {p_mcnemar_12:.5f}")

    # M2 vs M3
    b_23 = int(np.sum(succ_m2 & ~succ_m3))
    c_23 = int(np.sum(~succ_m2 & succ_m3))
    p_mcnemar_23 = exact_mcnemar(b_23, c_23)
    print(f"B1 Seq-32 vs G1-Adaptive : Discordant pairs (b={b_23}, c={c_23}) -> Exact McNemar p = {p_mcnemar_23:.5f}")

    # M2 vs M4
    b_24 = int(np.sum(succ_m2 & ~succ_m4))
    c_24 = int(np.sum(~succ_m2 & succ_m4))
    p_mcnemar_24 = exact_mcnemar(b_24, c_24)
    print(f"B1 Seq-32 vs R1 Tail     : Discordant pairs (b={b_24}, c={c_24}) -> Exact McNemar p = {p_mcnemar_24:.5f}")

    # 9. Save All Output JSON Files
    full_results = {
        "manifest": manifest_data,
        "validation_metadata": {
            "dataset_id": "speck32_64_attack_phase10_validation_r6",
            "seed": VALIDATION_SEED,
            "instance_count": INSTANCE_COUNT,
            "rounds": ROUNDS,
            "pairs_per_instance": PAIR_COUNT,
            "timestamp": datetime.now(UTC).isoformat(),
            "git_commit": git_commit,
            "total_execution_seconds": t_total_experiment,
        },
        "containment_summary": {
            "failures_total": n_failures,
            "failure_rate": failure_rate,
            "containment_top16": int(np.sum(y_fail_all == 0)),
            "containment_top32": sum(1 for r in atomic_records if r["in_top32"]),
        },
        "reliability_inference": {
            "g0_scalar": {
                "roc_auc": auc_g0,
                "roc_auc_95_ci": boot_res["auc_g0_ci"],
                "pr_auc": pr_g0,
                "pr_auc_95_ci": boot_res["pr_auc_g0_ci"],
                "brier_score": brier_g0,
                "permutation_test_p_value": p_val_g0_perm,
            },
            "g1_stability": {
                "roc_auc": auc_g1,
                "roc_auc_95_ci": boot_res["auc_g1_ci"],
                "pr_auc": pr_g1,
                "pr_auc_95_ci": boot_res["pr_auc_g1_ci"],
                "brier_score": brier_g1,
                "permutation_test_p_value": p_val_g1_perm,
            },
            "comparative": {
                "delta_roc_auc": delta_auc,
                "delta_roc_auc_95_ci": boot_res["delta_auc_ci"],
                "delta_pr_auc": delta_pr,
                "delta_pr_auc_95_ci": boot_res["delta_pr_auc_ci"],
                "brier_prevalence_baseline": brier_prev,
            },
            "frozen_threshold_metrics": {
                "theta": FROZEN_THETA,
                "trusted_coverage": trusted_coverage,
                "escalation_frequency": escalation_freq,
                "false_trust_count": false_trust_count,
                "false_trust_rate": false_trust_rate,
                "false_trust_rate_95_ci": [ft_ci_low, ft_ci_high],
                "failure_escalation_recall": failure_recall,
            },
        },
        "recovery_summary": summary_methods,
        "paired_significance_tests": {
            "b1_16_vs_b1_seq32": {"b": b_12, "c": c_12, "mcnemar_p": p_mcnemar_12},
            "b1_seq32_vs_g1_adaptive": {"b": b_23, "c": c_23, "mcnemar_p": p_mcnemar_23},
            "b1_seq32_vs_r1_tail": {"b": b_24, "c": c_24, "mcnemar_p": p_mcnemar_24},
        },
    }

    results_file = OUTPUT_DIR / "prospective_validation_results.json"
    results_file.write_text(json.dumps(full_results, indent=2) + "\n", encoding="utf-8")
    print(f"\n[✓] Saved summary results to {results_file}")

    atomic_file = OUTPUT_DIR / "validation_atomic_records.json"
    atomic_file.write_text(json.dumps(atomic_records, indent=2) + "\n", encoding="utf-8")
    print(f"[✓] Saved per-instance atomic records to {atomic_file}")

    print("\n==================================================================")
    print("PHASE 10 PROSPECTIVE VALIDATION COMPLETE")
    print("==================================================================")


if __name__ == "__main__":
    main()
