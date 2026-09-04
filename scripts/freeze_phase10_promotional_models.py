#!/usr/bin/env python3
"""Freeze Phase 10 promotional models fit on all N=192 development keys.

Fits:
1. Final G1 estimator on all 192 development keys using frozen lambda=2.0.
2. Final R1 rescue net on all development hard negatives for M=32.
Saves checkpoints and outputs SHA256 hashes.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from neural_sat_speck.attacks.r1 import R1HardNegativeMiner  # noqa: E402
from neural_sat_speck.data.attack import load_attack_shard  # noqa: E402
from neural_sat_speck.data.seed_policy import Phase10DataRole  # noqa: E402
from neural_sat_speck.neural.group_model import PairGroupRanker  # noqa: E402
from neural_sat_speck.neural.numpy_inference import (  # noqa: E402
    vectorized_final_round_features,
)
from scripts.forensic_development_audit import (  # noqa: E402
    R1RescueNet,
    fit_logistic_regression,
)

DATASET_PATH = REPO_ROOT / "datasets" / "phase10_development_r6" / "development.json"
CACHE_DIR = REPO_ROOT / "results" / "phase10_development" / "ranking_cache"
MODELS_DIR = REPO_ROOT / "models"
B1_MODEL_PATH = MODELS_DIR / "speck32_64_phase6a_b1_mean_g64.pt"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    print("=== Freezing Phase 10 Promotional Models on All 192 Development Keys ===")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    cache_files = sorted(CACHE_DIR.glob("instance_*.json"))
    records = [json.loads(f.read_text(encoding="utf-8")) for f in cache_files]
    N = len(records)
    instances = load_attack_shard(
        DATASET_PATH,
        expected_dataset_id="speck32_64_attack_phase10_development_r6",
    ).instances

    y_fail = np.array([r["y_fail"] for r in records], dtype=np.int32)
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

    # 1. Fit Final G1 on all 192 keys
    mu_g1 = np.mean(X_g1, axis=0)
    std_g1 = np.std(X_g1, axis=0) + 1e-8
    w_g1, b_g1 = fit_logistic_regression(
        (X_g1 - mu_g1) / std_g1, y_fail, l2_lambda=2.0
    )

    g1_ckpt = {
        "model_type": "G1_LogisticRegression",
        "feature_names": feature_names,
        "mean": mu_g1.tolist(),
        "std": std_g1.tolist(),
        "weights": w_g1.tolist(),
        "bias": float(b_g1),
        "frozen_threshold": 0.0892,
        "n_training_samples": N,
        "seed": 100001,
    }
    g1_path = MODELS_DIR / "phase10_promotional_g1_estimator.json"
    g1_path.write_text(json.dumps(g1_ckpt, indent=2) + "\n", encoding="utf-8")
    g1_hash = sha256_file(g1_path)
    print(f"[✓] Saved G1 promotional model: {g1_path}")
    print(f"    SHA256: {g1_hash}")

    # 2. Fit Final R1 on all development hard negatives (M=32)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    b1_ckpt = torch.load(B1_MODEL_PATH, map_location="cpu", weights_only=False)
    b1_model = PairGroupRanker(variant=b1_ckpt["variant"])
    b1_model.load_state_dict(b1_ckpt["state_dict"])
    b1_model.to(device)
    b1_model.eval()

    miner = R1HardNegativeMiner(shortlist_size=32, target_rank_cutoff=16)
    mined_all = []
    for idx in range(N):
        rec = records[idx]
        inst = instances[idx]
        mined = miner.mine_instance(
            rec["instance_id"],
            rec["b1_candidate_order_top128"],
            inst.final_round_subkey,
            data_role=Phase10DataRole.DEVELOPMENT_CALIBRATION,
        )
        if mined.hard_negatives and mined.true_subkey_in_shortlist:
            mined_all.append((inst, mined))

    torch.manual_seed(100001)
    r1_net = R1RescueNet(in_dim=64, hidden_dim=64).to(device)
    opt = torch.optim.Adam(r1_net.parameters(), lr=1e-3, weight_decay=1e-4)

    train_embs_pos = []
    train_embs_neg = []
    with torch.no_grad():
        for inst_t, mined_t in mined_all:
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

    P_t = torch.cat(train_embs_pos, dim=0)
    N_t = torch.cat(train_embs_neg, dim=0)
    for _ in range(50):
        opt.zero_grad()
        s_pos = r1_net(P_t)
        s_neg = r1_net(N_t)
        loss = torch.mean(torch.clamp(0.1 - (s_pos - s_neg), min=0.0))
        loss.backward()
        opt.step()

    r1_path = MODELS_DIR / "phase10_promotional_r1_rescue_net.pt"
    torch.save({
        "model_type": "R1RescueNet",
        "in_dim": 64,
        "hidden_dim": 64,
        "state_dict": r1_net.state_dict(),
        "shortlist_size": 32,
        "seed": 100001,
    }, r1_path)
    r1_hash = sha256_file(r1_path)
    print(f"[✓] Saved R1 promotional model: {r1_path}")
    print(f"    SHA256: {r1_hash}")

    manifest = {
        "dataset_seed": 100001,
        "dataset_samples": 192,
        "dataset_sha256": sha256_file(DATASET_PATH),
        "g1_model": {
            "path": str(g1_path.relative_to(REPO_ROOT)),
            "sha256": g1_hash,
            "threshold": 0.0892,
        },
        "r1_model": {
            "path": str(r1_path.relative_to(REPO_ROOT)),
            "sha256": r1_hash,
            "shortlist_size": 32,
        },
        "b1_base_model": {
            "path": str(B1_MODEL_PATH.relative_to(REPO_ROOT)),
            "sha256": sha256_file(B1_MODEL_PATH),
        }
    }
    manifest_path = REPO_ROOT / "results" / "phase10_development" / "frozen_promotional_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[✓] Saved promotional manifest: {manifest_path}")


if __name__ == "__main__":
    main()
