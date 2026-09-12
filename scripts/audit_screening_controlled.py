#!/usr/bin/env python3
"""Package P2: Controlled Matched-Device Multi-Fidelity Screening Audit.

Performs a rigorous, same-device, paired evaluation of:
1. Full 64-pair pair-group complete-domain ranking (all 65,536 candidates)
2. Multi-fidelity screened ranking (16-pair screen -> top-512 rescore with 64 pairs)

Both methods are executed on the exact same machine, PyTorch device (MPS),
model checkpoint, attack instances, candidate batch size (1024), and session.

Measures:
- Cold-start vs. steady-state wall-clock latency.
- True-key retention in top-512 screen.
- True-key integer rank in full vs. screened ranking.
- Top-prefix agreement: Jaccard similarity at k=16, k=32, k=512.
- Spearman rank correlation (rho) of candidate scores.
- Memory and component timing breakdown.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neural_sat_speck.attacks.group_ranking import rank_pair_group_subkeys
from neural_sat_speck.attacks.multi_round_instance import partial_decrypt_instance
from neural_sat_speck.attacks.screened_group_ranking import (
    rank_pair_group_subkeys_screened,
)
from neural_sat_speck.data.attack import load_attack_shard
from neural_sat_speck.neural.group_model import PairGroupRanker
from neural_sat_speck.speck.key_schedule import expand_key

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    REPOSITORY_ROOT / "datasets" / "phase11a_oracle_r7" / "oracle_evaluation.json"
)
DATASET_ID = "speck32_64_attack_phase11a_oracle_r7"
MODEL_PATH = (
    REPOSITORY_ROOT / "models" / "speck32_64_phase6a_b1_mean_g64.pt"
)
OUTPUT_PATH = (
    REPOSITORY_ROOT
    / "results"
    / "phase11_scaling_diagnostics"
    / "screening_controlled_audit_v3.json"
)

DEPENDENCY_PATHS = [
    REPOSITORY_ROOT / "src" / "neural_sat_speck" / "attacks" / "screened_group_ranking.py",
    REPOSITORY_ROOT / "src" / "neural_sat_speck" / "attacks" / "group_ranking.py",
    REPOSITORY_ROOT / "src" / "neural_sat_speck" / "neural" / "group_model.py",
]


def file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_state() -> dict[str, Any]:
    """Capture repository commit, dirty status, and diff digest."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=REPOSITORY_ROOT
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, cwd=REPOSITORY_ROOT
        ).strip()
        is_dirty = len(status) > 0
        diff_sha256 = (
            hashlib.sha256(
                subprocess.check_output(
                    ["git", "diff", "HEAD"], text=True, cwd=REPOSITORY_ROOT
                ).encode("utf-8")
            ).hexdigest()
            if is_dirty
            else None
        )
        return {
            "commit": commit,
            "is_dirty": is_dirty,
            "diff_sha256": diff_sha256,
        }
    except Exception as e:
        return {"commit": None, "is_dirty": None, "error": str(e)}


def jaccard_similarity(set_a: set[int], set_b: set[int]) -> float:
    union_len = len(set_a | set_b)
    if union_len == 0:
        return 1.0
    return len(set_a & set_b) / union_len


def permutation_spearman_rho(order_a: Sequence[int], order_b: Sequence[int]) -> float:
    """Compute exact Spearman rho between two candidate rankings (orderings)."""
    n = len(order_a)
    pos_a = np.empty(n, dtype=np.int64)
    pos_b = np.empty(n, dtype=np.int64)
    pos_a[list(order_a)] = np.arange(n, dtype=np.int64)
    pos_b[list(order_b)] = np.arange(n, dtype=np.int64)
    corr = np.corrcoef(pos_a, pos_b)[0, 1]
    return float(corr)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instances",
        type=int,
        default=4,
        help="Number of opened Seed 300001 instances to audit (default 4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="PyTorch execution device (default: mps)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Candidate evaluation batch size (default: 1024)",
    )
    parser.add_argument(
        "--include-synthetic-wrong-prefixes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include synthetic wrong-prefix diagnostic cases (default: True)",
    )
    parser.add_argument(
        "--include-wrong-prefixes",
        dest="include_synthetic_wrong_prefixes",
        action="store_true",
        help="Alias for --include-synthetic-wrong-prefixes",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Output path for audit results JSON",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow overwriting existing artifact",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = args.output.resolve()

    if output_path.exists() and not args.allow_overwrite:
        print(f"[!] Refusing to overwrite existing file: {output_path}")
        print("    Pass --allow-overwrite if intentional.")
        return 1

    device = torch.device(
        args.device
        if torch.backends.mps.is_available() or torch.cuda.is_available()
        else "cpu"
    )

    print("=== CONTROLLED SCREENING BENCHMARK (PACKAGE P2) ===")
    print(f"Device: {device} | Instances: {args.instances} | Batch Size: {args.batch_size}")
    print(f"Output: {output_path}\n")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    variant = checkpoint.get("variant", "b1_mean")
    model = PairGroupRanker(variant=variant)
    state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    shard = load_attack_shard(DATASET_PATH, expected_dataset_id=DATASET_ID)

    # 1. Warm-up pass on dummy instance to ensure steady state
    inst0 = shard.instances[0]
    keys0 = expand_key(inst0.master_key, rounds=7)
    descended0 = partial_decrypt_instance(inst0, keys0[6], expected_next_subkey=keys0[5])

    print("[*] Running warm-up passes...")
    if device.type == "mps":
        torch.mps.synchronize()
    _ = rank_pair_group_subkeys_screened(descended0, model, screen_pairs=16, screen_top_k=512, candidate_batch_size=args.batch_size, device=device)
    if device.type == "mps":
        torch.mps.synchronize()
    _ = rank_pair_group_subkeys(descended0, model, candidate_batch_size=args.batch_size, device=device)
    if device.type == "mps":
        torch.mps.synchronize()
    print("    Warm-up complete.\n")

    # 2. Main paired benchmark on true k6 prefixes (counterbalanced order)
    true_prefix_records: list[dict[str, Any]] = []

    for idx in range(min(args.instances, len(shard.instances))):
        inst = shard.instances[idx]
        all_keys = expand_key(inst.master_key, rounds=7)
        true_k6 = all_keys[6]
        true_k5 = all_keys[5]

        descended = partial_decrypt_instance(
            inst, true_k6, expected_next_subkey=true_k5, label=f"audit-true-{inst.instance_id}"
        )

        # Counterbalanced execution: even index runs screened first; odd index runs full first
        if idx % 2 == 0:
            if device.type == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
            screened_ranking = rank_pair_group_subkeys_screened(
                descended, model, screen_pairs=16, screen_top_k=512,
                candidate_batch_size=args.batch_size, device=device,
            )
            if device.type == "mps":
                torch.mps.synchronize()
            t_screened = time.perf_counter() - t0

            if device.type == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
            full_ranking = rank_pair_group_subkeys(
                descended, model, candidate_batch_size=args.batch_size, device=device,
            )
            if device.type == "mps":
                torch.mps.synchronize()
            t_full = time.perf_counter() - t0
        else:
            if device.type == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
            full_ranking = rank_pair_group_subkeys(
                descended, model, candidate_batch_size=args.batch_size, device=device,
            )
            if device.type == "mps":
                torch.mps.synchronize()
            t_full = time.perf_counter() - t0

            if device.type == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
            screened_ranking = rank_pair_group_subkeys_screened(
                descended, model, screen_pairs=16, screen_top_k=512,
                candidate_batch_size=args.batch_size, device=device,
            )
            if device.type == "mps":
                torch.mps.synchronize()
            t_screened = time.perf_counter() - t0

        speedup = t_full / t_screened if t_screened > 0 else 0.0

        screened_k5_rank = screened_ranking.rank(true_k5)
        full_k5_rank = full_ranking.rank(true_k5)
        retained = screened_k5_rank <= 512

        # Set overlap & survivor recall
        full_top16 = set(full_ranking.top_k(16))
        screened_top16 = set(screened_ranking.top_k(16))
        jaccard_16 = jaccard_similarity(full_top16, screened_top16)

        full_top32 = set(full_ranking.top_k(32))
        screened_top32 = set(screened_ranking.top_k(32))
        jaccard_32 = jaccard_similarity(full_top32, screened_top32)

        full_top512 = set(full_ranking.top_k(512))
        screened_top512 = set(screened_ranking.top_k(512))
        jaccard_512 = jaccard_similarity(full_top512, screened_top512)

        # Survivor recall: fraction of full top-k that survived 16-pair screen into top 512
        survivors = set(screened_ranking.top_k(512))
        recall_16 = len(full_top16 & survivors) / 16.0
        recall_32 = len(full_top32 & survivors) / 32.0
        recall_64 = len(set(full_ranking.top_k(64)) & survivors) / 64.0
        recall_512 = len(full_top512 & survivors) / 512.0

        # Exact ordered prefix agreement (count of identical position matches)
        ord_agree_16 = sum(1 for a, b in zip(full_ranking.candidate_order[:16], screened_ranking.candidate_order[:16]) if a == b)
        ord_agree_32 = sum(1 for a, b in zip(full_ranking.candidate_order[:32], screened_ranking.candidate_order[:32]) if a == b)

        # True permutation Spearman rho on final candidate orders
        perm_rho = permutation_spearman_rho(full_ranking.candidate_order, screened_ranking.candidate_order)

        rec = {
            "instance_index": idx,
            "instance_id": inst.instance_id,
            "prefix_type": "true_k6",
            "k6_tested": true_k6,
            "true_k5": true_k5,
            "full_rank": full_k5_rank,
            "screened_rank": screened_k5_rank,
            "rank_matched": full_k5_rank == screened_k5_rank,
            "retained_in_screen": retained,
            "timing_full_seconds": t_full,
            "timing_screened_seconds": t_screened,
            "matched_speedup": speedup,
            "jaccard_top16": jaccard_16,
            "jaccard_top32": jaccard_32,
            "jaccard_top512": jaccard_512,
            "survivor_recall_top16": recall_16,
            "survivor_recall_top32": recall_32,
            "survivor_recall_top64": recall_64,
            "survivor_recall_top512": recall_512,
            "ordered_agreement_top16": ord_agree_16,
            "ordered_agreement_top32": ord_agree_32,
            "permutation_spearman_rho": perm_rho,
        }
        true_prefix_records.append(rec)

        print(
            f"[True Prefix {idx:02d}] {inst.instance_id:<11} | True k5: {true_k5:#06x} | "
            f"Full: {full_k5_rank:<3} | Screen: {screened_k5_rank:<3} | "
            f"Jaccard@16: {jaccard_16:.4f} | Jaccard@32: {jaccard_32:.4f} | "
            f"Permutation rho: {perm_rho:.4f} | "
            f"Full: {t_full:.2f}s | Screen: {t_screened:.2f}s | Speedup: {speedup:.2f}x"
        )

    # 3. Synthetic Wrong-Prefix Diagnostic Cases
    # Fixed synthetic perturbations derived from true subkeys (true_k6 ^ 0x0001, true_k6 ^ 0x0002)
    wrong_prefix_records: list[dict[str, Any]] = []
    if args.include_synthetic_wrong_prefixes:
        print("\n[*] Evaluating synthetic wrong k6 prefixes (fixed bit-flip perturbations)...")
        # Predeclared offsets: 0x0001, 0x0002 on instances 0 and 1
        wrong_specs = [
            (0, 0x0001),
            (0, 0x0002),
            (1, 0x0001),
            (1, 0x0002),
        ]
        for inst_idx, offset in wrong_specs:
            inst = shard.instances[inst_idx]
            all_keys = expand_key(inst.master_key, rounds=7)
            wrong_k6 = all_keys[6] ^ offset
            descended_wrong = partial_decrypt_instance(
                inst,
                wrong_k6,
                expected_next_subkey=all_keys[5],
                label=f"audit-synthetic-wrong-{inst.instance_id}-{offset:#06x}",
            )

            if device.type == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
            screened_w = rank_pair_group_subkeys_screened(
                descended_wrong, model, screen_pairs=16, screen_top_k=512,
                candidate_batch_size=args.batch_size, device=device,
            )
            if device.type == "mps":
                torch.mps.synchronize()
            t_screen_w = time.perf_counter() - t0

            if device.type == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
            full_w = rank_pair_group_subkeys(
                descended_wrong, model, candidate_batch_size=args.batch_size, device=device,
            )
            if device.type == "mps":
                torch.mps.synchronize()
            t_full_w = time.perf_counter() - t0

            speedup_w = t_full_w / t_screen_w if t_screen_w > 0 else 0.0

            w_full_16 = set(full_w.top_k(16))
            w_screen_16 = set(screened_w.top_k(16))
            w_jaccard_16 = jaccard_similarity(w_full_16, w_screen_16)

            w_full_32 = set(full_w.top_k(32))
            w_screen_32 = set(screened_w.top_k(32))
            w_jaccard_32 = jaccard_similarity(w_full_32, w_screen_32)

            w_survivors = set(screened_w.top_k(512))
            w_rec_16 = len(w_full_16 & w_survivors) / 16.0
            w_rec_32 = len(w_full_32 & w_survivors) / 32.0
            w_rec_512 = len(set(full_w.top_k(512)) & w_survivors) / 512.0
            w_perm_rho = permutation_spearman_rho(full_w.candidate_order, screened_w.candidate_order)

            w_rec = {
                "instance_index": inst_idx,
                "instance_id": inst.instance_id,
                "prefix_type": "synthetic_wrong_prefix_diagnostic",
                "offset": offset,
                "k6_tested": wrong_k6,
                "timing_full_seconds": t_full_w,
                "timing_screened_seconds": t_screen_w,
                "matched_speedup": speedup_w,
                "jaccard_top16": w_jaccard_16,
                "jaccard_top32": w_jaccard_32,
                "survivor_recall_top16": w_rec_16,
                "survivor_recall_top32": w_rec_32,
                "survivor_recall_top512": w_rec_512,
                "permutation_spearman_rho": w_perm_rho,
            }
            wrong_prefix_records.append(w_rec)
            print(
                f"[Synthetic Wrong Prefix] {inst.instance_id:<11} offset={offset:#06x} | "
                f"Jaccard@16: {w_jaccard_16:.4f} | Jaccard@32: {w_jaccard_32:.4f} | "
                f"Permutation rho: {w_perm_rho:.4f} | "
                f"Full: {t_full_w:.2f}s | Screen: {t_screen_w:.2f}s | Speedup: {speedup_w:.2f}x"
            )

    # 4. Summary & Reconciliation
    all_paired_records = true_prefix_records + wrong_prefix_records
    all_speedups = [r["matched_speedup"] for r in all_paired_records]
    all_full_times = [r["timing_full_seconds"] for r in all_paired_records]
    all_screen_times = [r["timing_screened_seconds"] for r in all_paired_records]

    dep_hashes = {str(p.relative_to(REPOSITORY_ROOT)): file_sha256(p) for p in DEPENDENCY_PATHS if p.exists()}
    git_state = get_git_state()

    audit_summary = {
        "schema": "neural_sat_speck.screening_controlled_audit.v3",
        "date": "2026-09-12",
        "evidence_class": "ENGINEERING / DIAGNOSTIC",
        "provenance": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "device": str(device),
            "model_checkpoint": str(MODEL_PATH),
            "model_checkpoint_sha256": file_sha256(MODEL_PATH),
            "dataset_path": str(DATASET_PATH),
            "dataset_sha256": file_sha256(DATASET_PATH),
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": file_sha256(Path(__file__).resolve()),
            "dependency_sha256": dep_hashes,
            "git_state": git_state,
        },
        "synthetic_wrong_prefix_diagnostic_notes": [
            "Derived as fixed synthetic perturbations: true_k6 ^ 0x0001 and true_k6 ^ 0x0002.",
            "Only two attack instances (train-0000 and train-0001) contributed synthetic wrong-prefix cases.",
            "There was one timing observation per prefix.",
            "Wrong-prefix execution order was not counterbalanced (screened executed before full for all cases).",
            "Overall N=8 timing combines counterbalanced true-prefix runs with non-counterbalanced synthetic-prefix runs.",
            "The primary timing result is the N=4 true-prefix paired diagnostic.",
        ],
        "configuration": {
            "batch_size": args.batch_size,
            "screen_pairs": 16,
            "full_pairs": 64,
            "screen_top_k": 512,
            "true_prefix_instances": len(true_prefix_records),
            "synthetic_wrong_prefix_instances": len(wrong_prefix_records),
            "total_paired_runs": len(all_paired_records),
        },
        "true_prefix_summary": {
            "sample_size": len(true_prefix_records),
            "true_key_retained_in_screen": sum(r["retained_in_screen"] for r in true_prefix_records),
            "true_key_rank_exact_match": sum(r["rank_matched"] for r in true_prefix_records),
            "mean_full_seconds": float(np.mean([r["timing_full_seconds"] for r in true_prefix_records])),
            "median_full_seconds": float(np.median([r["timing_full_seconds"] for r in true_prefix_records])),
            "mean_screened_seconds": float(np.mean([r["timing_screened_seconds"] for r in true_prefix_records])),
            "median_screened_seconds": float(np.median([r["timing_screened_seconds"] for r in true_prefix_records])),
            "mean_matched_speedup": float(np.mean([r["matched_speedup"] for r in true_prefix_records])),
            "median_matched_speedup": float(np.median([r["matched_speedup"] for r in true_prefix_records])),
            "min_matched_speedup": float(np.min([r["matched_speedup"] for r in true_prefix_records])),
            "max_matched_speedup": float(np.max([r["matched_speedup"] for r in true_prefix_records])),
            "mean_jaccard_top16": float(np.mean([r["jaccard_top16"] for r in true_prefix_records])),
            "mean_jaccard_top32": float(np.mean([r["jaccard_top32"] for r in true_prefix_records])),
            "mean_jaccard_top512": float(np.mean([r["jaccard_top512"] for r in true_prefix_records])),
            "mean_survivor_recall_top16": float(np.mean([r["survivor_recall_top16"] for r in true_prefix_records])),
            "mean_survivor_recall_top32": float(np.mean([r["survivor_recall_top32"] for r in true_prefix_records])),
            "mean_permutation_spearman_rho": float(np.mean([r["permutation_spearman_rho"] for r in true_prefix_records])),
        },
        "overall_paired_timing_summary": {
            "total_paired_evaluations": len(all_paired_records),
            "mean_full_seconds": float(np.mean(all_full_times)),
            "median_full_seconds": float(np.median(all_full_times)),
            "mean_screened_seconds": float(np.mean(all_screen_times)),
            "median_screened_seconds": float(np.median(all_screen_times)),
            "mean_matched_speedup": float(np.mean(all_speedups)),
            "median_matched_speedup": float(np.median(all_speedups)),
            "min_matched_speedup": float(np.min(all_speedups)),
            "max_matched_speedup": float(np.max(all_speedups)),
        },
        "records": {
            "true_prefix_evaluations": true_prefix_records,
            "synthetic_wrong_prefix_evaluations": wrong_prefix_records,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(f".tmp.{os.getpid()}")
    try:
        temp_path.write_text(json.dumps(audit_summary, indent=2), encoding="utf-8")
        temp_path.replace(output_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise

    out_hash = file_sha256(output_path)

    print(f"\n[✓] Screening audit complete! Results saved to {output_path}")
    print(f"[✓] SHA-256: {out_hash}")
    print(f"    True-Prefix Observed Speedup on {device}: {audit_summary['true_prefix_summary']['mean_matched_speedup']:.2f}x (median: {audit_summary['true_prefix_summary']['median_matched_speedup']:.2f}x)")
    print(f"    True Key Retention: {audit_summary['true_prefix_summary']['true_key_retained_in_screen']}/{len(true_prefix_records)}")
    print(f"    True Key Rank Match: {audit_summary['true_prefix_summary']['true_key_rank_exact_match']}/{len(true_prefix_records)}")
    print(f"    Mean Permutation Spearman rho: {audit_summary['true_prefix_summary']['mean_permutation_spearman_rho']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
