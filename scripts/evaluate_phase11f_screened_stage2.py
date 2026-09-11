#!/usr/bin/env python3
"""Evaluate Screened Multi-Fidelity Pair-Group Ranker (Phase 11F).

Compares full-domain 64-pair ranking against 16-pair screened + 512 rescored
ranking on opened Seed 300001 instances with true k6 supplied.

Measures:
- Containment of true k5 in screening top-512.
- Top-8, top-16, and top-32 retention.
- Wall-clock speedup factor.
"""

from __future__ import annotations

import argparse
import json
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

from neural_sat_speck.attacks.multi_round_instance import partial_decrypt_instance
from neural_sat_speck.attacks.screened_group_ranking import (
    rank_pair_group_subkeys_screened,
)
from neural_sat_speck.data.attack import load_attack_shard
from neural_sat_speck.neural.group_model import PairGroupRanker, build_pair_group_model
from neural_sat_speck.neural.model import GohrResNet
from neural_sat_speck.speck.key_schedule import expand_key

DATASET_PATH = (
    REPOSITORY_ROOT / "datasets" / "phase11a_oracle_r7" / "oracle_evaluation.json"
)
DATASET_ID = "speck32_64_attack_phase11a_oracle_r7"
MODEL_PATH = (
    REPOSITORY_ROOT / "models" / "speck32_64_phase6a_b1_mean_g64.pt"
)
RESULTS_DIR = REPOSITORY_ROOT / "results" / "phase11_scaling_diagnostics"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=int, default=12, help="Number of instances to evaluate (default 12)")
    parser.add_argument("--screen-pairs", type=int, default=16, help="Screening pair count (default 16)")
    parser.add_argument("--screen-top-k", type=int, default=512, help="Screening survival budget (default 512)")
    parser.add_argument("--batch-size", type=int, default=1024, help="Candidate batch size (default 1024)")
    parser.add_argument("--device", type=str, default="mps", help="PyTorch device (default mps)")
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "screened_stage2_evaluation.json",
        help="Output JSON path",
    )
    return parser.parse_args(argv)


def load_model(device: torch.device) -> Any:
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    variant = checkpoint.get("variant", "b1_mean")
    model = PairGroupRanker(variant=variant)
    state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.backends.mps.is_available() or torch.cuda.is_available() else "cpu")

    print(f"=== EVALUATING SCREENED MULTI-FIDELITY STAGE-2 RANKER (Phase 11F) ===")
    print(f"Device: {device}, Instances: {args.instances}, Screen Pairs: {args.screen_pairs}, Screen Top-K: {args.screen_top_k}\n")

    shard = load_attack_shard(DATASET_PATH, expected_dataset_id=DATASET_ID)
    model = load_model(device)

    # Reference containment from stage2_oracle_k6_containment.json
    ref_path = RESULTS_DIR / "stage2_oracle_k6_containment.json"
    ref_records = {}
    if ref_path.exists():
        with open(ref_path) as f:
            ref_data = json.load(f)
            ref_records = {r["instance_id"]: r for r in ref_data.get("records", [])}

    records: list[dict[str, Any]] = []

    for idx in range(min(args.instances, len(shard.instances))):
        inst = shard.instances[idx]
        all_keys = expand_key(inst.master_key, rounds=7)
        true_k6 = all_keys[6]
        true_k5 = all_keys[5]

        # Descend 7-round instance by true k6 to produce 6-round instance for k5 ranking
        descended = partial_decrypt_instance(
            inst,
            true_k6,
            expected_next_subkey=true_k5,
            label=f"{inst.instance_id}-stage2",
        )

        t0 = time.perf_counter()
        screened_ranking = rank_pair_group_subkeys_screened(
            descended,
            model,
            screen_pairs=args.screen_pairs,
            screen_top_k=args.screen_top_k,
            candidate_batch_size=args.batch_size,
            device=device,
        )
        elapsed = time.perf_counter() - t0

        screened_rank = screened_ranking.rank(true_k5)
        in_screen = screened_rank <= args.screen_top_k

        ref_rank = ref_records.get(inst.instance_id, {}).get("true_k5_rank")
        ref_time = ref_records.get(inst.instance_id, {}).get("ranking_seconds")

        rec = {
            "instance_index": idx,
            "instance_id": inst.instance_id,
            "true_k5": true_k5,
            "screened_rank": screened_rank,
            "reference_rank": ref_rank,
            "retained_in_screen": in_screen,
            "screened_seconds": round(elapsed, 3),
            "reference_seconds": round(ref_time, 3) if ref_time else None,
            "speedup": round(ref_time / elapsed, 2) if ref_time else None,
        }
        records.append(rec)

        speedup_str = f"{rec['speedup']}x" if rec['speedup'] else "N/A"
        ref_rank_str = str(ref_rank) if ref_rank is not None else "?"
        print(
            f"[{idx:02d}] {inst.instance_id:<11} | True k5: {true_k5:#06x} | "
            f"Screened Rank: {screened_rank:<3} (Ref: {ref_rank_str:<3}) | "
            f"In Screen: {str(in_screen):<5} | "
            f"Time: {rec['screened_seconds']:<5.2f}s (Speedup: {speedup_str})"
        )

    # Aggregate statistics
    screened_ranks = [r["screened_rank"] for r in records]
    times = [r["screened_seconds"] for r in records]
    speedups = [r["speedup"] for r in records if r["speedup"] is not None]

    summary = {
        "experiment": "phase11f_screened_stage2_evaluation",
        "timestamp": datetime.now(UTC).isoformat(),
        "screen_pairs": args.screen_pairs,
        "screen_top_k": args.screen_top_k,
        "device": str(device),
        "instances_evaluated": len(records),
        "containment": {
            "top_1": sum(r <= 1 for r in screened_ranks),
            "top_8": sum(r <= 8 for r in screened_ranks),
            "top_16": sum(r <= 16 for r in screened_ranks),
            "top_32": sum(r <= 32 for r in screened_ranks),
            "top_512_screen_retention": sum(r["retained_in_screen"] for r in records),
            "screen_retention_rate": sum(r["retained_in_screen"] for r in records) / len(records),
        },
        "timing": {
            "mean_seconds": round(float(np.mean(times)), 3),
            "median_seconds": round(float(np.median(times)), 3),
            "min_seconds": round(float(np.min(times)), 3),
            "max_seconds": round(float(np.max(times)), 3),
            "mean_speedup": round(float(np.mean(speedups)), 2) if speedups else None,
        },
        "records": records,
    }

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[✓] Results saved to {args.output}")
    print("\nSummary:")
    print(f"  Top-512 Screen Retention: {summary['containment']['top_512_screen_retention']}/{len(records)} ({summary['containment']['screen_retention_rate']*100:.1f}%)")
    print(f"  Top-16 Containment: {summary['containment']['top_16']}/{len(records)} ({summary['containment']['top_16']/len(records)*100:.1f}%)")
    print(f"  Top-32 Containment: {summary['containment']['top_32']}/{len(records)} ({summary['containment']['top_32']/len(records)*100:.1f}%)")
    print(f"  Median Ranking Time: {summary['timing']['median_seconds']}s (Mean speedup: {summary['timing']['mean_speedup']}x)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
