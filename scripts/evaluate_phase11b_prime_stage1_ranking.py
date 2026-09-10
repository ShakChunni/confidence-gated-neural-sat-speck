#!/usr/bin/env python3
"""Evaluate the Phase 11B-prime stage-1 ranker on opened seven-round instances.

The stage-1 model is trained on six-round pair groups. For a seven-round attack
instance it ranks the final round key `k6` by decrypting one round under each
candidate and scoring the resulting six-round state, which is exactly what
`rank_pair_group_subkeys` does. The true `k6` is read only after the complete
domain has been ranked, and is used solely for post-hoc containment reporting.

Diagnostic only: this writes to `results/phase11_scaling_diagnostics/` and never
touches a frozen artefact. The opened Seed 300001 family is the only attack data
it reads.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import torch  # noqa: E402

from neural_sat_speck.attacks.group_ranking import rank_pair_group_subkeys  # noqa: E402
from neural_sat_speck.data.attack import load_attack_shard  # noqa: E402
from neural_sat_speck.neural.group_model import PairGroupRanker  # noqa: E402
from neural_sat_speck.neural.numpy_inference import (  # noqa: E402
    NumpyInferenceModule,
    NumpyRankerInference,
)
from neural_sat_speck.speck.key_schedule import expand_key  # noqa: E402

DATASET_PATH = (
    REPOSITORY_ROOT / "datasets" / "phase11a_oracle_r7" / "oracle_evaluation.json"
)
DATASET_ID = "speck32_64_attack_phase11a_oracle_r7"
RESULTS_DIR = REPOSITORY_ROOT / "results" / "phase11_scaling_diagnostics"
CONTAINMENT_KS = (1, 4, 8, 16, 32, 64)
DEFAULT_BATCH_SIZE = 1024


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--instances", type=int, default=16)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON output path; defaults to a timestamped file in the diagnostics directory",
    )
    return parser.parse_args(argv)


def load_stage1_model(path: Path) -> NumpyInferenceModule:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    variant = payload.get("variant")
    if not isinstance(variant, str):
        raise SystemExit(f"{path} does not contain a pair-group variant")
    model = PairGroupRanker(variant=variant)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return NumpyInferenceModule(NumpyRankerInference(model))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.model.exists():
        raise SystemExit(f"model checkpoint not found at {args.model}")

    ranker = load_stage1_model(args.model)
    shard = load_attack_shard(DATASET_PATH, expected_dataset_id=DATASET_ID)
    instances = shard.instances[args.start : args.start + args.instances]
    if not instances:
        raise SystemExit("no instances selected")

    print(
        f"stage-1 ranking evaluation: model={args.model.name} "
        f"instances={len(instances)} batch={args.batch_size}"
    )
    records: list[dict[str, Any]] = []
    for offset, instance in enumerate(instances):
        started = time.perf_counter()
        ranking = rank_pair_group_subkeys(
            instance,
            ranker,
            candidate_batch_size=args.batch_size,
        )
        ranking_seconds = time.perf_counter() - started

        # Ground truth is consulted only here, after the domain is ranked.
        true_subkey = expand_key(instance.master_key, rounds=7)[-1]
        rank = ranking.rank(true_subkey)
        record = {
            "index": args.start + offset,
            "instance_id": instance.instance_id,
            "true_final_round_subkey": true_subkey,
            "true_rank": rank,
            "containment": {f"top_{k}": int(rank <= k) for k in CONTAINMENT_KS},
            "ranking_seconds": round(ranking_seconds, 6),
            "top_candidate": ranking.top_k(1)[0],
        }
        records.append(record)
        print(
            f"  [{args.start + offset:02d}] {instance.instance_id} "
            f"true_rank={rank:>6} time={ranking_seconds:6.1f}s",
            flush=True,
        )

    ranks = [record["true_rank"] for record in records]
    try:
        model_label = str(args.model.resolve().relative_to(REPOSITORY_ROOT))
    except ValueError:
        model_label = str(args.model)
    summary = {
        "model": model_label,
        "instances": len(records),
        "start_index": args.start,
        "median_true_rank": statistics.median(ranks),
        "mean_true_rank": round(statistics.fmean(ranks), 3),
        "min_true_rank": min(ranks),
        "max_true_rank": max(ranks),
        "containment": {
            f"top_{k}": sum(int(rank <= k) for rank in ranks) for k in CONTAINMENT_KS
        },
        "median_ranking_seconds": round(
            statistics.median(record["ranking_seconds"] for record in records), 3
        ),
    }
    payload = {
        "protocol": "phase11b_prime_stage1_ranking_evaluation",
        "generated_at": datetime.now(UTC).isoformat(),
        "stage": 1,
        "rounds": 7,
        "note": (
            "ground truth is read only after the complete candidate domain is ranked; "
            "rank_pair_group_subkeys never consults the stored final-round subkey"
        ),
        "summary": summary,
        "records": records,
    }

    output_path = args.output
    if output_path is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_path = RESULTS_DIR / f"stage1_ranking_{stamp}.json"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("\nsummary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
