#!/usr/bin/env python3
"""End-to-end hierarchical seven-round evaluation (development diagnostic).

Runs the Phase 11B' architecture on the opened Seed 300001 family: a six-round
stage-1 ranker scores the final round key, the frozen five-round `B1` model ranks
the round key below it per surviving candidate, and every candidate pair is tested
through the exact seven-round model with fresh-pair verification.

Development evidence only: Seed 300001 is opened and retired, so results here are
DEVELOPMENT/DIAGNOSTIC and can never serve as confirmation. Frozen protocols are
untouched; this writes a JSON record under results/phase11_scaling_diagnostics/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import torch  # noqa: E402

from neural_sat_speck.attacks.hierarchical_multi_round import (  # noqa: E402
    HierarchicalStage,
    run_hierarchical_multi_round_attack,
)
from neural_sat_speck.attacks.recovery import (  # noqa: E402
    independent_verification_pairs,
    known_pairs_for_attack_instance,
)
from neural_sat_speck.data.attack import load_attack_shard  # noqa: E402
from neural_sat_speck.evaluation.metrics import wilson_score_interval  # noqa: E402
from neural_sat_speck.neural.group_model import PairGroupRanker  # noqa: E402
from neural_sat_speck.neural.numpy_inference import (  # noqa: E402
    NumpyInferenceModule,
    NumpyRankerInference,
)

DATASET_PATH = (
    REPOSITORY_ROOT / "datasets" / "phase11a_oracle_r7" / "oracle_evaluation.json"
)
DATASET_ID = "speck32_64_attack_phase11a_oracle_r7"
RESULTS_DIR = REPOSITORY_ROOT / "results" / "phase11_scaling_diagnostics"
ROUNDS = 7
KNOWN_PAIRS = 4
VERIFICATION_PAIRS = 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-model", type=Path, required=True)
    parser.add_argument(
        "--stage2-model",
        type=Path,
        default=REPOSITORY_ROOT / "models" / "speck32_64_phase6a_b1_mean_g64.pt",
    )
    parser.add_argument("--instances", type=int, default=8)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--m1", type=int, default=8)
    parser.add_argument("--m2", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def load_ranker(path: Path) -> NumpyInferenceModule:
    if not path.exists():
        raise SystemExit(f"checkpoint not found at {path}")
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
    stage1_model = load_ranker(args.stage1_model)
    stage2_model = load_ranker(args.stage2_model)
    stages = (
        HierarchicalStage(round_index=ROUNDS - 1, model=stage1_model, beam_width=args.m1),
        HierarchicalStage(round_index=ROUNDS - 2, model=stage2_model, beam_width=args.m2),
    )
    shard = load_attack_shard(DATASET_PATH, expected_dataset_id=DATASET_ID)
    instances = shard.instances[args.start : args.start + args.instances]
    if not instances:
        raise SystemExit("no instances selected")

    print(
        f"hierarchical evaluation: instances={len(instances)} M1={args.m1} M2={args.m2} "
        f"timeout={args.timeout}s stage1={args.stage1_model.name} stage2={args.stage2_model.name}"
    )
    records: list[dict[str, Any]] = []
    for offset, instance in enumerate(instances):
        known = known_pairs_for_attack_instance(instance)[:KNOWN_PAIRS]
        verification = independent_verification_pairs(
            instance.master_key,
            rounds=ROUNDS,
            pair_count=VERIFICATION_PAIRS,
            seed=999999,
        )
        started = time.perf_counter()
        result = run_hierarchical_multi_round_attack(
            instance,
            rounds=ROUNDS,
            stages=stages,
            known_pairs=known,
            verification_pairs=verification,
            solver_timeout=args.timeout,
            candidate_batch_size=args.batch_size,
        )
        wall_seconds = time.perf_counter() - started
        payload = result.to_dict()
        payload["index"] = args.start + offset
        payload["instance_id"] = instance.instance_id
        payload["wall_seconds"] = round(wall_seconds, 3)
        records.append(payload)
        stage_ranks = [
            stage.get("true_round_key_rank") for stage in payload.get("stages", [])
        ]
        print(
            f"  [{args.start + offset:02d}] {instance.instance_id} success={result.success} "
            f"status={result.status} cause={result.failure_cause} "
            f"stage_ranks={stage_ranks} tuples={result.candidates_attempted} "
            f"wall={wall_seconds:6.1f}s",
            flush=True,
        )

    successes = sum(1 for record in records if record["success"])
    low, high = wilson_score_interval(successes, len(records))
    causes: dict[str, int] = {}
    for record in records:
        if not record["success"]:
            cause = record.get("failure_cause") or "unknown"
            causes[cause] = causes.get(cause, 0) + 1
    summary = {
        "stage1_model": str(args.stage1_model),
        "stage2_model": str(args.stage2_model),
        "instances": len(records),
        "m1": args.m1,
        "m2": args.m2,
        "solver_timeout_seconds": args.timeout,
        "verified_recoveries": successes,
        "verified_recovery_rate": round(successes / len(records), 4),
        "wilson_95_ci": [round(low, 4), round(high, 4)],
        "failure_causes": causes,
        "median_wall_seconds": round(
            sorted(record["wall_seconds"] for record in records)[len(records) // 2], 3
        ),
        "total_ranking_seconds": round(
            sum(record["timing_seconds"]["ranking"] for record in records), 3
        ),
        "total_solver_seconds": round(
            sum(record["timing_seconds"]["solver"] for record in records), 3
        ),
    }
    output = {
        "protocol": "phase11b_prime_hierarchical_evaluation",
        "evidence_class": "DEVELOPMENT_DIAGNOSTIC",
        "generated_at": datetime.now(UTC).isoformat(),
        "rounds": ROUNDS,
        "known_pairs": KNOWN_PAIRS,
        "verification_pairs": VERIFICATION_PAIRS,
        "note": (
            "opened Seed 300001 family; development diagnostics only, never "
            "confirmation; ground-truth ranks are post-hoc stage diagnostics"
        ),
        "summary": summary,
        "records": records,
    }
    output_path = args.output
    if output_path is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_path = RESULTS_DIR / f"hierarchical_eval_{stamp}.json"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    print("\nsummary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
