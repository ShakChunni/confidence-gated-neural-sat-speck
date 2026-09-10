#!/usr/bin/env python3
"""Beam-selection analysis for the Phase 11B-prime hierarchical attack.

Combines two development measurements — the stage-1 rank of the true final round
key `k6` and the stage-2 rank of the true `k5` given a correct `k6` — into the
containment/cost curve that a beam choice implies:

    an instance is reachable iff  rank(k6) <= M1  and  rank(k5 | k6) <= M2

Cost follows the measured model in `docs/PHASE11B_PRIME_PROTOCOL.md`:
one stage-1 ranking plus `M1` stage-2 rankings, at a measured per-ranking cost.

This is development-only analysis on the opened Seed 300001 family. It selects no
method by itself; it produces the evidence a protocol freeze must cite.

Usage:
    python scripts/analyze_phase11b_prime_beam_curve.py \
        --stage1 results/phase11_scaling_diagnostics/stage1_ranking_*.json \
        --stage2 results/phase11_scaling_diagnostics/stage2_oracle_k6_containment.json \
        --m1 1,2,4,8,16 --m2 4,8,16,32
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SECONDS_PER_RANKING = 75.0


def _load_records(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not records:
        raise SystemExit(f"{path} does not contain per-instance records")
    # Stage-1 and stage-2 diagnostics name the rank field differently; accept both
    # rather than forcing one file to be rewritten.
    rank_keys = ("true_rank", "true_k5_rank", "true_k6_rank")
    ranks: dict[str, int] = {}
    for record in records:
        instance_id = record.get("instance_id")
        rank = next(
            (record[key] for key in rank_keys if isinstance(record.get(key), int)),
            None,
        )
        if isinstance(instance_id, str) and rank is not None:
            ranks[instance_id] = rank
    if not ranks:
        raise SystemExit(f"{path} contains no usable true_rank entries")
    return ranks


def _parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--m1", type=_parse_int_list, default=_parse_int_list("1,2,4,8,16"))
    parser.add_argument("--m2", type=_parse_int_list, default=_parse_int_list("4,8,16,32"))
    parser.add_argument("--seconds-per-ranking", type=float, default=DEFAULT_SECONDS_PER_RANKING)
    args = parser.parse_args()

    stage1 = _load_records(args.stage1)
    stage2 = _load_records(args.stage2)
    shared = sorted(set(stage1) & set(stage2))
    if not shared:
        raise SystemExit("stage-1 and stage-2 records share no instance ids")
    stage1_only = sorted(set(stage1) - set(stage2))
    stage2_only = sorted(set(stage2) - set(stage1))

    print(f"paired instances: {len(shared)}")
    if stage1_only:
        print(f"  stage-1 without stage-2 (not counted): {len(stage1_only)}")
    if stage2_only:
        print(f"  stage-2 without stage-1 (not counted): {len(stage2_only)}")

    reachable_all_m1 = sum(int(stage1[i] <= max(args.m1)) for i in shared)
    reachable_all_m2 = sum(int(stage2[i] <= max(args.m2)) for i in shared)
    print(
        f"  true k6 within top-{max(args.m1)}: {reachable_all_m1}/{len(shared)}; "
        f"true k5 within top-{max(args.m2)} given k6: {reachable_all_m2}/{len(shared)}"
    )
    print(f"  median stage-1 rank: {statistics.median(stage1[i] for i in shared)}; "
          f"median stage-2 rank: {statistics.median(stage2[i] for i in shared)}")

    rows: list[dict[str, Any]] = []
    print("\n  M1   M2   reachable   fraction   est. ranking minutes/instance")
    for m1 in args.m1:
        for m2 in args.m2:
            reachable = sum(
                int(stage1[i] <= m1 and stage2[i] <= m2) for i in shared
            )
            fraction = reachable / len(shared)
            minutes = (1 + m1) * args.seconds_per_ranking / 60.0
            rows.append(
                {
                    "m1": m1,
                    "m2": m2,
                    "reachable": reachable,
                    "instances": len(shared),
                    "fraction": round(fraction, 4),
                    "estimated_ranking_minutes_per_instance": round(minutes, 2),
                }
            )
            print(f"  {m1:>3}  {m2:>3}   {reachable:>3}/{len(shared):<3}    "
                  f"{fraction:7.3f}   {minutes:8.2f}")

    best = max(rows, key=lambda row: (row["fraction"], -row["estimated_ranking_minutes_per_instance"]))
    print(
        f"\nbest reachable fraction {best['fraction']:.3f} at M1={best['m1']}, M2={best['m2']} "
        f"(~{best['estimated_ranking_minutes_per_instance']:.1f} min/instance ranking)"
    )
    print(
        "NOTE: stage-2 ranks here are measured with the true k6 supplied, so the "
        "reachable fraction is an upper bound on an end-to-end run; and this analysis "
        "uses the opened Seed 300001 family for development only."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
