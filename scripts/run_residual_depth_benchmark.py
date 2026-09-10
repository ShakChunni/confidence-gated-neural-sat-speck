#!/usr/bin/env python3
"""Residual-depth benchmark for the exact SPECK32/64 backend.

Diagnostic (not a frozen protocol). For a given round count and number of
trailing round keys assumed to their true values, measure whether the exact
isolated CryptoMiniSat backend can recover and independently verify the master
key. The Phase 11A gate ran this configuration at 7 rounds with one assumed
round key and timed out on all 64 instances; this benchmark maps the boundary.

Examples
--------
    # 7 rounds, opened Seed 300001 family, 1/2/3 trailing round keys assumed
    python scripts/run_residual_depth_benchmark.py --source opened --rounds 7 \\
        --assume 1 2 3 --instances 8 --budget 10

    # 8 and 9 rounds on ephemeral demo-seed instances
    python scripts/run_residual_depth_benchmark.py --source ephemeral --rounds 8 9 \\
        --assume 2 3 4 --instances 2 --budget 10

Outputs a JSON record under results/phase11_scaling_diagnostics/ unless
--no-write is given.
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

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from neural_sat_speck.attacks.recovery import (  # noqa: E402
    independent_verification_pairs,
    known_pairs_for_attack_instance,
    verify_master_key,
)
from neural_sat_speck.data.attack import (  # noqa: E402
    AttackInstanceConfig,
    generate_attack_instances,
    load_attack_shard,
)
from neural_sat_speck.sat.isolation import IsolatedSolverFactory  # noqa: E402
from neural_sat_speck.sat.multi_round import assumptions_for_round_keys  # noqa: E402
from neural_sat_speck.speck.key_schedule import expand_key  # noqa: E402

DATASET_PATH = REPO_ROOT / "datasets" / "phase11a_oracle_r7" / "oracle_evaluation.json"
DATASET_ID = "speck32_64_attack_phase11a_oracle_r7"
RESULTS_DIR = REPO_ROOT / "results" / "phase11_scaling_diagnostics"
EPHEMERAL_SEED = 999999
VERIFICATION_SEED = 999999
KNOWN_PAIRS_COUNT = 4
VERIFICATION_PAIRS_COUNT = 2


def load_instances(source: str, rounds: int, instances: int):
    if source == "opened":
        shard = load_attack_shard(DATASET_PATH, expected_dataset_id=DATASET_ID)
        if rounds != 7:
            raise ValueError("the opened Seed 300001 family is a seven-round family")
        return shard.instances[:instances]
    config = AttackInstanceConfig(
        instance_count=instances,
        pair_count=64,
        rounds=rounds,
        seed=EPHEMERAL_SEED,
        split_role="validation",
    )
    return generate_attack_instances(config).instances


def evaluate(
    instance,
    rounds: int,
    assume_last: int,
    known_pair_count: int,
    budget: float,
) -> dict[str, Any]:
    all_pairs = known_pairs_for_attack_instance(instance)
    known_pairs = all_pairs[:known_pair_count]
    verification_pairs = independent_verification_pairs(
        instance.master_key,
        rounds=rounds,
        pair_count=VERIFICATION_PAIRS_COUNT,
        seed=VERIFICATION_SEED,
    )
    factory = IsolatedSolverFactory.from_pairs(
        known_pairs,
        rounds=rounds,
        instance_id=instance.instance_id,
    )
    model = factory.build_isolated_model(
        f"residual_{rounds}_{assume_last}_{instance.instance_id}"
    )
    schedule = expand_key(instance.master_key, rounds=rounds)
    round_key_values = {
        index: schedule[index]
        for index in range(rounds - assume_last, rounds)
    }
    assumptions = assumptions_for_round_keys(model, round_key_values)

    started = time.perf_counter()
    result = model.solve(assumptions, time_limit=budget)
    solve_seconds = time.perf_counter() - started

    verified = False
    key_match = False
    if result.satisfiable and result.solution is not None:
        recovered = model.master_key_from_solution(result.solution)
        key_match = tuple(recovered) == tuple(instance.master_key)
        verified = verify_master_key(
            recovered,
            known_pairs,
            verification_pairs,
            rounds,
        ).verified

    if verified:
        status = "SAT + verified"
    elif result.status == "solver_timeout" or result.timed_out:
        status = "timeout"
    elif result.status == "unsat":
        status = "UNSAT"
    elif result.satisfiable:
        status = "SAT + verification failed"
    else:
        status = "backend/error"

    return {
        "instance_id": instance.instance_id,
        "rounds": rounds,
        "assumed_trailing_round_keys": assume_last,
        "residual_rounds": rounds - assume_last,
        "known_pairs": known_pair_count,
        "solver_timeout_seconds": budget,
        "status": status,
        "verified": verified,
        "key_match": key_match,
        "solve_seconds": round(solve_seconds, 6),
        "variables": model.variable_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("opened", "ephemeral"), default="opened")
    parser.add_argument("--rounds", type=int, nargs="+", default=[7])
    parser.add_argument("--assume", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--instances", type=int, default=4)
    parser.add_argument("--known-pairs", type=int, default=KNOWN_PAIRS_COUNT)
    parser.add_argument("--budget", type=float, default=10.0)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    if args.source == "opened" and args.rounds != [7]:
        parser.error("--source opened only supports --rounds 7")

    records: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    print(f"residual-depth benchmark: source={args.source} budget={args.budget}s "
          f"known_pairs={args.known_pairs} instances={args.instances}")
    for rounds in args.rounds:
        instances = load_instances(args.source, rounds, args.instances)
        for assume_last in args.assume:
            if assume_last >= rounds:
                continue
            group_records = [
                evaluate(instance, rounds, assume_last, args.known_pairs, args.budget)
                for instance in instances
            ]
            solved = [r for r in group_records if r["verified"]]
            times = [r["solve_seconds"] for r in group_records]
            solved_times = [r["solve_seconds"] for r in solved]
            group = {
                "rounds": rounds,
                "assumed_trailing_round_keys": assume_last,
                "residual_rounds": rounds - assume_last,
                "instances": len(group_records),
                "verified_recoveries": len(solved),
                "median_solve_seconds": (
                    round(statistics.median(solved_times), 6) if solved_times else None
                ),
                "min_solve_seconds": round(min(times), 6),
                "max_solve_seconds": round(max(times), 6),
                "statuses": {},
            }
            for record in group_records:
                group["statuses"][record["status"]] = (
                    group["statuses"].get(record["status"], 0) + 1
                )
            groups.append(group)
            records.extend(group_records)
            print(
                f"  rounds={rounds} assumed_last={assume_last} "
                f"(residual {rounds - assume_last}): "
                f"{len(solved)}/{len(group_records)} verified, "
                f"median {group['median_solve_seconds']}s, "
                f"statuses {group['statuses']}",
                flush=True,
            )

    payload = {
        "protocol": "phase11_residual_depth_benchmark",
        "generated_at": datetime.now(UTC).isoformat(),
        "source": args.source,
        "known_pairs": args.known_pairs,
        "solver_timeout_seconds": args.budget,
        "ephemeral_seed": EPHEMERAL_SEED if args.source == "ephemeral" else None,
        "groups": groups,
        "records": records,
    }
    if not args.no_write:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_path = RESULTS_DIR / f"residual_depth_{args.source}_{stamp}.json"
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
