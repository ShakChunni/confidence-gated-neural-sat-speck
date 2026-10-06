"""Reproduce a frozen, retrospective history observation without running SAT."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from neural_sat_speck.simon.attack_instance import SimonAttackInstance
from neural_sat_speck.simon.cipher import encrypt_block

LEDGER_SHA = "8cc9da7fbdf5207d94e0b108452819244a9dd4aee3dd28de14419af05e397b61"
DATASET_SHA = "095ca0dc8356f79232019670f3555b9f00c1fc6c4e1203f397f7dfc437a89cc9"
ORACLE = "oracle_2key_n4"
METHODS = (
    "fully_classical_8parent",
    "neural_classical_hybrid_8parent",
    "fully_neural_1parent",
    "faithful_b1_hybrid",
    "random_control_128",
    "random_control_1024",
    "sat_only_n4",
    "sat_only_n128",
    ORACLE,
)


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def unique_object(items: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in items:
        require(key not in result, f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def checked_json(path: Path, expected_sha: str) -> dict | list:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    require(digest == expected_sha, f"Frozen input changed: {path}")
    require(
        path.with_name(path.name + ".sha256").read_text().split()
        == [digest, path.name],
        f"Sidecar mismatch: {path}",
    )
    return json.loads(raw, object_pairs_hook=unique_object)


def seconds(value: object) -> float:
    require(
        type(value) in (int, float) and math.isfinite(value) and value > 0,
        "Time must be a finite positive number",
    )
    return float(value)


def tuple_value(attempt: dict) -> tuple[int, int]:
    values = (attempt["candidate_k8"], attempt["candidate_k9"])
    require(
        all(type(v) is int and 0 <= v < 65536 for v in values),
        "Malformed candidate tuple",
    )
    return values


def derive_rows(records: list[dict], instance_ids: set[str]) -> list[dict]:
    grid = {method: {} for method in METHODS}
    for record in records:
        method, key = record["method"], record["instance_id"]
        require(method in grid and key in instance_ids, "Unexpected method or key")
        require(key not in grid[method], "Duplicate method/key cell")
        require(type(record["verified_on_fresh_pairs"]) is bool, "Non-Boolean flag")
        recovered = record["outcome"] == "verified_recovery"
        require(record["verified_on_fresh_pairs"] == recovered, "Outcome/flag conflict")
        seconds(record["complete_online_wall_seconds"])
        grid[method][key] = record
    require(
        all(set(cells) == instance_ids for cells in grid.values()),
        "Incomplete method/key grid",
    )
    rows = []
    for key in sorted(instance_ids):
        oracle = grid[ORACLE][key]
        if oracle["outcome"] == "verified_recovery":
            continue
        require(
            oracle["outcome"] == "candidate_unresolved", "Unexpected oracle failure"
        )
        oracle_attempts = oracle["candidate_attempts"]
        require(len(oracle_attempts) == 1, "Oracle must have exactly one request")
        target = oracle_attempts[0]
        require(target["status"] == "solver_timeout", "Oracle request not unresolved")
        for method in METHODS[:-1]:
            record = grid[method][key]
            if record["outcome"] != "verified_recovery":
                continue
            attempts = record["candidate_attempts"]
            require(len(attempts) > 1, "Recovery has no preceding search")
            require(attempts[-1]["status"] == "sat", "No final SAT attempt")
            require(
                tuple_value(attempts[-1]) == tuple_value(target), "Target tuple differs"
            )
            require(
                all(a["status"] == "unsat" for a in attempts[:-1]),
                "Prefix contains a non-UNSAT call",
            )
            require(
                all(a["rebuild_occurred"] is False for a in attempts),
                "Intervening solver rebuild",
            )
            rows.append(
                {
                    "instance_id": key,
                    "method": method,
                    "preceding_unsat_calls": len(attempts) - 1,
                    "preceding_solve_wall_seconds": sum(
                        seconds(a["solve_wall_seconds"]) for a in attempts[:-1]
                    ),
                    "target_solve_wall_seconds": seconds(
                        attempts[-1]["solve_wall_seconds"]
                    ),
                    "oracle_request_wall_seconds": seconds(
                        target["solve_wall_seconds"]
                    ),
                    "method_online_wall_seconds": seconds(
                        record["complete_online_wall_seconds"]
                    ),
                    "oracle_online_wall_seconds": seconds(
                        oracle["complete_online_wall_seconds"]
                    ),
                    "ranking_wall_seconds": seconds(record["ranking_wall_seconds"]),
                }
            )
    return rows


def verify_recoveries(records: list[dict], instances: dict) -> int:
    count = 0
    for record in records:
        if record["outcome"] != "verified_recovery":
            continue
        instance = instances[record["instance_id"]]
        key = record["recovered_master_key"]
        require(
            isinstance(key, list)
            and len(key) == 4
            and all(type(v) is int and 0 <= v < 65536 for v in key)
            and tuple(key) == instance.master_key,
            "Recovered key is not the assigned master key",
        )
        require(seconds(record["complete_online_wall_seconds"]) <= 300, "Late recovery")
        for plaintext, ciphertext in instance.known_pairs + instance.verification_pairs:
            require(
                encrypt_block(plaintext, key, instance.rounds) == ciphertext,
                "Known/fresh scalar verification failed",
            )
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "Output directory already exists")
    ledger = checked_json(args.ledger, LEDGER_SHA)
    dataset = checked_json(args.dataset, DATASET_SHA)
    identity = ledger["execution_identity"]
    require(
        identity["dataset_sha256"] == DATASET_SHA
        and identity["selected_instances"] == 64
        and identity["global_deadline_seconds"] == 300
        and identity["candidate_cutoff_soft_seconds"] == 45,
        "Frozen execution profile mismatch",
    )
    instances = {d["instance_id"]: SimonAttackInstance.from_dict(d) for d in dataset}
    require(len(dataset) == len(instances) == 64, "Dataset key IDs must be unique")
    require(
        all(
            i.rounds == 10
            and len(i.known_pairs) == 128
            and len(i.verification_pairs) == 2
            for i in instances.values()
        ),
        "Unexpected data profile",
    )
    rows = derive_rows(ledger["records"], set(instances))
    for record in ledger["records"]:
        if record["method"] == ORACLE:
            require(
                len(record["candidate_attempts"]) == 1
                and tuple_value(record["candidate_attempts"][0])
                == instances[record["instance_id"]].expected_trailing_subkeys(2),
                "Oracle tuple is not schedule-derived truth",
            )
    require(
        ledger["completed_count"] == len(ledger["records"]) == 576, "Incomplete ledger"
    )
    verified = verify_recoveries(ledger["records"], instances)
    unresolved = sum(
        r["method"] == ORACLE and r["outcome"] != "verified_recovery"
        for r in ledger["records"]
    )
    report = {
        "schema": "simon_candidate_history_observation_v1",
        "evidence_class": "RETROSPECTIVE_FROZEN_LEDGER_ANALYSIS",
        "ledger_sha256": LEDGER_SHA,
        "dataset_sha256": DATASET_SHA,
        "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "all_recoveries_independently_reverified": verified,
        "method_key_rows": len(rows),
        "distinct_keys": len({row["instance_id"] for row in rows}),
        "oracle_unresolved_keys": unresolved,
        "eligible_autonomous_method_key_cells": unresolved * (len(METHODS) - 1),
        "rows_by_method": dict(Counter(row["method"] for row in rows)),
        "rows": rows,
        "interpretation": "Observed counterexamples, not a causal history effect or speedup. Oracle used one 45s advisory request, not 300s of solving.",
    }
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "observation.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    fields = list(rows[0]) if rows else ["instance_id", "method"]
    with (args.output_dir / "cells.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for file in (output, args.output_dir / "cells.csv"):
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        file.with_name(file.name + ".sha256").write_text(f"{digest}  {file.name}\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "method_key_rows",
                    "distinct_keys",
                    "all_recoveries_independently_reverified",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
