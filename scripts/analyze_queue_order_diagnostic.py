"""Predeclared descriptive paired analysis; truth joins only after grid closure."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_sat_speck.analysis.paired_stats import (  # noqa: E402
    exact_mcnemar_test,
    newcombe_method10_paired_ci,
)
from neural_sat_speck.backend.guard import atomic_json, read_checked_json  # noqa: E402
from neural_sat_speck.followups.queue_order import (  # noqa: E402
    POLICIES,
    fixed_jobs,
    load_manifest,
)
from neural_sat_speck.followups.runner import validate_ledger  # noqa: E402

UNAVAILABLE = {"infrastructure_interrupted", "environment_rejected", "worker_exception"}


def analyze(ledger: dict, manifest: dict, digest: str, dataset: dict) -> dict:
    validate_ledger(ledger, manifest, digest, dataset)
    if ledger.get("status") != "COMPLETE" or set(ledger["records"]) != {
        j["job_id"] for j in fixed_jobs()
    }:
        raise ValueError("Analysis requires the complete96-job grid")
    summaries, cells = [], []
    for cipher, policy, rounds, _ in POLICIES:
        outcomes = {order: [] for order in ("ranked", "shuffled")}
        for key in range(8):
            base = f"{cipher}_k{key}_{policy}"
            prep = ledger["records"][base + "__prepare"]
            truth = dataset["ciphers"][cipher]["instances"][key]
            target = tuple(truth["rounds_data"][str(rounds)]["round_keys"][-2:])
            for order in outcomes:
                rec = ledger["records"][base + "__" + order]
                available = rec["outcome"] not in UNAVAILABLE
                success = rec["success"]
                if success and rec["recovered_key"] != truth["master_key"]:
                    raise ValueError(
                        "A verified distinct key is not assigned exact-key recovery"
                    )
                candidates = (
                    [tuple(v) for v in prep[order]]
                    if prep["outcome"] == "queue_ready"
                    else None
                )
                pos = (
                    candidates.index(target) + 1
                    if candidates and target in candidates
                    else None
                )
                accounting = rec.get("phase_accounting")
                phase_costs = {}
                for phase in (
                    "initial_formula_construction",
                    "solving",
                    "rebuilding_formula",
                    "decoding",
                    "verification",
                ):
                    phase_costs[phase + "_seconds"] = (
                        accounting["completed_seconds"].get(phase, 0.0)
                        + (
                            accounting["inflight_seconds"]
                            if accounting["inflight_phase"] == phase
                            else 0.0
                        )
                        if accounting is not None
                        else None
                    )
                cells.append(
                    {
                        "cipher": cipher,
                        "policy": policy,
                        "key_index": key,
                        "order": order,
                        "outcome": rec["outcome"],
                        "success": success,
                        "completed_trial": available,
                        "common_preparation_seconds": rec["common_preparation_seconds"],
                        "replay_wall_seconds": rec["replay_wall_seconds"],
                        "charged_online_seconds": rec["charged_online_seconds"],
                        "true_tuple_contained": target in candidates
                        if candidates
                        else None,
                        "true_tuple_position": pos,
                        "candidates_tested": rec.get("candidates_tested"),
                        "set_sha256": prep.get("set_sha256"),
                        "order_sha256": rec.get("queue_sha256"),
                        "ac_power_samples": sum(
                            s.get("power", {}).get("power_source") == "AC Power"
                            for s in rec.get("environment_samples", [])
                        ),
                        "battery_power_samples": sum(
                            s.get("power", {}).get("power_source") == "Battery Power"
                            for s in rec.get("environment_samples", [])
                        ),
                        "par2_assigned_seconds": rec["charged_online_seconds"]
                        if success
                        else 600.0,
                        **phase_costs,
                    }
                )
                outcomes[order].append((success, available, rec))
        pairs = [
            (a[0], b[0])
            for a, b in zip(outcomes["ranked"], outcomes["shuffled"], strict=True)
            if a[1] and b[1]
        ]
        n11 = sum(a and b for a, b in pairs)
        n10 = sum(a and not b for a, b in pairs)
        n01 = sum(not a and b for a, b in pairs)
        n00 = sum(not a and not b for a, b in pairs)
        ci = (
            newcombe_method10_paired_ci(n11, n10, n01, n00, len(pairs))
            if pairs
            else None
        )
        row = {
            "cipher": cipher,
            "policy": policy,
            "assigned_pairs": 8,
            "completed_pairs": len(pairs),
            "unavailable_pairs": 8 - len(pairs),
            "n11": n11,
            "n10": n10,
            "n01": n01,
            "n00": n00,
            "exact_mcnemar_raw_p": exact_mcnemar_test(n10, n01)["p_value"]
            if pairs
            else None,
            "paired_risk_difference_ci": ci,
            "evidence_class": "opened_demo_diagnostic",
            "multiplicity": "four descriptive contrasts; no familywise rejection claim",
        }
        for order, results in outcomes.items():
            row[order + "_verified_of8"] = sum(r[0] for r in results)
            row[order + "_unavailable_of8"] = sum(not r[1] for r in results)
            row[order + "_assigned_par2_seconds"] = statistics.mean(
                r[2]["charged_online_seconds"] if r[0] else 600.0 for r in results
            )
        summaries.append(row)
    return {
        "schema": "queue_order_analysis_v1",
        "manifest_sha256": digest,
        "summaries": summaries,
        "cells": cells,
        "limits": [
            "Eight opened keys per cipher; one shuffle per set",
            "Shared rank cost-accounted replay, not autonomous replication",
            "Order includes effects on incremental solver history",
            "Missing trials are unavailable; assigned operational metrics remain separate",
            "Author allowed battery before outcomes; timings depend on recorded laptop power conditions",
        ],
    }


def main() -> None:
    import hashlib

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or args.output.with_suffix(".csv").exists():
        raise FileExistsError("Preserve versioned derived analyses")
    manifest, paths = load_manifest(args.manifest, ROOT)
    ledger = read_checked_json(args.ledger)
    if ledger.get("evidence_root") != str(args.ledger.parent.absolute()):
        raise ValueError("Analysis evidence directory differs")
    result = analyze(
        ledger,
        manifest,
        hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        json.loads(paths["cohort"].read_bytes()),
    )
    atomic_json(args.output, result)
    with args.output.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result["cells"][0]))
        writer.writeheader()
        writer.writerows(result["cells"])
    csv_path = args.output.with_suffix(".csv")
    csv_path.with_name(csv_path.name + ".sha256").write_text(
        hashlib.sha256(csv_path.read_bytes()).hexdigest() + "  " + csv_path.name + "\n"
    )
    print(json.dumps(result["summaries"], indent=2))


if __name__ == "__main__":
    main()
