"""Re-audit D-055 and report descriptive within-cipher paired interactions."""

import argparse
import csv
import io
import statistics
from collections import Counter
from pathlib import Path

from neural_sat_speck.backend.artifacts import cipher_functions, verify_key
from neural_sat_speck.backend.guard import atomic_json, read_checked_json, sha256_file
from neural_sat_speck.followups.ranking_depth_audit import audit_ledger
from neural_sat_speck.followups.ranking_depth_control import (
    load_manifest,
    optional_terminal_json,
)


def analyze(ledger_path: Path, manifest_path: Path) -> dict:
    audit = audit_ledger(ledger_path, manifest_path)
    manifest, paths = load_manifest(manifest_path, hashes=False)
    if manifest["mode"] != "full":
        raise ValueError("Pilot is excluded from scientific analysis")
    ledger = read_checked_json(ledger_path)
    cells, groups = [], {}
    for cell in manifest["cells"]:
        row = ledger["records"][cell["cell_id"]]
        directory = Path(row["directory"])
        raw = row["worker_result"]
        truth = read_checked_json(paths[cell["case_id"] + ":truth"])
        public = read_checked_json(paths[cell["case_id"] + ":public"])
        recovered = False
        outcome = row["outcome"]
        if outcome == "recovered":
            recovered = verify_key(
                cell["cipher"], cell["rounds"], raw["recovered_key"], public, truth
            )["assigned_key_match"]
            if not recovered:
                outcome = "distinct_verified_key"
        qpath = directory / "queue.json"
        ranked, _ = optional_terminal_json(qpath, outcome == "global_deadline")
        _, expand = cipher_functions(cell["cipher"])
        true_tuple = list(
            expand(tuple(truth["master_key"]), cell["rounds"])[-cell["trailing_keys"] :]
        )
        queue = ranked["queue"] if ranked else []
        position = queue.index(true_tuple) if true_tuple in queue else None
        attempts = sorted((directory / "attempts").glob("[0-9][0-9][0-9][0-9]"))
        dispatched = []
        for attempt in attempts:
            process, _ = optional_terminal_json(
                attempt / "native_process.json", outcome == "global_deadline"
            )
            if process is not None:
                dispatched.append(int(attempt.name))
        true_dispatched = position in dispatched if position is not None else False
        if (
            position is not None
            and position < len(attempts)
            and position not in dispatched
        ):
            true_dispatched = None  # Interrupted dispatch receipt; no claim either way.
        phase = row["phase_accounting"]
        times = dict(phase["completed_seconds"]) if phase is not None else {}
        if phase is not None and phase["inflight_phase"]:
            name = phase["inflight_phase"]
            times[name] = times.get(name, 0.0) + phase["inflight_seconds"]
        record = {
            **cell,
            "outcome": outcome,
            "assigned_verified_recovery": recovered,
            "parent_wall_seconds": row["parent_wall_seconds"],
            "par2_seconds": row["parent_wall_seconds"] if recovered else 600.0,
            "stage1_seconds": times.get("stage1_ranking", 0.0)
            if phase is not None
            else None,
            "stage2_seconds": times.get("stage2_ranking", 0.0)
            if phase is not None
            else None,
            "component_accounting_available": phase is not None,
            "phase_seconds": times,
            "startup_and_supervision_seconds": max(
                0.0, row["parent_wall_seconds"] - sum(times.values())
            ),
            "candidate_attempts_started": len(dispatched),
            "attempt_directories_retained": len(attempts),
            "true_tuple_queue_index_zero_based": position,
            "true_tuple_dispatched": true_dispatched,
            "queue_containment_measured": ranked is not None,
            "sampled_max_group_rss_bytes": max(
                s.get("memory", {}).get("group_rss_bytes", 0)
                for s in row["environment_samples"]
            ),
        }
        cells.append(record)
        groups.setdefault(
            (cell["cipher"], cell["trailing_keys"], cell["profile"]), []
        ).append(record)
    summaries = []
    interactions = []
    for (cipher, depth, backend), records in groups.items():
        summaries.append(
            {
                "cipher": cipher,
                "trailing_keys": depth,
                "profile": backend,
                "n": len(records),
                "recovered": sum(r["assigned_verified_recovery"] for r in records),
                "mean_par2_seconds": statistics.mean(
                    r["par2_seconds"] for r in records
                ),
                "median_parent_seconds": statistics.median(
                    r["parent_wall_seconds"] for r in records
                ),
                "outcomes": dict(Counter(r["outcome"] for r in records)),
            }
        )
    for cipher in ("speck32_64", "simon32_64"):
        differences = {}
        for backend in ("cms_native_gauss", "cms_native_no_gauss"):
            one = sorted(groups[cipher, 1, backend], key=lambda r: r["key_index"])
            two = sorted(groups[cipher, 2, backend], key=lambda r: r["key_index"])
            differences[backend] = {
                "recovery_difference_two_minus_one": statistics.mean(
                    int(b["assigned_verified_recovery"])
                    - int(a["assigned_verified_recovery"])
                    for a, b in zip(one, two, strict=True)
                ),
                "par2_difference_two_minus_one_seconds": statistics.mean(
                    b["par2_seconds"] - a["par2_seconds"]
                    for a, b in zip(one, two, strict=True)
                ),
                "paired_recovery_differences": [
                    int(b["assigned_verified_recovery"])
                    - int(a["assigned_verified_recovery"])
                    for a, b in zip(one, two, strict=True)
                ],
            }
        g, x = differences["cms_native_gauss"], differences["cms_native_no_gauss"]
        interactions.append(
            {
                "cipher": cipher,
                "n": 8,
                "within_backend": differences,
                "recovery_interaction_G_minus_X": g["recovery_difference_two_minus_one"]
                - x["recovery_difference_two_minus_one"],
                "par2_interaction_G_minus_X_seconds": g[
                    "par2_difference_two_minus_one_seconds"
                ]
                - x["par2_difference_two_minus_one_seconds"],
            }
        )
    return {
        "schema": "backend_ranking_depth_analysis_v1",
        "evidence_role": "opened_demo_diagnostic",
        "audit": audit,
        "cells": cells,
        "summaries": summaries,
        "interactions": interactions,
        "interpretation": "descriptive paired effects on eight opened keys per cipher; no population test or pooled confirmatory claim",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Analysis exists; write a new version")
    result = analyze(args.ledger, args.manifest)
    atomic_json(args.output, result)
    columns = [
        "cell_id",
        "cipher",
        "key_index",
        "trailing_keys",
        "profile",
        "outcome",
        "assigned_verified_recovery",
        "parent_wall_seconds",
        "par2_seconds",
        "stage1_seconds",
        "stage2_seconds",
        "startup_and_supervision_seconds",
        "candidate_attempts_started",
        "true_tuple_queue_index_zero_based",
        "true_tuple_dispatched",
        "sampled_max_group_rss_bytes",
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(result["cells"])
    csv_path = args.output.with_suffix(".csv")
    if csv_path.exists():
        raise SystemExit("CSV exists; write a new version")
    csv_path.write_text(output.getvalue())
    csv_path.with_suffix(".csv.sha256").write_text(
        sha256_file(csv_path) + "  " + csv_path.name + "\n"
    )


if __name__ == "__main__":
    main()
