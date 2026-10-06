"""Read-only final audit of D-054; never constructs or calls a SAT solver.

Run with PYTHONPATH pointing at the manifest's frozen source snapshot. The frozen
validator checks provenance and receipts; this script separately evaluates every
returned Boolean assignment, decodes keys and recomputes descriptive statistics.
Outputs are new evidence only. Raw results and the executed analysis stay intact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def unique_object(items: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in items:
        require(key not in result, f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def checked_json(path: Path) -> tuple[dict | list, str]:
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    sidecar = path.with_name(path.name + ".sha256").read_text().split()
    require(sidecar == [sha, path.name], f"Checksum/filename mismatch: {path}")
    return json.loads(raw, object_pairs_hook=unique_object), sha


def check_assignment(formula: dict, model: list) -> list[int]:
    require(
        len(model) == formula["num_vars"] + 1
        and model[0] is None
        and all(type(value) is bool for value in model[1:]),
        "Model must contain every variable as a genuine Boolean",
    )

    def literal_value(literal: int) -> int:
        require(type(literal) is int and 0 < abs(literal) < len(model), "Bad literal")
        return int(model[abs(literal)]) if literal > 0 else 1 - int(model[-literal])

    for clause in formula["cnf_clauses"]:
        require(sum(map(literal_value, clause)) >= 1, "Original CNF clause violated")
    for clause in formula["xor_clauses"]:
        require(
            sum(map(literal_value, clause["literals"])) % 2 == int(clause["rhs"]),
            "Original XOR parity violated",
        )
    require(
        all(literal_value(lit) == 1 for lit in formula["assumption_literals"]),
        "Conditioned trailing-key literal violated",
    )
    return [
        sum(int(model[var]) * 2**bit for bit, var in enumerate(word))
        for word in formula["varmap"]["master_key"]
    ]


def recompute_summary(rows: list[dict], cipher: str) -> dict:
    """All rows must be eligible; no denominator silently omits a failure."""
    arms = ("permanent_units", "temporary_assumptions")
    result = {"arms": {}, "pairs": []}
    for arm in arms:
        selected = [
            r for r in rows if r["cell"]["cipher"] == cipher and r["cell"]["arm"] == arm
        ]
        require(len(selected) == 8, "Arm denominator differs from eight keys")
        success = [r for r in selected if r["outcome"] == "recovered"]
        result["arms"][arm] = {
            "scheduled": 8,
            "eligible": 8,
            "recovered": len(success),
            "outcomes": dict(Counter(r["outcome"] for r in selected)),
            "median_parent_seconds_eligible": median(
                min(r["parent_wall_seconds"], 300.0) for r in selected
            ),
            "par2_seconds_eligible": mean(
                r["parent_wall_seconds"] if r["outcome"] == "recovered" else 600.0
                for r in selected
            ),
            "mean_parent_seconds_success": mean(
                r["parent_wall_seconds"] for r in success
            )
            if success
            else None,
        }
    for index in range(8):
        pair = {
            r["cell"]["arm"]: r
            for r in rows
            if r["cell"]["cipher"] == cipher and r["cell"]["key_index"] == index
        }
        require(set(pair) == set(arms), "Incomplete paired key")
        result["pairs"].append(
            {
                "key_index": index,
                "eligible": True,
                "outcomes": {arm: pair[arm]["outcome"] for arm in arms},
                "restricted_parent_seconds": {
                    arm: min(pair[arm]["parent_wall_seconds"], 300.0) for arm in arms
                },
            }
        )
    result["discordances"] = {
        arm + "_only": sum(
            p["outcomes"][arm] == "recovered" and p["outcomes"][other] != "recovered"
            for p in result["pairs"]
        )
        for arm, other in (arms, arms[::-1])
    }
    result["twofold_cost_flag"] = {
        arm: sum(
            2 * p["restricted_parent_seconds"][arm]
            <= p["restricted_parent_seconds"][other]
            for p in result["pairs"]
        )
        >= 4
        for arm, other in (arms, arms[::-1])
    }
    return result


def same_values(actual: object, expected: object) -> bool:
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            same_values(actual[k], expected[k]) for k in actual
        )
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            same_values(a, b) for a, b in zip(actual, expected, strict=True)
        )
    if type(actual) is float and type(expected) is float:
        return math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9)
    return type(actual) is type(expected) and actual == expected


def audit(
    manifest_path: Path, ledger_path: Path, analysis_path: Path
) -> tuple[dict, list[dict]]:
    from neural_sat_speck.backend.artifacts import cipher_functions
    from neural_sat_speck.backend.formula import build_formula
    from neural_sat_speck.followups import conditioning_runner
    from neural_sat_speck.followups.conditioning_runner import (
        audit_ledger,
        require_pilot,
    )

    manifest, manifest_sha = checked_json(manifest_path)
    ledger, ledger_sha = checked_json(ledger_path)
    analysis, analysis_sha = checked_json(analysis_path)
    root = Path(manifest["source_root"])
    require(
        Path(conditioning_runner.__file__).resolve().is_relative_to(root / "src"),
        "Audit must use manifest-pinned source imports",
    )
    frozen_ledger, frozen_manifest = audit_ledger(ledger_path, manifest_path)
    require(
        frozen_ledger == ledger and frozen_manifest == manifest,
        "Frozen validator read different inputs",
    )
    require_pilot(manifest, root)
    require(
        ledger["status"] == "COMPLETE" and manifest["mode"] == "full",
        "Expected complete full campaign",
    )
    require(
        ledger["stop_reason"] is None
        and len(ledger["records"]) == len(ledger["starts"]) == 32,
        "Incomplete/stopped grid",
    )
    inventory = {}
    for directory in (ledger_path.parent, manifest_path.parent):
        for path in sorted(directory.rglob("*.json")):
            _, sha = checked_json(path)
            inventory[str(path)] = sha
    rows = [ledger["records"][c["cell_id"]] for c in manifest["cells"]]
    pilot, _ = checked_json(Path(manifest["pilot_ledger"]["path"]))
    pilot_tokens = {Path(r["directory"]).name for r in pilot["records"].values()}
    tokens, pids, cell_csv, previous_end, positive_models = set(), set(), [], 0.0, 0
    exit_only_samples, memory_max, sample_count, power_counts = 0, 0, 0, Counter()
    for row in rows:
        cell = row["cell"]
        require(
            row["outcome"] in {"recovered", "global_deadline"},
            "Unexpected outcome requires a separate closure review",
        )
        directory = Path(row["directory"])
        token = directory.name
        require(token not in tokens | pilot_tokens, "Reused invocation/pilot evidence")
        tokens.add(token)
        receipt, _ = checked_json(directory / "guardian_receipt.json")
        require(receipt["worker_pid"] not in pids, "Reused worker process identity")
        pids.add(receipt["worker_pid"])
        require(
            row["started_monotonic"] >= previous_end,
            "Workers overlap or violate serial schedule",
        )
        previous_end = receipt["cleanup_completed_monotonic"]
        request, _ = checked_json(directory / "solver_dispatch.json")
        public, _ = checked_json(
            Path(manifest["cases"][cell["case_id"]]["public"]["path"])
        )
        truth, _ = checked_json(
            Path(manifest["cases"][cell["case_id"]]["truth"]["path"])
        )
        formula = build_formula(cell["cipher"], public, 4, 1, include_cnf=False)
        require(
            request["invocation_id"] == token
            and request["cell_id"] == cell["cell_id"]
            and request["arm"] == cell["arm"],
            "Solver dispatch identity differs",
        )
        require(
            request["formula_sha256"]
            == manifest["cases"][cell["case_id"]]["formula_sha256"],
            "Dispatch formula differs",
        )
        require(
            request["solver_configuration"] == {"verbose": 0, "threads": 1},
            "Solver settings differ",
        )
        require(
            request["assumptions"]
            == (
                formula["assumption_literals"]
                if cell["arm"] == "temporary_assumptions"
                else []
            )
            and request["permanent_unit_count"]
            == (0 if cell["arm"] == "temporary_assumptions" else 16),
            "Conditioning interface differs",
        )
        require(
            row["started_monotonic"]
            <= request["requested_at_monotonic"]
            <= row["deadline_monotonic"] - 1,
            "No early solver request",
        )
        phases = row["phase_accounting"]
        if row["outcome"] == "global_deadline":
            require(
                row["worker_result"] is None and phases["inflight_phase"] == "solving",
                "Deadline hides a finished result or a different phase",
            )
        else:
            raw = row["worker_result"]
            key = check_assignment(formula, raw["model"])
            require(
                key == raw["recovered_key"] == truth["master_key"],
                "Decoded/reported/assigned key mismatch",
            )
            encrypt, expand = cipher_functions(cell["cipher"])
            require(
                list(expand(key, cell["rounds"]))[-1:] == public["trailing_round_keys"],
                "Trailing key schedule mismatch",
            )
            for pair in public["pairs"]:
                for side in (0, 1):
                    require(
                        list(
                            encrypt(
                                tuple(pair[f"plaintext{side}"]), key, cell["rounds"]
                            )
                        )
                        == pair[f"ciphertext{side}"],
                        "Known reference block mismatch",
                    )
            for pt, ct in truth["verification_pairs"]:
                require(
                    list(encrypt(tuple(pt), key, cell["rounds"])) == ct,
                    "Fresh reference block mismatch",
                )
            require(
                all(
                    raw[n] is True
                    for n in (
                        "formula_valid",
                        "verified_known",
                        "verified_fresh",
                        "assigned_key_match",
                    )
                ),
                "Positive status flags differ",
            )
            require(
                phases["inflight_phase"] is None
                and raw["phase_seconds"] == phases["completed_seconds"],
                "Completed phase accounting differs",
            )
            require(
                sum(raw["phase_seconds"].values())
                <= raw["elapsed_seconds"]
                <= row["parent_wall_seconds"],
                "Worker phase/runtime exceeds parent time",
            )
            positive_models += 1
        for sample in row["environment_samples"]:
            sample_count += 1
            power_counts[sample["power"]["power_source"]] += 1
            if "memory" in sample:
                memory_max = max(memory_max, sample["memory"]["group_rss_bytes"])
            policy = sample.get("worker_policy", {})
            observations = policy.get("observations", {})
            if observations and all(v.get("errno") == 3 for v in observations.values()):
                require(
                    receipt["status"] == "worker_exit"
                    and receipt["process_group_absent"] is True,
                    "ESRCH lacks matched terminal exit proof",
                )
                exit_only_samples += 1
        cell_csv.append(
            {
                **cell,
                "outcome": row["outcome"],
                "parent_wall_seconds": row["parent_wall_seconds"],
                "restricted_parent_seconds": min(row["parent_wall_seconds"], 300.0),
                "par2_seconds": row["parent_wall_seconds"]
                if row["outcome"] == "recovered"
                else 600.0,
                "completed_solving_seconds": phases["completed_seconds"].get(
                    "solving", 0.0
                ),
                "inflight_solving_seconds": phases["inflight_seconds"]
                if phases["inflight_phase"] == "solving"
                else 0.0,
                "cleanup_wall_seconds": row["cleanup_wall_seconds"],
                "formula_sha256": request["formula_sha256"],
                "invocation_id": token,
                "directory": str(directory),
            }
        )
    expected = {
        "role": "finite_opened_cohort_descriptive",
        "mode": "full",
        "status": "COMPLETE",
        "manifest_sha256": manifest_sha,
        "source_commit": manifest["source_commit"],
        "ciphers": {
            cipher: recompute_summary(rows, cipher)
            for cipher in ("speck32_64", "simon32_64")
        },
    }
    require(
        same_values(analysis, expected),
        "Independent paired summary differs from saved analysis",
    )
    return {
        "status": "PASS",
        "classification": "ENGINEERING_and_RETROSPECTIVE_opened_diagnostic",
        "checked_utc": datetime.now(UTC).isoformat(),
        "source_commit": manifest["source_commit"],
        "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "manifest_sha256": manifest_sha,
        "ledger_sha256": ledger_sha,
        "analysis_sha256": analysis_sha,
        "checked_cells": 32,
        "positive_models_independently_checked": positive_models,
        "outcomes": dict(Counter(r["outcome"] for r in rows)),
        "checked_json_sidecars": len(inventory) + 1,
        "json_inventory": inventory,
        "environment_sample_count": sample_count,
        "terminal_exit_only_samples": exit_only_samples,
        "sampled_max_group_rss_bytes": memory_max,
        "sampled_power_sources": dict(power_counts),
        "parent_wall_sum_seconds": sum(r["parent_wall_seconds"] for r in rows),
        "analysis_recomputed": expected,
        "limitations": [
            "Eight already opened keys per cipher, one repetition; no population or reproducibility inference",
            "Formula reconstruction and reference encryption share frozen repository encoders/ciphers; clause parity and key decoding are separate computations",
            "Solver request receipt precedes the API call; alone it does not trace C++ execution",
            "Receipts certify sampled resource/process policy and initial calling-thread QoS, not every thread, continuous power, or core affinity",
            "This audit checks terminal absence receipts; an actual OS process check is a separate receipt",
            "One true-candidate query; wrong-candidate histories and autonomous attacks unmeasured",
        ],
    }, cell_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("manifest", "ledger", "analysis", "output-dir"):
        parser.add_argument("--" + option, type=Path, required=True)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "Refusing to overwrite an earlier audit")
    report, rows = audit(
        args.manifest.absolute(), args.ledger.absolute(), args.analysis.absolute()
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "audit.json"
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    csv_path = args.output_dir / "cells.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for path in (output, csv_path):
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        path.with_name(path.name + ".sha256").write_text(f"{sha}  {path.name}\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in {"json_inventory", "analysis_recomputed"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
