#!/usr/bin/env python3
"""Post-run evidence audit; never launches a solver or generates a dataset.

See docs/BACKEND_CLOSURE_AUDIT_NOTE_20261004.md. Reuses the executed analyzer
for pinned provenance, then independently checks archived native assignments
against submitted DIMACS records and the cipher key schedule. Wrapper complete
assignments were not archived and cannot receive this extra model check.
"""

from __future__ import annotations

import argparse
import importlib.util
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path


def check_dimacs_assignment(text: str, assignment: dict[int, bool]) -> dict[str, int]:
    """Evaluate canonical one-record-per-line DIMACS, including signed XORs.

    Native `x` rows require odd parity of signed literal truth values. Ordinary
    rows require at least one true literal. This evaluator does not call the
    encoder, CNF converter or the worker's assignment checker.
    """
    lines = [line.split() for line in text.splitlines() if line.strip()]
    if not lines or len(lines[0]) != 4 or lines[0][:2] != ["p", "cnf"]:
        raise ValueError("Missing canonical DIMACS header")
    variables, records = map(int, lines[0][2:])
    if variables < 0 or records < 0 or set(assignment) != set(range(1, variables + 1)):
        raise ValueError("Assignment/header variable mismatch")
    if any(type(value) is not bool for value in assignment.values()):
        raise ValueError("Assignment values must be genuine booleans")
    if len(lines) - 1 != records:
        raise ValueError("DIMACS record count mismatch")
    counts = {"variables": variables, "cnf_records": 0, "xor_records": 0}
    for row in lines[1:]:
        xor = row[0].startswith("x")
        if xor:
            row = [row[0][1:], *row[1:]]
            if not row[0]:
                row = row[1:]
        literals = list(map(int, row))
        if not literals or literals[-1] != 0 or 0 in literals[:-1]:
            raise ValueError("Malformed canonical DIMACS record")
        literals = literals[:-1]
        if any(abs(literal) > variables for literal in literals):
            raise ValueError("DIMACS literal outside declared range")
        values = [assignment[abs(literal)] == (literal > 0) for literal in literals]
        satisfied = sum(values) % 2 == 1 if xor else any(values)
        if not satisfied:
            raise ValueError("Saved assignment violates submitted formula")
        counts["xor_records" if xor else "cnf_records"] += 1
    return counts


def decode_words(assignment: dict[int, bool], words: list[list[int]]) -> list[int]:
    """Decode the documented LSB-first variable maps without worker helpers."""
    return [
        sum(int(assignment[variable]) << bit for bit, variable in enumerate(word))
        for word in words
    ]


def extend_witness(cnf: list, xors: list, initial: dict, variables: int) -> dict:
    """Extend reference cipher values by forced literals only; no SAT search."""
    assignment = dict(initial)
    while True:
        before = len(assignment)
        for clause in cnf:
            if any(assignment.get(abs(lit)) == (lit > 0) for lit in clause):
                continue
            unknown = [lit for lit in clause if abs(lit) not in assignment]
            if not unknown:
                raise ValueError("Reference witness contradicts a CNF clause")
            if len(unknown) == 1:
                assignment[abs(unknown[0])] = unknown[0] > 0
        for literals, rhs in xors:
            target = int(rhs) ^ (sum(lit < 0 for lit in literals) % 2)
            active = [
                v
                for v, count in Counter(abs(lit) for lit in literals).items()
                if count % 2
            ]
            known = sum(assignment[v] for v in active if v in assignment) % 2
            unknown = [v for v in active if v not in assignment]
            if not unknown and known != target:
                raise ValueError("Reference witness contradicts an XOR row")
            if len(unknown) == 1:
                assignment[unknown[0]] = bool(known ^ target)
        if len(assignment) == before:
            break
    if set(assignment) != set(range(1, variables + 1)):
        raise ValueError("Forced extension left reference witness incomplete")
    return assignment


def reference_witness_checks(paths: dict, cells: list) -> dict:
    """Construct both representations' true-key witnesses, including timeouts."""
    from neural_sat_speck.backend.artifacts import cipher_functions
    from neural_sat_speck.backend.guard import read_checked_json
    from neural_sat_speck.sat.pure_cnf import _parse_native_dimacs
    from neural_sat_speck.simon.cipher import encrypt_round as simon_round
    from neural_sat_speck.speck.cipher import encrypt_round as speck_round

    checked = {}
    for cell in cells:
        ident = cell["case_id"]
        if ident in checked:
            continue
        public = read_checked_json(paths[f"{ident}:public"])
        truth = read_checked_json(paths[f"{ident}:truth"])
        varmap = read_checked_json(paths[f"{ident}:varmap"])
        mapping = varmap["mapping"]
        _, expand = cipher_functions(cell["cipher"])
        keys = expand(tuple(truth["master_key"]), cell["rounds"])
        assignment = {}

        def bind(variables, value):
            for bit, variable in enumerate(variables):
                expected = bool((value >> bit) & 1)
                if variable in assignment and assignment[variable] != expected:
                    raise ValueError(
                        "Conflicting reference cipher values in variable map"
                    )
                assignment[variable] = expected

        for variables, value in zip(
            mapping["master_key"], truth["master_key"], strict=True
        ):
            bind(variables, value)
        for variables, value in zip(mapping["round_keys"], keys, strict=True):
            bind(variables, value)
        observations = [
            pair[f"plaintext{s}"] for pair in public["pairs"] for s in (0, 1)
        ]
        round_fn = speck_round if cell["cipher"] == "speck32_64" else simon_round
        labels = ("x", "y") if cell["cipher"] == "speck32_64" else ("left", "right")
        for plaintext, state_map in zip(observations, mapping["states"], strict=True):
            state = tuple(plaintext)
            for index, round_state in enumerate(state_map["round_states"]):
                if index:
                    state = round_fn(state, keys[index - 1])
                for label, value in zip(labels, state, strict=True):
                    bind(round_state[label], value)
        witness = {}
        for kind, count_name in (
            ("native", "native_num_vars"),
            ("cnf", "pure_cnf_num_vars"),
        ):
            text = paths[f"{ident}:{kind}"].read_text()
            count = varmap[count_name]
            cnf, xors = _parse_native_dimacs(text, count)
            model = extend_witness(cnf, xors, assignment, count)
            witness[kind] = check_dimacs_assignment(text, model)
        checked[ident] = witness
    return checked


def audit(source: Path, manifest: Path, ledger_path: Path, analysis_path: Path) -> dict:
    """Reproduce the frozen analysis and add independent saved-model checks."""
    spec = importlib.util.spec_from_file_location(
        "frozen_backend_analysis", source / "scripts/analyze_backend_comparison.py"
    )
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load frozen analyzer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from neural_sat_speck.backend.artifacts import cipher_functions
    from neural_sat_speck.backend.guard import read_checked_json, sha256_file
    from neural_sat_speck.backend.solver_io import parse_solver_output

    recomputed = module.analyze(manifest, ledger_path)
    if recomputed != read_checked_json(analysis_path):
        raise ValueError("Saved analysis does not match frozen recomputation")
    ledger = read_checked_json(ledger_path)
    cells = read_checked_json(manifest)["cells"]
    records = ledger["records"]
    ordered = sorted(ledger["starts"], key=lambda k: ledger["starts"][k]["started_utc"])
    if ordered != [cell["cell_id"] for cell in cells]:
        raise ValueError("Actual cell-start order differs from fixed schedule")
    for before, after in zip(ordered, ordered[1:]):
        if datetime.fromisoformat(
            records[before]["completed_utc"]
        ) > datetime.fromisoformat(records[after]["started_utc"]):
            raise ValueError("Overlapping recorded backend cells")

    # These paths were already validated against both frozen source segments.
    paths = module.validate_manifest(read_checked_json(manifest), source)
    reference_witnesses = reference_witness_checks(paths, cells)
    model_checks = {}
    known_blocks = fresh_blocks = wrapper_recoveries = 0
    power_failures = []
    initial_power_only = []
    for ident, record in records.items():
        directory = Path(record["worker_directory"])
        # Fast cells can finish before the first periodic sampling interval.
        # Their initial observation is available; a later observation is not.
        if "power_latest.json" in record["raw_artifacts"]:
            power = read_checked_json(directory / "power_latest.json")
        else:
            initial_power_only.append(ident)
            power = read_checked_json(directory / "power_at_worker_start.json")
        if power.get("accepted") is not True:
            power_failures.append(ident)
        if record["outcome"] != "recovered":
            continue
        public = read_checked_json(paths[f"{record['case_id']}:public"])
        truth = read_checked_json(paths[f"{record['case_id']}:truth"])
        key = tuple(record["recovered_key"])
        encrypt, expand = cipher_functions(record["cipher"])
        if key != tuple(truth["master_key"]):
            raise ValueError("Assigned master key mismatch")
        for pair in public["pairs"]:
            for side in (0, 1):
                if encrypt(
                    tuple(pair[f"plaintext{side}"]), key, record["rounds"]
                ) != tuple(pair[f"ciphertext{side}"]):
                    raise ValueError("Known-block cipher mismatch")
                known_blocks += 1
        for plaintext, ciphertext in truth["verification_pairs"]:
            if encrypt(tuple(plaintext), key, record["rounds"]) != tuple(ciphertext):
                raise ValueError("Fresh-block cipher mismatch")
            fresh_blocks += 1
        if record["profile"] == "cms_wrapper":
            wrapper_recoveries += 1
            continue
        pure = record["profile"] in {"cms_cnf_no_gauss", "cadical_cnf"}
        submitted = directory / ("input.cnf" if pure else "input.dimacs")
        text = submitted.read_text()
        variables = int(text.splitlines()[0].split()[2])
        parsed = parse_solver_output(
            "cadical" if record["profile"] == "cadical_cnf" else "cms",
            record["solver_returncode"],
            (directory / "solver_stdout.txt").read_text(),
            required_variables=list(range(1, variables + 1)),
            num_vars=variables,
        )
        if parsed["status"] != "sat":
            raise ValueError("Recovery has no complete saved SAT assignment")
        assignment = parsed["assignment"]
        checked = check_dimacs_assignment(text, assignment)
        mapping = read_checked_json(paths[f"{record['case_id']}:varmap"])["mapping"]
        if pure:
            varmap = read_checked_json(paths[f"{record['case_id']}:varmap"])
            original = paths[f"{record['case_id']}:native"].read_text()
            checked["original_native_records_rechecked"] = sum(
                value
                for name, value in check_dimacs_assignment(
                    original,
                    {v: assignment[v] for v in range(1, varmap["native_num_vars"] + 1)},
                ).items()
                if name != "variables"
            )
        if decode_words(assignment, mapping["master_key"]) != list(key):
            raise ValueError("Saved native model does not decode to reported key")
        if decode_words(assignment, mapping["round_keys"]) != list(
            expand(key, record["rounds"])
        ):
            raise ValueError("Saved native round keys disagree with cipher reference")
        model_checks[ident] = checked
    expected_power_failures = {"simon32_64_k1_o128_m0__cms_wrapper"}
    if set(power_failures) != expected_power_failures:
        raise ValueError("Unexpected final raw power observation failure")
    deadlines = [
        r["online_seconds"]
        for r in records.values()
        if r["outcome"] == "global_deadline"
    ]
    return {
        "schema": "backend_root_closure_audit_v1",
        "classification": "OPENED_DEMO_DIAGNOSTIC",
        "ledger_sha256": sha256_file(ledger_path),
        "manifest_sha256": sha256_file(manifest),
        "saved_analysis_sha256": sha256_file(analysis_path),
        "frozen_reanalysis_matches_saved": True,
        "cells": len(records),
        "starts": len(ledger["starts"]),
        "outcomes": dict(Counter(r["outcome"] for r in records.values())),
        "known_recovery_blocks_verified": known_blocks,
        "fresh_recovery_blocks_verified": fresh_blocks,
        "native_saved_models_verified": len(model_checks),
        "wrapper_recoveries_without_archived_complete_model": wrapper_recoveries,
        "native_model_checks": model_checks,
        "case_native_and_cnf_reference_witnesses": reference_witnesses,
        "raw_artifacts_checked": sum(
            a["raw_artifacts_verified"]
            for a in recomputed["cell_evidence_audits"].values()
        ),
        "actual_start_order_matches_fixed_grid": True,
        "recorded_cells_do_not_overlap": True,
        "final_raw_power_failure_cells": power_failures,
        "cells_with_initial_power_observation_only": initial_power_only,
        "parent_cell_seconds_sum": sum(r["online_seconds"] for r in records.values()),
        "parent_deadline_seconds_min": min(deadlines),
        "parent_deadline_seconds_max": max(deadlines),
        "coordinator_pre_dispatch_seconds_sum": sum(
            r["pre_dispatch_seconds"] for r in records.values()
        ),
        "coordinator_post_audit_seconds_sum": sum(
            r["post_dispatch_audit_seconds"] for r in records.values()
        ),
        "run_receipts": ledger["run_receipts"],
        "global_par2_seconds": statistics.mean(
            r["online_seconds"] if r["outcome"] == "recovered" else 600.0
            for r in records.values()
        ),
        "limits": recomputed["interpretation_limits"]
        + [
            "Wrapper complete assignments cannot be rechecked from saved models",
            "Live process-group absence is a separate OS-level closure check",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "manifest", "ledger", "analysis", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite a closure audit")
    result = audit(
        args.source.resolve(),
        args.manifest.resolve(),
        args.ledger.resolve(),
        args.analysis.resolve(),
    )
    from neural_sat_speck.backend.guard import atomic_json

    atomic_json(args.output, result)
    print(
        f"Audited {result['cells']} cells, {result['native_saved_models_verified']} saved native models and all recovery blocks."
    )


if __name__ == "__main__":
    main()
