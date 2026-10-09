"""Reproduce descriptive environment checks from completed, hash-bound D-056 data."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from neural_sat_speck.backend.guard import atomic_json, read_checked_json, sha256_file
from scripts.analyze_backend_ranking_depth import summarize


def check(path: Path) -> dict:
    analysis = read_checked_json(path)
    audit = analysis["audit"]
    if (
        analysis["evidence_role"] != "development"
        or audit["passed"] is not True
        or audit["cells"] != 256
    ):
        raise ValueError("Expected the completed audited D-056 development analysis")
    raw = {}
    for segment in audit["segments"]:
        ledger_path = Path(segment["ledger_path"])
        if sha256_file(ledger_path) != segment["ledger_sha256"]:
            raise ValueError("Raw ledger changed since analysis")
        ledger = read_checked_json(ledger_path)
        if raw.keys() & ledger["records"].keys():
            raise ValueError("Duplicate terminal cells across segments")
        raw.update(ledger["records"])
    if set(raw) != {c["cell_id"] for c in analysis["cells"]} or len(raw) != 256:
        raise ValueError("Raw/derived assigned-cell sets differ")
    cases = defaultdict(list)
    paths = Counter()
    for cell in analysis["cells"]:
        cases[cell["cipher"], cell["key_index"]].append(cell)
        label = (
            "verified_recovery"
            if cell["assigned_verified_recovery"]
            else "tuple_absent_from_queue"
            if cell["true_tuple_queue_index_zero_based"] is None
            else "correct_tuple_dispatched_without_bounded_recovery"
            if cell["true_tuple_dispatched"] is True
            else "contained_tuple_not_dispatched"
        )
        paths[label] += 1
    all_ac, matched, mismatched = [], [], []
    for (cipher, key), cells in cases.items():
        source_sets = [
            {
                s["power"]["power_source"]
                for s in raw[c["cell_id"]]["environment_samples"]
            }
            for c in cells
        ]
        if all(s == {"AC Power"} for s in source_sets):
            all_ac.extend(cells)
        if all(s == source_sets[0] for s in source_sets):
            matched.extend(cells)
        else:
            mismatched.append({"cipher": cipher, "key_index": key})

    def paired(rows):
        summaries, interactions = summarize(rows)
        return {"summaries": summaries, "interactions": interactions}

    return {
        "schema": "d056_additional_audit_checks_v1",
        "role": "descriptive_audit_sensitivity_not_primary_or_power_causal_estimate",
        "failure_path_counts": dict(paths),
        "power_mismatched_key_clusters": mismatched,
        "all_ac_key_sensitivity": paired(all_ac),
        "matched_power_label_sensitivity": paired(matched),
        "segment_key_block_checks": {
            label: paired([c for c in analysis["cells"] if c["key_index"] in indices])
            for label, indices in (
                ("early_keys_0_through_16", set(range(17))),
                ("late_keys_18_through_31", set(range(18, 32))),
            )
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Audit checks exist; use a new output path")
    atomic_json(args.output, check(args.analysis))


if __name__ == "__main__":
    main()
