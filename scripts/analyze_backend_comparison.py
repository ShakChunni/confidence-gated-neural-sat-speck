#!/usr/bin/env python3
"""Descriptive paired backend audit; four opened keys are not confirmation."""

from __future__ import annotations

import argparse
import itertools
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_sat_speck.backend.guard import (  # noqa: E402
    atomic_json,
    read_checked_json,
    sha256_file,
    validate_manifest,
)
from neural_sat_speck.backend.runner import (  # noqa: E402
    _verify_recovery,
    validate_ledger,
)


def analyze(manifest_path: Path, ledger_path: Path) -> dict:
    import json

    from neural_sat_speck.backend.artifacts import validate_case_inputs, validate_cohort
    from neural_sat_speck.backend.evidence import audit_cell_evidence
    from neural_sat_speck.backend.reviewed_resume import (
        load_prefix,
        provenance,
        validate_preserved_prefix,
        validate_review_history,
    )

    manifest = read_checked_json(manifest_path)
    paths = validate_manifest(manifest, ROOT)
    cohort = json.loads(paths["cohort"].read_bytes())
    validate_cohort(cohort)
    validate_case_inputs(cohort, paths)
    ledger = read_checked_json(ledger_path)
    prefix = load_prefix(manifest, ROOT, ledger_path)
    validate_preserved_prefix(ledger, prefix)
    validate_review_history(ledger, prefix, sha256_file(manifest_path))
    validate_ledger(ledger, manifest, sha256_file(manifest_path))
    records = ledger["records"]
    if ledger.get("status") != "complete" or len(records) != 160:
        raise ValueError("Final analysis requires the complete 160-cell grid")
    evidence_audits = {}
    for ident, record in records.items():
        recorded_manifest, recorded_paths = provenance(ident, manifest, paths, prefix)
        _verify_recovery(record, recorded_paths)
        evidence_audits[ident] = audit_cell_evidence(
            record,
            ledger["starts"][ident],
            recorded_manifest,
            recorded_paths,
            ledger_path,
            reviewed_environment_stop=prefix is not None
            and ident in prefix["power_stop_cells"],
        )
    groups = defaultdict(list)
    blocks = defaultdict(dict)
    for record in records.values():
        block = (
            f"{record['cipher']}_o{record['observations']}_m{record['trailing_keys']}"
        )
        groups[f"{block}__{record['profile']}"].append(record)
        blocks[block].setdefault(record["profile"], {})[record["key_index"]] = record
    summaries = {}
    for name, group in sorted(groups.items()):
        durations = [
            r["online_seconds"] for r in group if r.get("duration_available", True)
        ]
        summaries[name] = {
            "n": len(group),
            "recoveries": sum(r["outcome"] == "recovered" for r in group),
            "outcomes": dict(Counter(r["outcome"] for r in group)),
            "median_available_online_seconds": statistics.median(durations)
            if durations
            else None,
            "unavailable_durations": len(group) - len(durations),
            "par2_seconds": statistics.mean(
                r["online_seconds"] if r["outcome"] == "recovered" else 600.0
                for r in group
            ),
        }
    paired = {}
    for block, profiles in sorted(blocks.items()):
        paired[block] = []
        for first, second in itertools.combinations(sorted(profiles), 2):
            a, b = profiles[first], profiles[second]
            if set(a) != set(b) or set(a) != set(range(4)):
                raise ValueError("Paired key grid mismatch")
            both = a_only = b_only = neither = 0
            par2_differences = []
            for key in range(4):
                sa, sb = (
                    a[key]["outcome"] == "recovered",
                    b[key]["outcome"] == "recovered",
                )
                both += sa and sb
                a_only += sa and not sb
                b_only += sb and not sa
                neither += not sa and not sb
                par2_differences.append(
                    (a[key]["online_seconds"] if sa else 600.0)
                    - (b[key]["online_seconds"] if sb else 600.0)
                )
            paired[block].append(
                {
                    "first": first,
                    "second": second,
                    "both": both,
                    "first_only": a_only,
                    "second_only": b_only,
                    "neither": neither,
                    "recovery_difference": (a_only - b_only) / 4,
                    "mean_paired_par2_difference_seconds": statistics.mean(
                        par2_differences
                    ),
                }
            )
    return {
        "classification": "OPENED_DEMO_DIAGNOSTIC",
        "manifest_sha256": sha256_file(manifest_path),
        "ledger_sha256": sha256_file(ledger_path),
        "completed_cells": 160,
        "resume_reviews": ledger.get("resume_reviews", []),
        "summaries": summaries,
        "paired_descriptive_contrasts": paired,
        "cell_evidence_audits": evidence_audits,
        "interpretation_limits": [
            "Four fixed keys, one repetition; no population or cipher-mechanism claim",
            "Unit assumptions are shared by all profiles and differ from frozen attack calls",
            "Wrapper/native builds and internal defaults may differ",
            "Requested Gauss policy requires raw-log evidence of matrix construction/use",
            "Native pre-exec and process samples do not certify all threads continuously",
            "Any retained power-stop failure and reviewed source continuation remain part of the evidence",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_json(args.output, analyze(args.manifest.resolve(), args.ledger.resolve()))
    print("Complete grid and recoveries audited; descriptive backend analysis written.")


if __name__ == "__main__":
    main()
