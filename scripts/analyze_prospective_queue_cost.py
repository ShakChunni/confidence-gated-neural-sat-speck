#!/usr/bin/env python3
"""Analysis of the prospective queue-cost campaign (D-060 model test, D-061 comparator).

Implements docs/PROSPECTIVE_QUEUE_COST_TEST_PROTOCOL_20261009.md sections 7 and 8 on a
completed full-campaign ledger. It refuses to run unless all 320 cells are terminal
with the design's exact grid. Every cell is in every denominator.

Run as:
    python3 -I scripts/analyze_prospective_queue_cost.py --campaign-dir DIR --output FILE

After the D-062 continuation, pass the interrupted run's execution directory as
--campaign-dir and the continuation output root as --continuation-root. Rows are
merged only if the combined audit (continuation_audit.json) passed and binds both
ledgers byte for byte.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

from fit_queue_cost_model import (  # noqa: E402
    as_bool,
    cp_interval,
    dists,
    load_dataset,
    p_recover,
)

from neural_sat_speck.backend.artifacts import cipher_functions  # noqa: E402
from neural_sat_speck.backend.guard import read_checked_json, sha256_file  # noqa: E402
from neural_sat_speck.followups.ranking_depth_control import (  # noqa: E402
    SCIENTIFIC_OUTCOMES,
    prospective_cells,
)

EVIDENCE_DIR = ROOT / "docs" / "queue_cost_model_evidence_20261009"
EXPECTED_CELLS = 320
KEYS = 32
BUDGET_SECONDS = 300.0
PAR2_PENALTY_SECONDS = 600.0
CONDITIONAL_SEED = 59201
CONDITIONAL_PASS_ACCURACY = 0.85
Z = 1.959963984540054

# Frozen predictions: protocol section 6 (90% intervals for the count out of 32).
FROZEN_INTERVALS = {
    "S-G": (13, 23),
    "S-X": (15, 25),
    "S-G-c": (0, 5),
    "S-X-c": (13, 25),
    "M-G": (23, 30),
    "M-X": (17, 27),
    "M-G-c": (15, 27),
    "M-X-c": (5, 16),
}
MODEL_ARMS = ("S-G", "S-X", "S-G-c", "S-X-c", "M-G", "M-X", "M-G-c", "M-X-c")
D056_ARMS = {
    "S-G": ("speck32_64", "cms_native_gauss"),
    "S-X": ("speck32_64", "cms_native_no_gauss"),
    "M-G": ("simon32_64", "cms_native_gauss"),
    "M-X": ("simon32_64", "cms_native_no_gauss"),
}
CLASSICAL_FAMILY = (("S-C-G", "S-G"), ("S-C-X", "S-X"))
MODEL_FAMILY = (
    ("S-X-c", "S-G-c"),
    ("M-G-c", "M-X-c"),
    ("S-G-c", "S-G"),
    ("S-X-c", "S-X"),
    ("M-G-c", "M-G"),
    ("M-X-c", "M-X"),
)
ARM_LABELS = {
    ("speck32_64", "neural", "cms_native_gauss", None): "S-G",
    ("speck32_64", "neural", "cms_native_no_gauss", None): "S-X",
    ("speck32_64", "neural", "cms_native_gauss", 1.5): "S-G-c",
    ("speck32_64", "neural", "cms_native_no_gauss", 1.5): "S-X-c",
    ("speck32_64", "classical_sparse_r6", "cms_native_gauss", None): "S-C-G",
    ("speck32_64", "classical_sparse_r6", "cms_native_no_gauss", None): "S-C-X",
    ("simon32_64", "neural", "cms_native_gauss", None): "M-G",
    ("simon32_64", "neural", "cms_native_no_gauss", None): "M-X",
    ("simon32_64", "neural", "cms_native_gauss", 10.0): "M-G-c",
    ("simon32_64", "neural", "cms_native_no_gauss", 10.0): "M-X-c",
}


def arm_key(cell: dict) -> tuple:
    return (
        cell["cipher"],
        cell.get("stage1", "neural"),
        cell["profile"],
        cell["candidate_cap_seconds"],
    )


def is_terminal(outcome: str) -> bool:
    return outcome in SCIENTIFIC_OUTCOMES


def load_campaign_rows(campaign_dir: Path) -> list[dict]:
    """One row per cell, read from the ledger, its manifest and its cell directory."""
    ledger = read_checked_json(campaign_dir / "ledger.json")
    if ledger.get("mode") != "full":
        raise ValueError("Analysis needs the full campaign ledger")
    manifest = read_checked_json(Path(ledger["manifest_path"]))
    rows = []
    for cell_id, record in ledger["records"].items():
        cell = record["cell"]
        row = {
            "cell_id": cell_id,
            "case_id": cell["case_id"],
            "key_index": cell["key_index"],
            "cipher": cell["cipher"],
            "stage1": cell.get("stage1", "neural"),
            "profile": cell["profile"],
            "candidate_cap_seconds": cell["candidate_cap_seconds"],
            "outcome": record["outcome"],
            "recovered": record["outcome"] == "recovered",
            "parent_wall_seconds": record.get("parent_wall_seconds"),
        }
        directory = Path(record["directory"]) if record.get("directory") else None
        if directory is None or not directory.is_dir():
            row.update(
                contained=False,
                true_queue_position=None,
                prep_seconds=None,
                capped_attempts=0,
            )
        else:
            row.update(_cell_evidence(cell, record, manifest, directory))
        rows.append(row)
    return rows


def load_continued_rows(campaign_dir: Path, continuation_root: Path) -> list[dict]:
    """D-062: original recorded prefix plus continuation suffix, audit-gated."""
    audit = read_checked_json(continuation_root / "continuation_audit.json")
    original = (campaign_dir / "ledger.json").absolute()
    suffix = (continuation_root / "execution" / "ledger.json").absolute()
    if (
        audit.get("schema") != "backend_ranking_depth_continuation_audit_v1"
        or audit.get("passed") is not True
        or audit.get("original_ledger_path") != str(original)
        or audit.get("original_ledger_sha256") != sha256_file(original)
        or audit.get("continuation_ledger_path") != str(suffix)
        or audit.get("continuation_ledger_sha256") != sha256_file(suffix)
    ):
        raise ValueError("Continuation audit does not bind these two ledgers")
    return load_campaign_rows(campaign_dir) + load_campaign_rows(suffix.parent)


def _cell_evidence(cell: dict, record: dict, manifest: dict, directory: Path) -> dict:
    _, expand = cipher_functions(cell["cipher"])
    truth_path = Path(manifest["cases"][cell["case_id"]]["truth"]["path"])
    master_key = read_checked_json(truth_path)["master_key"]
    truth = list(expand(tuple(master_key), cell["rounds"])[-cell["trailing_keys"] :])
    stage1 = directory / "stage1.json"
    parents = read_checked_json(stage1)["stage1_parents"] if stage1.exists() else []
    queue_path = directory / "queue.json"
    queue = read_checked_json(queue_path)["queue"] if queue_path.exists() else []
    attempts = sorted((directory / "attempts").glob("[0-9][0-9][0-9][0-9]"))
    capped = 0
    for attempt in attempts:
        if (attempt / "attempt.json").exists():
            capped += read_checked_json(attempt / "attempt.json")["status"] == "capped"
    prep = None
    if attempts and (attempts[0] / "native_process.json").exists():
        started = read_checked_json(attempts[0] / "native_process.json")[
            "started_monotonic"
        ]
        prep = started - record["started_monotonic"]
    return {
        "contained": truth[-1] in parents,
        "true_queue_position": queue.index(truth) if truth in queue else None,
        "prep_seconds": prep,
        "capped_attempts": capped,
    }


def validate_grid(rows: list[dict]) -> None:
    """Refuse unless every design cell is present exactly once and terminal."""
    if (
        len(rows) != EXPECTED_CELLS
        or len({r["cell_id"] for r in rows}) != EXPECTED_CELLS
    ):
        raise ValueError(
            f"Refusing to analyse: expected {EXPECTED_CELLS} distinct cells, found {len(rows)}"
        )
    design = {c["cell_id"]: arm_key(c) for c in prospective_cells("full", KEYS)}
    if {r["cell_id"]: arm_key(r) for r in rows} != design:
        raise ValueError(
            "Refusing to analyse: cell grid differs from the prospective design"
        )
    pending = [r["cell_id"] for r in rows if not is_terminal(r["outcome"])]
    if pending:
        raise ValueError(f"Refusing to analyse: {len(pending)} cells are not terminal")


def arms_of(rows: list[dict]) -> dict[str, list[dict]]:
    arms: dict[str, list[dict]] = {}
    for row in rows:
        arms.setdefault(ARM_LABELS[arm_key(row)], []).append(row)
    for label, members in arms.items():
        members.sort(key=lambda r: r["key_index"])
        if [r["key_index"] for r in members] != list(range(KEYS)):
            raise ValueError(f"Arm {label} does not cover all 32 keys")
    return arms


def summarize_arm(members: list[dict]) -> dict:
    prep = [r["prep_seconds"] for r in members if r["prep_seconds"] is not None]
    par2 = [
        r["parent_wall_seconds"] if r["recovered"] else PAR2_PENALTY_SECONDS
        for r in members
    ]
    return {
        "recovered": sum(r["recovered"] for r in members),
        "of": len(members),
        "contained": sum(r["contained"] for r in members),
        "par2_mean_seconds": statistics.mean(par2),
        "median_prep_seconds": statistics.median(prep) if prep else None,
        "capped_attempts": sum(r["capped_attempts"] for r in members),
    }


def frozen_check(counts: dict) -> dict:
    result = {}
    for label, (lo, hi) in FROZEN_INTERVALS.items():
        observed = counts[label]["recovered"]
        result[label] = {
            "recovered": observed,
            "frozen_interval": [lo, hi],
            "inside": lo <= observed <= hi,
        }
    return result


def directional_checks(counts: dict) -> dict:
    speck = counts["S-X-c"]["recovered"] - counts["S-G-c"]["recovered"]
    simon = counts["M-G-c"]["recovered"] - counts["M-X-c"]["recovered"]
    return {
        "speck_capped_S-X-c_minus_S-G-c": {
            "difference": speck,
            "threshold": 10,
            "holds": speck >= 10,
        },
        "simon_capped_M-G-c_minus_M-X-c": {
            "difference": simon,
            "threshold": 5,
            "holds": simon >= 5,
        },
    }


def primary_verdict(frozen: dict, directional: dict) -> str:
    inside = sum(v["inside"] for v in frozen.values())
    holds = sum(v["holds"] for v in directional.values())
    if inside >= 7 and holds == len(directional):
        return "PASS"
    if 5 <= inside <= 6 or holds == 1:
        return "PARTIAL"
    return "FAIL"


def mcnemar_exact(n10: int, n01: int) -> float:
    """Two-sided exact McNemar p-value (binomial test on the discordant pairs)."""
    n = n10 + n01
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(n10, n01) + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def holm(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in input order."""
    m = len(p_values)
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(sorted(range(m), key=lambda i: p_values[i])):
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def wilson(x: int, n: int) -> tuple[float, float]:
    p = x / n
    denom = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / denom
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / denom
    return centre - half, centre + half


def newcombe_paired(n11: int, n10: int, n01: int, n00: int) -> tuple[float, float]:
    """Newcombe (1998) method 10: hybrid Wilson interval for a paired difference."""
    n = n11 + n10 + n01 + n00
    p1, p2 = (n11 + n10) / n, (n11 + n01) / n
    l1, u1 = wilson(n11 + n10, n)
    l2, u2 = wilson(n11 + n01, n)
    denom = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    phi = (n11 * n00 - n10 * n01) / denom if denom else 0.0
    delta = p1 - p2
    lower = delta - math.sqrt(
        max(0.0, (p1 - l1) ** 2 + (u2 - p2) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2))
    )
    upper = delta + math.sqrt(
        max(0.0, (u1 - p1) ** 2 + (p2 - l2) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2))
    )
    return lower, upper


def pair_result(a: str, b: str, arms: dict[str, list[dict]]) -> dict:
    table = {"n11": 0, "n10": 0, "n01": 0, "n00": 0}
    for ra, rb in zip(arms[a], arms[b], strict=True):
        assert ra["key_index"] == rb["key_index"]
        table[f"n{int(ra['recovered'])}{int(rb['recovered'])}"] += 1
    n = KEYS
    lower, upper = newcombe_paired(
        table["n11"], table["n10"], table["n01"], table["n00"]
    )
    return {
        "a": a,
        "b": b,
        **table,
        "difference_a_minus_b": (table["n10"] - table["n01"]) / n,
        "newcombe_95": [lower, upper],
        "p_raw": mcnemar_exact(table["n10"], table["n01"]),
    }


def family(pairs, arms: dict[str, list[dict]]) -> list[dict]:
    results = [pair_result(a, b, arms) for a, b in pairs]
    for result, adjusted in zip(
        results, holm([r["p_raw"] for r in results]), strict=True
    ):
        result["p_holm"] = adjusted
    return results


def conditional_test(rows: list[dict], arms: dict[str, list[dict]], evidence) -> dict:
    """Frozen model, unchanged, applied to each fresh model-arm cell."""
    predicted, contained, majority, actual = [], [], [], []
    for label in MODEL_ARMS:
        members = arms[label]
        first = members[0]
        rkm, skm, oarr = dists(evidence, (first["cipher"], 2, first["profile"]))
        rng = np.random.default_rng(CONDITIONAL_SEED)
        recovered = sum(r["recovered"] for r in members)
        arm_majority = 2 * recovered > len(members)
        for r in members:
            prep = (
                r["prep_seconds"]
                if r["prep_seconds"] is not None
                else r["parent_wall_seconds"]
            )
            p = p_recover(
                float(prep),
                r["true_queue_position"],
                rkm,
                skm,
                oarr,
                rng,
                tau=r["candidate_cap_seconds"],
            )
            predicted.append(p >= 0.5)
            contained.append(r["contained"])
            majority.append(arm_majority)
            actual.append(r["recovered"])
    actual_arr = np.array(actual)

    def accuracy(prediction) -> float:
        return float(np.mean(np.array(prediction) == actual_arr))

    model, by_contained, by_majority = (
        accuracy(predicted),
        accuracy(contained),
        accuracy(majority),
    )
    passed = (
        model >= CONDITIONAL_PASS_ACCURACY
        and model > by_contained
        and model > by_majority
    )
    return {
        "cells": len(actual),
        "accuracy_model": model,
        "accuracy_recovered_if_contained": by_contained,
        "accuracy_arm_majority": by_majority,
        "verdict": "PASS" if passed else "FAIL",
    }


def analyze(rows: list[dict], evidence) -> dict:
    validate_grid(rows)
    arms = arms_of(rows)
    counts = {label: summarize_arm(members) for label, members in sorted(arms.items())}
    frozen = frozen_check(counts)
    directional = directional_checks(counts)
    d056 = {}
    for label, (cipher, profile) in D056_ARMS.items():
        cells = [
            c
            for c in evidence.cells
            if c["cipher"] == cipher
            and int(c["depth"]) == 2
            and c["profile"] == profile
        ]
        x = sum(as_bool(c["recovered"]) for c in cells)
        lower, upper = cp_interval(x, len(cells))
        d056[label] = {"recovered": x, "of": len(cells), "exact_95": [lower, upper]}
    return {
        "schema": "prospective_queue_cost_analysis_v1",
        "cells": len(rows),
        "counts": counts,
        "frozen_interval_check": frozen,
        "directional_checks": directional,
        "primary_model_test": {
            "verdict": primary_verdict(frozen, directional),
            "arms_inside_frozen_interval": sum(v["inside"] for v in frozen.values()),
        },
        "classical_comparison_D061": {
            "holm_family_size": len(CLASSICAL_FAMILY),
            "tests": family(CLASSICAL_FAMILY, arms),
        },
        "model_contrasts_D060": {
            "holm_family_size": len(MODEL_FAMILY),
            "tests": family(MODEL_FAMILY, arms),
        },
        "d056_replication_descriptive": d056,
        "conditional_model_test": conditional_test(rows, arms, evidence),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--continuation-root", type=Path, default=None)
    parser.add_argument("--evidence-dir", type=Path, default=EVIDENCE_DIR)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        raise SystemExit("Refusing to overwrite existing analysis output")
    try:
        rows = (
            load_campaign_rows(args.campaign_dir)
            if args.continuation_root is None
            else load_continued_rows(args.campaign_dir, args.continuation_root)
        )
        result = analyze(rows, load_dataset(args.evidence_dir))
    except ValueError as error:
        raise SystemExit(str(error)) from error
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
