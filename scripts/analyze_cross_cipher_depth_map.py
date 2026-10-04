#!/usr/bin/env python3
"""Statistical analysis and synthesis of the 192-cell cross-cipher residual-depth map.

Generates:
1. Complete grid recovery rates, medians, IQRs, and formula complexity.
2. Controlled paired contrasts at matched nominal residual depth:
   - SPECK at d=6: (6,0) vs (7,1) vs (8,2)
   - SPECK at d=5: (6,1) vs (7,2) vs (8,3)
   - SIMON at d=9: (9,0) vs (10,1) vs (11,2)
   - SIMON at d=8: (9,1) vs (10,2) vs (11,3)
3. Observation sensitivity: O=4 vs O=128 impact across depth regimes.
4. Output analysis JSON and markdown synthesis report.
"""

from __future__ import annotations

import argparse
import math
import statistics as stats
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_sat_speck.backend.guard import (  # noqa: E402
    atomic_json,
    read_checked_json,
    sha256_file,
)

RESULTS_FILE = (
    ROOT
    / "results"
    / "phase11_scaling_diagnostics"
    / "cross_cipher_residual_depth_map.json"
)
ANALYSIS_OUTPUT = (
    ROOT
    / "results"
    / "phase11_scaling_diagnostics"
    / "cross_cipher_residual_depth_analysis_reviewed_v2.json"
)


def validate_grid(data: dict) -> list[dict]:
    points = {
        "speck32_64": ((6, 0), (6, 1), (7, 1), (7, 2), (8, 2), (8, 3)),
        "simon32_64": ((9, 0), (9, 1), (10, 1), (10, 2), (11, 2), (11, 3)),
    }
    expected = {
        f"{cipher}_r{r}_m{m}_o{o}_k{k}": (cipher, r, m, r - m, o, k)
        for cipher, pairs in points.items()
        for r, m in pairs
        for o in (4, 128)
        for k in range(8)
    }
    rows = data.get("cells")
    if not isinstance(rows, dict) or set(rows) != set(expected):
        raise ValueError("Expected the exact complete 192-cell grid")
    fields = (
        "cipher",
        "rounds",
        "assumed_keys_count",
        "nominal_residual_depth",
        "observations",
        "instance_index",
    )
    for ident, row in rows.items():
        if row.get("cell_id") != ident or any(
            type(row.get(f)) is not type(v) or row.get(f) != v
            for f, v in zip(fields, expected[ident], strict=True)
        ):
            raise ValueError("Duplicate or inconsistent depth cell identity")
        if row.get("outcome") not in {"verified_recovery", "parent_watchdog_timeout"}:
            raise ValueError("Additional outcome requires explicit analysis review")
        recovered = row["outcome"] == "verified_recovery"
        if (
            type(row.get("verified_recovery")) is not bool
            or row["verified_recovery"] != recovered
        ):
            raise ValueError("Contradictory recovery flag")
        elapsed = row.get("parent_wall_seconds")
        if (
            type(elapsed) not in (float, int)
            or not math.isfinite(elapsed)
            or elapsed <= 0
            or (recovered and elapsed >= 300)
            or (not recovered and elapsed < 300)
        ):
            raise ValueError("Invalid parent duration or late recovery")
    return list(rows.values())


def analyze_map(
    results_file: Path | None = None,
    output: Path | None = None,
    dataset: Path | None = None,
) -> dict[str, Any]:
    results_file, output = results_file or RESULTS_FILE, output or ANALYSIS_OUTPUT
    if output.exists():
        raise FileExistsError("Reviewed analysis output already exists; preserve it")
    data = read_checked_json(results_file)
    cells = validate_grid(data)
    cohort = None
    if dataset is not None:
        from neural_sat_speck.backend.artifacts import validate_cohort

        cohort = read_checked_json(dataset)
        validate_cohort(cohort)

    # 1. Overall counts
    total_cells = len(cells)
    total_recovered = sum(1 for c in cells if c["outcome"] == "verified_recovery")
    total_timeouts = sum(1 for c in cells if c["outcome"] == "parent_watchdog_timeout")
    total_other = total_cells - total_recovered - total_timeouts

    # 2. Group by cell configuration
    # (cipher, rounds, assumed_keys, nominal_depth, observations)
    groups: dict[tuple[str, int, int, int, int], list[dict[str, Any]]] = defaultdict(
        list
    )
    for c in cells:
        key = (
            c["cipher"],
            c["rounds"],
            c["assumed_keys_count"],
            c["nominal_residual_depth"],
            c["observations"],
        )
        groups[key].append(c)

    table_rows = []
    for key in sorted(groups.keys()):
        cipher, r, m, d_val, o = key
        cell_list = groups[key]
        n = len(cell_list)
        rec = sum(1 for c in cell_list if c["outcome"] == "verified_recovery")
        timeouts = sum(
            1 for c in cell_list if c["outcome"] == "parent_watchdog_timeout"
        )

        p_times = [c["parent_wall_seconds"] for c in cell_list]
        completed = [c for c in cell_list if c["outcome"] == "verified_recovery"]
        s_times = [c["solve_wall_seconds"] for c in completed]
        if any(
            type(t) not in (float, int) or not math.isfinite(t) or t <= 0
            for t in s_times
        ):
            raise ValueError("Completed solve duration is invalid")

        med_p = stats.median(p_times)
        med_s = stats.median(s_times) if s_times else None
        q25_p = stats.quantiles(p_times, n=4)[0] if len(p_times) >= 4 else min(p_times)
        q75_p = stats.quantiles(p_times, n=4)[2] if len(p_times) >= 4 else max(p_times)

        sizes = {
            (c["variables_count"], c["cnf_clauses_count"], c["xor_clauses_count"])
            for c in completed
        }
        if len(sizes) > 1 or any(
            type(v) is not int or v <= 0 for values in sizes for v in values
        ):
            raise ValueError("Inconsistent completed formula counts")
        v_cnt, cnf_cnt, xor_cnt = next(iter(sizes)) if sizes else (None, None, None)
        count_source = (
            "completed_worker_receipts" if sizes else "unavailable_in_raw_workers"
        )
        if not sizes and cohort is not None:
            from neural_sat_speck.backend.artifacts import case_inputs
            from neural_sat_speck.backend.formula import build_formula

            public, _ = case_inputs(
                cohort,
                {
                    "cipher": cipher,
                    "rounds": r,
                    "observations": o,
                    "trailing_keys": m,
                    "key_index": 0,
                    "case_id": f"depth_reconstruction_{cipher}_{r}_{m}_{o}",
                },
            )
            formula = build_formula(cipher, public, o, m, include_cnf=False)
            v_cnt, cnf_cnt, xor_cnt = (
                formula["num_vars"],
                len(formula["cnf_clauses"]),
                len(formula["xor_clauses"]),
            )
            count_source = "offline_encoder_reconstruction_not_worker_measurement"

        table_rows.append(
            {
                "cipher": cipher,
                "rounds": r,
                "assumed_keys": m,
                "nominal_residual_depth": d_val,
                "observations": o,
                "n_instances": n,
                "verified_recovery_count": rec,
                "verified_recovery_rate": rec / n,
                "parent_timeout_count": timeouts,
                "median_parent_seconds": med_p,
                "iqr_parent_seconds": q75_p - q25_p,
                "median_solve_seconds": med_s,
                "solve_median_scope": "completed_verified_recoveries_only",
                "solve_duration_available_count": len(s_times),
                "solve_duration_unavailable_count": n - len(s_times),
                "par2_seconds": stats.mean(
                    c["parent_wall_seconds"]
                    if c["outcome"] == "verified_recovery"
                    else 600.0
                    for c in cell_list
                ),
                "formula_count_source": count_source,
                "variables_count": v_cnt,
                "cnf_clauses_count": cnf_cnt,
                "xor_clauses_count": xor_cnt,
            }
        )

    # 3. Controlled Paired Contrasts at Matched Nominal Depth d
    # For each key k (0..7), compare solve times across same d
    paired_contrasts: dict[str, Any] = {}

    def get_cell(cipher: str, r: int, m: int, o: int, k: int) -> dict[str, Any]:
        return next(
            c
            for c in cells
            if c["cipher"] == cipher
            and c["rounds"] == r
            and c["assumed_keys_count"] == m
            and c["observations"] == o
            and c["instance_index"] == k
        )

    # Contrast 1: SPECK d=6 at O=4: (6,0) vs (7,1) vs (8,2)
    speck_d6_o4 = []
    for k in range(8):
        c60 = get_cell("speck32_64", 6, 0, 4, k)
        c71 = get_cell("speck32_64", 7, 1, 4, k)
        c82 = get_cell("speck32_64", 8, 2, 4, k)
        speck_d6_o4.append(
            {
                "key": k,
                "(6,0)": {
                    "time": c60["parent_wall_seconds"],
                    "outcome": c60["outcome"],
                },
                "(7,1)": {
                    "time": c71["parent_wall_seconds"],
                    "outcome": c71["outcome"],
                },
                "(8,2)": {
                    "time": c82["parent_wall_seconds"],
                    "outcome": c82["outcome"],
                },
            }
        )
    paired_contrasts["speck_d6_o4"] = speck_d6_o4

    # Contrast 2: SPECK d=5 at O=4: (6,1) vs (7,2) vs (8,3)
    speck_d5_o4 = []
    for k in range(8):
        c61 = get_cell("speck32_64", 6, 1, 4, k)
        c72 = get_cell("speck32_64", 7, 2, 4, k)
        c83 = get_cell("speck32_64", 8, 3, 4, k)
        speck_d5_o4.append(
            {
                "key": k,
                "(6,1)": {
                    "time": c61["parent_wall_seconds"],
                    "outcome": c61["outcome"],
                },
                "(7,2)": {
                    "time": c72["parent_wall_seconds"],
                    "outcome": c72["outcome"],
                },
                "(8,3)": {
                    "time": c83["parent_wall_seconds"],
                    "outcome": c83["outcome"],
                },
            }
        )
    paired_contrasts["speck_d5_o4"] = speck_d5_o4

    # Contrast 3: SIMON d=9 at O=4: (9,0) vs (10,1) vs (11,2)
    simon_d9_o4 = []
    for k in range(8):
        c90 = get_cell("simon32_64", 9, 0, 4, k)
        c101 = get_cell("simon32_64", 10, 1, 4, k)
        c112 = get_cell("simon32_64", 11, 2, 4, k)
        simon_d9_o4.append(
            {
                "key": k,
                "(9,0)": {
                    "time": c90["parent_wall_seconds"],
                    "outcome": c90["outcome"],
                },
                "(10,1)": {
                    "time": c101["parent_wall_seconds"],
                    "outcome": c101["outcome"],
                },
                "(11,2)": {
                    "time": c112["parent_wall_seconds"],
                    "outcome": c112["outcome"],
                },
            }
        )
    paired_contrasts["simon_d9_o4"] = simon_d9_o4

    # Contrast 4: SIMON d=8 at O=4: (9,1) vs (10,2) vs (11,3)
    simon_d8_o4 = []
    for k in range(8):
        c91 = get_cell("simon32_64", 9, 1, 4, k)
        c102 = get_cell("simon32_64", 10, 2, 4, k)
        c113 = get_cell("simon32_64", 11, 3, 4, k)
        simon_d8_o4.append(
            {
                "key": k,
                "(9,1)": {
                    "time": c91["parent_wall_seconds"],
                    "outcome": c91["outcome"],
                },
                "(10,2)": {
                    "time": c102["parent_wall_seconds"],
                    "outcome": c102["outcome"],
                },
                "(11,3)": {
                    "time": c113["parent_wall_seconds"],
                    "outcome": c113["outcome"],
                },
            }
        )
    paired_contrasts["simon_d8_o4"] = simon_d8_o4

    # Contrast 5: O=4 vs O=128 Sensitivity
    observation_impact = []
    for cipher in ("speck32_64", "simon32_64"):
        r_points = (
            [(6, 0), (6, 1), (7, 1), (7, 2), (8, 2), (8, 3)]
            if cipher == "speck32_64"
            else [(9, 0), (9, 1), (10, 1), (10, 2), (11, 2), (11, 3)]
        )
        for r, m in r_points:
            o4_cells = [
                c
                for c in cells
                if c["cipher"] == cipher
                and c["rounds"] == r
                and c["assumed_keys_count"] == m
                and c["observations"] == 4
            ]
            o128_cells = [
                c
                for c in cells
                if c["cipher"] == cipher
                and c["rounds"] == r
                and c["assumed_keys_count"] == m
                and c["observations"] == 128
            ]
            rec_o4 = sum(1 for c in o4_cells if c["outcome"] == "verified_recovery")
            rec_o128 = sum(1 for c in o128_cells if c["outcome"] == "verified_recovery")
            med_o4 = stats.median([c["parent_wall_seconds"] for c in o4_cells])
            med_o128 = stats.median([c["parent_wall_seconds"] for c in o128_cells])
            observation_impact.append(
                {
                    "cipher": cipher,
                    "rounds": r,
                    "assumed_keys": m,
                    "residual_depth": r - m,
                    "o4_recovery": f"{rec_o4}/8",
                    "o128_recovery": f"{rec_o128}/8",
                    "o4_median_seconds": med_o4,
                    "o128_median_seconds": med_o128,
                }
            )

    analysis = {
        "classification": "OPENED_DEMO_DIAGNOSTIC",
        "ledger_sha256": sha256_file(results_file),
        "dataset_sha256": sha256_file(dataset) if dataset else None,
        "analysis_source_sha256": sha256_file(Path(__file__)),
        "summary": {
            "total_cells": total_cells,
            "total_verified_recovery": total_recovered,
            "total_parent_watchdog_timeout": total_timeouts,
            "total_other_outcomes": total_other,
            "recovery_rate_overall": total_recovered / total_cells,
            "sum_parent_seconds": sum(c["parent_wall_seconds"] for c in cells),
            "event_span_seconds": (
                datetime.fromisoformat(data["events"][-1]["timestamp"])
                - datetime.fromisoformat(data["events"][0]["timestamp"])
            ).total_seconds()
            if data.get("events")
            else None,
        },
        "grid_breakdown": table_rows,
        "paired_contrasts": paired_contrasts,
        "observation_impact": observation_impact,
        "interpretation_limits": [
            "Eight opened demo keys per cipher; not a population or universal law",
            "Conditioned trailing keys are oracle information",
            "Timeout solve/formula metrics are unavailable, not measured zero",
            "PAR-2 is a post-run descriptive summary under the common 300-second cutoff",
            "O4 precedes O128 in every key block; observation differences are chronology-confounded",
            "Missing timeout receipts and continuous runtime telemetry limit execution certification",
            "No carry/AND mechanism or unconditional tractability is established",
        ],
    }

    # Save analysis JSON
    atomic_json(output, analysis)

    return analysis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS_FILE)
    parser.add_argument("--output", type=Path, default=ANALYSIS_OUTPUT)
    parser.add_argument("--dataset", type=Path)
    args = parser.parse_args()
    res = analyze_map(args.results, args.output, args.dataset)
    print("=" * 115)
    print("CROSS-CIPHER RESIDUAL-DEPTH EMPIRICAL MAP: FULL 192-CELL BREAKDOWN")
    print("=" * 115)
    print(
        f"{'Cipher':10s} | {'(R, m)':8s} | {'d':3s} | {'O':4s} | {'Recovered':9s} | {'Med Solve(s)':12s} | {'Med Parent(s)':13s} | {'Variables':9s} | {'CNF Cls':8s} | {'XOR Cls':8s}"
    )
    print("-" * 115)
    for r in res["grid_breakdown"]:
        c_short = "SPECK" if "speck" in r["cipher"] else "SIMON"
        rm = f"({r['rounds']}, {r['assumed_keys']})"
        rec_str = f"{r['verified_recovery_count']}/{r['n_instances']} ({r['verified_recovery_rate'] * 100:4.1f}%)"
        med_s = (
            f"{r['median_solve_seconds']:.2f}"
            if r["median_solve_seconds"] is not None
            else "unavailable"
        )
        print(
            f"{c_short:10s} | {rm:8s} | {r['nominal_residual_depth']:3d} | {r['observations']:4d} | {rec_str:13s} | {med_s:12s} | {r['median_parent_seconds']:13.2f} | {str(r['variables_count']):9s} | {str(r['cnf_clauses_count']):8s} | {str(r['xor_clauses_count']):8s}"
        )

    print("\n" + "=" * 115)
    print("OBSERVATION IMPACT (O=4 vs O=128)")
    print("=" * 115)
    for oi in res["observation_impact"]:
        c_short = "SPECK" if "speck" in oi["cipher"] else "SIMON"
        print(
            f"{c_short:8s} ({oi['rounds']},{oi['assumed_keys']}) d={oi['residual_depth']} | O=4: {oi['o4_recovery']:4s} (med {oi['o4_median_seconds']:6.2f}s) -> O=128: {oi['o128_recovery']:4s} (med {oi['o128_median_seconds']:6.2f}s)"
        )
