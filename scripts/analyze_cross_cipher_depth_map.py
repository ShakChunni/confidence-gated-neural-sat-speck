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

import json
import statistics as stats
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE = ROOT / "results" / "phase11_scaling_diagnostics" / "cross_cipher_residual_depth_map.json"
ANALYSIS_OUTPUT = ROOT / "results" / "phase11_scaling_diagnostics" / "cross_cipher_residual_depth_analysis.json"


def analyze_map() -> dict[str, Any]:
    with open(RESULTS_FILE, encoding="utf-8") as f:
        data = json.load(f)

    cells = list(data["cells"].values())
    assert len(cells) == 192, f"Expected 192 cells, found {len(cells)}"

    # 1. Overall counts
    total_cells = len(cells)
    total_recovered = sum(1 for c in cells if c["outcome"] == "verified_recovery")
    total_timeouts = sum(1 for c in cells if c["outcome"] == "parent_watchdog_timeout")
    total_other = total_cells - total_recovered - total_timeouts

    # 2. Group by cell configuration
    # (cipher, rounds, assumed_keys, nominal_depth, observations)
    groups: dict[tuple[str, int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
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
        timeouts = sum(1 for c in cell_list if c["outcome"] == "parent_watchdog_timeout")

        p_times = [c["parent_wall_seconds"] for c in cell_list]
        s_times = [c["solve_wall_seconds"] for c in cell_list]

        med_p = stats.median(p_times)
        med_s = stats.median(s_times)
        q25_p = stats.quantiles(p_times, n=4)[0] if len(p_times) >= 4 else min(p_times)
        q75_p = stats.quantiles(p_times, n=4)[2] if len(p_times) >= 4 else max(p_times)

        v_cnt = cell_list[0]["variables_count"]
        cnf_cnt = cell_list[0]["cnf_clauses_count"]
        xor_cnt = cell_list[0]["xor_clauses_count"]

        table_rows.append({
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
            "variables_count": v_cnt,
            "cnf_clauses_count": cnf_cnt,
            "xor_clauses_count": xor_cnt,
        })

    # 3. Controlled Paired Contrasts at Matched Nominal Depth d
    # For each key k (0..7), compare solve times across same d
    paired_contrasts: dict[str, Any] = {}

    def get_cell(cipher: str, r: int, m: int, o: int, k: int) -> dict[str, Any]:
        return next(
            c for c in cells
            if c["cipher"] == cipher and c["rounds"] == r and c["assumed_keys_count"] == m
            and c["observations"] == o and c["instance_index"] == k
        )

    # Contrast 1: SPECK d=6 at O=4: (6,0) vs (7,1) vs (8,2)
    speck_d6_o4 = []
    for k in range(8):
        c60 = get_cell("speck32_64", 6, 0, 4, k)
        c71 = get_cell("speck32_64", 7, 1, 4, k)
        c82 = get_cell("speck32_64", 8, 2, 4, k)
        speck_d6_o4.append({
            "key": k,
            "(6,0)": {"time": c60["parent_wall_seconds"], "outcome": c60["outcome"]},
            "(7,1)": {"time": c71["parent_wall_seconds"], "outcome": c71["outcome"]},
            "(8,2)": {"time": c82["parent_wall_seconds"], "outcome": c82["outcome"]},
        })
    paired_contrasts["speck_d6_o4"] = speck_d6_o4

    # Contrast 2: SPECK d=5 at O=4: (6,1) vs (7,2) vs (8,3)
    speck_d5_o4 = []
    for k in range(8):
        c61 = get_cell("speck32_64", 6, 1, 4, k)
        c72 = get_cell("speck32_64", 7, 2, 4, k)
        c83 = get_cell("speck32_64", 8, 3, 4, k)
        speck_d5_o4.append({
            "key": k,
            "(6,1)": {"time": c61["parent_wall_seconds"], "outcome": c61["outcome"]},
            "(7,2)": {"time": c72["parent_wall_seconds"], "outcome": c72["outcome"]},
            "(8,3)": {"time": c83["parent_wall_seconds"], "outcome": c83["outcome"]},
        })
    paired_contrasts["speck_d5_o4"] = speck_d5_o4

    # Contrast 3: SIMON d=9 at O=4: (9,0) vs (10,1) vs (11,2)
    simon_d9_o4 = []
    for k in range(8):
        c90 = get_cell("simon32_64", 9, 0, 4, k)
        c101 = get_cell("simon32_64", 10, 1, 4, k)
        c112 = get_cell("simon32_64", 11, 2, 4, k)
        simon_d9_o4.append({
            "key": k,
            "(9,0)": {"time": c90["parent_wall_seconds"], "outcome": c90["outcome"]},
            "(10,1)": {"time": c101["parent_wall_seconds"], "outcome": c101["outcome"]},
            "(11,2)": {"time": c112["parent_wall_seconds"], "outcome": c112["outcome"]},
        })
    paired_contrasts["simon_d9_o4"] = simon_d9_o4

    # Contrast 4: SIMON d=8 at O=4: (9,1) vs (10,2) vs (11,3)
    simon_d8_o4 = []
    for k in range(8):
        c91 = get_cell("simon32_64", 9, 1, 4, k)
        c102 = get_cell("simon32_64", 10, 2, 4, k)
        c113 = get_cell("simon32_64", 11, 3, 4, k)
        simon_d8_o4.append({
            "key": k,
            "(9,1)": {"time": c91["parent_wall_seconds"], "outcome": c91["outcome"]},
            "(10,2)": {"time": c102["parent_wall_seconds"], "outcome": c102["outcome"]},
            "(11,3)": {"time": c113["parent_wall_seconds"], "outcome": c113["outcome"]},
        })
    paired_contrasts["simon_d8_o4"] = simon_d8_o4

    # Contrast 5: O=4 vs O=128 Sensitivity
    observation_impact = []
    for cipher in ("speck32_64", "simon32_64"):
        r_points = [(6, 0), (6, 1), (7, 1), (7, 2), (8, 2), (8, 3)] if cipher == "speck32_64" else [(9, 0), (9, 1), (10, 1), (10, 2), (11, 2), (11, 3)]
        for r, m in r_points:
            o4_cells = [c for c in cells if c["cipher"] == cipher and c["rounds"] == r and c["assumed_keys_count"] == m and c["observations"] == 4]
            o128_cells = [c for c in cells if c["cipher"] == cipher and c["rounds"] == r and c["assumed_keys_count"] == m and c["observations"] == 128]
            rec_o4 = sum(1 for c in o4_cells if c["outcome"] == "verified_recovery")
            rec_o128 = sum(1 for c in o128_cells if c["outcome"] == "verified_recovery")
            med_o4 = stats.median([c["parent_wall_seconds"] for c in o4_cells])
            med_o128 = stats.median([c["parent_wall_seconds"] for c in o128_cells])
            observation_impact.append({
                "cipher": cipher,
                "rounds": r,
                "assumed_keys": m,
                "residual_depth": r - m,
                "o4_recovery": f"{rec_o4}/8",
                "o128_recovery": f"{rec_o128}/8",
                "o4_median_seconds": med_o4,
                "o128_median_seconds": med_o128,
            })

    analysis = {
        "summary": {
            "total_cells": total_cells,
            "total_verified_recovery": total_recovered,
            "total_parent_watchdog_timeout": total_timeouts,
            "total_other_outcomes": total_other,
            "recovery_rate_overall": total_recovered / total_cells,
        },
        "grid_breakdown": table_rows,
        "paired_contrasts": paired_contrasts,
        "observation_impact": observation_impact,
    }

    # Save analysis JSON
    with open(ANALYSIS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    return analysis


if __name__ == "__main__":
    res = analyze_map()
    print("=" * 115)
    print("CROSS-CIPHER RESIDUAL-DEPTH EMPIRICAL MAP: FULL 192-CELL BREAKDOWN")
    print("=" * 115)
    print(f"{'Cipher':10s} | {'(R, m)':8s} | {'d':3s} | {'O':4s} | {'Recovered':9s} | {'Med Solve(s)':12s} | {'Med Parent(s)':13s} | {'Variables':9s} | {'CNF Cls':8s} | {'XOR Cls':8s}")
    print("-" * 115)
    for r in res["grid_breakdown"]:
        c_short = "SPECK" if "speck" in r["cipher"] else "SIMON"
        rm = f"({r['rounds']}, {r['assumed_keys']})"
        rec_str = f"{r['verified_recovery_count']}/{r['n_instances']} ({r['verified_recovery_rate']*100:4.1f}%)"
        print(f"{c_short:10s} | {rm:8s} | {r['nominal_residual_depth']:3d} | {r['observations']:4d} | {rec_str:13s} | {r['median_solve_seconds']:12.2f} | {r['median_parent_seconds']:13.2f} | {r['variables_count']:9d} | {r['cnf_clauses_count']:8d} | {r['xor_clauses_count']:8d}")

    print("\n" + "=" * 115)
    print("OBSERVATION IMPACT (O=4 vs O=128)")
    print("=" * 115)
    for oi in res["observation_impact"]:
        c_short = "SPECK" if "speck" in oi["cipher"] else "SIMON"
        print(f"{c_short:8s} ({oi['rounds']},{oi['assumed_keys']}) d={oi['residual_depth']} | O=4: {oi['o4_recovery']:4s} (med {oi['o4_median_seconds']:6.2f}s) -> O=128: {oi['o128_recovery']:4s} (med {oi['o128_median_seconds']:6.2f}s)")
