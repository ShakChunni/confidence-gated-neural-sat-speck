"""D-065 item 1: brute-force completion in the queue-cost rule (model-based, not a campaign).

For every Fig. 8 cell, replace the measured SAT rejection cost r and correct-solve
time s with native brute-force costs at the same number of unknown schedule bits:
r_BF = full-space enumeration time, s_BF = half of it (uniform true guess).
Report both affordable depths and the per-candidate speed ratio, in seconds and
in D-058 reference-encryption equivalents. Brute-force costs come from
scripts/bruteforce_completion.py benchmark output; SAT costs from the Fig. 8 CSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "figures/fig8_affordable_depth.csv"
RATE = {"speck32_64": 68.247e6, "simon32_64": 29.693e6}
BUDGET = 300.0
# Fig. 8 row -> (rounds, fixed trailing words m) of the completion problem.
SHAPE = {
    "SIMON_primary_classical": (10, 2), "SIMON_primary_hybrid": (10, 2), "SIMON_primary_neural": (10, 2),
    "SPECK_primary_hybrid": (7, 2), "SPECK_primary_screened": (7, 2),
    "N32_SPECK_1w_on": (7, 1), "N32_SPECK_1w_off": (7, 1), "N32_SPECK_2w_on": (7, 2), "N32_SPECK_2w_off": (7, 2),
    "N32_SIMON_1w_on": (10, 1), "N32_SIMON_1w_off": (10, 1), "N32_SIMON_2w_on": (10, 2), "N32_SIMON_2w_off": (10, 2),
}
# Eight-round SPECK, three words (D-063/D-064 R8-G, 64 opened keys): SAT costs from the ledgers.
EIGHT = [REPO / "docs/paper/evidence/d063_campaign_20261011/ledger_k00-31.jsonl",
         REPO / "docs/paper/evidence/d064_campaign_20261011/ledger_k00-31.jsonl"]


def lg(x: float) -> float | None:
    return round(math.log2(x), 2) if x and x > 0 and math.isfinite(x) else None


def kstar(P: float, r: float, s: float) -> float:
    return max(0.0, (BUDGET - P - s) / r) if math.isfinite(s) else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    bench = json.loads(args.benchmark.read_text())
    bf = {(c["cipher"], c["rounds"], c["m"]): c["rejection"] for c in bench["configs"]}
    rows = []

    def add(row_id, cipher, rounds, m, P, r, s, med_pos, queue, recovered, basis):
        rej = bf[(cipher.split("32")[0], rounds, m)]
        r_bf = rej["full_space_seconds"]
        k_sat, k_bf = kstar(P, r, s), kstar(P, r_bf, 0.5 * r_bf)
        rows.append({
            "row_id": row_id, "cipher": cipher, "rounds": rounds, "m": m, "unknown_bits": 16 * (4 - m),
            "P_s": round(P, 2), "r_sat_s": round(r, 4), "s_sat_s": round(s, 4) if math.isfinite(s) else None,
            "r_bf_s": r_bf, "r_bf_label": rej["full_space_label"],
            "log2_enc": {"P": lg(P * RATE[cipher]), "r_sat": lg(r * RATE[cipher]), "r_bf": lg(r_bf * RATE[cipher])},
            "bf_over_sat_per_candidate": round(r_bf / r, 3), "log2_bf_over_sat": lg(r_bf / r),
            "k_star_sat": round(k_sat, 2), "k_star_bf": round(k_bf, 4),
            "median_true_position": med_pos, "queue_length": queue, "observed_recovered": recovered,
            "rule_prefers": "neither (k* = 0)" if max(k_sat, k_bf) == 0 else ("SAT" if k_sat > k_bf else "BF"), "basis": basis,
        })

    for r in csv.DictReader(CSV.open()):
        rounds, m = SHAPE[r["row_id"]]
        s = float(r["solve_true_km_median_s"]) if r["solve_true_km_median_s"] not in ("", "inf") else math.inf
        add(r["row_id"], r["cipher"], rounds, m, float(r["prep_s"]), float(r["reject_s"]), s,
            r["median_true_position"], r["queue_length"], r["recovered_label"], "fig8_affordable_depth.csv")

    cells = [c for p in EIGHT for c in map(json.loads, p.open()) if c.get("arm") == "R8-G"]
    wrong = [c["wrong_mean_s"] for c in cells if c["wrong_mean_s"]]
    solve = sorted(c["correct_solve_s"] for c in cells if c["outcome"] == "recovered")
    prep = sorted(c["prep_s"] for c in cells)
    pos = sorted(c["true_queue_position"] for c in cells if c["true_queue_position"] is not None)
    add("SPECK_8R_net7", "speck32_64", 8, 3, prep[len(prep) // 2], sum(wrong) / len(wrong), solve[len(solve) // 2],
        pos[len(pos) // 2], 512, f"{sum(c['outcome'] == 'recovered' for c in cells)}/{len(cells)}", "D-063 + D-064 R8-G ledgers")

    out = {"schema": "d065_bruteforce_crossover_v1", "budget_s": BUDGET, "rates_per_s": RATE,
           "benchmark": str(args.benchmark), "benchmark_sha256": hashlib.sha256(args.benchmark.read_bytes()).hexdigest(),
           "note": "model-based substitution into the queue-cost rule; BF s = half the full-space time", "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    for x in rows:
        print(f"{x['row_id']:26s} {x['unknown_bits']:2d}b  r_sat={x['r_sat_s']:9.3f}s r_bf={x['r_bf_s']:12.4g}s "
              f"ratio=2^{x['log2_bf_over_sat']}  k*sat={x['k_star_sat']:8.1f} k*bf={x['k_star_bf']:10.3f} -> {x['rule_prefers']}")


if __name__ == "__main__":
    main()
