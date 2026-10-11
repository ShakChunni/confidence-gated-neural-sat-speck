#!/usr/bin/env python3
"""Convert measured pipeline costs to D-058 reference-encryption equivalents (D-065 item 2).

ENGINEERING. Each time t in seconds becomes t * E full-round encryptions, with E the
D-058 single-thread median rate (SPECK32/64 68.247e6/s, SIMON32/64 29.693e6/s).
The conversion is one constant per cipher: it changes units, not ratios, so the
affordable depth k* = (300 - P - s) / r is unchanged, and the result is not a
hardware-independent complexity (D-058 scope). Inputs: the audited affordable-depth
table (figures/fig8_affordable_depth.csv) and the D-063/D-064 ledgers.
"""

import csv
import json
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RATE = {"speck32_64": 68.247e6, "simon32_64": 29.693e6}  # docs/paper/06_ENCRYPTION_COST_CALIBRATION_REPORT.md


def log2_enc(seconds, cipher):
    return None if seconds is None or seconds <= 0 else round(math.log2(seconds * RATE[cipher]), 2)


def ledger_rows(path, arms, label):
    cells = [json.loads(l) for l in open(path)][1:]
    out = []
    for a in arms:
        g = [r for r in cells if r["arm"] == a]
        wrong = [r["wrong_mean_s"] for r in g if r["wrong_mean_s"]]
        solve = [r["correct_solve_s"] for r in g if r.get("correct_solve_s")]
        P, r, s = st.median(r["prep_s"] for r in g), st.mean(wrong), st.median(solve) if solve else None
        out.append({"row_id": f"{label}_{a}", "cipher": "speck32_64", "prep_s": P, "reject_s": r, "solve_s": s,
                    "k_star": (300 - P - (s or 0)) / r, "recovered": f"{sum(x['outcome'] == 'recovered' for x in g)}/{len(g)}"})
    return out


def main():
    rows = []
    for r in csv.DictReader(open(ROOT / "figures/fig8_affordable_depth.csv")):
        f = lambda k: float(r[k]) if r[k] not in ("", "None", "nan", "inf") else None
        rows.append({"row_id": r["row_id"], "cipher": r["cipher"], "prep_s": f("prep_s"), "reject_s": f("reject_s"),
                     "solve_s": f("solve_true_km_median_s"), "k_star": f("k_star"), "recovered": r["recovered_label"]})
    rows += ledger_rows(ROOT / "docs/paper/evidence/d063_campaign_20261011/ledger_k00-31.jsonl", ["R7-G", "R7-C", "R7-B1", "R8-G", "R8-C"], "D063")
    rows += ledger_rows(ROOT / "docs/paper/evidence/d064_campaign_20261011/ledger_k00-31.jsonl", ["R8-G", "R8-F"], "D064")
    for r in rows:
        for k in ("prep", "reject", "solve"):
            r[f"{k}_log2_enc"] = log2_enc(r[f"{k}_s"], r["cipher"])
    out = {"rates_per_s": RATE, "budget_log2_enc": {c: round(math.log2(300 * e), 2) for c, e in RATE.items()},
           "note": "unit conversion only; k* unchanged; not hardware independent", "rows": rows}
    dest = Path(sys.argv[1])
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print(f"{'row':28s} {'P s':>8s} {'log2':>6s} {'r s':>8s} {'log2':>6s} {'s s':>8s} {'log2':>6s} {'k*':>7s} rec")
    for r in rows:
        fmt = lambda v, n=8: f"{v:{n}.2f}" if isinstance(v, float) else f"{'-':>{n}s}"
        print(f"{r['row_id']:28s} {fmt(r['prep_s'])} {fmt(r['prep_log2_enc'],6)} {fmt(r['reject_s'])} {fmt(r['reject_log2_enc'],6)} "
              f"{fmt(r['solve_s'])} {fmt(r['solve_log2_enc'],6)} {fmt(r['k_star'],7)} {r['recovered']}")


if __name__ == "__main__":
    main()
