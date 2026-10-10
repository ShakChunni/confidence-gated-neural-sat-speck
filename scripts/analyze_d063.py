#!/usr/bin/env python3
"""Analyse the D-063 seven-/eight-round SPECK campaign as pre-registered.

docs/paper/12_D063_PROTOCOL.md: exact two-sided McNemar tests, Holm over the
two Q3 contrasts, Wilson 95% intervals, PAR-2 (600 s per failure) with paired
sign tests within a round count, Stage-1 rank and queue summaries.

Usage: analyze_d063.py LEDGER [LEDGER ...] > summary.json
"""

from __future__ import annotations

import json
import math
import statistics as st
import sys
from math import comb

ARMS = ["R7-B1", "R7-G", "R7-C", "R8-G", "R8-C"]
PREDICTED = {"R8-G": (23, 16, 28), "R8-C": (0, 0, 2), "R7-G": (31, 29, 32), "R7-C": (30, 27, 32), "R7-B1": (21, 15, 27)}


def exact_two_sided(b: int, c: int) -> float:
    """Exact two-sided binomial p for b vs c discordant pairs (McNemar / sign test)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def main() -> None:
    rows = []
    for path in sys.argv[1:]:
        rows += [r for r in map(json.loads, open(path)) if r.get("type") != "header"]
    cell = {(r["key_index"], r["arm"]): r for r in rows}
    keys = sorted({r["key_index"] for r in rows})
    assert len(cell) == len(rows) == len(keys) * len(ARMS), "missing or duplicate cells"
    ok = {a: {k: cell[k, a]["outcome"] == "recovered" for k in keys} for a in ARMS}
    par2 = {a: {k: cell[k, a]["total_s"] if ok[a][k] else 600.0 for k in keys} for a in ARMS}
    n = len(keys)

    out: dict = {"n_keys": n, "arms": {}}
    for a in ARMS:
        rs = [cell[k, a] for k in keys]
        rec = sum(ok[a].values())
        lo, hi = wilson(rec, n)
        ranks = [r["stage1_true_rank"] for r in rs]
        pos = [r["true_queue_position"] for r in rs]
        wrong = [r["wrong_median_s"] for r in rs if r["wrong_median_s"] is not None]
        point, plo, phi = PREDICTED[a]
        out["arms"][a] = {
            "recovered": rec,
            "wilson95": [round(lo, 3), round(hi, 3)],
            "predicted": {"point": point, "range": [plo, phi], "verdict": "inside" if plo <= rec <= phi else ("above" if rec > phi else "below")},
            "par2_mean_s": round(st.mean(par2[a].values()), 1),
            "median_total_s_recovered": round(st.median([r["total_s"] for r in rs if r["outcome"] == "recovered"]), 1) if rec else None,
            "outcomes": {o: sum(r["outcome"] == o for r in rs) for o in sorted({r["outcome"] for r in rs})},
            "stage1_in_top32": sum(x is not None for x in ranks),
            "stage1_top1": sum(x == 1 for x in ranks),
            "stage1_top8": sum(x is not None and x <= 8 for x in ranks),
            "stage1_median_rank_when_in_top32": st.median([x for x in ranks if x is not None]) if any(x is not None for x in ranks) else None,
            "true_chain_in_queue": sum(x is not None for x in pos),
            "in_queue_not_recovered": sum(pos[i] is not None and not ok[a][k] for i, k in enumerate(keys)),
            "median_stage1_s": round(st.median(r["stage1_s"] for r in rs), 2),
            "median_prep_s": round(st.median(r["prep_s"] for r in rs), 1),
            "median_wrong_candidate_s": round(st.median(wrong), 3) if wrong else None,
            "verification_rejections": sum(r["verification_rejections"] for r in rs),
        }

    def mcnemar(x: str, y: str) -> dict:
        b = sum(ok[x][k] and not ok[y][k] for k in keys)
        c = sum(ok[y][k] and not ok[x][k] for k in keys)
        return {"x": x, "y": y, "x_only": b, "y_only": c, "p_exact_two_sided": exact_two_sided(b, c)}

    def sign_par2(x: str, y: str) -> dict:
        b = sum(par2[x][k] < par2[y][k] for k in keys)
        c = sum(par2[y][k] < par2[x][k] for k in keys)
        return {"x": x, "y": y, "x_cheaper": b, "y_cheaper": c, "ties": n - b - c, "p_exact_two_sided": exact_two_sided(b, c)}

    q1 = mcnemar("R8-G", "R8-C")
    rg = out["arms"]["R8-G"]["recovered"]
    q1["verdict"] = ("pass" if rg >= 16 else "weak pass") if q1["p_exact_two_sided"] < 0.05 and q1["x_only"] > q1["y_only"] else "fail"
    q2 = mcnemar("R7-G", "R7-C")
    q2["noninferior_R7G_ge_R7C_minus_2"] = out["arms"]["R7-G"]["recovered"] >= out["arms"]["R7-C"]["recovered"] - 2
    q3 = [mcnemar("R7-G", "R7-B1"), mcnemar("R7-C", "R7-B1")]
    order = sorted(range(2), key=lambda i: q3[i]["p_exact_two_sided"])
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (2 - rank) * q3[i]["p_exact_two_sided"]))
        q3[i]["p_holm"] = running
    out["Q1_primary"] = q1
    out["Q2_secondary"] = q2
    out["Q3_holm"] = q3
    out["par2_sign_tests"] = [sign_par2("R8-G", "R8-C"), sign_par2("R7-G", "R7-C"), sign_par2("R7-G", "R7-B1"), sign_par2("R7-C", "R7-B1")]
    out["per_key"] = {k: {a: [cell[k, a]["outcome"], cell[k, a]["stage1_true_rank"], round(cell[k, a]["total_s"], 1)] for a in ARMS} for k in keys}
    json.dump(out, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
