#!/usr/bin/env python3
"""Pre-registered analysis of the D-064 eight-round campaign (docs/paper/18_D064_PROTOCOL.md).

Q1 (primary): exact two-sided sign test on paired Stage-1 true-word positions.
Q2: exact two-sided McNemar test on paired recovery. Q3: exact two-sided sign
test on paired PAR-2. Q2 and Q3 are Holm-adjusted as a family of two.
"""

import json
import math
import statistics as st
import sys


def binom_two_sided(k, n):
    if n == 0:
        return 1.0
    t = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(t + 1)) / 2 ** n)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - s) / d, 4), round((c + s) / d, 4)]


def par2(r):
    return r["total_s"] if r["outcome"] == "recovered" else 600.0


def main():
    lines = [json.loads(l) for l in open(sys.argv[1])]
    header, cells = lines[0], lines[1:]
    arms = {a: {r["key_index"]: r for r in cells if r["arm"] == a} for a in ("R8-G", "R8-F")}
    assert all(len(d) == 32 for d in arms.values())
    out = {"ledger_header": {k: header[k] for k in ("seed", "git_commit", "runner_dirty", "script_sha256", "started_local")}, "arms": {}}
    for a, d in arms.items():
        rs = list(d.values())
        rec = [r for r in rs if r["outcome"] == "recovered"]
        pos = [r["stage1_true_position"] for r in rs]
        queued = [r for r in rs if r["true_queue_position"] is not None]
        wrong = [r["wrong_mean_s"] for r in rs if r["wrong_mean_s"]]
        out["arms"][a] = {
            "recovered": len(rec), "wilson95": wilson(len(rec), 32),
            "outcomes": {o: sum(r["outcome"] == o for r in rs) for o in sorted({r["outcome"] for r in rs})},
            "stage1_top1": sum(p == 1 for p in pos), "stage1_top8": sum(p <= 8 for p in pos),
            "stage1_top32": sum(p <= 32 for p in pos), "stage1_median_position": st.median(pos), "stage1_max_position": max(pos),
            "true_chain_queued": len(queued), "recovered_when_queued": sum(r["outcome"] == "recovered" for r in queued),
            "median_total_s_recovered": round(st.median(r["total_s"] for r in rec), 1), "max_total_s": round(max(r["total_s"] for r in rs), 1),
            "mean_par2_s": round(st.mean(par2(r) for r in rs), 1),
            "median_stage1_s": round(st.median(r["stage1_s"] for r in rs), 2),
            "median_prep_s": round(st.median(r["prep_s"] for r in rs), 1),
            "median_wrong_median_s": round(st.median(r["wrong_median_s"] for r in rs if r["wrong_median_s"]), 3),
            "mean_wrong_mean_s": round(st.mean(wrong), 3),
            "verification_rejections": sum(r["verification_rejections"] for r in rs),
        }
        if a == "R8-F":
            out["arms"][a]["stage1_load_s_median_max"] = [round(st.median(r["stage1_load_s"] for r in rs), 2), round(max(r["stage1_load_s"] for r in rs), 2)]
    G, F = arms["R8-G"], arms["R8-F"]
    ks = sorted(G)
    g1 = sum(G[k]["stage1_true_position"] < F[k]["stage1_true_position"] for k in ks)
    f1 = sum(G[k]["stage1_true_position"] > F[k]["stage1_true_position"] for k in ks)
    g2 = sum(G[k]["outcome"] == "recovered" and F[k]["outcome"] != "recovered" for k in ks)
    f2 = sum(F[k]["outcome"] == "recovered" and G[k]["outcome"] != "recovered" for k in ks)
    g3 = sum(par2(G[k]) < par2(F[k]) for k in ks)
    f3 = sum(par2(G[k]) > par2(F[k]) for k in ks)
    p1, p2, p3 = binom_two_sided(g1, g1 + f1), binom_two_sided(g2, g2 + f2), binom_two_sided(g3, g3 + f3)
    lo, hi = sorted([("Q2", p2), ("Q3", p3)], key=lambda t: t[1])
    holm = {lo[0]: min(1.0, 2 * lo[1])}
    holm[hi[0]] = max(holm[lo[0]], min(1.0, hi[1]))
    out["tests"] = {
        "Q1_stage1_sign": {"net_better": g1, "table_better": f1, "ties": 32 - g1 - f1, "p": p1,
                           "verdict": "no detectable difference" if p1 >= 0.05 else ("network ranks better" if g1 > f1 else "table ranks better")},
        "Q2_recovery_mcnemar": {"net_only": g2, "table_only": f2, "both": sum(G[k]["outcome"] == F[k]["outcome"] == "recovered" for k in ks),
                                "neither": sum(G[k]["outcome"] != "recovered" and F[k]["outcome"] != "recovered" for k in ks),
                                "p": p2, "p_holm": holm["Q2"]},
        "Q3_par2_sign": {"net_cheaper": g3, "table_cheaper": f3, "ties": 32 - g3 - f3, "p": p3, "p_holm": holm["Q3"]},
    }
    out["per_key"] = [{"key_index": k, "net_stage1_position": G[k]["stage1_true_position"], "table_stage1_position": F[k]["stage1_true_position"],
                       "net_outcome": G[k]["outcome"], "table_outcome": F[k]["outcome"],
                       "net_queue_position": G[k]["true_queue_position"], "table_queue_position": F[k]["true_queue_position"]} for k in ks]
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
