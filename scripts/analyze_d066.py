#!/usr/bin/env python3
"""Pre-registered analysis of the D-066 eight-round replication (docs/paper/28_D066_PROTOCOL.md).

Q1 (primary): exact two-sided McNemar test on paired recovery at 300 s.
Q2: exact two-sided sign test on paired Stage-1 true-word positions.
Q3: exact two-sided sign test on paired PAR-2 (600 s on failure).
Q2 and Q3 are Holm-adjusted as a family of two. Descriptive: counts and paired
McNemar at 150, 200 and 300 s, Wilson intervals, Stage-1 positions, queueing,
predicted ranges and a pooled 96-key estimate with D-064 (secondary).

Written while the campaign ran, before any outcome was read; its SHA-256 is
recorded with an AWST time in docs/paper/29_D066_ANALYSIS_SCRIPT.sha256.

Usage: analyze_d066.py D066_LEDGER D064_LEDGER
"""

import json
import math
import statistics as st
import sys

ARMS = ("R8-G", "R8-F")
N = 64
LIMITS = (150, 200, 300)
PREDICTED = {"R8-G": (47, 37, 56), "R8-F": (36, 25, 47)}  # point, 90% low, 90% high
SHARED_ASSETS = ("net7", "t6", "t5", "tfull", "cms")


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


def ok(r, limit=300):
    """Recovered within the limit. At 300 s the runner's own outcome is the definition."""
    return r["outcome"] == "recovered" and (limit >= 300 or r["total_s"] <= limit)


def load(path):
    lines = [json.loads(line) for line in open(path)]
    header = lines[0]
    assert header["type"] == "header"
    cells = [r for r in lines[1:] if "arm" in r]
    return header, cells


def mcnemar(G, F, ks, limit=300):
    g = sum(ok(G[k], limit) and not ok(F[k], limit) for k in ks)
    f = sum(ok(F[k], limit) and not ok(G[k], limit) for k in ks)
    return {"net_only": g, "table_only": f,
            "both": sum(ok(G[k], limit) and ok(F[k], limit) for k in ks),
            "neither": sum(not ok(G[k], limit) and not ok(F[k], limit) for k in ks),
            "p": binom_two_sided(g, g + f)}


def main():
    header, cells = load(sys.argv[1])
    h64, cells64 = load(sys.argv[2])

    # Execution checks fixed by the protocol: one record per key and arm, and
    # infrastructure failures (no record, or total_s > 310) listed for one rerun.
    seen = {}
    for r in cells:
        seen.setdefault((r["arm"], r["key_index"]), []).append(r)
    duplicates = sorted(k for k, v in seen.items() if len(v) > 1)
    missing = sorted((a, k) for a in ARMS for k in range(N) if (a, k) not in seen)
    overrun = sorted((r["arm"], r["key_index"], r["total_s"]) for r in cells if r["total_s"] > 310)
    out = {
        "ledger_header": {k: header[k] for k in ("seed", "role", "keys", "git_commit", "runner_dirty",
                                                 "script_sha256", "started_local")},
        "execution": {"cells": len(cells), "duplicates": duplicates, "missing": missing,
                      "total_s_over_310": overrun,
                      "recovered_after_300s": sorted((r["arm"], r["key_index"], r["total_s"]) for r in cells
                                                     if r["outcome"] == "recovered" and r["total_s"] > 300),
                      "assets_match_d064": {a: header["assets"].get(a) == h64["assets"].get(a)
                                            for a in SHARED_ASSETS}},
        "arms": {},
    }
    if missing or duplicates:
        out["status"] = "incomplete: analysis not run"
        print(json.dumps(out, indent=1))
        return

    arms = {a: {r["key_index"]: r for r in cells if r["arm"] == a} for a in ARMS}
    for a, d in arms.items():
        rs = [d[k] for k in range(N)]
        rec = [r for r in rs if ok(r)]
        pos = [r["stage1_true_position"] for r in rs]
        queued = [r for r in rs if r["true_queue_position"] is not None]
        point, lo, hi = PREDICTED[a]
        out["arms"][a] = {
            "recovered": len(rec), "wilson95": wilson(len(rec), N),
            "recovered_by_limit": {str(t): sum(ok(r, t) for r in rs) for t in LIMITS},
            "outcomes": {o: sum(r["outcome"] == o for r in rs) for o in sorted({r["outcome"] for r in rs})},
            "mean_par2_s": round(st.mean(par2(r) for r in rs), 1),
            "stage1_median_position": st.median(pos), "stage1_top1": sum(p == 1 for p in pos),
            "stage1_top8": sum(p <= 8 for p in pos), "stage1_top32": sum(p <= 32 for p in pos),
            "stage1_max_position": max(pos),
            "true_chain_queued": len(queued),
            "true_queue_position_median": st.median(r["true_queue_position"] for r in queued) if queued else None,
            "recovered_when_queued": sum(ok(r) for r in queued),
            "median_total_s_recovered": round(st.median(r["total_s"] for r in rec), 1) if rec else None,
            "verification_rejections": sum(r["verification_rejections"] for r in rs),
            "prediction": {"point": point, "range90": [lo, hi],
                           "result": "inside" if lo <= len(rec) <= hi else ("above" if len(rec) > hi else "below")},
        }

    G, F = arms["R8-G"], arms["R8-F"]
    ks = list(range(N))
    q1 = mcnemar(G, F, ks)
    if q1["p"] >= 0.05:
        verdict = "no detectable recovery difference at N = 64"
    elif q1["net_only"] > q1["table_only"]:
        verdict = "network recovers more"
    else:
        verdict = "table recovers more"
    g2 = sum(G[k]["stage1_true_position"] < F[k]["stage1_true_position"] for k in ks)
    f2 = sum(G[k]["stage1_true_position"] > F[k]["stage1_true_position"] for k in ks)
    g3 = sum(par2(G[k]) < par2(F[k]) for k in ks)
    f3 = sum(par2(G[k]) > par2(F[k]) for k in ks)
    p2, p3 = binom_two_sided(g2, g2 + f2), binom_two_sided(g3, g3 + f3)
    first, second = sorted([("Q2", p2), ("Q3", p3)], key=lambda t: t[1])
    holm = {first[0]: min(1.0, 2 * first[1])}
    holm[second[0]] = max(holm[first[0]], min(1.0, second[1]))
    out["tests"] = {
        "Q1_recovery_mcnemar_300s": {**q1, "verdict": verdict},
        "Q2_stage1_sign": {"net_better": g2, "table_better": f2, "ties": N - g2 - f2, "p": p2, "p_holm": holm["Q2"]},
        "Q3_par2_sign": {"net_cheaper": g3, "table_cheaper": f3, "ties": N - g3 - f3, "p": p3, "p_holm": holm["Q3"]},
    }
    out["descriptive_by_limit"] = {str(t): mcnemar(G, F, ks, t) for t in LIMITS}

    # Secondary, descriptive: pool with the 32 D-064 keys (different seed, so keys are distinct).
    a64 = {a: {r["key_index"]: r for r in cells64 if r["arm"] == a} for a in ARMS}
    assert all(len(d) == 32 for d in a64.values())
    PG = {("d066", k): G[k] for k in ks} | {("d064", k): a64["R8-G"][k] for k in range(32)}
    PF = {("d066", k): F[k] for k in ks} | {("d064", k): a64["R8-F"][k] for k in range(32)}
    pooled = mcnemar(PG, PF, sorted(PG))
    out["pooled_96_secondary"] = {**pooled, "net_recovered": sum(ok(r) for r in PG.values()),
                                  "table_recovered": sum(ok(r) for r in PF.values()), "keys": len(PG)}
    out["per_key"] = [{"key_index": k, "net_outcome": G[k]["outcome"], "table_outcome": F[k]["outcome"],
                       "net_total_s": round(G[k]["total_s"], 1), "table_total_s": round(F[k]["total_s"], 1),
                       "net_stage1_position": G[k]["stage1_true_position"],
                       "table_stage1_position": F[k]["stage1_true_position"]} for k in ks]
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
