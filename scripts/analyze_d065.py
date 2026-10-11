#!/usr/bin/env python3
"""Pre-registered analysis of the D-065 prescriptive campaign (docs/paper/22_PRESCRIPTIVE_PROTOCOL.md).

H1 (primary): per-key pooled exact one-sided sign test, SAT > BF at seven rounds.
    Secondary: per-ranker exact one-sided McNemar, Holm over three.
H2 (co-primary): eight rounds, (a) BF recovered >= SAT recovered - 1 and
    (b) exact one-sided sign test on paired PAR-2, BF < SAT.
H3: how many of the 8 arm counts fall inside their frozen 90% intervals (pass >= 7).
H4: whether L_B1 >= L_table >= L_net6, with L = SAT recovered - BF recovered.

Usage: analyze_d065.py PREDICTIONS_JSON OUT_JSON LEDGER [LEDGER ...]
"""

import json
import math
import statistics as st
import sys

RATE = 68.247e6  # D-058 single-thread full-round SPECK32/64 encryptions per second
PAIRS = {"R7-B1": "R7-B1-BF", "R7-G": "R7-G-BF", "R7-C": "R7-C-BF", "R8-G": "R8-G-BF"}
SEVEN = ("R7-B1", "R7-G", "R7-C")


def sign_one_sided(a, b):
    """P(X >= a), X ~ Bin(a + b, 1/2)."""
    n = a + b
    return 1.0 if n == 0 else sum(math.comb(n, k) for k in range(a, n + 1)) / 2 ** n


def holm(ps):
    order = sorted(ps, key=ps.get)
    out, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * ps[k]))
        out[k] = run
    return out


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - s) / d, 4), round((c + s) / d, 4)]


def par2(r):
    return r["total_s"] if r["outcome"] == "recovered" else 600.0


def log2e(seconds):
    return round(math.log2(seconds * RATE), 2) if seconds and seconds > 0 else None


def main():
    pred = json.load(open(sys.argv[1]))
    cells, headers = [], []
    for path in sys.argv[3:]:
        for line in open(path):
            r = json.loads(line)
            (headers if r.get("type") == "header" else cells).append(r)
    arms = {a: {} for p in PAIRS.items() for a in p}
    for r in cells:
        if r["arm"] in arms:
            assert r["key_index"] not in arms[r["arm"]], f"duplicate cell {r['arm']} {r['key_index']}"
            arms[r["arm"]][r["key_index"]] = r
    assert all(sorted(d) == list(range(32)) for d in arms.values()), {a: len(d) for a, d in arms.items()}
    assert all(h["seed"] == 701003 and h["role"] == "fresh" for h in headers)
    rec = {a: {k: d[k]["outcome"] == "recovered" for k in d} for a, d in arms.items()}
    out = {"ledgers": [{k: h[k] for k in ("seed", "first_key", "keys", "git_commit", "runner_dirty", "script_sha256", "started_local")}
                       for h in headers], "arms": {}}

    for a, d in arms.items():
        rs = [d[k] for k in range(32)]
        n = sum(rec[a].values())
        wrong = [r["wrong_mean_s"] for r in rs if r["wrong_mean_s"]]
        solve = [r["correct_solve_s"] for r in rs if r["outcome"] == "recovered"]
        lo, hi = pred["fresh_interval_90"][a]
        out["arms"][a] = {
            "recovered": n, "wilson95": wilson(n, 32),
            "outcomes": {o: sum(r["outcome"] == o for r in rs) for o in sorted({r["outcome"] for r in rs})},
            "interval90": [lo, hi], "vs_interval": "inside" if lo <= n <= hi else ("above" if n > hi else "below"),
            "true_chain_queued": sum(r["true_queue_position"] is not None for r in rs),
            "true_position_median": st.median(r["true_queue_position"] for r in rs if r["true_queue_position"] is not None),
            "mean_par2_s": round(st.mean(par2(r) for r in rs), 1),
            "median_prep_s": round(st.median(r["prep_s"] for r in rs), 2),
            "mean_wrong_s": round(st.mean(wrong), 4) if wrong else None,
            "median_correct_solve_s": round(st.median(solve), 4) if solve else None,
            "attempts_total": sum(r["attempts"] for r in rs),
            "verification_rejections": sum(r["verification_rejections"] for r in rs),
            "max_total_s": round(max(r["total_s"] for r in rs), 1),
            "log2_encryptions": {"prep": log2e(st.median(r["prep_s"] for r in rs)),
                                 "wrong": log2e(st.mean(wrong)) if wrong else None,
                                 "correct": log2e(st.median(solve)) if solve else None},
        }

    # H1
    d = [sum(rec[s][k] - rec[PAIRS[s]][k] for s in SEVEN) for k in range(32)]
    pos, neg = sum(x > 0 for x in d), sum(x < 0 for x in d)
    per = {}
    for s in SEVEN:
        a = sum(rec[s][k] and not rec[PAIRS[s]][k] for k in range(32))
        b = sum(rec[PAIRS[s]][k] and not rec[s][k] for k in range(32))
        per[s] = {"sat_only": a, "bf_only": b, "p": sign_one_sided(a, b)}
    adj = holm({s: per[s]["p"] for s in SEVEN})
    for s in SEVEN:
        per[s]["p_holm"] = adj[s]
    p1 = sign_one_sided(pos, neg)
    h1 = {"keys_sat_ahead": pos, "keys_bf_ahead": neg, "keys_tied": 32 - pos - neg, "p": p1,
          "pass": p1 < 0.05, "per_ranker": per}

    # H2
    S, B = arms["R8-G"], arms["R8-G-BF"]
    nS, nB = sum(rec["R8-G"].values()), sum(rec["R8-G-BF"].values())
    cheaper = sum(par2(B[k]) < par2(S[k]) for k in range(32))
    dearer = sum(par2(B[k]) > par2(S[k]) for k in range(32))
    p2 = sign_one_sided(cheaper, dearer)
    h2 = {"sat_recovered": nS, "bf_recovered": nB, "a_noninferior": nB >= nS - 1,
          "bf_cheaper": cheaper, "bf_dearer": dearer, "ties": 32 - cheaper - dearer, "p_b": p2, "b_pass": p2 < 0.05,
          "discordant_recovery": {"sat_only": sum(rec["R8-G"][k] and not rec["R8-G-BF"][k] for k in range(32)),
                                  "bf_only": sum(rec["R8-G-BF"][k] and not rec["R8-G"][k] for k in range(32))}}
    h2["pass"] = h2["a_noninferior"] and h2["b_pass"]

    inside = sum(v["vs_interval"] == "inside" for v in out["arms"].values())
    loss = {s: out["arms"][s]["recovered"] - out["arms"][PAIRS[s]]["recovered"] for s in SEVEN}
    out["tests"] = {
        "H1_primary": h1, "H2_coprimary": h2,
        "H3_calibration": {"inside": inside, "of": 8, "pass": inside >= 7},
        "H4_loss_order": {"loss": loss, "B1_ge_table_ge_net6": loss["R7-B1"] >= loss["R7-C"] >= loss["R7-G"]},
        "verdict": ("rule wins at both depths" if h1["pass"] and h2["pass"] else
                    "rule wins at seven rounds only" if h1["pass"] else
                    "rule wins at eight rounds only" if h2["pass"] else "rule fails"),
    }
    with open(sys.argv[2], "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(json.dumps(out["tests"], indent=1))
    for a, v in out["arms"].items():
        print(a, v["recovered"], v["interval90"], v["vs_interval"], "PAR-2", v["mean_par2_s"])


if __name__ == "__main__":
    main()
