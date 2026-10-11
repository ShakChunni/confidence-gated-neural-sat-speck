"""Frozen D-065 predictions: does the cost rule pick the right completer on fresh keys?

Inputs are development and demo evidence only:
  * D-063 ledger (opened seed 701001): per-key preparation P, true queue position q,
    and per-arm SAT rejection cost r_bar and correct-solve time s;
  * the D-065 demo pilot (seed 999999): measured brute-force full-space rejection
    time R_BF inside the campaign runner.

Rule (paper Eq. 4): a key is predicted recovered when P + q * r + s <= 300 s.
  SAT arm: r = r_bar (development mean), s = development median correct-solve time.
  BF arm:  r = R_BF; s = U * R_BF with U ~ Uniform(0, 1), because the true schedule
           guess is uniform over the enumerated space for a random key.
Fresh-cohort intervals come from a paired key-level bootstrap of the 32 development
keys (the same resampled keys for every arm) with fresh U draws.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEV = REPO / "docs/paper/evidence/d063_campaign_20261011/ledger_k00-31.jsonl"
BUDGET = 300.0
FAIL = 600.0
PAIRS = {"R7-B1": "R7-B1-BF", "R7-G": "R7-G-BF", "R7-C": "R7-C-BF", "R8-G": "R8-G-BF"}
BF_FALLBACK = {7: 61.774133248, 8: 65536 * 1.9e-8}  # benchmark full-space seconds (m=2 at 7R, m=3 at 8R)


def load(path: Path) -> list[dict]:
    return [r for r in map(json.loads, path.open()) if "arm" in r]


def components(rows: list[dict]) -> dict:
    out = {}
    for arm in PAIRS:
        rs = sorted((r for r in rows if r["arm"] == arm), key=lambda r: r["key_index"])
        out[arm] = {
            "P": [r["prep_s"] for r in rs],
            "q": [r["true_queue_position"] for r in rs],
            "observed": [r["outcome"] == "recovered" for r in rs],
            "r_bar": mean(r["wrong_mean_s"] for r in rs if r["wrong_mean_s"]),
            "s": median(r["correct_solve_s"] for r in rs if r.get("correct_solve_s")),
        }
    return out


def bf_rejection(pilot: Path | None) -> dict:
    """Median per-candidate full-space BF time measured by the runner on the demo seed."""
    out = {}
    rows = load(pilot) if pilot else []
    for rounds in (7, 8):
        w = [r["wrong_mean_s"] for r in rows if r["arm"].endswith("-BF") and r["rounds"] == rounds and r["wrong_mean_s"]]
        out[rounds] = {"R_BF": median(w) if w else BF_FALLBACK[rounds], "source": "pilot" if w else "benchmark", "n_cells": len(w)}
    return out


def sign_p(a: int, b: int) -> float:
    """One-sided exact sign test: P(X >= a) for X ~ Bin(a + b, 1/2)."""
    n = a + b
    return 1.0 if n == 0 else sum(math.comb(n, k) for k in range(a, n + 1)) / 2**n


def holm_all(ps: list[float], alpha: float = 0.05) -> bool:
    return all(p <= alpha / (len(ps) - i) for i, p in enumerate(sorted(ps)))


def simulate(c: dict, rbf: dict, keys: np.ndarray, u: np.ndarray) -> dict:
    """Recovery indicators and per-key PAR-2 scores for every arm on the given key draw."""
    res = {}
    for sat, bf in PAIRS.items():
        rounds = 8 if sat.startswith("R8") else 7
        P = np.array(c[sat]["P"])[keys]
        q = np.array([np.inf if v is None else v for v in c[sat]["q"]])[keys]
        t_sat = P + q * c[sat]["r_bar"] + c[sat]["s"]
        R = rbf[rounds]["R_BF"]
        t_bf = P + q * R + u * R
        for arm, t in ((sat, t_sat), (bf, t_bf)):
            ok = t <= BUDGET
            res[arm] = {"rec": ok, "par2": np.where(ok, t, FAIL)}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=100_000)
    args = ap.parse_args()
    c = components(load(DEV))
    rbf = bf_rejection(args.pilot)
    rng = np.random.default_rng(65065)
    n = len(c["R7-G"]["P"])

    # Point expectation on the development keys themselves (U integrated by Monte Carlo).
    u = rng.random((4096, n))
    point = {}
    for arm in list(PAIRS) + list(PAIRS.values()):
        point[arm] = float(np.mean([simulate(c, rbf, np.arange(n), ui)[arm]["rec"].sum() for ui in u[:512]]))

    # Paired key-level bootstrap for fresh 32-key cohorts.
    counts = {a: np.empty(args.reps, dtype=np.int16) for a in point}
    par2 = {a: np.empty(args.reps) for a in point}
    disc = {s: np.empty((args.reps, 2), dtype=np.int16) for s in PAIRS}
    h1 = np.empty(args.reps, dtype=bool)
    h1_primary = np.empty(args.reps, dtype=bool)
    for i in range(args.reps):
        keys = rng.integers(0, n, n)
        sim = simulate(c, rbf, keys, rng.random(n))
        for a in point:
            counts[a][i] = sim[a]["rec"].sum()
            par2[a][i] = sim[a]["par2"].mean()
        for s, b in PAIRS.items():
            disc[s][i] = ((sim[s]["rec"] & ~sim[b]["rec"]).sum(), (sim[b]["rec"] & ~sim[s]["rec"]).sum())
        d = sum(sim[s]["rec"].astype(int) - sim[PAIRS[s]]["rec"].astype(int) for s in ("R7-B1", "R7-G", "R7-C"))
        h1_primary[i] = sign_p(int((d > 0).sum()), int((d < 0).sum())) <= 0.05
        h1[i] = holm_all([sign_p(int(disc[s][i, 0]), int(disc[s][i, 1])) for s in ("R7-B1", "R7-G", "R7-C")])

    def iv(x):
        return [float(np.percentile(x, 5)), float(np.percentile(x, 95))]

    out = {
        "schema": "d065_predictions_v1",
        "inputs": {"dev_ledger": str(DEV.relative_to(REPO)), "dev_sha256": hashlib.sha256(DEV.read_bytes()).hexdigest(),
                   "pilot": str(args.pilot) if args.pilot else None,
                   "pilot_sha256": hashlib.sha256(args.pilot.read_bytes()).hexdigest() if args.pilot else None},
        "components": {a: {"P_median": median(v["P"]), "r_bar": v["r_bar"], "s": v["s"],
                           "dev_observed_sat": int(sum(v["observed"]))} for a, v in c.items()},
        "bf_rejection": rbf,
        "k_star": {a: (BUDGET - median(v["P"]) - v["s"]) / v["r_bar"] for a, v in c.items()},
        "k_star_bf": {a: (BUDGET - median(c[a]["P"]) - 0.5 * rbf[8 if a.startswith("R8") else 7]["R_BF"])
                      / rbf[8 if a.startswith("R8") else 7]["R_BF"] for a in PAIRS},
        "dev_point_expectation": point,
        "fresh_interval_90": {a: iv(counts[a]) for a in point},
        "fresh_median": {a: float(np.median(counts[a])) for a in point},
        "fresh_par2_mean_interval_90": {a: iv(par2[a]) for a in point},
        "fresh_gap_sat_minus_bf_interval_90": {s: iv(counts[s].astype(int) - counts[b]) for s, b in PAIRS.items()},
        "fresh_discordant_median": {s: [float(np.median(disc[s][:, 0])), float(np.median(disc[s][:, 1]))] for s in PAIRS},
        "p_gap_order_B1_ge_C_ge_G": float(np.mean(
            ((counts["R7-B1"] - counts["R7-B1-BF"]) >= (counts["R7-C"] - counts["R7-C-BF"]))
            & ((counts["R7-C"] - counts["R7-C-BF"]) >= (counts["R7-G"] - counts["R7-G-BF"])))),
        "power_H1_primary_per_key_pooled": float(h1_primary.mean()),
        "power_H1_all_three_holm": float(h1.mean()),
        "power_H1_single_sign_test_p05": {s: float(np.mean([sign_p(int(x), int(y)) <= 0.05 for x, y in disc[s]]))
                                          for s in ("R7-B1", "R7-G", "R7-C")},
        "reps": args.reps, "rng_seed": 65065,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("bf_rejection", "k_star", "k_star_bf", "dev_point_expectation",
                                         "fresh_interval_90", "fresh_gap_sat_minus_bf_interval_90",
                                         "fresh_discordant_median", "p_gap_order_B1_ge_C_ge_G",
                                         "power_H1_primary_per_key_pooled", "power_H1_all_three_holm", "power_H1_single_sign_test_p05")}, indent=1))


if __name__ == "__main__":
    main()
