"""Queue-cost model for the N=32 backend x ranking-depth study.

Implements docs/QUEUE_COST_MODEL_PROTOCOL_20261009.md on the per-cell and
per-attempt tables written by scripts/extract_queue_cost_attempts.py. Writes
predictions.csv and summary.json to --output-dir and prints the summary as JSON.

Run as:
    python3 -I scripts/fit_queue_cost_model.py --input-dir DIR --output-dir DIR
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise
from math import comb
from pathlib import Path

import numpy as np

BUDGET = 300.0
NDRAW = 20000
SEED = 59001
TAUS = [1, 2, 5, 10, 20, 45, 100, 300]
METHODS = ("pred_primary", "pred_simple", "pred_B0", "pred_B1")
CAP_DEVIATION = (
    "Cap-curve estimability operationalised as: <10% of the arm's r and s observations "
    "censored below tau."
)


@dataclass
class Dataset:
    """Cells, their attempts (sorted by attempt index) and per-cell observations."""

    cells: list[dict]
    by_cell: dict[str, list[dict]]
    obs: dict[str, dict]
    arms: list[tuple]


def as_bool(text: str) -> bool:
    return text == "True"


def arm_of(c: dict) -> tuple:
    return (c["cipher"], int(c["depth"]), c["profile"])


def kq(c: dict) -> int | None:
    v = c["true_queue_position"]
    return int(v) if v not in ("", "None") else None


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def build_observations(cells: list[dict], by_cell: dict[str, list[dict]]) -> dict:
    """Per cell: r (wrong-candidate), s (true-candidate) and o (overhead) observations.

    r and s hold (seconds, censored) pairs; a censored attempt is timed to the deadline.
    o holds the gap between a completed attempt's end and the next attempt's start.
    """
    obs = {}
    for c in cells:
        r, s, o = [], [], []
        al = by_cell[c["cell_id"]]
        for j, a in enumerate(al):
            cen = as_bool(a["censored"])
            t = float(a["censor_seconds"]) if cen else float(a["solve_wall_seconds"])
            (s if as_bool(a["is_true"]) else r).append((t, cen))
            if j + 1 < len(al) and not cen:
                nxt = float(al[j + 1]["started_offset_seconds"])
                o.append(nxt - (float(a["started_offset_seconds"]) + t))
        obs[c["cell_id"]] = dict(r=r, s=s, o=o)
    return obs


def load_dataset(input_dir: Path) -> Dataset:
    cells = read_rows(input_dir / "cells.csv")
    atts = read_rows(input_dir / "attempts.csv")
    by_cell: dict[str, list[dict]] = defaultdict(list)
    for a in atts:
        by_cell[a["cell_id"]].append(a)
    for v in by_cell.values():
        v.sort(key=lambda a: int(a["attempt_index"]))
    obs = build_observations(cells, by_cell)
    arms = sorted({arm_of(c) for c in cells})
    return Dataset(cells=cells, by_cell=by_cell, obs=obs, arms=arms)


class KM:
    """Kaplan-Meier; sample() returns inf for mass beyond last event."""

    def __init__(self, data):
        data = sorted(data)
        n = len(data)
        self.n = n
        self.ncens = sum(1 for _, c in data if c)
        times, cdf = [], []
        surv, at_risk, i = 1.0, n, 0
        while i < n:
            t = data[i][0]
            d = m = 0
            while i < n and data[i][0] == t:
                m += 1
                d += 0 if data[i][1] else 1
                i += 1
            if d and at_risk:
                surv *= 1 - d / at_risk
                times.append(t)
                cdf.append(1 - surv)
            at_risk -= m
        self.t = np.array(times)
        self.F = np.array(cdf)
        self.data = data

    def sample(self, rng, size):
        u = rng.random(size)
        if self.t.size == 0:
            return np.full(size, np.inf)
        idx = np.searchsorted(self.F, u, side="left")
        out = np.full(size, np.inf)
        ok = idx < self.t.size
        out[ok] = self.t[idx[ok]]
        return out

    def quantile(self, q):
        if self.t.size == 0 or self.F[-1] < q:
            return math.inf
        return float(self.t[np.searchsorted(self.F, q, side="left")])

    def surv_at(self, x):
        if self.t.size == 0:
            return 1.0
        k = np.searchsorted(self.t, x, side="right")
        return 1.0 if k == 0 else float(1 - self.F[k - 1])


def pooled(ds: Dataset, arm, exclude_key=None, field="r"):
    out = []
    for c in ds.cells:
        if arm_of(c) == arm and c["key_index"] != exclude_key:
            out += ds.obs[c["cell_id"]][field]
    return out


def dists(ds: Dataset, arm, exclude_key=None):
    o = pooled(ds, arm, exclude_key, "o")
    return (
        KM(pooled(ds, arm, exclude_key, "r")),
        KM(pooled(ds, arm, exclude_key, "s")),
        np.array(o if o else [0.0]),
    )


def p_recover(P, k, rkm, skm, oarr, rng, tau=None):
    if k is None:
        return 0.0
    total = np.full(NDRAW, float(P))
    chunk = 64
    left = k
    while left > 0:
        m = min(chunk, left)
        rr = rkm.sample(rng, (NDRAW, m))
        if tau is not None:
            rr = np.minimum(rr, tau)
        oo = rng.choice(oarr, size=(NDRAW, m))
        total += (rr + oo).sum(axis=1)
        left -= m
        if np.all(total > BUDGET):
            return 0.0
    ss = skm.sample(rng, NDRAW)
    if tau is not None:
        ss = np.where(ss > tau, np.inf, ss)
    return float(np.mean(total + ss <= BUDGET))


def t_simple(P, k, rkm, skm, oarr):
    if k is None:
        return math.inf
    mr = min(rkm.quantile(0.5), BUDGET)
    ms = min(skm.quantile(0.5), BUDGET)
    return P + k * (mr + float(np.median(oarr))) + ms


def cp_interval(x, n, a=0.05):
    def cdf(k, p):
        return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))

    lo, hi = 0.0, 1.0
    if x > 0:
        left, right = 0.0, x / n
        for _ in range(60):
            m = (left + right) / 2
            if 1 - cdf(x - 1, m) < a / 2:
                left = m
            else:
                right = m
        lo = left
    if x < n:
        left, right = x / n, 1.0
        for _ in range(60):
            m = (left + right) / 2
            if cdf(x, m) < a / 2:
                right = m
            else:
                left = m
        hi = right
    return lo, hi


def metrics(rows, key):
    n = len(rows)
    acc = sum(r[key] == r["actual"] for r in rows)
    pos = [r for r in rows if r["actual"] == 1]
    neg = [r for r in rows if r["actual"] == 0]
    tpr = sum(r[key] == 1 for r in pos) / len(pos) if pos else float("nan")
    tnr = sum(r[key] == 0 for r in neg) / len(neg) if neg else float("nan")
    return dict(
        correct=acc,
        n=n,
        accuracy=acc / n,
        ci=cp_interval(acc, n),
        balanced_accuracy=(tpr + tnr) / 2,
    )


def spearman(x, y):
    if len(x) < 3:
        return None
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def assumption_diagnostics(ds: Dataset) -> dict:
    diag = {}
    for arm in ds.arms:
        pos, rt, a1, a2 = [], [], [], []
        for c in ds.cells:
            if arm_of(c) != arm:
                continue
            al = [
                a
                for a in ds.by_cell[c["cell_id"]]
                if not as_bool(a["is_true"]) and not as_bool(a["censored"])
            ]
            for a in al:
                pos.append(int(a["queue_position"]))
                rt.append(float(a["solve_wall_seconds"]))
            for x, y in pairwise(al):
                a1.append(float(x["solve_wall_seconds"]))
                a2.append(float(y["solve_wall_seconds"]))
        diag["|".join(map(str, arm))] = dict(
            spearman_r_vs_position=spearman(pos, rt),
            n=len(rt),
            spearman_neighbour=spearman(a1, a2),
            n_pairs=len(a1),
        )
    return diag


def cf_count(ds: Dataset, target_arm, r_from=None, P_from_arm=None):
    src = r_from or target_arm
    rkm = KM(pooled(ds, src))
    _, skm, oarr = dists(ds, target_arm)
    rng3 = np.random.default_rng(59003)
    tot = 0.0
    for c in ds.cells:
        if arm_of(c) != target_arm:
            continue
        P = float(c["prep_seconds"])
        if P_from_arm:
            twin = next(
                x
                for x in ds.cells
                if arm_of(x) == P_from_arm and x["key_index"] == c["key_index"]
            )
            P = float(twin["prep_seconds"])
        tot += p_recover(P, kq(c), rkm, skm, oarr, rng3)
    return round(tot, 2)


def counterfactuals(ds: Dataset) -> dict:
    # No leave-one-out needed: model-based predictions on full arm distributions.
    cf = {}
    for ci in ("speck32_64", "simon32_64"):
        for d in (1, 2):
            G, X = (ci, d, "cms_native_gauss"), (ci, d, "cms_native_no_gauss")
            cf[f"{ci} d{d} X with G rejection times"] = dict(
                base=cf_count(ds, X), counterfactual=cf_count(ds, X, r_from=G)
            )
            cf[f"{ci} d{d} G with X rejection times"] = dict(
                base=cf_count(ds, G), counterfactual=cf_count(ds, G, r_from=X)
            )
        for p in ("cms_native_gauss", "cms_native_no_gauss"):
            a1, a2 = (ci, 1, p), (ci, 2, p)
            cf[f"{ci} {p} d1 with d2 rejection times"] = dict(
                base=cf_count(ds, a1), counterfactual=cf_count(ds, a1, r_from=a2)
            )
            cf[f"{ci} {p} d2 with d1 preparation"] = dict(
                base=cf_count(ds, a2), counterfactual=cf_count(ds, a2, P_from_arm=a1)
            )
    return cf


def cap_curve(ds: Dataset) -> dict:
    cap = {}
    for arm in ds.arms:
        rlist, slist = pooled(ds, arm), pooled(ds, arm, field="s")
        rkm, skm, oarr = dists(ds, arm)
        row = {}
        for tau in TAUS:
            fr = sum(1 for t, c in rlist if c and t < tau) / max(len(rlist), 1)
            fs = sum(1 for t, c in slist if c and t < tau) / max(len(slist), 1)
            if fr >= 0.10 or fs >= 0.10:
                row[tau] = "not estimable"
                continue
            rng4 = np.random.default_rng(59004)
            tot = sum(
                p_recover(
                    float(c["prep_seconds"]), kq(c), rkm, skm, oarr, rng4, tau=tau
                )
                for c in ds.cells
                if arm_of(c) == arm
            )
            row[tau] = round(tot, 2)
        cap["|".join(map(str, arm))] = row
    return cap


def predict_all(ds: Dataset) -> list[dict]:
    """Primary model per cell, leave-one-key-out distributions, shared RNG stream."""
    rng = np.random.default_rng(SEED)
    pred = []
    for c in ds.cells:
        arm = arm_of(c)
        rkm, skm, oarr = dists(ds, arm, exclude_key=c["key_index"])
        P, k = float(c["prep_seconds"]), kq(c)
        p = p_recover(P, k, rkm, skm, oarr, rng)
        ts = t_simple(P, k, rkm, skm, oarr)
        pred.append(
            dict(
                cell_id=c["cell_id"],
                cipher=c["cipher"],
                key_index=c["key_index"],
                depth=c["depth"],
                profile=c["profile"],
                actual=int(as_bool(c["recovered"])),
                contained=int(k is not None),
                k=k if k is not None else "",
                P=round(P, 3),
                p_primary=round(p, 4),
                pred_primary=int(p >= 0.5),
                T_simple=round(ts, 2) if ts < math.inf else "inf",
                pred_simple=int(ts <= BUDGET),
            )
        )
    return pred


def add_baselines(ds: Dataset, pred: list[dict]) -> None:
    maj = {}
    for arm in ds.arms:
        ys = [int(as_bool(c["recovered"])) for c in ds.cells if arm_of(c) == arm]
        maj[arm] = int(sum(ys) * 2 > len(ys))
    for row, c in zip(pred, ds.cells):
        row["pred_B0"] = row["contained"]
        row["pred_B1"] = maj[arm_of(c)]


def add_cluster_bootstrap(pred: list[dict], overall: dict) -> None:
    """Cluster bootstrap over the 64 (cipher, key_index) clusters."""
    rng2 = np.random.default_rng(59002)
    clusters = defaultdict(list)
    for r in pred:
        clusters[(r["cipher"], r["key_index"])].append(r)
    ck = list(clusters)
    boot = defaultdict(list)
    for _ in range(10000):
        pick = rng2.integers(0, len(ck), len(ck))
        rows = [r for i in pick for r in clusters[ck[i]]]
        for key in METHODS:
            boot[key].append(np.mean([r[key] == r["actual"] for r in rows]))
    for key, v in boot.items():
        overall[key]["cluster_bootstrap_ci"] = [
            float(np.quantile(v, 0.025)),
            float(np.quantile(v, 0.975)),
        ]


def per_arm_summary(ds: Dataset, pred: list[dict]) -> tuple[dict, int]:
    per_arm = {}
    fits = 0
    for arm in ds.arms:
        rows = [r for r in pred if (r["cipher"], int(r["depth"]), r["profile"]) == arm]
        rkm, skm, oarr = dists(ds, arm)
        Ps = [r["P"] for r in rows]
        ks = [r["k"] for r in rows if r["k"] != ""]
        expected = sum(r["p_primary"] for r in rows)
        actual = sum(r["actual"] for r in rows)
        within3 = abs(expected - actual) <= 3
        fits += within3
        per_arm["|".join(map(str, arm))] = dict(
            actual=actual,
            expected_primary=round(expected, 2),
            within3=within3,
            pred_simple_count=sum(r["pred_simple"] for r in rows),
            acc_primary=metrics(rows, "pred_primary")["accuracy"],
            acc_simple=metrics(rows, "pred_simple")["accuracy"],
            acc_B0=metrics(rows, "pred_B0")["accuracy"],
            acc_B1=metrics(rows, "pred_B1")["accuracy"],
            r_n=rkm.n,
            r_censored=rkm.ncens,
            r_median=rkm.quantile(0.5),
            r_p90=rkm.quantile(0.9),
            s_n=skm.n,
            s_censored=skm.ncens,
            s_median=skm.quantile(0.5),
            s_p90=skm.quantile(0.9),
            o_median=float(np.median(oarr)),
            P_median=float(np.median(Ps)),
            k_median=float(np.median(ks)) if ks else None,
        )
    return per_arm, fits


def run_fit(ds: Dataset) -> tuple[list[dict], dict]:
    pred = predict_all(ds)
    add_baselines(ds, pred)

    summary = {"overall": {}, "per_arm": {}, "deviations": []}
    for key in METHODS:
        summary["overall"][key] = metrics(pred, key)
    summary["overall"]["brier_primary"] = float(
        np.mean([(r["p_primary"] - r["actual"]) ** 2 for r in pred])
    )
    add_cluster_bootstrap(pred, summary["overall"])

    summary["per_arm"], summary["arms_within3"] = per_arm_summary(ds, pred)
    summary["diagnostics"] = assumption_diagnostics(ds)
    summary["counterfactuals"] = counterfactuals(ds)
    summary["deviations"].append(CAP_DEVIATION)
    summary["cap_curve_expected_recoveries_of_32"] = cap_curve(ds)
    return pred, summary


def write_outputs(output_dir: Path, pred: list[dict], summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pred[0]))
        w.writeheader()
        w.writerows(pred)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            indent=2,
            default=lambda x: None if x == math.inf else x,
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fit the N=32 queue-cost model and write predictions and summary."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="directory containing cells.csv and attempts.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory receiving predictions.csv and summary.json",
    )
    args = parser.parse_args(argv)
    ds = load_dataset(args.input_dir)
    pred, summary = run_fit(ds)
    write_outputs(args.output_dir, pred, summary)
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
