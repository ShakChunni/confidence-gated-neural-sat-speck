"""POST HOC diagnostic for D-065 (not pre-registered; written after the fresh results).

Applies the frozen cost rule cell by cell with the OBSERVED preparation time and true
queue position, the frozen development r_bar and s for SAT, and the pilot BF rejection
time with the exact true-guess position (computed from the assigned key) for BF.
Separates cost-model error from the queue-position (containment) distribution shift.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from gohr_published_net import expand_key  # noqa: E402

from neural_sat_speck.data.attack import (  # noqa: E402
    AttackInstanceConfig,
    generate_attack_instances,
)

LEDGER = REPO / "docs/paper/evidence/d065_campaign_20261012/ledger_k00-31.jsonl"
PRED = REPO / "docs/paper/evidence/d065_predictions/predictions.json"
OUT = REPO / "docs/paper/evidence/d065_campaign_20261012/posthoc_cell_rule.json"


def main() -> None:
    cells = [r for r in map(json.loads, LEDGER.open()) if "arm" in r]
    pred = json.loads(PRED.read_text())
    comp = pred["components"]
    rbf = {7: pred["bf_rejection"]["7"]["R_BF"], 8: pred["bf_rejection"]["8"]["R_BF"]}
    inst = {R: generate_attack_instances(AttackInstanceConfig(instance_count=32, pair_count=64, rounds=R, seed=701003,
                                                              split_role="test")).instances for R in (7, 8)}
    rows, mismatches = [], []
    for r in cells:
        R, m, q, P = r["rounds"], r["words"], r["true_queue_position"], r["prep_s"]
        if q is None:
            t = None
        elif r["arm"].endswith("-BF"):
            ks = expand_key(list(inst[R][r["key_index"]].master_key), R)
            g = sum(ks[R - 4 + i] << (16 * i) for i in range(4 - m))
            t = P + q * rbf[R] + g / 2 ** (16 * (4 - m)) * rbf[R]
        else:
            c = comp[r["arm"]]
            t = P + q * c["r_bar"] + c["s"]
        predicted = t is not None and t <= 300
        observed = r["outcome"] == "recovered"
        row = {"arm": r["arm"], "key_index": r["key_index"], "q": q, "prep_s": P, "rule_time_s": t,
               "predicted": predicted, "observed": observed, "total_s": r["total_s"]}
        rows.append(row)
        if predicted != observed:
            mismatches.append(row)
    out = {"label": "POST HOC diagnostic, not pre-registered", "agree": len(rows) - len(mismatches), "cells": len(rows),
           "mismatches": mismatches, "rows": rows}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"agree {out['agree']}/{out['cells']}")
    for x in mismatches:
        print(" mismatch", x)


if __name__ == "__main__":
    main()
