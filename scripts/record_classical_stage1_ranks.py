#!/usr/bin/env python3
"""Per-key Stage-1 rank records for a sparse classical SPECK table.

Ranks the last round key of R-round demo instances (seed 999999, development
keys) with an (R-1)-round sparse table and writes one JSON line per key:
the true word, its score, how many words score higher or tie, the tie-aware
midrank, and the true word's position in the order the attack pipeline uses
(descending score, ties by ascending word). A summary line ends the file.

Usage: record_classical_stage1_ranks.py TABLE ROUNDS KEYS OUT.jsonl
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

from neural_sat_speck.attacks.classical_ranking import ClassicalDistributionTable, rank_classical_subkeys
from neural_sat_speck.data.attack import AttackInstanceConfig, generate_attack_instances


def main() -> None:
    table_path, rounds, n_keys, out = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
    if out.exists():
        raise SystemExit("Refusing to overwrite existing records")
    table = ClassicalDistributionTable.load(str(table_path))
    instances = generate_attack_instances(AttackInstanceConfig(
        instance_count=n_keys, pair_count=64, rounds=rounds, seed=999999, split_role="test")).instances
    rows = []
    for j, x in enumerate(instances):
        t = time.perf_counter()
        r = rank_classical_subkeys(x, table)
        secs = time.perf_counter() - t
        true_word = x.final_round_subkey
        true_score = r.candidate_scores[r.candidate_values.index(true_word)]
        above = sum(v > true_score for v in r.candidate_scores)
        ties = sum(v == true_score for v in r.candidate_scores)  # includes the true word
        rows.append({
            "key_index": j, "instance_id": x.instance_id, "true_last_word": true_word,
            "true_score": float(true_score), "words_scoring_higher": above, "words_tied_incl_true": ties,
            "midrank": above + (ties + 1) / 2,
            "pipeline_position": list(r.candidate_order).index(true_word) + 1,
            "in_pipeline_top32": list(r.candidate_order).index(true_word) < 32,
            "nonzero_scores": sum(v != 0 for v in r.candidate_scores),
            "rank_seconds": round(secs, 3),
        })
    summary = {
        "type": "summary", "table": str(table_path),
        "table_sha256": hashlib.sha256(table_path.read_bytes()).hexdigest(),
        "table_rounds": table.rounds, "attack_rounds": rounds, "keys": n_keys, "pairs": 64,
        "seed": 999999, "seed_role": "demo (development keys)",
        "in_pipeline_top32": sum(r["in_pipeline_top32"] for r in rows),
        "best_pipeline_position": min(r["pipeline_position"] for r in rows),
        "median_pipeline_position": statistics.median(r["pipeline_position"] for r in rows),
        "best_midrank": min(r["midrank"] for r in rows),
        "median_midrank": statistics.median(r["midrank"] for r in rows),
    }
    with out.open("x") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
        f.write(json.dumps(summary) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
