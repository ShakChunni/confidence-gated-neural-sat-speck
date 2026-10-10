"""Demo-seed feasibility probe: sparse classical Stage-1 containment one round deeper.

Engineering only (seed 999999, demo role). Ranks the final SPECK subkey of
R-round instances with an (R-1)-round sparse table and reports top-32
containment, median rank and ranking time. Not a protected or reportable result.
"""

import json
import statistics
import sys
import time

from neural_sat_speck.attacks.classical_ranking import ClassicalDistributionTable, rank_classical_subkeys
from neural_sat_speck.data.attack import AttackInstanceConfig, generate_attack_instances

table_path, rounds, n_keys, n_pairs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
t0 = time.perf_counter()
table = ClassicalDistributionTable.load(table_path)
load_s = time.perf_counter() - t0
cfg = AttackInstanceConfig(instance_count=n_keys, pair_count=n_pairs, rounds=rounds, seed=999999, split_role="test")
inst = generate_attack_instances(cfg).instances
ranks, secs = [], []
for x in inst:
    t = time.perf_counter()
    r = rank_classical_subkeys(x, table)
    secs.append(time.perf_counter() - t)
    true_s = r.candidate_scores[r.candidate_values.index(x.final_round_subkey)]
    above = sum(v > true_s for v in r.candidate_scores)
    ties = sum(v == true_s for v in r.candidate_scores)
    ranks.append(above + (ties + 1) / 2)  # tie-aware midrank
print(json.dumps({
    "table": table_path, "rounds": rounds, "keys": n_keys, "pairs": n_pairs,
    "load_s": round(load_s, 2),
    "top32": sum(k <= 32 for k in ranks), "top1": sum(k == 1 for k in ranks),
    "median_rank": statistics.median(ranks), "ranks": ranks,
    "median_rank_s": round(statistics.median(secs), 3),
}))
