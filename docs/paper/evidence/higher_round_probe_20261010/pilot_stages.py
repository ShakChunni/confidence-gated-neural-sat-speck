"""8-round pilot, part 1 (demo seed 999999, engineering only).

(a) Bimodality: re-rank each of the 32 demo keys with 3 fresh independent
    64-pair sets (same key) using Gohr net7. If success follows the key, it is
    key-dependent; if it follows the pair set, it is data-dependent.
(b) Stage 2/3 given the TRUE parents: rank k7 with the 6-round sparse table
    after 1-round decryption, and k6 with the 5-round table after 2-round
    decryption. Reports containment and per-call time.
"""

import json
import statistics
import sys
import time

import numpy as np
import torch

from gohr_torch import dec_one, enc, expand_key, load_gohr, to_bits
from neural_sat_speck.attacks.classical_ranking import ClassicalDistributionTable, rank_classical_subkeys
from neural_sat_speck.data.attack import AttackInstance, AttackPair, AttackInstanceConfig, generate_attack_instances

NPZ, T6, T5 = sys.argv[1], sys.argv[2], sys.argv[3]
R, NP = 8, 64
dev = torch.device("mps")
net = load_gohr(NPZ).to(dev)
inst = generate_attack_instances(AttackInstanceConfig(instance_count=32, pair_count=NP, rounds=R, seed=999999, split_role="test")).instances
cands = np.arange(65536, dtype=np.uint32)


def neural_rank(c0, c1, true_k):
    scores = np.empty(65536)
    for s in range(0, 65536, 4096):
        k = np.repeat(cands[s:s + 4096], len(c0))
        a = dec_one(np.tile(c0[:, 0], 4096), np.tile(c0[:, 1], 4096), k)
        b = dec_one(np.tile(c1[:, 0], 4096), np.tile(c1[:, 1], 4096), k)
        with torch.no_grad():
            z = net.logits(torch.from_numpy(to_bits(*a, *b)).to(dev)).cpu().numpy()
        scores[s:s + 4096] = z.reshape(4096, len(c0)).sum(1)
    t = scores[true_k]
    return float((scores > t).sum() + ((scores == t).sum() + 1) / 2)


def class_rank(c0, c1, table, true_k):
    pairs = tuple(AttackPair((0, 0), (0, 0), (int(a[0]), int(a[1])), (int(b[0]), int(b[1]))) for a, b in zip(c0, c1))
    x = AttackInstance("p", (0, 0, 0, 0), true_k, pairs)
    t = time.perf_counter()
    r = rank_classical_subkeys(x, table)
    dt = time.perf_counter() - t
    ts = r.candidate_scores[r.candidate_values.index(true_k)]
    return sum(v > ts for v in r.candidate_scores) + (sum(v == ts for v in r.candidate_scores) + 1) / 2, dt


rng = np.random.default_rng(999999 + 1)
t6, t5 = ClassicalDistributionTable.load(T6), ClassicalDistributionTable.load(T5)
rows = []
for i, x in enumerate(inst):
    ks = expand_key(list(x.master_key), R)
    resample = []
    for _ in range(3):
        p0 = rng.integers(0, 65536, (NP, 2), dtype=np.uint32)
        p1 = p0 ^ np.array([0x0040, 0], dtype=np.uint32)
        c0 = np.stack(enc(p0[:, 0], p0[:, 1], ks), 1)
        c1 = np.stack(enc(p1[:, 0], p1[:, 1], ks), 1)
        resample.append(neural_rank(c0, c1, ks[-1]))
    c0 = np.array([p.ciphertext0 for p in x.pairs], dtype=np.uint32)
    c1 = np.array([p.ciphertext1 for p in x.pairs], dtype=np.uint32)
    d0 = np.stack(dec_one(c0[:, 0], c0[:, 1], ks[-1]), 1)
    d1 = np.stack(dec_one(c1[:, 0], c1[:, 1], ks[-1]), 1)
    r7, s7 = class_rank(d0, d1, t6, ks[-2])
    e0 = np.stack(dec_one(d0[:, 0], d0[:, 1], ks[-2]), 1)
    e1 = np.stack(dec_one(d1[:, 0], d1[:, 1], ks[-2]), 1)
    r6, s6 = class_rank(e0, e1, t5, ks[-3])
    rows.append({"key": i + 1, "resample_ranks": resample, "stage2_rank": r7, "stage3_rank": r6, "s2_s": s7, "s3_s": s6})
    print(json.dumps(rows[-1]), flush=True)
print(json.dumps({
    "stage2_top32": sum(r["stage2_rank"] <= 32 for r in rows),
    "stage2_top8": sum(r["stage2_rank"] <= 8 for r in rows),
    "stage3_top32": sum(r["stage3_rank"] <= 32 for r in rows),
    "stage3_top8": sum(r["stage3_rank"] <= 8 for r in rows),
    "s2_median_s": statistics.median(r["s2_s"] for r in rows),
    "s3_median_s": statistics.median(r["s3_s"] for r in rows),
}))
