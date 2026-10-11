"""Pilot: dense truncated-difference classical table for SPECK32/64 Stage 1.

Engineering only (demo seed 999999). Build a (R-1)-round table over the b most
biased output-difference bits, then rank the last round key of R-round demo
instances by summing log2(p/2^-b) over the 64 pairs after one-round
decryption. Reports true-word ranks.

Usage: trunc_pilot.py ROUNDS(attack) BITS LOG2_SAMPLES KEYS OUT.json
"""

import json
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/ashfaq/Dev/comp-sci-project/scripts")
from gohr_published_net import dec_one, enc  # noqa: E402
from neural_sat_speck.data.attack import AttackInstanceConfig, generate_attack_instances  # noqa: E402

R, B, LOG2N, NKEYS, OUT = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
TR = R - 1  # table rounds
CHUNK = 1 << 24
rng = np.random.default_rng(20261011)


def sample_diffs(n):
    """Output differences after TR rounds from input difference (0x0040, 0), independent random round keys."""
    x0 = rng.integers(0, 1 << 16, n, dtype=np.uint32)
    y0 = rng.integers(0, 1 << 16, n, dtype=np.uint32)
    ks = [rng.integers(0, 1 << 16, n, dtype=np.uint32) for _ in range(TR)]
    a = enc(x0, y0, ks)
    b = enc(x0 ^ 0x0040, y0, ks)
    return ((a[0] ^ b[0]) << 16) | (a[1] ^ b[1])  # 32-bit difference, x in high half


t0 = time.perf_counter()
d = sample_diffs(CHUNK)
p1 = np.array([((d >> i) & 1).mean() for i in range(32)])
bits = [int(i) for i in np.argsort(-np.abs(p1 - 0.5))[:B]]
bias = {i: round(float(p1[i] - 0.5), 4) for i in bits}


def project(diff):
    idx = np.zeros(diff.shape, dtype=np.uint32)
    for j, i in enumerate(bits):
        idx |= ((diff >> i) & 1) << j
    return idx


counts = np.zeros(1 << B, dtype=np.int64)
n_total = 1 << LOG2N
for _ in range(n_total // CHUNK):
    counts += np.bincount(project(sample_diffs(CHUNK)), minlength=1 << B)
build_s = time.perf_counter() - t0
llr = np.log2((counts + 0.5) / (n_total + 0.5 * (1 << B)) * (1 << B)).astype(np.float32)

inst = generate_attack_instances(AttackInstanceConfig(instance_count=NKEYS, pair_count=64, rounds=R, seed=999999, split_role="test")).instances
cands = np.arange(1 << 16, dtype=np.uint32)
ranks, secs = [], []
CC = 4096
for x in inst:
    t = time.perf_counter()
    c0 = np.array([p.ciphertext0 for p in x.pairs], dtype=np.uint32)
    c1 = np.array([p.ciphertext1 for p in x.pairs], dtype=np.uint32)
    scores = np.empty(1 << 16, dtype=np.float64)
    for s in range(0, 1 << 16, CC):
        k = np.repeat(cands[s:s + CC], 64)
        a = dec_one(np.tile(c0[:, 0], CC), np.tile(c0[:, 1], CC), k)
        b = dec_one(np.tile(c1[:, 0], CC), np.tile(c1[:, 1], CC), k)
        diff = ((a[0] ^ b[0]) << 16) | (a[1] ^ b[1])
        scores[s:s + CC] = llr[project(diff)].reshape(CC, 64).sum(1)
    secs.append(time.perf_counter() - t)
    true_s = scores[x.final_round_subkey]
    above, ties = int((scores > true_s).sum()), int((scores == true_s).sum())
    order = np.lexsort((cands, -scores))
    pos = int(np.where(order == x.final_round_subkey)[0][0]) + 1
    ranks.append({"key_index": x.instance_id, "midrank": above + (ties + 1) / 2, "pipeline_position": pos})

pp = [r["pipeline_position"] for r in ranks]
res = {"attack_rounds": R, "table_rounds": TR, "bits": B, "log2_samples": LOG2N, "keys": NKEYS,
       "selected_bits": bits, "bias": bias, "build_s": round(build_s, 1),
       "top1": sum(p == 1 for p in pp), "top8": sum(p <= 8 for p in pp), "top32": sum(p <= 32 for p in pp),
       "top64": sum(p <= 64 for p in pp), "median_position": float(np.median(pp)),
       "median_rank_s": round(float(np.median(secs)), 2), "per_key": ranks}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ("per_key", "bias", "selected_bits")}))
