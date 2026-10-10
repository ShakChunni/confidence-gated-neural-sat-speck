"""Demo-seed probe: Gohr's published (R-1)-round net ranking the final SPECK subkey at R rounds.

Engineering only (seed 999999). Same instance generator, 64 plaintext-difference
pairs per key, as the classical probe. Score = sum of logits over all pairs
(Gohr's log2 odds up to a constant factor). Reports top-32 containment, tie-aware
rank and ranking time. Not a protected or reportable result.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


import json
import statistics
import time

import numpy as np
import torch

from gohr_published_net import dec_one, enc, expand_key, load_gohr, to_bits
from neural_sat_speck.data.attack import AttackInstanceConfig, generate_attack_instances

npz, rounds, n_keys, n_pairs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
net = load_gohr(npz).to(device)
inst = generate_attack_instances(AttackInstanceConfig(
    instance_count=n_keys, pair_count=n_pairs, rounds=rounds, seed=999999, split_role="test")).instances
cands = np.arange(65536, dtype=np.uint32)
CH = 512  # MPS gives wrong outputs for rows >= 65,536 in one batch (torch 2.13); 4096 x 64 corrupted results before 2026-10-10 18:00
ranks, secs = [], []
for x in inst:
    p0 = np.array([p.plaintext0 for p in x.pairs], dtype=np.uint32)
    c0 = np.array([p.ciphertext0 for p in x.pairs], dtype=np.uint32)
    c1 = np.array([p.ciphertext1 for p in x.pairs], dtype=np.uint32)
    ks = expand_key(list(x.master_key), rounds)
    assert ks[-1] == x.final_round_subkey
    assert np.array_equal(np.stack(enc(p0[:, 0], p0[:, 1], ks), 1), c0), "cipher convention mismatch"
    t = time.perf_counter()
    scores = np.empty(65536, dtype=np.float64)
    for s in range(0, 65536, CH):
        k = np.repeat(cands[s:s + CH], len(c0))
        a0l, a0r = dec_one(np.tile(c0[:, 0], CH), np.tile(c0[:, 1], CH), k)
        a1l, a1r = dec_one(np.tile(c1[:, 0], CH), np.tile(c1[:, 1], CH), k)
        with torch.no_grad():
            z = net.logits(torch.from_numpy(to_bits(a0l, a0r, a1l, a1r)).to(device)).float().cpu().numpy()
        scores[s:s + CH] = z.reshape(CH, len(c0)).sum(1)
    secs.append(time.perf_counter() - t)
    ts = scores[x.final_round_subkey]
    ranks.append(float((scores > ts).sum() + ((scores == ts).sum() + 1) / 2))
    print(len(ranks), ranks[-1], round(secs[-1], 1), flush=True)
print(json.dumps({
    "net": npz, "rounds": rounds, "keys": n_keys, "pairs": n_pairs, "device": str(device),
    "top32": sum(k <= 32 for k in ranks), "top1": sum(k == 1 for k in ranks),
    "median_rank": statistics.median(ranks), "ranks": ranks,
    "median_rank_s": round(statistics.median(secs), 2),
}))
