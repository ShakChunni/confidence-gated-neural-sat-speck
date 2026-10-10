"""Demo probe: Gohr's wrong-key-profile (Bayesian) ranking at 8 rounds, evaluated exhaustively.

For every candidate k, emp[k] = mean network output over the 64 pairs after
decrypting one round with k. Gohr's score for a true-key hypothesis t is
  D(t) = sum_k ((emp[k] - m[k^t]) * s[k^t])^2,  s = 1/std,
with m, std his published 7-round wrong-key profile. Lower is better. All 2^16
hypotheses are scored exactly via XOR-correlation (Walsh-Hadamard):
  D(t) = (emp^2 * s^2)(t) - 2 (emp * m s^2)(t) + const.
Also reports the plain summed-logit rank for comparison.
"""

import json
import statistics
import sys

import numpy as np
import torch

from gohr_torch import dec_one, load_gohr, to_bits
from neural_sat_speck.data.attack import AttackInstanceConfig, generate_attack_instances

G = sys.argv[1]
n_keys = int(sys.argv[2])
first = int(sys.argv[3]) if len(sys.argv) > 3 else 0
net = load_gohr(f"{G}/net7_small.npz").to("mps")
m = np.load(f"{G}/data_wrong_key_mean_7r.npy").astype(np.float64)
s = 1.0 / np.load(f"{G}/data_wrong_key_std_7r.npy").astype(np.float64)
inst = generate_attack_instances(AttackInstanceConfig(instance_count=first + n_keys, pair_count=64, rounds=8, seed=999999, split_role="test")).instances[first:]
cands = np.arange(65536, dtype=np.uint32)


def wht(a):
    a = a.copy()
    h = 1
    while h < len(a):
        a = a.reshape(-1, 2, h)
        a = np.stack([a[:, 0] + a[:, 1], a[:, 0] - a[:, 1]], 1).reshape(-1)
        h *= 2
    return a


def xcorr(f, g):  # (f * g)(t) = sum_k f(k) g(k ^ t)
    return wht(wht(f) * wht(g)) / len(f)


W, U = wht(s * s), wht(m * s * s)
out = []
for i, x in enumerate(inst):
    c0 = np.array([p.ciphertext0 for p in x.pairs], dtype=np.uint32)
    c1 = np.array([p.ciphertext1 for p in x.pairs], dtype=np.uint32)
    emp, logit = np.empty(65536), np.empty(65536)
    for st in range(0, 65536, 4096):
        k = np.repeat(cands[st:st + 4096], 64)
        a = dec_one(np.tile(c0[:, 0], 4096), np.tile(c0[:, 1], 4096), k)
        b = dec_one(np.tile(c1[:, 0], 4096), np.tile(c1[:, 1], 4096), k)
        with torch.no_grad():
            z = net.logits(torch.from_numpy(to_bits(*a, *b)).to("mps")).cpu().numpy().astype(np.float64).reshape(4096, 64)
        logit[st:st + 4096] = z.sum(1)
        emp[st:st + 4096] = (1 / (1 + np.exp(-z))).mean(1)
    D = wht(wht(emp * emp) * W) / 65536 - 2 * wht(wht(emp) * U) / 65536
    t = x.final_round_subkey

    def rank(score, higher_better):
        v = score[t]
        better = (score > v).sum() if higher_better else (score < v).sum()
        return float(better + ((np.isclose(score, v)).sum() + 1) / 2)

    rb, rl = rank(D, False), rank(logit, True)
    out.append({"key": first + i + 1, "bits11_10": int((t >> 10) & 3), "bayes_rank": rb, "logit_rank": rl})
    print(json.dumps(out[-1]), flush=True)
br = [o["bayes_rank"] for o in out]
lr = [o["logit_rank"] for o in out]
print(json.dumps({"keys": len(out), "bayes_top32": sum(r <= 32 for r in br), "bayes_top128": sum(r <= 128 for r in br),
                  "logit_top32": sum(r <= 32 for r in lr), "bayes_median": statistics.median(br)}))
