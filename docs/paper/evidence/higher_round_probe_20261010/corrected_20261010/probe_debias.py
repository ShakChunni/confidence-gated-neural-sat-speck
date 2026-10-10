"""Demo probe: subtract a per-candidate baseline score estimated on unrelated keys.

baseline[c] = mean over B calibration keys (demo instances disjoint from the test
keys) of the summed-logit score of candidate c. Corrected score = score - baseline.
"""
import json, statistics, sys
import numpy as np, torch
from gohr_torch import dec_one, load_gohr, to_bits
from neural_sat_speck.data.attack import AttackInstanceConfig, generate_attack_instances
G, n_test, n_cal = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
net = load_gohr(f"{G}/net7_small.npz").to("mps")
inst = generate_attack_instances(AttackInstanceConfig(instance_count=n_test + 1000 + n_cal, pair_count=64, rounds=8, seed=999999, split_role="test")).instances
test, cal = inst[:n_test], inst[1000:1000 + n_cal]
cands = np.arange(65536, dtype=np.uint32)
def scores(x):
    c0 = np.array([p.ciphertext0 for p in x.pairs], dtype=np.uint32); c1 = np.array([p.ciphertext1 for p in x.pairs], dtype=np.uint32)
    out = np.empty(65536)
    for st in range(0, 65536, 4096):
        k = np.repeat(cands[st:st+4096], 64)
        a = dec_one(np.tile(c0[:,0],4096), np.tile(c0[:,1],4096), k); b = dec_one(np.tile(c1[:,0],4096), np.tile(c1[:,1],4096), k)
        with torch.no_grad(): out[st:st+4096] = net.logits(torch.from_numpy(to_bits(*a,*b)).to("mps")).cpu().numpy().reshape(4096,64).sum(1)
    return out
base = np.mean([scores(x) for x in cal], axis=0)
np.save(f"baseline_cal{n_cal}.npy", base)
def rank(sc, t):
    v = sc[t]; return float((sc > v).sum() + ((sc == v).sum() + 1) / 2)
rows = []
for i, x in enumerate(test):
    sc = scores(x); t = x.final_round_subkey
    rows.append({"key": i+1, "bits11_10": int((t>>10)&3), "raw": rank(sc, t), "debiased": rank(sc - base, t)})
    print(json.dumps(rows[-1]), flush=True)
print(json.dumps({"n_cal": n_cal, "raw_top32": sum(r["raw"] <= 32 for r in rows), "debiased_top32": sum(r["debiased"] <= 32 for r in rows),
                  "debiased_top16": sum(r["debiased"] <= 16 for r in rows), "debiased_top128": sum(r["debiased"] <= 128 for r in rows)}))
