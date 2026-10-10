"""Demo probe: within-class rank of the true key, classes = bits 11..10 of the candidate.

Reports (a) rank of true key among the 16,384 candidates sharing its bits 11..10,
(b) rank after per-class z-scoring across all 65,536 candidates.
"""
import json, sys
import numpy as np, torch
from gohr_torch import dec_one, load_gohr, to_bits
from neural_sat_speck.data.attack import AttackInstanceConfig, generate_attack_instances
G, n = sys.argv[1], int(sys.argv[2])
net = load_gohr(f"{G}/net7_small.npz").to("mps")
inst = generate_attack_instances(AttackInstanceConfig(instance_count=n, pair_count=64, rounds=8, seed=999999, split_role="test")).instances
cands = np.arange(65536, dtype=np.uint32); cls = (cands >> 10) & 3
rows = []
for i, x in enumerate(inst):
    c0 = np.array([p.ciphertext0 for p in x.pairs], dtype=np.uint32); c1 = np.array([p.ciphertext1 for p in x.pairs], dtype=np.uint32)
    sc = np.empty(65536)
    for st in range(0, 65536, 4096):
        k = np.repeat(cands[st:st+4096], 64)
        a = dec_one(np.tile(c0[:,0],4096), np.tile(c0[:,1],4096), k); b = dec_one(np.tile(c1[:,0],4096), np.tile(c1[:,1],4096), k)
        with torch.no_grad(): sc[st:st+4096] = net.logits(torch.from_numpy(to_bits(*a,*b)).to("mps")).cpu().numpy().reshape(4096,64).sum(1)
    t = int(x.final_round_subkey); ct = cls[t]
    inc = sc[cls == ct]; within = float((inc > sc[t]).sum() + 1)
    z = sc.copy()
    for c in range(4):
        sel = cls == c; z[sel] = (sc[sel] - sc[sel].mean()) / sc[sel].std()
    zr = float((z > z[t]).sum() + 1)
    means = [round(float(sc[cls == c].mean()), 2) for c in range(4)]
    rows.append({"key": i+1, "bits11_10": int(ct), "within_class_rank": within, "znorm_rank": zr, "class_means": means})
    print(json.dumps(rows[-1]), flush=True)
print(json.dumps({"within_top8": sum(r["within_class_rank"] <= 8 for r in rows), "within_top32": sum(r["within_class_rank"] <= 32 for r in rows),
                  "znorm_top32": sum(r["znorm_rank"] <= 32 for r in rows), "znorm_top64": sum(r["znorm_rank"] <= 64 for r in rows)}))
