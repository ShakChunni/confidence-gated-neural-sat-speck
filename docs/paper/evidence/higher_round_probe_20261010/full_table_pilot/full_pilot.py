"""Pilot: full 32-bit sampled 7-round table as SPECK 8-round Stage 1 (demo seed 999999)."""
import json, sys, time
import numpy as np
sys.path.insert(0, "/Users/ashfaq/Dev/comp-sci-project/scripts")
from gohr_published_net import dec_one
from neural_sat_speck.data.attack import AttackInstanceConfig, generate_attack_instances
path, nkeys, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
t0 = time.perf_counter(); llr = np.load(path); load_s = time.perf_counter() - t0
inst = generate_attack_instances(AttackInstanceConfig(instance_count=nkeys, pair_count=64, rounds=8, seed=999999, split_role="test")).instances
cands = np.arange(1 << 16, dtype=np.uint32); CC = 4096; rows = []
for x in inst:
    t = time.perf_counter()
    c0 = np.array([p.ciphertext0 for p in x.pairs], dtype=np.uint32); c1 = np.array([p.ciphertext1 for p in x.pairs], dtype=np.uint32)
    sc = np.empty(1 << 16, dtype=np.int64)
    for s in range(0, 1 << 16, CC):
        k = np.repeat(cands[s:s + CC], 64)
        a = dec_one(np.tile(c0[:, 0], CC), np.tile(c0[:, 1], CC), k); b = dec_one(np.tile(c1[:, 0], CC), np.tile(c1[:, 1], CC), k)
        sc[s:s + CC] = llr[((a[0] ^ b[0]) << 16) | (a[1] ^ b[1])].astype(np.int32).reshape(CC, 64).sum(1)
    secs = time.perf_counter() - t
    order = np.lexsort((cands, -sc)); pos = int(np.where(order == x.final_round_subkey)[0][0]) + 1
    rows.append({"key": x.instance_id, "position": pos, "rank_s": round(secs, 2)})
pp = [r["position"] for r in rows]
res = {"table": path, "load_s": round(load_s, 1), "keys": nkeys, "top1": sum(p == 1 for p in pp), "top8": sum(p <= 8 for p in pp),
       "top16": sum(p <= 16 for p in pp), "top32": sum(p <= 32 for p in pp), "top64": sum(p <= 64 for p in pp),
       "median_position": float(np.median(pp)), "median_rank_s": float(np.median([r["rank_s"] for r in rows])), "per_key": rows}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "per_key"}))
print("positions:", sorted(pp))
