"""For each demo key at 8 rounds: top-5 neural candidates XOR the true final subkey, and best rank among k ^ {0,0x400,0x800,0xC00}."""
import numpy as np, torch
from gohr_torch import dec_one, expand_key, load_gohr, to_bits
from neural_sat_speck.data.attack import AttackInstanceConfig, generate_attack_instances
net = load_gohr("/Users/ashfaq/Dev/comp-sci-project/runtime/gohr_deep_speck/net7_small.npz").to("mps")
inst = generate_attack_instances(AttackInstanceConfig(instance_count=32, pair_count=64, rounds=8, seed=999999, split_role="test")).instances
cands = np.arange(65536, dtype=np.uint32)
for i, x in enumerate(inst):
    c0 = np.array([p.ciphertext0 for p in x.pairs], dtype=np.uint32); c1 = np.array([p.ciphertext1 for p in x.pairs], dtype=np.uint32)
    sc = np.empty(65536)
    for s in range(0, 65536, 4096):
        k = np.repeat(cands[s:s+4096], 64)
        a = dec_one(np.tile(c0[:,0],4096), np.tile(c0[:,1],4096), k); b = dec_one(np.tile(c1[:,0],4096), np.tile(c1[:,1],4096), k)
        with torch.no_grad(): sc[s:s+4096] = net.logits(torch.from_numpy(to_bits(*a,*b)).to("mps")).cpu().numpy().reshape(4096,64).sum(1)
    order = np.lexsort((cands, -sc)); rank = np.empty(65536, int); rank[order] = np.arange(1, 65537)
    t = x.final_round_subkey
    top = [f"{int(order[j]) ^ t:04x}" for j in range(5)]
    nb = {f"{m:03x}": int(rank[t ^ m]) for m in (0, 0x400, 0x800, 0xC00)}
    print(i+1, f"true bits11..10={(t>>10)&3:02b}", "top5 xor true:", top, "ranks k^m:", nb, flush=True)
