"""Which round-key bits separate neural-success keys (top-32) from failures? 128 demo keys, 8 rounds."""
import json, math
from gohr_torch import expand_key
from neural_sat_speck.data.attack import AttackInstanceConfig, generate_attack_instances
from math import comb
def fisher_exact(t):
    (a, b), (c, d) = t
    n1, n2, k, n = a + b, c + d, a + c, a + b + c + d
    pmf = lambda x: comb(n1, x) * comb(n2, k - x) / comb(n, k)
    p0 = pmf(a)
    return None, min(1.0, sum(pmf(x) for x in range(max(0, k - n2), min(k, n1) + 1) if pmf(x) <= p0 * (1 + 1e-9)))
ranks = json.loads(open("neural_r8_n128.log").read().strip().splitlines()[-1])["ranks"]
inst = generate_attack_instances(AttackInstanceConfig(instance_count=128, pair_count=64, rounds=8, seed=999999, split_role="test")).instances
ok = [r <= 32 for r in ranks]
ks = [expand_key(list(x.master_key), 8) for x in inst]
res = []
for rnd in range(8):
    for b in range(16):
        bit = [(k[rnd] >> b) & 1 for k in ks]
        t = [[sum(1 for o, v in zip(ok, bit) if o and v), sum(1 for o, v in zip(ok, bit) if o and not v)],
             [sum(1 for o, v in zip(ok, bit) if not o and v), sum(1 for o, v in zip(ok, bit) if not o and not v)]]
        res.append((fisher_exact(t)[1], rnd + 1, b, t))
res.sort()
for p, r, b, t in res[:10]:
    print(f"k{r} bit{b:2d}  p={p:.2e}  success(bit=1,bit=0)={t[0]}  fail(bit=1,bit=0)={t[1]}")
print("Bonferroni 0.05 threshold:", 0.05 / 128)
