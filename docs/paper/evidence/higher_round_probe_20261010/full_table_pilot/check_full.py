import sys, time, numpy as np
sys.path.insert(0, "scripts")
from gohr_published_net import dec_one, enc
from neural_sat_speck.data.attack import AttackInstanceConfig, generate_attack_instances
llr = np.load(sys.argv[1])
# 1) brute check: score true key and 3 random keys pair-by-pair in pure python for key 0
x = generate_attack_instances(AttackInstanceConfig(instance_count=1, pair_count=64, rounds=8, seed=999999, split_role="test")).instances[0]
def ror(v,r): return ((v>>r)|(v<<(16-r)))&0xFFFF
def rol(v,r): return ((v<<r)|(v>>(16-r)))&0xFFFF
def dec(c,k):
    x,y=c; y=ror(y^x,2); x=rol(((x^k)-y)&0xFFFF,7); return x,y
def score(k):
    s=0
    for p in x.pairs:
        a=dec(p.ciphertext0,k); b=dec(p.ciphertext1,k); s+=int(llr[((a[0]^b[0])<<16)|(a[1]^b[1])])
    return s
true=x.final_round_subkey
print("true key", true, "score", score(true), "others", [score(k) for k in (1,2,12345,(true^1))])
# 2) check penultimate-round difference distribution: decrypt with true key, how many pairs land on positive-LLR entries
vals=[]
for p in x.pairs:
    a=dec(p.ciphertext0,true); b=dec(p.ciphertext1,true); vals.append(int(llr[((a[0]^b[0])<<16)|(a[1]^b[1])]))
print("true-key per-pair llr (1/16 bit):", sorted(vals)[:5], "...", sorted(vals)[-5:], "mean bits", np.mean(vals)/16)
# 3) timing of full ranking, cold-ish, repeated
c0=np.array([p.ciphertext0 for p in x.pairs],dtype=np.uint32); c1=np.array([p.ciphertext1 for p in x.pairs],dtype=np.uint32)
cands=np.arange(1<<16,dtype=np.uint32); CC=4096
t=time.perf_counter()
sc=np.empty(1<<16,dtype=np.int64)
for s in range(0,1<<16,CC):
    k=np.repeat(cands[s:s+CC],64)
    a=dec_one(np.tile(c0[:,0],CC),np.tile(c0[:,1],CC),k); b=dec_one(np.tile(c1[:,0],CC),np.tile(c1[:,1],CC),k)
    sc[s:s+CC]=llr[((a[0]^b[0])<<16)|(a[1]^b[1])].astype(np.int32).reshape(CC,64).sum(1)
print("rank seconds", round(time.perf_counter()-t,3), "vector score of true matches:", sc[true]==score(true))
print("table stats: mean llr bits", float(llr[::997].astype(np.float64).mean()/16), "frac at floor -128", float((llr[::997]==-128).mean()))
