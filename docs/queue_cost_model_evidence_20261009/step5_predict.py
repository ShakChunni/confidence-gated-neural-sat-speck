# Step 5 unconditional predictions (frozen-parameter model from N=32 study).
import sys, json, numpy as np
src=open(sys.argv[1]+'/fit_model.py').read().split('rng = np.random.default_rng(SEED)')[0]
sys.argv=['x',sys.argv[1]]; g={}; exec(compile(src,'fm','exec'), g)
KM,pooled,dists,p_recover,cells,arm_of,kq=[g[k] for k in ('KM','pooled','dists','p_recover','cells','arm_of','kq')]
ARMS=[('speck32_64',2,'cms_native_gauss',None),('speck32_64',2,'cms_native_no_gauss',None),
      ('speck32_64',2,'cms_native_gauss',1.5),('speck32_64',2,'cms_native_no_gauss',1.5),
      ('simon32_64',2,'cms_native_gauss',None),('simon32_64',2,'cms_native_no_gauss',None),
      ('simon32_64',2,'cms_native_gauss',10.0),('simon32_64',2,'cms_native_no_gauss',10.0)]
out={}
for scale in (0.9,1.0,1.1):
  for ci,d,p,tau in ARMS:
    arm=(ci,d,p)
    rkm=KM([(t*scale,c) for t,c in pooled(arm)]); skm=KM([(t*scale,c) for t,c in pooled(arm,field='s')]); _,_,o=dists(arm)
    rng=np.random.default_rng(59101)
    pk={c['key_index']:p_recover(float(c['prep_seconds']),kq(c),rkm,skm,o,rng,tau=tau) for c in cells if arm_of(c)==arm}
    keys=list(pk); pv=np.array([pk[k] for k in keys])
    # paired keys across arms: resample key indices identically per cipher via fixed seed
    rr=np.random.default_rng(59102 if ci=='speck32_64' else 59103)
    idx=rr.integers(0,32,(20000,32))
    counts=(rr.random((20000,32))<pv[idx]).sum(1)
    lo,med,hi=np.quantile(counts,[0.05,0.5,0.95])
    out.setdefault(f'{ci}|{p}|cap={tau}',{})[scale]=dict(expected=round(float(pv.sum()),2),pi90=[int(lo),int(hi)],median=int(med))
for k,v in out.items(): print(k, v)
json.dump(out,open(sys.argv[1]+'/step5_predictions.json','w'),indent=1)
