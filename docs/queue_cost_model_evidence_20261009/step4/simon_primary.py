# Step 4 (EXPLORATORY): apply the frozen queue-cost model form to the primary SIMON
# replacement campaign (incremental wrapper, 45 s advisory candidate cap).
import sys, json, csv, numpy as np
from collections import defaultdict
sys.path.insert(0, sys.argv[1])
src=open(sys.argv[1]+'/step3/fit_model.py').read()
KMsrc=src[src.index('class KM'):src.index('def pooled')]
g={'np':np,'math':__import__('math')}; exec(KMsrc,g); KM=g['KM']
BUDGET=300.0; NDRAW=20000
d=json.load(open(sys.argv[2])); recs=d['records']
truth={r['instance_id']:tuple(r['full_queue'][0]) for r in recs if r['method']=='oracle_2key_n4'}
ARMS=['fully_classical_8parent','neural_classical_hybrid_8parent','fully_neural_1parent','random_control_1024','random_control_128','faithful_b1_hybrid']
cells=[]
for r in recs:
    if r['method'] not in ARMS: continue
    t=truth[r['instance_id']]
    q=[tuple(x) for x in r['full_queue']]
    k=q.index(t) if t in q else None
    P=r['complete_online_wall_seconds']-(r['solving_wall_seconds'] or 0)-(r['verification_wall_seconds'] or 0)
    rr,ss=[],[]
    atts=list(r['candidate_attempts'])+([r['in_flight_attempt']] if r['in_flight_attempt'] else [])
    for a in atts:
        is_t=(a['candidate_k8'],a['candidate_k9'])==t
        st=a['status']; tt=a['solve_wall_seconds']
        if st=='in_flight': obs=(tt,True)
        elif is_t and st!='sat': obs=(1e9,False)   # abandoned true candidate = loss
        else: obs=(tt,False)
        (ss if is_t else rr).append(obs)
    cells.append(dict(arm=r['method'],key=r['instance_id'],P=P,k=k,y=int(r['outcome']=='verified_recovery'),r=rr,s=ss,ntest=r['candidates_tested_count']))
def p_rec(P,k,rkm,skm,rng):
    if k is None or P>=BUDGET: return 0.0
    tot=np.full(NDRAW,P); left=k
    while left>0:
        m=min(64,left); tot+=rkm.sample(rng,(NDRAW,m)).sum(1); left-=m
        if np.all(tot>BUDGET): return 0.0
    return float(np.mean(tot+skm.sample(rng,NDRAW)<=BUDGET))
rng=np.random.default_rng(59201)
out=[]
for c in cells:
    o=[x for x in cells if x['arm']==c['arm'] and x['key']!=c['key']]
    rkm=KM([v for x in o for v in x['r']]); skm=KM([v for x in o for v in x['s']])
    p=p_rec(c['P'],c['k'],rkm,skm,rng); out.append(dict(arm=c['arm'],key=c['key'],y=c['y'],p=p,pred=int(p>=0.5),k=c['k'],P=c['P'],contained=int(c['k'] is not None)))
summ={}
for a in ARMS:
    rows=[x for x in out if x['arm']==a]; cs=[x for x in cells if x['arm']==a]
    rkm=KM([v for x in cs for v in x['r']]); skm=KM([v for x in cs for v in x['s']])
    ks=[x['k'] for x in cs if x['k'] is not None]
    summ[a]=dict(actual=sum(x['y'] for x in rows),expected=round(sum(x['p'] for x in rows),2),
      acc=sum(x['pred']==x['y'] for x in rows)/64, acc_B0=sum(x['contained']==x['y'] for x in rows)/64,
      contained=len(ks),P_med=round(float(np.median([x['P'] for x in cs])),1),
      r_med=rkm.quantile(.5), r_p90=rkm.quantile(.9), r_n=rkm.n, r_cens=rkm.ncens,
      s_med=skm.quantile(.5), s_n=skm.n, k_med=float(np.median(ks)) if ks else None,
      k_q=[float(np.quantile(ks,q)) for q in (.25,.75,.9)] if ks else None)
tot=sum(x['pred']==x['y'] for x in out); B0=sum(x['contained']==x['y'] for x in out)
maj={a:int(summ[a]['actual']*2>64) for a in ARMS}; B1=sum(maj[x['arm']]==x['y'] for x in out)
print('overall acc',tot,'/',len(out),round(tot/len(out),3),'B0',round(B0/len(out),3),'B1',round(B1/len(out),3))
for a,v in summ.items(): print(a,v)
json.dump(dict(summary=summ,overall=dict(correct=tot,n=len(out),B0=B0,B1=B1)),open(sys.argv[1]+'/step4/simon_primary_summary.json','w'),indent=1,default=str)
with open(sys.argv[1]+'/step4/simon_primary_predictions.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
