import numpy as np, pandas as pd
from collections import defaultdict
pd.set_option("display.width",250); rng=np.random.default_rng(3); W=20
d=pd.read_csv("_scratch/_r4c1_margin_integral.csv")
aw=pd.read_csv("_scratch/_vfy_aw_fullaw.csv")[["seq","frame","track_id","m_trk","m_det"]]
f10=pd.read_csv("_scratch/_r21_features.csv")[["seq","track_id","frame","std10","mean10"]]
d=d.merge(aw,on=["seq","track_id","frame"],how="left").merge(f10,on=["seq","track_id","frame"],how="left")
d=d.sort_values(["seq","track_id","frame"]).reset_index(drop=True)
sw=pd.read_csv("_scratch/_r21_switch_events_full.csv"); sw["sid"]=sw.seq+"|"+sw.frame.astype(str)+"|"+sw.gt_id.astype(str)
ridx={k:i for i,k in enumerate(zip(d.seq,d.track_id,d.frame))}
trk=pd.factorize(d.seq+"|"+d.track_id.astype(str))[0]
row2sids=defaultdict(set)
for r in sw.itertuples():
    for hid in (r.new_hid,r.old_hid):
        if pd.isna(hid): continue
        for dt in range(1,W+1):
            i=ridx.get((r.seq,int(hid),r.frame-dt))
            if i is not None: row2sids[i].add(r.sid)
N=len(d); n=int(.10*N)
def fbc(v,k):
    o=np.argsort(np.where(np.isfinite(v),-v,np.inf),kind="stable"); f=np.zeros(len(v),bool); f[o[:k]]=True; return f
def reach(f):
    g=set()
    for i in np.flatnonzero(f): g|=row2sids.get(i,set())
    return len(g)
sigs={"AW":d.AW.to_numpy(float),"AW_full":-(np.minimum(d.m_trk,.5)+np.minimum(d.m_det,.5)).to_numpy()/2,
      "std30":d.std30.to_numpy(float),"std10":d.std10.to_numpy(float),"mean30":d.mean30.to_numpy(float),
      "mean10":d.mean10.to_numpy(float),"resid":d.resid.to_numpy(float),"d_second(-)":-d.d_second.to_numpy(float),
      "area":d.area.to_numpy(float),"random":rng.random(N)}
rows=[]
for nm,v in sigs.items():
    f=fbc(v,n); idx=np.flatnonzero(f)
    runs=1
    for a,b in zip(idx[:-1],idx[1:]):
        if not(b==a+1 and trk[a]==trk[b]): runs+=1
    ac=[]
    for g in np.split(np.argsort(trk,kind="stable"),np.flatnonzero(np.diff(trk[np.argsort(trk,kind='stable')]))+1):
        s=pd.Series(v[g]).astype(float)
        if s.notna().sum()>40: ac.append(s.autocorr(1))
    rows.append({"signal":nm,"lag1 autocorr":round(np.nanmedian(ac),3),"flag runs":runs,
                 "mean run len":round(n/runs,2),"distinct tracks":len(np.unique(trk[f])),
                 "reach@10%":reach(f)})
t=pd.DataFrame(rows).sort_values("reach@10%",ascending=False)
print("=== reach@10% is a function of flag SCATTER, not of predictive quality ===")
print(t.to_string(index=False))
print("\nspearman(reach, #runs)      = %+.3f"%t["reach@10%"].corr(t["flag runs"],method="spearman"))
print("spearman(reach, lag1 rho)   = %+.3f"%t["reach@10%"].corr(t["lag1 autocorr"],method="spearman"))
print("spearman(reach, #tracks)    = %+.3f"%t["reach@10%"].corr(t["distinct tracks"],method="spearman"))
