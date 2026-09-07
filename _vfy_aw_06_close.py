import numpy as np, pandas as pd
from collections import defaultdict
pd.set_option("display.width",270); rng=np.random.default_rng(23); W=20
d=pd.read_csv("_scratch/_r4c1_margin_integral.csv")
aw=pd.read_csv("_scratch/_vfy_aw_fullaw.csv")[["seq","frame","track_id","m_trk","m_det"]]
d=d.merge(aw,on=["seq","track_id","frame"],how="left").sort_values(["seq","track_id","frame"]).reset_index(drop=True)
sw=pd.read_csv("_scratch/_r21_switch_events_full.csv"); sw["sid"]=sw.seq+"|"+sw.frame.astype(str)+"|"+sw.gt_id.astype(str)
ridx={k:i for i,k in enumerate(zip(d.seq,d.track_id,d.frame))}
trk=pd.factorize(d.seq+"|"+d.track_id.astype(str))[0]
order=np.argsort(trk,kind="stable"); groups=np.split(order,np.flatnonzero(np.diff(trk[order]))+1)
win_rows=defaultdict(list); row2sids=defaultdict(set)
for r in sw.itertuples():
    for hid in (r.new_hid,r.old_hid):
        if pd.isna(hid): continue
        for dt in range(1,W+1):
            i=ridx.get((r.seq,int(hid),r.frame-dt))
            if i is not None: win_rows[r.sid].append((i,dt)); row2sids[i].add(r.sid)
N=len(d); pm=np.zeros(N,bool); pm[list(row2sids.keys())]=True
cls=d.cname.to_numpy()
def fbc(v,n):
    o=np.argsort(np.where(np.isfinite(v),-v,np.inf),kind="stable"); f=np.zeros(len(v),bool); f[o[:n]]=True; return f
def reach(f):
    g=set()
    for i in np.flatnonzero(f): g|=row2sids.get(i,set())
    return len(g)
def zsc_by_class(v):
    z=np.full(N,np.nan)
    for c in np.unique(cls):
        m=cls==c; x=v[m]; z[m]=(x-np.nanmean(x))/(np.nanstd(x)+1e-12)
    return z
AW=d.AW.to_numpy(float); SD=d.std30.to_numpy(float)
COMB=0.5*zsc_by_class(AW)+0.5*zsc_by_class(SD)

print("=== Q3: within-track autocorrelation of each signal (lag 1 and lag 10) ===")
for nm,v in (("AW",AW),("AW_full",-(np.minimum(d.m_trk,.5)+np.minimum(d.m_det,.5)).to_numpy()/2),("std30",SD),("comb",COMB)):
    a1=[];a10=[]
    for g in groups:
        x=pd.Series(v[g]).astype(float)
        if x.notna().sum()>40:
            a1.append(x.autocorr(1)); a10.append(x.autocorr(10))
    print("  %-8s lag1 rho=%.3f   lag10 rho=%.3f  (median over %d tracks)"
          %(nm,np.nanmedian(a1),np.nanmedian(a10),len(a1)))

print("\n=== reach at matched budget, WITH both nulls (400 draws) ===")
rows=[]
for p in (0.05,0.10,0.20):
    n=int(round(p*N))
    uni=np.array([reach(fbc(rng.random(N),n)) for _ in range(400)])
    r={"rate":f"{p:.0%}","uniform-random null":"%.1f±%.1f"%(uni.mean(),uni.std())}
    for nm,v in (("AW",AW),("std30",SD),("0.5AW+0.5std30",COMB)):
        f=fbc(v,n); obs=reach(f)
        nl=[]
        for _ in range(200):
            g=np.zeros(N,bool)
            for grp in groups:
                k=int(f[grp].sum())
                if k: g[rng.choice(grp,k,replace=False)]=True
            nl.append(reach(g))
        nl=np.array(nl)
        r[nm]="%d (shuffle %.0f, z=%+.1f)"%(obs,nl.mean(),(obs-nl.mean())/max(nl.std(),1e-9))
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== the metric I consider correct: within-track AUC at lead >= 5 (no onset leak, no")
print("    between-track confound), and pooled AUC over the whole 20f window ===")
def auc(p,n):
    p=np.asarray(p,float);n=np.asarray(n,float);p=p[np.isfinite(p)];n=n[np.isfinite(n)]
    if len(p)<3 or len(n)<3: return np.nan
    r=pd.Series(np.concatenate([p,n])).rank().to_numpy();n1,n0=len(p),len(n)
    return (r[:n1].sum()-n1*(n1+1)/2)/(n1*n0)
def within_auc(v,mask,neg):
    U=[];Wt=[]
    for g in groups:
        a=auc(v[g][mask[g]],v[g][neg[g]])
        if np.isnan(a): continue
        w=np.isfinite(v[g][mask[g]]).sum()*np.isfinite(v[g][neg[g]]).sum(); U.append(a*w);Wt.append(w)
    return sum(U)/sum(Wt) if Wt else np.nan
# pre-window restricted to leads 5..20
m5=np.zeros(N,bool)
for sid,lst in win_rows.items():
    for i,l in lst:
        if l>=5: m5[i]=True
neg=~pm
rows=[]
for nm,v in (("AW (yours)",AW),("AW_full",-(np.minimum(d.m_trk,.5)+np.minimum(d.m_det,.5)).to_numpy()/2),
             ("std30",SD),("resid",d.resid.to_numpy(float)),("mean30",d.mean30.to_numpy(float)),("0.5AW+0.5std30",COMB)):
    rows.append({"signal":nm,"pooled AUC lead1-20":round(auc(v[pm],v[neg]),4),
                 "pooled AUC lead5-20":round(auc(v[m5],v[neg]),4),
                 "WITHIN-track AUC lead5-20":round(within_auc(v,m5,neg),4),
                 "reach@10%":reach(fbc(v,int(.10*N))),"reach@20%":reach(fbc(v,int(.20*N)))})
print(pd.DataFrame(rows).to_string(index=False))
print("\n(uniform-random reach@10%% ~200, @20%% ~230 -> the reach column is below chance for every signal)")
