"""Bonus attack the user did not ask for: within track, does DISPERSION (std) add anything
over the rolling LEVEL (mean) of the same residual?"""
import numpy as np, pandas as pd
from collections import defaultdict
pd.set_option("display.width",250); W=20
d=pd.read_csv("_scratch/_r4c1_margin_integral.csv")
f10=pd.read_csv("_scratch/_r21_features.csv")[["seq","track_id","frame","std10","mean10"]]
d=d.merge(f10,on=["seq","track_id","frame"],how="left").sort_values(["seq","track_id","frame"]).reset_index(drop=True)
sw=pd.read_csv("_scratch/_r21_switch_events_full.csv")
ridx={k:i for i,k in enumerate(zip(d.seq,d.track_id,d.frame))}
trk=pd.factorize(d.seq+"|"+d.track_id.astype(str))[0]
groups=np.split(np.argsort(trk,kind="stable"),np.flatnonzero(np.diff(trk[np.argsort(trk,kind='stable')]))+1)
N=len(d)
pairs=[]
for r in sw.itertuples():
    for hid in (r.new_hid,r.old_hid):
        if pd.isna(hid): continue
        pairs.append((r.seq,int(hid),r.frame))
pm=np.zeros(N,bool)
for s,t,fr in pairs:
    for dt in range(1,W+1):
        i=ridx.get((s,t,fr-dt))
        if i is not None: pm[i]=True
def auc(p,n):
    p=np.asarray(p,float);n=np.asarray(n,float);p=p[np.isfinite(p)];n=n[np.isfinite(n)]
    if len(p)<3 or len(n)<3: return np.nan
    r=pd.Series(np.concatenate([p,n])).rank().to_numpy();n1,n0=len(p),len(n)
    return (r[:n1].sum()-n1*(n1+1)/2)/(n1*n0)
def within(v,m,neg):
    U=[];Wt=[]
    for g in groups:
        a=auc(v[g][m[g]],v[g][neg[g]])
        if np.isnan(a): continue
        w=np.isfinite(v[g][m[g]]).sum()*np.isfinite(v[g][neg[g]]).sum(); U.append(a*w);Wt.append(w)
    return sum(U)/sum(Wt) if Wt else np.nan
cols=["std30","mean30","std10","mean10","resid","AW"]
rows=[]
for dl in (1,3,5,10,15,20,30):
    m=np.zeros(N,bool)
    sel=[ridx.get((s,t,fr-dl)) for s,t,fr in pairs]; sel=[i for i in sel if i is not None]
    m[sel]=True; neg=(~pm)|m; neg=neg&~m
    r={"lead":dl,"n_pos":len(sel)}
    for c in cols: r[c+" (within)"]=round(within(d[c].to_numpy(float),m,neg),4)
    rows.append(r)
print("=== WITHIN-TRACK AUC at fixed lead (negatives = frames of the same track outside ANY pre-window) ===")
print(pd.DataFrame(rows).to_string(index=False))
print("\npartial: within-track AUC of std30 among frames stratified by mean30 tercile (lead 5-20)")
m5=np.zeros(N,bool)
for s,t,fr in pairs:
    for dt in range(5,W+1):
        i=ridx.get((s,t,fr-dt))
        if i is not None: m5[i]=True
neg=~pm
q=pd.qcut(d.mean30.rank(method="first"),3,labels=["low","mid","high"])
for lab in ["low","mid","high"]:
    sub=(q==lab).to_numpy()
    v=d.std30.to_numpy(float); va=d.AW.to_numpy(float)
    print("  mean30 %-5s  std30 pooled AUC=%.4f  AW pooled AUC=%.4f  (n_pos=%d)"
          %(lab,auc(v[sub&m5],v[sub&neg]),auc(va[sub&m5],va[sub&neg]),int((sub&m5).sum())))
print("\nspearman(std30, mean30) = %.3f"%d.std30.corr(d.mean30,method="spearman"))
