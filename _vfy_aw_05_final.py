"""Head-to-head with the FULL (symmetric, clipped) Deep OC-SORT AW, + class-conditional."""
import numpy as np, pandas as pd
from collections import defaultdict
pd.set_option("display.width",270); rng=np.random.default_rng(11); W=20
d=pd.read_csv("_scratch/_r4c1_margin_integral.csv")
aw=pd.read_csv("_scratch/_vfy_aw_fullaw.csv")[["seq","frame","track_id","m_trk","m_det","AW_full","AW_trk","AW_det","n_trk"]]
f10=pd.read_csv("_scratch/_r21_features.csv")[["seq","track_id","frame","std10","tracklet_len"]]
d=d.merge(aw,on=["seq","track_id","frame"],how="left").merge(f10,on=["seq","track_id","frame"],how="left")
d=d.sort_values(["seq","track_id","frame"]).reset_index(drop=True)
print("proxy check: corr(cached reid_margin, replayed m_trk) pearson=%.6f  max|diff|=%.2e"
      %(d.reid_margin.corr(d.m_trk), np.abs(d.reid_margin-d.m_trk).max()))
print("spearman(AW_trk-proxy, AW_full) = %.4f ; spearman(AW_det, AW_trk) = %.4f"
      %(d.AW.corr(d.AW_full,method="spearman"), d.AW_det.corr(d.AW_trk,method="spearman")))
print("frac m_trk clipped at 0.5: %.4f%% ; m_det clipped: %.4f%%"
      %(100*(d.m_trk>0.5).mean(),100*(d.m_det>0.5).mean()))

sw=pd.read_csv("_scratch/_r21_switch_events_full.csv"); sw["sid"]=sw.seq+"|"+sw.frame.astype(str)+"|"+sw.gt_id.astype(str)
ridx={k:i for i,k in enumerate(zip(d.seq,d.track_id,d.frame))}
trk=pd.factorize(d.seq+"|"+d.track_id.astype(str))[0]
order=np.argsort(trk,kind="stable"); groups=np.split(order,np.flatnonzero(np.diff(trk[order]))+1)
win_rows=defaultdict(list); row2sids=defaultdict(set); sid_cls={}
for r in sw.itertuples():
    sid_cls[r.sid]=r.cls
    for hid in (r.new_hid,r.old_hid):
        if pd.isna(hid): continue
        for dt in range(1,W+1):
            i=ridx.get((r.seq,int(hid),r.frame-dt))
            if i is not None: win_rows[r.sid].append((i,dt)); row2sids[i].add(r.sid)
N=len(d); CEIL=len(win_rows)
pm=np.zeros(N,bool); pm[list(row2sids.keys())]=True
def fbc(v,n):
    o=np.argsort(np.where(np.isfinite(v),-v,np.inf),kind="stable"); f=np.zeros(len(v),bool); f[o[:n]]=True; return f
def reach(f):
    g=set()
    for i in np.flatnonzero(f): g|=row2sids.get(i,set())
    return g
def auc(p,n):
    p=np.asarray(p,float);n=np.asarray(n,float);p=p[np.isfinite(p)];n=n[np.isfinite(n)]
    if len(p)<3 or len(n)<3: return np.nan
    r=pd.Series(np.concatenate([p,n])).rank().to_numpy();n1,n0=len(p),len(n)
    return (r[:n1].sum()-n1*(n1+1)/2)/(n1*n0)
def within_auc(v,mask,negmask):
    U=[];Wt=[]
    for g in groups:
        a=auc(v[g][mask[g]],v[g][negmask[g]])
        if np.isnan(a): continue
        w=np.isfinite(v[g][mask[g]]).sum()*np.isfinite(v[g][negmask[g]]).sum(); U.append(a*w);Wt.append(w)
    return sum(U)/sum(Wt) if Wt else np.nan

SIG={"AW (trk half, yours)":d.AW.to_numpy(float),"AW_full (both halves)":-((np.minimum(d.m_trk,.5)+np.minimum(d.m_det,.5))/2).to_numpy(float),
     "AW_det (det half only)":d.AW_det.to_numpy(float),"std30":d.std30.to_numpy(float)}
print("\n=== Q1: does the missing det-wise half change anything? ===")
rows=[]
for nm,v in SIG.items():
    r={"signal":nm,"pooled AUC(pre-window)":round(auc(v[pm],v[~pm]),4),
       "within-track AUC":round(within_auc(v,pm,~pm),4)}
    for p in (0.05,0.10,0.20):
        r[f"reach@{p:.0%}"]=len(reach(fbc(v,int(round(p*N)))))
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))
print("ceiling=%d of 282; uniform-random reach@10%% = 200.5+-5.8 (from _vfy_aw_03)"%CEIL)

print("\n=== Q5: class-conditional. per-class within-track & pooled AUC (20f pre-window) ===")
rows=[]
for c,sub in d.groupby("cname"):
    gi=sub.index.to_numpy()
    npos=int(pm[gi].sum())
    if npos<40: 
        rows.append({"class":c,"n_rows":len(sub),"n_pos":npos,"note":"too few"}); continue
    r={"class":c,"n_rows":len(sub),"n_pos":npos}
    for nm,v in (("AW",d.AW.to_numpy(float)),("AW_full",SIG["AW_full (both halves)"]),("std30",d.std30.to_numpy(float))):
        r[nm]=round(auc(v[gi][pm[gi]],v[gi][~pm[gi]]),4)
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))
sw_by_cls=sw.cls.value_counts().to_dict(); print("switches per class:",sw_by_cls)

print("\n=== Q5b: class-conditional COMBINATION at matched total budget (reach + lift) ===")
print("per-class z-scored signals, weight w on std30 vs (1-w) on AW, chosen PER CLASS;")
print("budget spent per class proportional to that class's share of rows (so total budget matched)\n")
cls=d.cname.to_numpy()
def zsc(v,mask):
    x=v[mask]; mu=np.nanmean(x); sd=np.nanstd(x); return (v-mu)/(sd+1e-12)
Z={}
for nm,v in (("AW",d.AW.to_numpy(float)),("std30",d.std30.to_numpy(float))):
    z=np.full(N,np.nan)
    for c in np.unique(cls):
        m=cls==c; z[m]=zsc(v,m)[m]
    Z[nm]=z
best=None; res=[]
for w in (0.0,0.25,0.5,0.75,1.0):
    comb=(1-w)*Z["AW"]+w*Z["std30"]
    r={"w(std30)":w,"pooled AUC":round(auc(comb[pm],comb[~pm]),4),"within AUC":round(within_auc(comb,pm,~pm),4)}
    for p in (0.05,0.10,0.20): r[f"reach@{p:.0%}"]=len(reach(fbc(comb,int(round(p*N)))))
    res.append(r)
print(pd.DataFrame(res).to_string(index=False))
# per-class best-w oracle combination
print("\n oracle per-class weight (pedestrian uses w_p, vehicles use w_v), matched budget:")
ped=cls=="pedestrian"
rows=[]
for wp in (0.0,0.5,1.0):
    for wv in (0.0,0.5,1.0):
        comb=np.where(ped,(1-wp)*Z["AW"]+wp*Z["std30"],(1-wv)*Z["AW"]+wv*Z["std30"])
        rows.append({"w_ped":wp,"w_veh":wv,"pooled AUC":round(auc(comb[pm],comb[~pm]),4),
                     "reach@10%":len(reach(fbc(comb,int(round(.10*N))))),
                     "reach@20%":len(reach(fbc(comb,int(round(.20*N)))))})
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== which switches does each signal uniquely reach @10% (class breakdown) ===")
rA=reach(fbc(d.AW.to_numpy(float),int(.10*N))); rS=reach(fbc(d.std30.to_numpy(float),int(.10*N)))
rF=reach(fbc(SIG["AW_full (both halves)"],int(.10*N)))
onlyS=rS-rA; onlyA=rA-rS
print("  only std30 (%d): %s"%(len(onlyS),pd.Series([sid_cls[s] for s in onlyS]).value_counts().to_dict()))
print("  only AW    (%d): %s"%(len(onlyA),pd.Series([sid_cls[s] for s in onlyA]).value_counts().to_dict()))
print("  AW_full reach@10%% = %d (vs AW-proxy %d)"%(len(rF),len(rA)))
print("\n=== the 30 switches with NO pre-window rows at all ===")
miss=set(sw.sid)-set(win_rows)
print(pd.Series([sid_cls[s] for s in miss]).value_counts().to_dict())
print("  per seq:",pd.Series([s.split("|")[0] for s in miss]).value_counts().to_dict())
