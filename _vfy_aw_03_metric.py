"""Is 'reach' informative at all? Random nulls + null-corrected reach + a threshold-free
operating metric appropriate to a CONTINUOUS per-track lambda modulator."""
import numpy as np, pandas as pd
from collections import defaultdict
pd.set_option("display.width",260)
rng=np.random.default_rng(7); W=20
d=pd.read_csv("_scratch/_r4c1_margin_integral.csv")
f10=pd.read_csv("_scratch/_r21_features.csv")[["seq","track_id","frame","std10","tracklet_len","n_det","n_pool","n_valid"]].rename(columns={"n_valid":"nv2"})
d=d.merge(f10,on=["seq","track_id","frame"],how="left").sort_values(["seq","track_id","frame"]).reset_index(drop=True)
sw=pd.read_csv("_scratch/_r21_switch_events_full.csv"); sw["sid"]=sw.seq+"|"+sw.frame.astype(str)+"|"+sw.gt_id.astype(str)
ridx={k:i for i,k in enumerate(zip(d.seq,d.track_id,d.frame))}
trk=pd.factorize(d.seq+"|"+d.track_id.astype(str))[0]
win_rows=defaultdict(list); row2sids=defaultdict(set)
for r in sw.itertuples():
    for hid in (r.new_hid,r.old_hid):
        if pd.isna(hid): continue
        for dt in range(1,W+1):
            i=ridx.get((r.seq,int(hid),r.frame-dt))
            if i is not None: win_rows[r.sid].append((i,dt)); row2sids[i].add(r.sid)
CEIL=len(win_rows); N=len(d)
order=np.argsort(trk,kind="stable"); groups=np.split(order,np.flatnonzero(np.diff(trk[order]))+1)
def flag_by_count(v,n):
    fin=np.isfinite(v); o=np.argsort(np.where(fin,-v,np.inf),kind="stable")
    f=np.zeros(len(v),bool); f[o[:n]]=True; return f
def reach(f):
    g=set()
    for i in np.flatnonzero(f): g|=row2sids.get(i,set())
    return len(g)
AW=d.AW.to_numpy(float); SD=d.std30.to_numpy(float)

print("=== E5: how much of 'reach' is information? ===")
print("nulls: (U) uniform random flags over all rows; (T) within-track shuffle of the real flags\n")
rows=[]
for p in (0.02,0.05,0.10,0.20):
    n=int(round(p*N))
    fA=flag_by_count(AW,n); fS=flag_by_count(SD,n)
    uni=[]
    for _ in range(200):
        g=np.zeros(N,bool); g[rng.choice(N,n,replace=False)]=True; uni.append(reach(g))
    uni=np.array(uni)
    r={"rate":f"{p:.0%}","AW":reach(fA),"std30":reach(fS),"uniform-random":"%.1f±%.1f"%(uni.mean(),uni.std())}
    for nm,f in (("AW",fA),("std30",fS)):
        nl=[]
        for _ in range(200):
            g=np.zeros(N,bool)
            for grp in groups:
                k=int(f[grp].sum())
                if k: g[rng.choice(grp,k,replace=False)]=True
            nl.append(reach(g))
        nl=np.array(nl); r[nm+" excess/shuffle"]="%+.1f (z=%+.1f)"%(reach(f)-nl.mean(),(reach(f)-nl.mean())/max(nl.std(),1e-9))
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))
print("ceiling (flag everything) = %d of 282"%CEIL)

print("\n=== E6: threshold-free operating metrics for a CONTINUOUS per-track lambda ===")
print("g = global rank of the signal in [0,1]; a modulator applies Delta*g on EVERY frame.")
print("  LIFT_sum = mean(g | pre-window) / mean(g | all)      [total perturbation-normalised benefit]")
print("  LIFT_max = mean_over_switches(max g in window) / mean(g|all)  [saturating: one frame suffices]")
print("  and the same computed WITHIN TRACK (g ranked inside each track) to strip the between-track part\n")
res=[]
for nm,v in (("AW",AW),("AW_s5",d.AW_s5.to_numpy(float)),("std30",SD),("std10",d.std10.to_numpy(float)),
             ("mean30",d.mean30.to_numpy(float)),("resid",d.resid.to_numpy(float))):
    s=pd.Series(v); g=s.rank(pct=True).to_numpy()
    g=np.where(np.isfinite(v),g,np.nan)
    base=np.nanmean(g)
    pm=np.zeros(N,bool); pm[list(row2sids.keys())]=True
    lift_sum=np.nanmean(g[pm])/base
    mx=[np.nanmax([g[i] for i,_ in lst]) for lst in win_rows.values()]
    lift_max=np.nanmean(mx)/base
    # within-track ranks
    gw=np.full(N,np.nan)
    for grp in groups:
        vv=pd.Series(v[grp])
        if vv.notna().sum()>=3: gw[grp]=vv.rank(pct=True).to_numpy()
    basew=np.nanmean(gw)
    lw_sum=np.nanmean(gw[pm])/basew
    mxw=[np.nanmax([gw[i] for i,_ in lst]) for lst in win_rows.values()]
    lw_max=np.nanmean(mxw)/basew
    res.append({"signal":nm,"LIFT_sum":round(lift_sum,4),"LIFT_max":round(lift_max,4),
                "LIFT_sum within":round(lw_sum,4),"LIFT_max within":round(lw_max,4)})
print(pd.DataFrame(res).to_string(index=False))

print("\n=== E7: is AW just a crowding/density proxy? ===")
for c in ("n_det","n_pool","tracklet_len","area"):
    if c in d: print("  spearman(AW,%-12s) = %+.3f    spearman(std30,%-12s) = %+.3f"
        %(c,d.AW.corr(d[c],method="spearman"),c,d.std30.corr(d[c],method="spearman")))
print("  spearman(AW, std30) = %+.3f"%d.AW.corr(d.std30,method="spearman"))
print("  spearman(AW, resid) = %+.3f   spearman(std30, resid) = %+.3f"%(d.AW.corr(d.resid,method="spearman"),d.std30.corr(d.resid,method="spearman")))
