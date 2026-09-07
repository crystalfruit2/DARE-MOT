"""Within-track structure: is either signal a TEMPORAL forewarning, or only a
between-track/scene discriminator? Plus null-corrected lead-stratified reach."""
import numpy as np, pandas as pd
from collections import defaultdict
pd.set_option("display.width",260)
rng=np.random.default_rng(1)
W=20
d=pd.read_csv("_scratch/_r4c1_margin_integral.csv")
f10=pd.read_csv("_scratch/_r21_features.csv")[["seq","track_id","frame","std10","mean10","tracklet_len","n_det","n_pool"]]
d=d.merge(f10,on=["seq","track_id","frame"],how="left")
d=d.sort_values(["seq","track_id","frame"]).reset_index(drop=True)
sw=pd.read_csv("_scratch/_r21_switch_events_full.csv"); sw["sid"]=sw.seq+"|"+sw.frame.astype(str)+"|"+sw.gt_id.astype(str)
key=list(zip(d.seq,d.track_id,d.frame)); ridx={k:i for i,k in enumerate(key)}
trk=pd.factorize(d.seq+"|"+d.track_id.astype(str))[0]
win_rows=defaultdict(list); row2sids=defaultdict(set)
for r in sw.itertuples():
    for hid in (r.new_hid,r.old_hid):
        if pd.isna(hid): continue
        for dt in range(1,W+1):
            i=ridx.get((r.seq,int(hid),r.frame-dt))
            if i is not None: win_rows[r.sid].append((i,dt)); row2sids[i].add(r.sid)
pos_mask=np.zeros(len(d),bool); pos_mask[list(row2sids.keys())]=True

def auc(pos,neg):
    pos=np.asarray(pos,float); neg=np.asarray(neg,float)
    pos=pos[np.isfinite(pos)]; neg=neg[np.isfinite(neg)]
    if len(pos)<3 or len(neg)<3: return np.nan
    r=pd.Series(np.concatenate([pos,neg])).rank().to_numpy()
    n1,n0=len(pos),len(neg)
    return (r[:n1].sum()-n1*(n1+1)/2)/(n1*n0)

print("=== E4a: POOLED (global) AUC of pre-window frames vs all other frames ===")
for c in ("AW","AW_s5","AW_s10","std30","std10","mean30","resid"):
    v=d[c].to_numpy(float)
    print("  %-8s pooled AUC = %.4f"%(c,auc(v[pos_mask],v[~pos_mask])))

print("\n=== E4b: WITHIN-TRACK AUC (pre-window frames vs OTHER frames of the SAME track) ===")
print("this removes every between-track / between-scene effect; 0.5 = no temporal forewarning")
order=np.argsort(trk,kind="stable"); bounds=np.flatnonzero(np.diff(trk[order]))+1
groups=np.split(order,bounds)
for c in ("AW","AW_s5","AW_s10","std30","std10","mean30","resid"):
    v=d[c].to_numpy(float)
    Us=[]; Ns=[]; npos_tot=0; ntr=0
    for grp in groups:
        p=v[grp][pos_mask[grp]]; n=v[grp][~pos_mask[grp]]
        a=auc(p,n)
        if np.isnan(a): continue
        w=len(p[np.isfinite(p)])*len(n[np.isfinite(n)])
        Us.append(a*w); Ns.append(w); npos_tot+=len(p); ntr+=1
    print("  %-8s within-track AUC = %.4f   (%d tracks, %d pos frames)"%(c,sum(Us)/sum(Ns),ntr,npos_tot))

print("\n=== E4c: within-track AUC at FIXED lead offsets (circularity / onset-leak test) ===")
pairs=[]
for r in sw.itertuples():
    for hid in (r.new_hid,r.old_hid):
        if pd.isna(hid): continue
        pairs.append((r.seq,int(hid),r.frame))
rows=[]
for dl in (1,2,3,5,8,10,15,20,30,40):
    sel=[ridx.get((s,t,fr-dl)) for s,t,fr in pairs]; sel=[i for i in sel if i is not None]
    m=np.zeros(len(d),bool); m[sel]=True
    # exclude all other pre-window frames from the negatives so leads don't contaminate
    negok=~pos_mask|m
    r={"lead":dl,"n_pos":len(sel)}
    for c in ("AW","std30"):
        v=d[c].to_numpy(float)
        r["%s pooled"%c]=round(auc(v[m],v[negok&~m]),4)
        Us=[];Ns=[]
        for grp in groups:
            p=v[grp][m[grp]]; n=v[grp][negok[grp]&~m[grp]]
            a=auc(p,n)
            if np.isnan(a): continue
            w=len(p[np.isfinite(p)])*len(n[np.isfinite(n)]); Us.append(a*w); Ns.append(w)
        r["%s within"%c]=round(sum(Us)/sum(Ns),4) if Ns else np.nan
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))
