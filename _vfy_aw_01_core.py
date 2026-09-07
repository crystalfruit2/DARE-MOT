"""Adversarial re-verification of the AW-vs-std30 dominance claim. Offline only."""
import numpy as np, pandas as pd
from collections import defaultdict
pd.set_option("display.width", 260)
rng = np.random.default_rng(0)

W = 20
d = pd.read_csv("_scratch/_r4c1_margin_integral.csv")
sw = pd.read_csv("_scratch/_r21_switch_events_full.csv")
d = d.sort_values(["seq","track_id","frame"]).reset_index(drop=True)
sw["sid"] = sw.seq + "|" + sw.frame.astype(str) + "|" + sw.gt_id.astype(str)
assert sw.sid.nunique() == len(sw), sw.sid.nunique()
N_SW = len(sw)

# map (seq,tid,frame) -> row index
key = list(zip(d.seq, d.track_id, d.frame))
ridx = {k:i for i,k in enumerate(key)}
trk_of_row = pd.factorize(d.seq + "|" + d.track_id.astype(str))[0]

# For each switch: the set of row indices in its pre-window, per involved track
# win_rows[sid] = list of row indices ; also record lead (frames before switch)
win_rows = defaultdict(list)   # sid -> list of (rowidx, lead)
row2sids = defaultdict(set)
for r in sw.itertuples():
    for hid in (r.new_hid, r.old_hid):
        if pd.isna(hid): continue
        for dt in range(1, W+1):
            i = ridx.get((r.seq, int(hid), r.frame-dt))
            if i is not None:
                win_rows[r.sid].append((i, dt))
                row2sids[i].add(r.sid)

CEIL = len(win_rows)
print("=== SUPPORT / CEILING ===")
print("switches: %d ; switches with ANY track-frame in their 20-frame pre-window: %d (%.1f%%)"
      % (N_SW, CEIL, 100*CEIL/N_SW))
print("rows inside at least one pre-window: %d (%.2f%% of %d rows)"
      % (len(row2sids), 100*len(row2sids)/len(d), len(d)))
nrows_per_sw = pd.Series([len(v) for v in win_rows.values()])
print("pre-window rows per reachable switch:", nrows_per_sw.describe().to_dict())

SIGS = {"AW": d.AW.to_numpy(float), "std30": d.std30.to_numpy(float)}

def flag_by_count(v, n):
    """flag exactly n rows with the largest finite v (ties broken deterministically)."""
    fin = np.isfinite(v)
    order = np.argsort(np.where(fin, -v, np.inf), kind="stable")
    f = np.zeros(len(v), bool); f[order[:n]] = True
    return f

def reach(f, minlead=1, maxlead=W):
    got = set()
    for i in np.flatnonzero(f):
        for sid in row2sids.get(i, ()):
            got.add(sid)
    if minlead == 1 and maxlead == W:
        return len(got)
    # lead-restricted
    got = set()
    for sid, lst in win_rows.items():
        for i, lead in lst:
            if minlead <= lead <= maxlead and f[i]:
                got.add(sid); break
    return len(got)

print("\n=== E1: reach vs matched ABSOLUTE flag budget (counts, not percentiles) ===")
rows=[]
for p in (0.02,0.05,0.10,0.15,0.20,0.30,0.50,1.00):
    n = int(round(p*len(d)))
    r={"budget_frames":n, "rate":f"{p:.0%}"}
    for name,v in SIGS.items():
        f=flag_by_count(v,n); r[name]=reach(f)
    # union of half-budget each
    fa=flag_by_count(SIGS["AW"], n//2); fs=flag_by_count(SIGS["std30"], n//2)
    r["union50/50"]=reach(fa|fs)
    r["AW-std30"]=r["AW"]-r["std30"]
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))
print("(ceiling = %d)"%CEIL)

print("\n=== E2: LEAD-STRATIFIED reach -- drop flags closer than k frames to the switch ===")
print("if AW's edge is the switch's own onset leaking in, it collapses as k grows\n")
rows=[]
n = int(round(0.10*len(d)))
fA=flag_by_count(SIGS["AW"],n); fS=flag_by_count(SIGS["std30"],n)
for k in (1,2,3,5,8,11,15):
    rows.append({"flags allowed in lead range":f"[{k},20]",
                 "AW":reach(fA,k,W),"std30":reach(fS,k,W),
                 "diff":reach(fA,k,W)-reach(fS,k,W)})
print(pd.DataFrame(rows).to_string(index=False))
n2 = int(round(0.20*len(d)))
fA2=flag_by_count(SIGS["AW"],n2); fS2=flag_by_count(SIGS["std30"],n2)
print("\n same at 20% budget:")
rows=[{"lead range":f"[{k},20]","AW":reach(fA2,k,W),"std30":reach(fS2,k,W),
       "diff":reach(fA2,k,W)-reach(fS2,k,W)} for k in (1,3,5,8,11,15)]
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== E3: burstiness / track-coverage control ===")
for name,f in (("AW",fA),("std30",fS)):
    tr = trk_of_row[f]
    u,c = np.unique(tr, return_counts=True)
    # run lengths
    idx = np.flatnonzero(f)
    runs=1; 
    for a,b in zip(idx[:-1],idx[1:]):
        if not (b==a+1 and trk_of_row[a]==trk_of_row[b]): runs+=1
    print("  %-6s flags=%d  distinct tracks=%d  flags/track=%.1f  contiguous runs=%d  mean run len=%.2f"
          % (name, f.sum(), len(u), c.mean(), runs, f.sum()/runs))
    # how many distinct (track,switch) pre-windows are hit
    hitw=0; totw=0
    for sid,lst in win_rows.items():
        bytrk=defaultdict(list)
        for i,l in lst: bytrk[trk_of_row[i]].append(i)
        for t,ii in bytrk.items():
            totw+=1; hitw+= any(f[i] for i in ii)
    print("         pre-windows hit: %d / %d"%(hitw,totw))

print("\n  -- within-track permutation null (flags/track preserved, timing destroyed), 200 draws --")
for name,f in (("AW",fA),("std30",fS)):
    obs=reach(f)
    # group row indices by track
    order = np.argsort(trk_of_row, kind="stable")
    bounds = np.flatnonzero(np.diff(trk_of_row[order]))+1
    groups = np.split(order, bounds)
    nulls=[]
    for _ in range(200):
        g=np.zeros(len(d),bool)
        for grp in groups:
            k=int(f[grp].sum())
            if k: g[rng.choice(grp,k,replace=False)]=True
        nulls.append(reach(g))
    nulls=np.array(nulls)
    print("  %-6s observed %3d   null %.1f +- %.1f   excess %+.1f   z=%+.1f"
          % (name,obs,nulls.mean(),nulls.std(),obs-nulls.mean(),(obs-nulls.mean())/max(nulls.std(),1e-9)))
