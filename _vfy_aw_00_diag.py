import numpy as np, pandas as pd
pd.set_option("display.width",250)
d = pd.read_csv("_scratch/_r4c1_margin_integral.csv")
print("rows", len(d), "cols", list(d.columns))
print("\nNaN counts:")
for c in ["reid_margin","AW","std30","mean30","area","n_valid","resid","d_second"]:
    print("  %-12s NaN=%6d  (%.1f%%)" % (c, d[c].isna().sum(), 100*d[c].isna().mean()))
print("\nn tracks:", d.groupby(['seq','track_id']).ngroups)
tl = d.groupby(['seq','track_id']).size()
print("track-frame count distribution:", tl.describe().to_dict())
print("tracks with >=30 frames:", (tl>=30).sum(), " frames in them:", tl[tl>=30].sum())
print("\nreid_margin describe:"); print(d.reid_margin.describe())
print("frac reid_margin<0:", (d.reid_margin<0).mean(), " frac >0.5:", (d.reid_margin>0.5).mean())
print("\nstd30 describe:"); print(d.std30.describe())
# flag counts at nanpercentile 90
for c in ["AW","std30"]:
    v=d[c].to_numpy(float); thr=np.nanpercentile(v,90)
    print("%s thr90=%.5f  flagged=%d  (%.2f%% of all rows, %.2f%% of non-nan)"%(c,thr,(v>=thr).sum(),100*(v>=thr).mean(),100*(v>=thr).sum()/np.isfinite(v).sum()))
    thr=np.nanpercentile(v,80); print("   thr80=%.5f flagged=%d"%(thr,(v>=thr).sum()))
