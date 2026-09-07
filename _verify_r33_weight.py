"""(b)+(c): re-derive the expected net IDSw, with and without the corrections."""
import numpy as np, pandas as pd
pd.set_option("display.width",250)

# ---- real gap distribution (recomputed in _verify_r33_idsw_locate.py) ----
import os.path as osp
TRK=osp.join("YOLOX_outputs","mc_dare_cv_rerun0803","track_results")
SEQS=["uav0000086_00000_v","uav0000117_02622_v","uav0000137_00458_v","uav0000182_00000_v",
      "uav0000268_05773_v","uav0000305_00000_v","uav0000339_00001_v"]
lens=[]
for s in SEQS:
    r=np.loadtxt(osp.join(TRK,s+".txt"),delimiter=",",ndmin=2)
    d=pd.DataFrame({"f":r[:,0].astype(int),"t":r[:,1].astype(int)})
    for t,g in d.groupby("t"):
        f=np.sort(g.f.to_numpy()); df=np.diff(f)
        for i in np.where(df>1)[0]: lens.append(int(df[i])-1)
lens=np.array(lens)
print("real gaps: n=%d  L>=3: %d  L1-2: %d"%(len(lens),(lens>=3).sum(),(lens<3).sum()))

# ---- per-L event rates ----
dn=pd.read_csv("_scratch/_r33_downstream.csv")          # his, L in {3,5,8,12,18,25}, STRIDE=5
dn["swC"]=(dn.bestC>dn.iouC)&(dn.bestC>0.1); dn["swD"]=(dn.bestD>dn.iouD)&(dn.bestD>0.1)
short=pd.read_csv("_scratch/_verify_r33_shortdown.csv") # mine, L in 1..5, STRIDE=1
short["swC"]=(short.bestC>short.iouC)&(short.bestC>0.1)
short["swD"]=(short.bestD>short.iouD)&(short.bestD>0.1)
print("\nHIS pooled event rates: plain %.4f%%  cmc %.4f%%  fixed %d broken %d  (n=%d)"
      %(100*dn.swC.mean(),100*dn.swD.mean(),int((dn.swC&~dn.swD).sum()),
        int((~dn.swC&dn.swD).sum()),len(dn)))

rateH=dn.groupby("L").agg(nC=("swC","size"),pC=("swC","mean"),pD=("swD","mean"))
rateS=short.groupby("L").agg(nC=("swC","size"),pC=("swC","mean"),pD=("swD","mean"))
print("\nper-L per-check event rates (%):")
comb=pd.concat([rateS.assign(src="mine S=1"),rateH.assign(src="his S=5")]).sort_index()
comb[["pC","pD"]]*=100
print(comb.round(4).to_string())
print("  >>> STRIDE cross-check at L=3/L=5: mine(S=1) pC %.4f/%.4f vs his(S=5) pC %.4f/%.4f"
      %(100*rateS.pC[3],100*rateS.pC[5],100*rateH.pC[3],100*rateH.pC[5]))

# build an interpolator over L for pC,pD (his grid for L>=3, mine for L=1,2)
grid=sorted(set(list(rateH.index)+[1,2]))
def r(L,col):
    if L<=2: return rateS.loc[L,col]
    xs=np.array(sorted(rateH.index)); ys=np.array([rateH.loc[x,col] for x in xs])
    return float(np.interp(min(L,xs.max()),xs,ys))

HOR=3
def expected(mask,label):
    eC=eD=0.0
    for L in lens[mask]:
        eC+=HOR*r(L,"pC"); eD+=HOR*r(L,"pD")
    print("  %-28s gaps=%3d   plain-ORU %5.2f   CMC-ORU %5.2f   DELTA %5.2f"
          %(label,mask.sum(),eC,eD,eC-eD))
    return eC-eD
print("\n=== expected ID-switch-shaped events over a 3-frame post-recovery horizon ===")
d_long =expected(lens>=3,"L>=3 only (HIS population)")
d_short=expected(lens<3 ,"L=1,2 (he excluded these)")
d_all  =expected(lens>=0,"ALL 330 gaps")

print("\n=== (c) selection-bias corrections ===")
pg=pd.read_csv("_scratch/_r33_pseudogap2.csv")
print("  _r33_pseudogap2 cols:",list(pg.columns)[:14])
if "gapconc" in pg.columns:
    for L in sorted(pg.L.unique()):
        g=pg[pg.L==L]
        a=g[~g.gapconc.astype(bool)]; b=g[g.gapconc.astype(bool)]
        print("   L=%2d  n_easy=%6d n_gapconc=%5d"%(L,len(a),len(b)))
# difficulty multiplier from the REAL data: real post-recovery switch rate vs proxy rate
real_sw_1to3 = 30      # measured in _verify_r33_attrib.py, age in [1,3], either side
print("\n  CALIBRATION (the strongest bias estimate available):")
print("    real IDSw at post-recovery age 1..3, all 330 gaps : %d  -> %.4f per gap"%(real_sw_1to3,real_sw_1to3/330))
proxy_per_gap = (HOR*np.mean([r(L,"pC") for L in lens]))
print("    proxy plain-ORU events per gap (same horizon)     : %.4f"%proxy_per_gap)
mult = (real_sw_1to3/330)/proxy_per_gap
print("    => real gaps are %.1fx harder than the pseudo-gaps the ladder samples"%mult)

print("\n=== FINAL ESTIMATES ===")
print("  his figure (L>=3 only, no difficulty correction)            : %.1f IDSw"%d_long)
print("  + L=1,2 population                                          : %.1f IDSw"%d_all)
print("  + his own gap-concurrent difficulty factor (1.7x)           : %.1f IDSw"%(d_all*1.7))
print("  + calibrated difficulty factor (%.1fx, from real switches)   : %.1f IDSw"%(mult,d_all*mult))
print("  HARD CEILING (all IDSw in the addressable age-1..3 window)  : %d IDSw"%real_sw_1to3)
