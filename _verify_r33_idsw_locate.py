"""Score the cached headline run against REAL multi-class GT, extract every actual
ID-switch event, and ask: how many of them are post-gap-recovery events at all?

That number is a HARD CEILING on anything an ORU/CMC-ORU trajectory reconstruction
can buy, independent of any per-event effect size."""
import os, sys, tempfile
import numpy as np
if not hasattr(np,"asfarray"): np.asfarray = lambda a,dtype=np.float64: np.asarray(a,dtype=dtype)
if not hasattr(np,"float_"): np.float_ = np.float64
import pandas as pd
import motmetrics as mm
mm.lap.default_solver = 'scipy'

MC_GT = (r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone_MOT_Format_MC"
         r"\VisDrone2019-MOT-val\{seq}\gt\gt.txt")
SEQS = ["uav0000086_00000_v","uav0000117_02622_v","uav0000137_00458_v","uav0000182_00000_v",
        "uav0000268_05773_v","uav0000305_00000_v","uav0000339_00001_v"]
MC_NAMES = {1:"pedestrian",2:"car",3:"van",4:"truck",5:"bus"}
CAT=7
EXP = sys.argv[1] if len(sys.argv)>1 else "mc_dare_cv_rerun0803"

def respath(seq): return os.path.join("YOLOX_outputs",EXP,"track_results",seq+".txt")

def filt(src,cls,tmp,tag):
    out=os.path.join(tmp,tag+".txt"); n=0
    with open(out,"w") as fo:
        if os.path.exists(src):
            for line in open(src):
                p=line.strip().split(",")
                if len(p)<=CAT: continue
                try:
                    if int(float(p[CAT]))==cls: fo.write(line); n+=1
                except ValueError: pass
    return out,n

# ---- gap table straight from the tracker output (independent of _r33_gap_divergence.csv)
gaps={}   # (seq,tid) -> list of (t1,t2)
for seq in SEQS:
    raw=np.loadtxt(respath(seq),delimiter=",",ndmin=2)
    df=pd.DataFrame({"frame":raw[:,0].astype(int),"tid":raw[:,1].astype(int)})
    for tid,g in df.groupby("tid"):
        f=np.sort(g.frame.to_numpy()); d=np.diff(f)
        for i in np.where(d>1)[0]:
            gaps.setdefault((seq,int(tid)),[]).append((int(f[i]),int(f[i+1])))

allsw=[]; per_class={}
with tempfile.TemporaryDirectory() as tmp:
    for cls,name in MC_NAMES.items():
        accs=[];names=[];hasgt=False
        for seq in SEQS:
            gf,ng=filt(MC_GT.format(seq=seq),cls,tmp,"gt")
            tf,nt=filt(respath(seq),cls,tmp,"ts")
            if ng==0 and nt==0: continue
            if ng>0: hasgt=True
            gt=mm.io.loadtxt(gf,fmt="mot15-2D",min_confidence=1)
            ts=mm.io.loadtxt(tf,fmt="mot15-2D",min_confidence=-1)
            acc=mm.utils.compare_to_groundtruth(gt,ts,"iou",distth=0.5)
            ev=acc.mot_events
            sw=ev[ev.Type=="SWITCH"]
            for fid,r in sw.iterrows():
                allsw.append(dict(seq=seq,cls=cls,frame=int(fid[0]),
                                  oid=r.OId,hid=int(r.HId)))
            accs.append(acc);names.append(seq)
        if accs and hasgt:
            mh=mm.metrics.create()
            s=mh.compute_many(accs,names=names,metrics=["num_switches","idf1","mota","num_objects"],
                              generate_overall=True).loc["OVERALL"]
            per_class[cls]=dict(idsw=int(s.num_switches),idf1=float(s.idf1),
                                mota=float(s.mota),gt=int(s.num_objects))

print("=== %s scored vs REAL MC GT (scipy LSA in motmetrics) ==="%EXP)
tot=0
for c,v in per_class.items():
    print("  %-11s IDSw %5d  IDF1 %5.1f%%  MOTA %6.1f%%  GT %7d"%(MC_NAMES[c],v["idsw"],v["idf1"]*100,v["mota"]*100,v["gt"]))
    tot+=v["idsw"]
print("  SUM IDSw = %d"%tot)
sw=pd.DataFrame(allsw)
print("  extracted SWITCH events = %d"%len(sw))

# ---- how many switches are POST-RECOVERY?
def age_since_gap(seq,hid,frame):
    """frames since this hypothesis track resumed from a gap; None if never gapped before."""
    best=None
    for (t1,t2) in gaps.get((seq,hid),[]):
        if t2<=frame:
            a=frame-t2
            if best is None or a<best: best=a
    return best
sw["age"]=[age_since_gap(r.seq,r.hid,r.frame) for r in sw.itertuples()]
n=len(sw)
print("\n=== is an actual ID switch a post-recovery event? ===")
print("  switches on a track that NEVER had a gap before it: %d (%.1f%%)"
      %( (sw.age.isna()).sum(), 100*(sw.age.isna()).mean()))
for h in (0,1,2,3,5,10,30):
    m=(sw.age<=h).sum()
    print("  switch within %2d frames of a gap-recovery: %4d (%.1f%%)"%(h,m,100*m/n))
sw.to_csv("_scratch/_verify_r33_switches.csv",index=False)

# gaps-per-track distribution
gl=pd.Series({k:len(v) for k,v in gaps.items()})
ntracks=sum(len(pd.read_csv(respath(s),header=None)[1].unique()) for s in SEQS)
print("\n=== gap population (recomputed independently) ===")
print("  tracks total %d ; tracks with >=1 gap %d ; total gaps %d"%(ntracks,len(gl),int(gl.sum())))
lens=[t2-t1-1 for v in gaps.values() for (t1,t2) in v]
lens=pd.Series(lens)
print("  gap-length hist:"); print(lens.value_counts().sort_index().to_string())
print("  L>=3 : %d ;  L in 1..2 : %d ; median %.0f p90 %.0f max %d"
      %((lens>=3).sum(),(lens<3).sum(),lens.median(),lens.quantile(.9),lens.max()))
print("  gaps-per-track: "); print(gl.value_counts().sort_index().to_string())
