"""Refine: for each real SWITCH, look at BOTH the hypothesis that gained the GT object
and the one that lost it, and ask how recently either resumed from a gap.

Also: is the switch AT the recovery frame (age 0 = the re-association itself chose wrong
=> NOT addressable by ORU, which runs after re-association and only edits velocity), or
at age>=1 (=> in ORU's actual channel)?"""
import os,sys,tempfile
import numpy as np
if not hasattr(np,"asfarray"): np.asfarray=lambda a,dtype=np.float64: np.asarray(a,dtype=dtype)
if not hasattr(np,"float_"): np.float_=np.float64
import pandas as pd, motmetrics as mm
mm.lap.default_solver='scipy'
MC_GT=(r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone_MOT_Format_MC"
       r"\VisDrone2019-MOT-val\{seq}\gt\gt.txt")
SEQS=["uav0000086_00000_v","uav0000117_02622_v","uav0000137_00458_v","uav0000182_00000_v",
      "uav0000268_05773_v","uav0000305_00000_v","uav0000339_00001_v"]
CAT=7; EXP="mc_dare_cv_rerun0803"
def respath(s): return os.path.join("YOLOX_outputs",EXP,"track_results",s+".txt")
def filt(src,cls,tmp,tag):
    out=os.path.join(tmp,tag+".txt");n=0
    with open(out,"w") as fo:
        for line in open(src):
            p=line.strip().split(",")
            if len(p)<=CAT: continue
            try:
                if int(float(p[CAT]))==cls: fo.write(line);n+=1
            except ValueError: pass
    return out,n

gaps={}
for seq in SEQS:
    raw=np.loadtxt(respath(seq),delimiter=",",ndmin=2)
    df=pd.DataFrame({"frame":raw[:,0].astype(int),"tid":raw[:,1].astype(int)})
    for tid,g in df.groupby("tid"):
        f=np.sort(g.frame.to_numpy());d=np.diff(f)
        for i in np.where(d>1)[0]:
            gaps.setdefault((seq,int(tid)),[]).append((int(f[i]),int(f[i+1])))
def age(seq,hid,fr):
    b=None
    for t1,t2 in gaps.get((seq,hid),[]):
        if t2<=fr:
            a=fr-t2
            if b is None or a<b: b=a
    return b if b is not None else 10**6

rows=[]
with tempfile.TemporaryDirectory() as tmp:
    for cls in (1,2,3,4,5):
        for seq in SEQS:
            gf,ng=filt(MC_GT.format(seq=seq),cls,tmp,"gt")
            tf,nt=filt(respath(seq),cls,tmp,"ts")
            if ng==0: continue
            gt=mm.io.loadtxt(gf,fmt="mot15-2D",min_confidence=1)
            ts=mm.io.loadtxt(tf,fmt="mot15-2D",min_confidence=-1)
            acc=mm.utils.compare_to_groundtruth(gt,ts,"iou",distth=0.5)
            ev=acc.mot_events.reset_index()
            # track previous hypothesis per GT object
            prev={}
            for r in ev.itertuples():
                if r.Type in ("MATCH","SWITCH"):
                    if r.Type=="SWITCH":
                        rows.append(dict(seq=seq,cls=cls,frame=int(r.FrameId),oid=r.OId,
                                         hid_new=int(r.HId),hid_old=prev.get(r.OId,-1)))
                    prev[r.OId]=int(r.HId)
sw=pd.DataFrame(rows)
sw["age_new"]=[age(r.seq,r.hid_new,r.frame) for r in sw.itertuples()]
sw["age_old"]=[age(r.seq,r.hid_old,r.frame) if r.hid_old>=0 else 10**6 for r in sw.itertuples()]
sw["age_min"]=sw[["age_new","age_old"]].min(axis=1)
N=len(sw)
print("total SWITCH events = %d"%N)
print("\n--- age of the NEAREST gap-recovery on either the gaining or losing hypothesis ---")
for h in (0,1,2,3,5,10,30):
    print("  <= %2d frames : %4d (%.1f%%)"%(h,(sw.age_min<=h).sum(),100*(sw.age_min<=h).mean()))
print("  never gapped : %4d (%.1f%%)"%((sw.age_min>=10**6).sum(),100*(sw.age_min>=10**6).mean()))
print("\n--- ORU's ACTUAL channel: switch strictly AFTER the recovery frame ---")
print("  at the recovery frame itself (age==0, re-association error, ORU cannot touch): %d"%((sw.age_min==0).sum()))
for h in (1,2,3,5,10):
    m=((sw.age_min>=1)&(sw.age_min<=h)).sum()
    print("  age in [1,%2d] (ORU-addressable window): %3d  (%.1f%% of all IDSw)"%(h,m,100*m/N))
sw.to_csv("_scratch/_verify_r33_switch_attrib.csv",index=False)

# --- compounding: do gaps cluster on the same track?
print("\n=== compounding check: inter-gap intervals on multi-gap tracks ===")
iv=[]
for k,v in gaps.items():
    v=sorted(v)
    for i in range(1,len(v)):
        iv.append(v[i][0]-v[i-1][1])   # observed frames between end of recovery i-1 and start of gap i
iv=pd.Series(iv)
print("  %d consecutive gap pairs; observed-run length between them: median %.0f, p25 %.0f, p10 %.0f"
      %(len(iv),iv.median(),iv.quantile(.25),iv.quantile(.10)))
for h in (1,2,3,5,10):
    print("    next gap starts within %2d observed frames of the previous recovery: %d (%.1f%%)"
          %(h,(iv<=h).sum(),100*(iv<=h).mean()))
