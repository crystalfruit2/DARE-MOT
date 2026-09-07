"""THE measurement the pseudo-gap ladder was a proxy for:
run arms C (plain ORU lerp) and D (CMC-composed lerp) on the ACTUAL 330 gaps of the
cached run, scored against the REAL multi-class ground truth in the hidden frames.

No pseudo-gaps, no selection bias, no proxy for 'difficulty'."""
import os.path as osp, numpy as np, pandas as pd
TRK=osp.join("YOLOX_outputs","mc_dare_cv_rerun0803","track_results")
GT=(r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone_MOT_Format_MC"
    r"\VisDrone2019-MOT-val\{seq}\gt\gt.txt")
AFF="_scratch/_r33_affines"
SEQS=["uav0000086_00000_v","uav0000117_02622_v","uav0000137_00458_v","uav0000182_00000_v",
      "uav0000268_05773_v","uav0000305_00000_v","uav0000339_00001_v"]
def rd(p,cols):
    r=np.loadtxt(p,delimiter=",",ndmin=2)
    return pd.DataFrame({c:r[:,i] for i,c in cols})
def affs(s):
    d=pd.read_csv(osp.join(AFF,s+".csv")); H={}
    for r in d.itertuples():
        M=np.eye(3); M[0,0],M[0,1],M[0,2]=r.a,r.b,r.tx; M[1,0],M[1,1],M[1,2]=r.c,r.d,r.ty
        H[int(r.frame)]=M
    return H
def wp(M,p): 
    q=M@np.array([p[0],p[1],1.0]); return q[:2]
def iou_xywh(a,b):
    ax,ay,aw,ah=a; bx,by,bw,bh=b
    iw=max(0.,min(ax+aw,bx+bw)-max(ax,bx)); ih=max(0.,min(ay+ah,by+bh)-max(ay,by))
    i=iw*ih; return i/max(aw*ah+bw*bh-i,1e-9)
def iou_c(pc,w,h,tc,tw,th):
    iw=max(0.,min(pc[0]+w/2,tc[0]+tw/2)-max(pc[0]-w/2,tc[0]-tw/2))
    ih=max(0.,min(pc[1]+h/2,tc[1]+th/2)-max(pc[1]-h/2,tc[1]-th/2))
    i=iw*ih; return i/max(w*h+tw*th-i,1e-9)

HOR=3; rows=[]; nogt=0; ngap=0
for seq in SEQS:
    H=affs(seq)
    tr=rd(osp.join(TRK,seq+".txt"),list(enumerate(["frame","tid","x","y","w","h","sc","cls"])))
    gt=rd(GT.format(seq=seq),list(enumerate(["frame","gid","x","y","w","h","cf","cls","vis"])))
    tr["frame"]=tr.frame.astype(int); tr["tid"]=tr.tid.astype(int)
    gt["frame"]=gt.frame.astype(int); gt["gid"]=gt.gid.astype(int); gt["cls"]=gt.cls.astype(int)
    gtf={f:g for f,g in gt.groupby("frame")}
    trf={f:g for f,g in tr.groupby("frame")}
    # map hypothesis track -> GT id by majority IoU>=0.5 vote over its observed frames
    for tid,g in tr.groupby("tid"):
        g=g.sort_values("frame"); f=g.frame.to_numpy()
        brk=np.where(np.diff(f)>1)[0]
        if len(brk)==0: continue
        votes={}
        for r in g.itertuples():
            gg=gtf.get(r.frame)
            if gg is None: continue
            best=0.;bid=None
            for q in gg.itertuples():
                v=iou_xywh((r.x,r.y,r.w,r.h),(q.x,q.y,q.w,q.h))
                if v>best: best,bid=v,q.gid
            if bid is not None and best>=0.5: votes[bid]=votes.get(bid,0)+1
        if not votes: continue
        gid=max(votes,key=votes.get)
        gsub=gt[gt.gid==gid].set_index("frame")
        for i in brk:
            ngap+=1
            t1,t2=int(f[i]),int(f[i+1]); L=t2-t1-1
            hid_frames=[t for t in range(t1+1,t2) if t in gsub.index]
            if t1 not in gsub.index or t2 not in gsub.index or not hid_frames:
                nogt+=1; continue
            M={t1:np.eye(3)}; cur=np.eye(3); ok=True
            for t in range(t1+1,t2+HOR+1):
                if t not in H:
                    if t<=t2: ok=False;break
                    else: break
                cur=H[t]@cur; M[t]=cur
            if not ok or t2 not in M: continue
            try: inv2=np.linalg.inv(M[t2])
            except np.linalg.LinAlgError: continue
            # anchors: the tracker's OWN last box before the gap and first box after
            r1=g[g.frame==t1].iloc[0]; r2=g[g.frame==t2].iloc[0]
            p1=np.array([r1.x+r1.w/2,r1.y+r1.h/2]); p2=np.array([r2.x+r2.w/2,r2.y+r2.h/2])
            p2b=wp(inv2,p2); span=t2-t1; size=np.sqrt(max(r1.w*r1.h,1.))
            eC=[];eD=[];iC=[];iD=[]
            for t in hid_frames:
                a=(t-t1)/float(span); gq=gsub.loc[t]
                if isinstance(gq,pd.DataFrame): gq=gq.iloc[0]
                truth=np.array([gq.x+gq.w/2,gq.y+gq.h/2])
                pw=r1.w+a*(r2.w-r1.w); ph=r1.h+a*(r2.h-r1.h)
                C=p1+a*(p2-p1); D=wp(M[t],p1+a*(p2b-p1))
                eC.append(np.linalg.norm(C-truth)/size); eD.append(np.linalg.norm(D-truth)/size)
                iC.append(iou_c(C,pw,ph,truth,gq.w,gq.h)); iD.append(iou_c(D,pw,ph,truth,gq.w,gq.h))
            # post-recovery velocity from each arm's terminal step
            a_last=(span-1)/float(span)
            lastC=p1+a_last*(p2-p1)
            lastD=wp(M[t2-1],p1+a_last*(p2b-p1)) if (t2-1) in M else lastC
            vC=p2-lastC; vD=p2-lastD
            ev={"C":0,"D":0}
            for k in range(1,HOR+1):
                t=t2+k
                if t not in gsub.index or t not in gtf: break
                gq=gsub.loc[t]
                if isinstance(gq,pd.DataFrame): gq=gq.iloc[0]
                tc=np.array([gq.x+gq.w/2,gq.y+gq.h/2])
                own={"C":iou_c(p2+k*vC,r2.w,r2.h,tc,gq.w,gq.h),
                     "D":iou_c(p2+k*vD,r2.w,r2.h,tc,gq.w,gq.h)}
                comp={"C":0.,"D":0.}
                for q in gtf[t].itertuples():
                    if q.gid==gid: continue
                    o=np.array([q.x+q.w/2,q.y+q.h/2])
                    comp["C"]=max(comp["C"],iou_c(p2+k*vC,r2.w,r2.h,o,q.w,q.h))
                    comp["D"]=max(comp["D"],iou_c(p2+k*vD,r2.w,r2.h,o,q.w,q.h))
                for kk in "CD":
                    if comp[kk]>own[kk] and comp[kk]>0.1: ev[kk]+=1
            rows.append(dict(seq=seq,tid=tid,gid=gid,t1=t1,t2=t2,L=L,cls=int(r1.cls),
                             nhid=len(hid_frames),e_C=np.mean(eC),e_D=np.mean(eD),
                             iou_C=np.mean(iC),iou_D=np.mean(iD),evC=ev["C"],evD=ev["D"]))
d=pd.DataFrame(rows); d.to_csv("_scratch/_verify_r33_realgaps.csv",index=False)
print("gaps found %d ; usable with GT in hidden frames %d ; dropped (no GT) %d"%(ngap,len(d),nogt))
print("\n=== REAL GAPS, REAL GT: reconstruction of the hidden frames ===")
t=d.groupby(pd.cut(d.L,[0,2,4,8,15,30],labels=["1-2","3-4","5-8","9-15","16-30"]),observed=True).apply(
    lambda g:pd.Series({"n":len(g),"e_C":g.e_C.median(),"e_D":g.e_D.median(),
      "D_wins_%":100*(g.e_D<g.e_C).mean(),"iouC":g.iou_C.median(),"iouD":g.iou_D.median(),
      "catC_%":100*(g.iou_C<0.5).mean(),"catD_%":100*(g.iou_D<0.5).mean()}),include_groups=False)
print(t.round(3).to_string())
print("\npooled: n=%d  e_C %.4f  e_D %.4f  D wins %.1f%%  IoU<0.5: C %.1f%% D %.1f%%"
      %(len(d),d.e_C.median(),d.e_D.median(),100*(d.e_D<d.e_C).mean(),
        100*(d.iou_C<0.5).mean(),100*(d.iou_D<0.5).mean()))
print("\n=== post-recovery ID-switch-shaped events, REAL gaps, REAL GT competitors ===")
print("  plain-ORU total events %d over %d gaps (%.4f/gap)"%(d.evC.sum(),len(d),d.evC.mean()))
print("  CMC-ORU   total events %d over %d gaps (%.4f/gap)"%(d.evD.sum(),len(d),d.evD.mean()))
print("  NET DELTA (plain - cmc) = %d events on the whole val7 gap population"%(d.evC.sum()-d.evD.sum()))
print("  gaps where C had an event and D did not: %d ; D had one and C did not: %d"
      %(int(((d.evC>0)&(d.evD==0)).sum()),int(((d.evD>0)&(d.evC==0)).sum())))
print("\n  by gap length:")
print(d.groupby(pd.cut(d.L,[0,2,4,8,15,30],labels=["1-2","3-4","5-8","9-15","16-30"]),observed=True)
      .apply(lambda g:pd.Series({"n":len(g),"evC":g.evC.sum(),"evD":g.evD.sum(),
                                 "delta":g.evC.sum()-g.evD.sum()}),include_groups=False).to_string())
