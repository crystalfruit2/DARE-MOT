"""Isolate the MECHANISM on real gaps: use GT boxes as the anchors at t1 and t2 too, so
tracker localisation error at the endpoints cannot mask the C-vs-D difference."""
import os.path as osp, numpy as np, pandas as pd
exec(open("_verify_r33_realgaps.py").read().split("HOR=3; rows=[]")[0])
HOR=3; rows=[]
for seq in SEQS:
    H=affs(seq)
    tr=rd(osp.join(TRK,seq+".txt"),list(enumerate(["frame","tid","x","y","w","h","sc","cls"])))
    gt=rd(GT.format(seq=seq),list(enumerate(["frame","gid","x","y","w","h","cf","cls","vis"])))
    for c in ("frame","tid"): tr[c]=tr[c].astype(int)
    for c in ("frame","gid","cls"): gt[c]=gt[c].astype(int)
    gtf={f:g for f,g in gt.groupby("frame")}
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
        gsub=gt[gt.gid==gid].drop_duplicates("frame").set_index("frame")
        for i in brk:
            t1,t2=int(f[i]),int(f[i+1]); L=t2-t1-1; span=t2-t1
            hid=[t for t in range(t1+1,t2) if t in gsub.index]
            if t1 not in gsub.index or t2 not in gsub.index or not hid: continue
            M={t1:np.eye(3)}; cur=np.eye(3); ok=True
            for t in range(t1+1,t2+1):
                if t not in H: ok=False;break
                cur=H[t]@cur; M[t]=cur
            if not ok: continue
            try: inv2=np.linalg.inv(M[t2])
            except np.linalg.LinAlgError: continue
            b1=gsub.loc[t1]; b2=gsub.loc[t2]
            p1=np.array([b1.x+b1.w/2,b1.y+b1.h/2]); p2=np.array([b2.x+b2.w/2,b2.y+b2.h/2])
            p2b=wp(inv2,p2); size=np.sqrt(max(b1.w*b1.h,1.))
            eC=[];eD=[];iC=[];iD=[]
            for t in hid:
                a=(t-t1)/float(span); q=gsub.loc[t]
                truth=np.array([q.x+q.w/2,q.y+q.h/2])
                pw=b1.w+a*(b2.w-b1.w); ph=b1.h+a*(b2.h-b1.h)
                C=p1+a*(p2-p1); D=wp(M[t],p1+a*(p2b-p1))
                eC.append(np.linalg.norm(C-truth)/size); eD.append(np.linalg.norm(D-truth)/size)
                iC.append(iou_c(C,pw,ph,truth,q.w,q.h)); iD.append(iou_c(D,pw,ph,truth,q.w,q.h))
            rows.append(dict(seq=seq,L=L,e_C=np.mean(eC),e_D=np.mean(eD),
                             iou_C=np.mean(iC),iou_D=np.mean(iD)))
d=pd.DataFrame(rows)
print("=== REAL gaps, GT ANCHORS + GT truth (pure mechanism test), n=%d ==="%len(d))
t=d.groupby(pd.cut(d.L,[0,2,4,8,15,30],labels=["1-2","3-4","5-8","9-15","16-30"]),observed=True).apply(
  lambda g:pd.Series({"n":len(g),"e_C":g.e_C.median(),"e_D":g.e_D.median(),
   "D_wins_%":100*(g.e_D<g.e_C).mean(),"catC_%":100*(g.iou_C<0.5).mean(),
   "catD_%":100*(g.iou_D<0.5).mean()}),include_groups=False)
print(t.round(3).to_string())
print("pooled: e_C %.4f e_D %.4f  D wins %.1f%%  IoU<0.5 C %.1f%% D %.1f%%"
      %(d.e_C.median(),d.e_D.median(),100*(d.e_D<d.e_C).mean(),
        100*(d.iou_C<0.5).mean(),100*(d.iou_D<0.5).mean()))
