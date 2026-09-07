"""(a) Test the 'endpoint-anchored lerp exactly absorbs affine-linear camera motion' claim,
then measure L=1 and L=2 directly (ladder + downstream event proxy), STRIDE=1."""
import os.path as osp, numpy as np, pandas as pd
np.set_printoptions(suppress=True)

# ---------- PART 1: the algebra, checked numerically ----------
print("="*78)
print("PART 1  -- is 'z_cmc == z_plain for M_t = I + (t-t1)V' true?")
print("="*78)
def run(V, p1, q, T):
    M={}; 
    for s in range(T+1): M[s]=np.eye(3)+s*V
    p2=M[T]@np.array([q[0],q[1],1.0]); p2=p2[:2]/p2[2] if p2[2]!=0 else p2[:2]
    out=[]
    for s in range(1,T):
        a=s/T
        C=p1+a*(p2-p1)
        inv2=np.linalg.inv(M[T]); qq=inv2@np.array([p2[0],p2[1],1.0]); qq=qq[:2]/qq[2]
        v=M[s]@np.array([*(p1+a*(qq-p1)),1.0]); D=v[:2]/v[2]
        out.append(np.linalg.norm(D-C))
    return max(out)
p1=np.array([100.,200.]); q=np.array([160.,260.])   # object moved 60,60 in stabilised frame
Vt=np.zeros((3,3)); Vt[0,2]=3.0; Vt[1,2]=-2.0                      # pure translation rate
Vr=np.zeros((3,3)); Vr[0,1]=0.004; Vr[1,0]=-0.004                  # rotation rate ~0.23 deg/frame
Vs=np.zeros((3,3)); Vs[0,0]=0.003; Vs[1,1]=0.003                   # zoom rate 0.3%/frame
for name,V in [("pure translation",Vt),("+ rotation 0.23deg/f",Vt+Vr),("+ zoom 0.3%/f",Vt+Vs)]:
    print("  %-22s : max |D-C| over hidden frames  T=2:%8.5f px   T=9:%8.5f px   T=26:%8.5f px"
          %(name,run(V,p1,q,2),run(V,p1,q,9),run(V,p1,q,26)))
print("  closed form: D - C = s(1-a) * V * (p1 - q)  [homogeneous]. For PURE TRANSLATION")
print("  V*(p1-q)=0 because (p1-q) has zero homogeneous coord -> identically 0 (his claim holds).")
print("  For a linear-in-time ROTATION or ZOOM it is NOT 0: it scales with the object's own")
print("  displacement (p1-q) and peaks at s=T/2 with magnitude ~ (T/4)*|Vlin|*|p1-q|.")

# ---------- PART 2: measure L=1,2 (and 3 as a control) directly ----------
AFF="_scratch/_r33_affines"; TRK=osp.join("YOLOX_outputs","mc_dare_cv_rerun0803","track_results")
SEQS=["uav0000086_00000_v","uav0000117_02622_v","uav0000137_00458_v","uav0000182_00000_v",
      "uav0000268_05773_v","uav0000305_00000_v","uav0000339_00001_v"]
def boxes(s):
    r=np.loadtxt(osp.join(TRK,s+".txt"),delimiter=",",ndmin=2)
    return pd.DataFrame({"frame":r[:,0].astype(int),"tid":r[:,1].astype(int),
                         "x":r[:,2],"y":r[:,3],"w":r[:,4],"h":r[:,5]})
def affs(s):
    d=pd.read_csv(osp.join(AFF,s+".csv")); H={}
    for r in d.itertuples():
        M=np.eye(3); M[0,0],M[0,1],M[0,2]=r.a,r.b,r.tx; M[1,0],M[1,1],M[1,2]=r.c,r.d,r.ty
        H[int(r.frame)]=M
    return H
def wp(M,p):
    q=M@np.array([p[0],p[1],1.0]); return q[:2]
def iou(pc,w,h,tc,tw,th):
    iw=max(0.,min(pc[0]+w/2,tc[0]+tw/2)-max(pc[0]-w/2,tc[0]-tw/2))
    ih=max(0.,min(pc[1]+h/2,tc[1]+th/2)-max(pc[1]-h/2,tc[1]-th/2))
    i=iw*ih; return i/max(w*h+tw*th-i,1e-9)

LENGTHS=[1,2,3,4,5]; STRIDE=1; HOR=3
rows=[]; drows=[]
for seq in SEQS:
    H=affs(seq); b=boxes(seq)
    b["cx"]=b.x+b.w/2; b["cy"]=b.y+b.h/2
    byf={int(t):(g.tid.to_numpy(),g.cx.to_numpy(),g.cy.to_numpy(),g.w.to_numpy(),g.h.to_numpy())
         for t,g in b.groupby("frame")}
    for tid,g in b.groupby("tid"):
        g=g.sort_values("frame"); f=g.frame.to_numpy()
        cx=g.cx.to_numpy(); cy=g.cy.to_numpy(); w=g.w.to_numpy(); h=g.h.to_numpy()
        brk=np.where(np.diff(f)!=1)[0]
        for s0,e in zip(np.r_[0,brk+1],np.r_[brk,len(f)-1]):
            for L in LENGTHS:
                span=L+1
                for i0 in range(s0,e-span+1,STRIDE):
                    i1=i0+span
                    t1,t2=int(f[i0]),int(f[i1])
                    cur=np.eye(3); M={t1:np.eye(3)}; ok=True
                    for t in range(t1+1,t2+1):
                        if t not in H: ok=False;break
                        cur=H[t]@cur; M[t]=cur
                    if not ok: continue
                    try: inv2=np.linalg.inv(M[t2])
                    except np.linalg.LinAlgError: continue
                    p1=np.array([cx[i0],cy[i0]]); p2=np.array([cx[i1],cy[i1]])
                    p2b=wp(inv2,p2); size=float(np.sqrt(max(w[i0]*h[i0],1.0)))
                    eC=[];eD=[];iC=[];iD=[]
                    for k in range(1,span):
                        t=t1+k; a=k/float(span)
                        tr=np.array([cx[i0+k],cy[i0+k]]); tw,th=w[i0+k],h[i0+k]
                        pw=w[i0]+a*(w[i1]-w[i0]); ph=h[i0]+a*(h[i1]-h[i0])
                        C=p1+a*(p2-p1); D=wp(M[t],p1+a*(p2b-p1))
                        eC.append(np.linalg.norm(C-tr)); eD.append(np.linalg.norm(D-tr))
                        iC.append(iou(C,pw,ph,tr,tw,th)); iD.append(iou(D,pw,ph,tr,tw,th))
                    rows.append(dict(seq=seq,L=L,e_C=np.mean(eC)/size,e_D=np.mean(eD)/size,
                                     iou_C=np.mean(iC),iou_D=np.mean(iD)))
                    # downstream event proxy (needs HOR observed frames after t2)
                    a_last=(span-1)/float(span)
                    lastC=p1+a_last*(p2-p1); lastD=wp(M[t2-1],p1+a_last*(p2b-p1))
                    vC=p2-lastC; vD=p2-lastD
                    for k in range(1,HOR+1):
                        j=i1+k
                        if j>e: break
                        t=int(f[j]); tc=np.array([cx[j],cy[j]]); tw,th=w[j],h[j]
                        icc=iou(p2+k*vC,w[i1],h[i1],tc,tw,th); idd=iou(p2+k*vD,w[i1],h[i1],tc,tw,th)
                        ids,ccx,ccy,cw,ch=byf[t]; msk=ids!=tid
                        bC=bD=0.
                        if msk.any():
                            pc=p2+k*vC; pdd=p2+k*vD
                            for xx,yy,ww,hh in zip(ccx[msk],ccy[msk],cw[msk],ch[msk]):
                                o=np.array([xx,yy])
                                bC=max(bC,iou(pc,w[i1],h[i1],o,ww,hh)); bD=max(bD,iou(pdd,w[i1],h[i1],o,ww,hh))
                        drows.append(dict(seq=seq,L=L,k=k,iouC=icc,iouD=idd,bestC=bC,bestD=bD))
d=pd.DataFrame(rows); dd=pd.DataFrame(drows)
print("\n"+"="*78); print("PART 2 -- SHORT gaps measured directly (STRIDE=1)"); print("="*78)
t=d.groupby("L").agg(n=("L","size"),C=("e_C","median"),D=("e_D","median"),
                     iouC=("iou_C","median"),iouD=("iou_D","median"))
t["D_beats_C_%"]=d.groupby("L").apply(lambda g:100*(g.e_D<g.e_C).mean(),include_groups=False)
t["cat_C_%"]=d.groupby("L").apply(lambda g:100*(g.iou_C<0.5).mean(),include_groups=False)
t["cat_D_%"]=d.groupby("L").apply(lambda g:100*(g.iou_D<0.5).mean(),include_groups=False)
print(t.round(4).to_string())
print("\nmedian |D-C| in px at L=1,2 (how big is the correction at all):")
print("  (reconstruction error medians above are in BOX units)")
swC=(dd.bestC>dd.iouC)&(dd.bestC>0.1); swD=(dd.bestD>dd.iouD)&(dd.bestD>0.1)
dd["swC"]=swC; dd["swD"]=swD
print("\nID-switch-shaped event rate by L (post-recovery k=1..3):")
e=dd.groupby("L").apply(lambda g:pd.Series({"n":len(g),
    "plain_%":100*g.swC.mean(),"cmc_%":100*g.swD.mean(),
    "fixed":int((g.swC&~g.swD).sum()),"broken":int((~g.swC&g.swD).sum())}),include_groups=False)
print(e.round(4).to_string())
d.to_csv("_scratch/_verify_r33_shortladder.csv",index=False); dd.to_csv("_scratch/_verify_r33_shortdown.csv",index=False)
