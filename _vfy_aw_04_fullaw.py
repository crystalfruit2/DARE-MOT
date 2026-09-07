"""Recompute Deep OC-SORT's ACTUAL Adaptive Weighting quantity, both halves.

DOC-SORT compute_aw_new_metric: emb_cost is a similarity matrix [det x trk].
  row_weight_i (detection-wise) = min(top1 - top2 over TRACKS,      max_diff)
  col_weight_j (track-wise)     = min(top1 - top2 over DETECTIONS,  max_diff)
  w(i,j) = w_base + (row_weight_i + col_weight_j)/2
Higher margin -> MORE appearance weight. Risk = -(row+col)/2.

The cached _verify_out_E_margins.csv has only the track-wise half (and uses the realised
match rather than the argmin). This rebuilds the full square matrix per frame from the same
cached tracker output, snapshotting all templates BEFORE any update in that frame.
"""
import os, os.path as osp, sys
import numpy as np, pandas as pd, cv2
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from _verify_common import SEQS

DATA_ROOT = r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone2019-MOT-val\sequences"
TRK = osp.join('YOLOX_outputs','mc_dare_cv_rerun0803','track_results')
MAX_DIFF = 0.5
for k,v in dict(DARE_AGG='B',DARE_TAU='0.5',DARE_AGG_ORDER='2',DARE_STATIC_EMA='-1',DARE_STATIC_GAMMAS='').items():
    os.environ.setdefault(k,v)
import torch
from torchreid.reid.utils import FeatureExtractor
from yolox.tracker.byte_tracker import STrack
dev='cuda' if torch.cuda.is_available() else 'cpu'
EX=FeatureExtractor(model_name='osnet_ain_x1_0',
                    model_path=r'reid_weights\osnet_ain_x1_0_visdrone_ft.pth',device=dev,verbose=False)

def crop_boxes(img,boxes):
    H,W=img.shape[:2]; crops=[];idxs=[]
    for i,(x,y,w,h) in enumerate(boxes):
        x,y,w,h=int(x),int(y),int(w),int(h)
        x1,y1,x2,y2=max(0,x),max(0,y),min(W,x+w),min(H,y+h)
        c=img[y1:y2,x1:x2]
        if c.size>0: crops.append(cv2.cvtColor(c,cv2.COLOR_BGR2RGB)); idxs.append(i)
    return crops,idxs

def two_smallest_excl(D):
    """for each row of D, the smallest and second smallest OFF-DIAGONAL entry index/value"""
    n=D.shape[0]
    M=D.copy(); np.fill_diagonal(M,np.inf)
    return M.min(axis=1)

allrows=[]
for seq in SEQS:
    raw=np.loadtxt(osp.join(TRK,seq+'.txt'),delimiter=',',ndmin=2)
    fr=raw[:,0].astype(int); o=np.argsort(fr,kind='stable'); raw,fr=raw[o],fr[o]
    tracks={}; rows=[]
    for f in np.unique(fr):
        sel=raw[fr==f]
        img=cv2.imread(osp.join(DATA_ROOT,seq,'%07d.jpg'%f))
        if img is None: continue
        boxes=sel[:,2:6]
        crops,idxs=crop_boxes(img,boxes)
        if len(crops)<2: continue
        feats=np.zeros((len(sel),512),np.float32); got=np.zeros(len(sel),bool)
        for b in range(0,len(crops),256):
            with torch.no_grad(): out=EX(crops[b:b+256]).cpu().numpy()
            for k,i in enumerate(idxs[b:b+256]): feats[i],got[i]=out[k],True
        Fn=feats/(np.linalg.norm(feats,axis=1,keepdims=True)+1e-12)
        tids=sel[:,1].astype(int)
        # rows of the matrix = existing tracks present in this frame (snapshot templates)
        have=[i for i in range(len(sel)) if got[i] and tids[i] in tracks]
        cols=np.where(got)[0]                       # all detections with a feature
        if len(have)>=1 and len(cols)>=2:
            T=np.stack([tracks[tids[i]].smooth_feat/(np.linalg.norm(tracks[tids[i]].smooth_feat)+1e-12)
                        for i in have])
            D=np.maximum(0.0,1.0-T@Fn[cols].T)      # [n_have x n_cols] distance
            colpos={c:k for k,c in enumerate(cols)}
            for a,i in enumerate(have):
                j=colpos[i]                          # its own detection
                d_own=float(D[a,j])
                # track-wise (column of DOC-SORT = this track over all detections)
                row=np.delete(D[a],j); d2_trk=float(row.min())
                m_trk=d2_trk-d_own
                # detection-wise (row of DOC-SORT = this detection over all tracks)
                if len(have)>=2:
                    col=np.delete(D[:,j],a); d2_det=float(col.min())
                    m_det=d2_det-d_own
                else:
                    m_det=np.nan
                rows.append((int(f),int(tids[i]),int(sel[i,7]),d_own,m_trk,m_det,len(have),len(cols)))
        for i in range(len(sel)):
            if not got[i]: continue
            t=tracks.get(tids[i])
            if t is None:
                tr=STrack(list(boxes[i]),float(sel[i,6]),feature=feats[i].astype(np.float64),cls=int(sel[i,7]))
                tr.tracklet_len=1; tracks[tids[i]]=tr
            else:
                t.update_features(feats[i].astype(np.float64),float(sel[i,6])); t.tracklet_len+=1
    R=pd.DataFrame(rows,columns=['frame','track_id','cls','resid','m_trk','m_det','n_trk','n_det'])
    R['seq']=seq; allrows.append(R)
    print('%-22s %6d rows'%(seq,len(R)),flush=True)
A=pd.concat(allrows,ignore_index=True)
A['m_trk_clip']=np.minimum(A.m_trk,MAX_DIFF); A['m_det_clip']=np.minimum(A.m_det,MAX_DIFF)
A['AW_full']=-(A.m_trk_clip+A.m_det_clip)/2
A['AW_trk']=-A.m_trk_clip
A['AW_det']=-A.m_det_clip
A.to_csv('_scratch/_vfy_aw_fullaw.csv',index=False)
print('wrote _vfy_aw_fullaw.csv rows=%d'%len(A))
print(A[['m_trk','m_det']].describe())
print('corr(m_trk,m_det) spearman = %.3f'%A.m_trk.corr(A.m_det,method='spearman'))
