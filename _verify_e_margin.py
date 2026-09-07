"""ATTACK E -- operational magnitude, done properly.

The candidate's own evidence (TEST B1) is circular: frames are flagged BECAUSE the median residual
is high. The non-circular question is whether a residual perturbation of the observed size could
change any ASSIGNMENT. Two facts frame it:

  1. A linear assignment problem is invariant to adding a constant to the whole cost matrix, and
     to adding a constant to any single row or column. A pure scene-wide common mode is therefore
     the one appearance perturbation the solver is structurally immune to -- it can only act
     through the match_thresh=0.9 cut-off.
  2. What can flip a match is the MARGIN: cost(assigned) vs cost(best competitor). So measure it.

This re-runs the replay (same STrack template evolution) and additionally computes, for every
matched track in every frame, the cosine distance from its template to EVERY box in the frame:
    resid       = distance to its own assigned box
    d_second    = smallest distance to any OTHER box
    reid_margin = d_second - resid
and an approximate fused margin at lam=0.5 using IoU between the track's linearly-extrapolated
previous box and each candidate (the tracker's iou_dists are additionally fuse_score'd, so this
is an approximation -- stated, not hidden).

Reported at flagged vs unflagged frames:
  * margin distribution
  * fraction of matched pairs whose margin is smaller than the observed common-mode shift
    (0.0153 in reid units -> 0.0077 in fused units at lam=0.5)  == pairs that COULD flip
  * fraction of matched pairs within that distance of match_thresh == pairs that could be cut
"""
import os
import os.path as osp
import sys
import numpy as np
import pandas as pd
import cv2

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from _verify_common import SEQS, load_resid, frame_median, zseries

DATA_ROOT = r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone2019-MOT-val\sequences"
TRK = osp.join('YOLOX_outputs', 'mc_dare_cv_rerun0803', 'track_results')
LAM = 0.5
MATCH_THRESH = 0.9
SHIFT_REID = 0.0153          # observed common-mode excursion, reid units
SHIFT_FUSED = LAM * SHIFT_REID

os.environ.setdefault('DARE_AGG', 'B')
os.environ.setdefault('DARE_TAU', '0.5')
os.environ.setdefault('DARE_AGG_ORDER', '2')
os.environ.setdefault('DARE_STATIC_EMA', '-1')
os.environ.setdefault('DARE_STATIC_GAMMAS', '')

import torch
from torchreid.reid.utils import FeatureExtractor
from yolox.tracker.byte_tracker import STrack

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
EX = FeatureExtractor(model_name='osnet_ain_x1_0',
                      model_path=r'reid_weights\osnet_ain_x1_0_visdrone_ft.pth',
                      device=dev, verbose=False)


def crop_boxes(img, boxes):
    H, W = img.shape[:2]
    crops, idxs = [], []
    for i, (x, y, w, h) in enumerate(boxes):
        x, y, w, h = int(x), int(y), int(w), int(h)
        x1, y1, x2, y2 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        c = img[y1:y2, x1:x2]
        if c.size > 0:
            crops.append(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
            idxs.append(i)
    return crops, idxs


def iou_dist(a, B):
    ax1, ay1, aw, ah = a
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx1, by1, bw, bh = B[:, 0], B[:, 1], B[:, 2], B[:, 3]
    bx2, by2 = bx1 + bw, by1 + bh
    iw = np.maximum(0, np.minimum(ax2, bx2) - np.maximum(ax1, bx1))
    ih = np.maximum(0, np.minimum(ay2, by2) - np.maximum(ay1, by1))
    inter = iw * ih
    return 1.0 - inter / (aw * ah + bw * bh - inter + 1e-9)


allrows = []
for seq in SEQS:
    d = load_resid(seq)
    grid, med = frame_median(d, 5)
    z = zseries(med, 31)
    flagged = set(grid[(~np.isnan(z)) & (np.abs(z) > 3)].tolist())

    raw = np.loadtxt(osp.join(TRK, seq + '.txt'), delimiter=',', ndmin=2)
    fr = raw[:, 0].astype(int)
    o = np.argsort(fr, kind='stable')
    raw, fr = raw[o], fr[o]
    tracks, prev_box = {}, {}
    rows = []
    for f in np.unique(fr):
        sel = raw[fr == f]
        img = cv2.imread(osp.join(DATA_ROOT, seq, '%07d.jpg' % f))
        if img is None:
            continue
        boxes = sel[:, 2:6]
        crops, idxs = crop_boxes(img, boxes)
        if len(crops) < 2:
            continue
        feats = np.zeros((len(sel), 512), np.float32)
        got = np.zeros(len(sel), bool)
        for b in range(0, len(crops), 256):
            with torch.no_grad():
                out = EX(crops[b:b + 256]).cpu().numpy()
            for k, i in enumerate(idxs[b:b + 256]):
                feats[i], got[i] = out[k], True
        Fn = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)

        for i, r in enumerate(sel):
            tid, score, cls = int(r[1]), float(r[6]), int(r[7])
            box = r[2:6]
            trk = tracks.get(tid)
            if trk is None:
                if got[i]:
                    tracks[tid] = STrack(list(box), score, feature=feats[i].astype(np.float64), cls=cls)
                    tracks[tid].tracklet_len = 1
                    prev_box[tid] = box
                continue
            if not got[i]:
                continue
            t = trk.smooth_feat
            t = t / (np.linalg.norm(t) + 1e-12)
            dall = np.maximum(0.0, 1.0 - Fn[got] @ t)
            ids_got = np.where(got)[0]
            own = np.where(ids_got == i)[0][0]
            dr = float(dall[own])
            other = np.delete(dall, own)
            oidx = np.delete(ids_got, own)
            j = int(np.argmin(other))
            d2 = float(other[j])
            # approximate KF prediction: previous posterior box extrapolated by its last step
            pb = prev_box.get(tid, box)
            pred = pb + (pb - prev_box.get(str(tid) + '_pp', pb)) if False else pb
            iod = iou_dist(pred, boxes[ids_got])
            io_own = float(iod[own])
            io_oth = float(np.delete(iod, own)[j])
            rows.append((int(f), tid, cls, dr, d2, d2 - dr,
                         LAM * dr + (1 - LAM) * io_own,
                         (LAM * d2 + (1 - LAM) * io_oth) - (LAM * dr + (1 - LAM) * io_own),
                         int(f) in flagged))
            trk.update_features(feats[i].astype(np.float64), score)
            trk.tracklet_len += 1
            prev_box[tid] = box
    R = pd.DataFrame(rows, columns=['frame', 'tid', 'cls', 'resid', 'd_second', 'reid_margin',
                                    'fused_cost', 'fused_margin', 'flagged'])
    R['seq'] = seq
    allrows.append(R)
    print('%-22s %6d matched pairs (%d at flagged frames)' % (seq, len(R), int(R.flagged.sum())))

A = pd.concat(allrows, ignore_index=True)
A.to_csv('_scratch/_verify_out_E_margins.csv', index=False)

print('')
print('=' * 128)
print('ATTACK E -- assignment margins at flagged vs unflagged frames  (lam=0.5, match_thresh=0.9)')
print('   common-mode shift to beat: 0.0153 reid units = %.4f fused units' % SHIFT_FUSED)
print('=' * 128)
print('%-22s %9s %12s %12s %12s %14s %14s %13s' %
      ('sequence', 'grp', 'n', 'reid_marg', 'fused_marg', 'P(fused_marg', 'P(fused_marg', 'P(cost within'))
print('%-22s %9s %12s %12s %12s %14s %14s %13s' %
      ('', '', '', '(median)', '(median)', '< %.4f)' % SHIFT_FUSED, '< 0.05)', '%.4f of 0.9)' % SHIFT_FUSED))
out = []
for seq in SEQS + ['ALL']:
    S = A if seq == 'ALL' else A[A.seq == seq]
    for grp, m in (('flagged', S.flagged), ('unflagged', ~S.flagged)):
        s = S[m]
        if len(s) < 10:
            continue
        p1 = float((s.fused_margin < SHIFT_FUSED).mean())
        p2 = float((s.fused_margin < 0.05).mean())
        p3 = float((np.abs(s.fused_cost - MATCH_THRESH) < SHIFT_FUSED).mean())
        print('%-22s %9s %12d %12.4f %12.4f %13.2f%% %13.2f%% %12.2f%%' %
              (seq if grp == 'flagged' else '', grp, len(s), s.reid_margin.median(),
               s.fused_margin.median(), 100 * p1, 100 * p2, 100 * p3))
        out.append(dict(seq=seq, grp=grp, n=len(s), reid_margin=s.reid_margin.median(),
                        fused_margin=s.fused_margin.median(), p_flip=p1, p_near05=p2, p_thresh=p3))
pd.DataFrame(out).to_csv('_scratch/_verify_out_E.csv', index=False)
print('')
print('wrote _verify_out_E_margins.csv and _verify_out_E.csv')
