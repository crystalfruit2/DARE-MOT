"""ATTACK A, part 3 -- DECISIVE artifact test.

The replay crops the KF POSTERIOR box; the tracker crops the DETECTION box. During abrupt global
motion the posterior lags the detection for EVERY track at once (same sign, same frame), so the
crop error is cross-track CORRELATED, not independent. A1/A2-step1 already showed the flagged
frames sit at the 92nd-96th percentile of global acceleration, i.e. exactly where that lag is
largest.

What is still missing is the SCALE: does a plausible few-pixel misalignment actually move a
cosine residual by the observed excursion (~+0.015)? This measures it directly by re-embedding
the same crops displaced by delta pixels and taking the cosine distance to the unshifted
embedding. If d_shift(2-3 px) is of order 0.015 or more, the replay can manufacture the entire
observed common mode and the finding is uninterpretable without the in-tracker logger.

Also splits flagged vs unflagged frames: if crops at flagged frames are MORE shift-sensitive, the
artifact is amplified exactly where the finding lives.
"""
import os
import os.path as osp
import sys
import numpy as np
import pandas as pd
import cv2

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from _verify_common import SEQS, load_resid, frame_median, zseries, load_boxes, geometry_series

DATA_ROOT = r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone2019-MOT-val\sequences"
DELTAS = [1.0, 2.0, 3.0, 5.0]
NSAMP_UNFLAG = 60
RNG = np.random.default_rng(7)

os.environ.setdefault('DARE_AGG', 'B')
os.environ.setdefault('DARE_TAU', '0.5')
os.environ.setdefault('DARE_AGG_ORDER', '2')

import torch
from torchreid.reid.utils import FeatureExtractor
from scipy.spatial.distance import cdist

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
EX = FeatureExtractor(model_name='osnet_ain_x1_0',
                      model_path=r'reid_weights\osnet_ain_x1_0_visdrone_ft.pth',
                      device=dev, verbose=False)


def crops_at(img, boxes, ox, oy):
    H, W = img.shape[:2]
    out, idx = [], []
    for i, (x, y, w, h) in enumerate(boxes):
        x, y, w, h = int(x + ox), int(y + oy), int(w), int(h)
        x1, y1, x2, y2 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        c = img[y1:y2, x1:x2]
        if c.size > 0 and c.shape[0] > 4 and c.shape[1] > 4:
            out.append(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
            idx.append(i)
    return out, idx


def embed(crops):
    fs = []
    for b in range(0, len(crops), 256):
        with torch.no_grad():
            fs.append(EX(crops[b:b + 256]).cpu().numpy())
    return np.concatenate(fs, 0) if fs else np.zeros((0, 512), np.float32)


print('=' * 118)
print('ATTACK A3 -- cosine-residual induced by a pure crop misalignment of delta pixels')
print('=' * 118)
hdr = '%-22s %7s %9s' % ('sequence', 'grp', 'nbox') + ''.join('%12s' % ('d=%gpx' % d) for d in DELTAS)
print(hdr)

rows = []
for seq in SEQS:
    df = load_resid(seq)
    grid, med = frame_median(df, 5)
    z = zseries(med, 31)
    G = geometry_series(seq, grid)
    ok = ~np.isnan(z)
    A = ok & (np.abs(z) > 3)
    fl = grid[A].tolist()
    unf = grid[ok & ~A]
    unf = RNG.choice(unf, min(NSAMP_UNFLAG, len(unf)), replace=False).tolist()
    B = load_boxes(seq)
    bf = {int(f): g for f, g in B.groupby('frame')}
    # global motion direction per frame (shift ALONG it = worst-case KF lag direction)
    gdx = G['glob_dx'].to_dict()
    gdy = G['glob_dy'].to_dict()

    for grp, frames in (('flagged', fl), ('unflagged', unf)):
        acc = {d: [] for d in DELTAS}
        nbox = 0
        for f in frames:
            g = bf.get(int(f))
            if g is None or len(g) < 3:
                continue
            img = cv2.imread(osp.join(DATA_ROOT, seq, '%07d.jpg' % int(f)))
            if img is None:
                continue
            boxes = g[['x', 'y', 'w', 'h']].to_numpy()
            c0, i0 = crops_at(img, boxes, 0, 0)
            if len(c0) < 3:
                continue
            f0 = embed(c0)
            vx, vy = gdx.get(int(f), 0.0), gdy.get(int(f), 0.0)
            n = np.hypot(vx, vy)
            ux, uy = (vx / n, vy / n) if n > 1e-6 else (1.0, 0.0)
            for d in DELTAS:
                cd, idd = crops_at(img, boxes, ux * d, uy * d)
                common = [k for k, i in enumerate(idd) if i in set(i0)]
                if len(common) < 3:
                    continue
                fd = embed([cd[k] for k in common])
                pos = {i: k for k, i in enumerate(i0)}
                base = f0[[pos[idd[k]] for k in common]]
                dd = np.maximum(0.0, 1.0 - np.sum(base * fd, 1) /
                                (np.linalg.norm(base, axis=1) * np.linalg.norm(fd, axis=1) + 1e-12))
                acc[d].extend(dd.tolist())
            nbox += len(c0)
        if nbox == 0:
            continue
        vals = [np.median(acc[d]) if acc[d] else np.nan for d in DELTAS]
        print('%-22s %7s %9d' % (seq, grp, nbox) + ''.join('%12.4f' % v for v in vals))
        rows.append(dict(seq=seq, grp=grp, nbox=nbox,
                         **{'d%g' % d: v for d, v in zip(DELTAS, vals)}))

R = pd.DataFrame(rows)
print('-' * 118)
for grp in ('flagged', 'unflagged'):
    s = R[R.grp == grp]
    print('%-22s %7s %9d' % ('POOLED MEDIAN', grp, s.nbox.sum()) +
          ''.join('%12.4f' % s['d%g' % d].median() for d in DELTAS))
R.to_csv('_scratch/_verify_out_A3.csv', index=False)
print('')
print('observed excursion to explain: median resid 0.022 -> 0.037 at flagged frames (+0.0153)')
print('wrote _verify_out_A3.csv')
