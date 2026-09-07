"""Round-3 candidate #1 -- offline REPLAY of per-track appearance-cost residuals.

Why a replay instead of in-tracker logging: `lap._lapjv` is currently blocked by this machine's
Smart App Control policy, so tools/track.py cannot run at all (see the 2026-09-07 experiment-log
entry). The residual this diagnostic needs, however, does not require the solver -- only the
identity assignment, which the cached run already recorded. So this script re-derives the exact
quantity byte_tracker.py computes at association time:

    resid(track i, det j) = cosine_distance(F^{t-1}_i, f^t_j)   [= reid_dists[i, j], line ~899]

by walking a cached track_results file frame by frame, cropping each output box, embedding it
with the same OSNet-AIN checkpoint, and evolving each track's DARE aggregated template with the
very same STrack.update_features() the tracker uses (imported, not reimplemented).

FIDELITY -- stated up front because the verdict depends on it:
  * The tracker crops the DETECTION box; a cached track_results row stores the KF-CORRECTED
    posterior box. They are close but not identical, so replayed residuals carry a small extra
    per-track perturbation the real pipeline does not have.
  * That perturbation is independent across tracks. It can therefore DILUTE a real common-mode
    (scene-wide) signal but cannot MANUFACTURE one.
  * Hence the asymmetry this diagnostic is used under: a NEGATIVE result is trustworthy and kills
    the candidate; a POSITIVE result is a lead that must be reconfirmed with the in-tracker
    logger (byte_tracker.py::_log_appearance_residuals, already written, needs lapjv restored).
  * Rows are output tracks only, so boxes below --min-box-area and never-confirmed tracklets are
    absent, and stage-1 vs stage-2 matches are indistinguishable here.

Emits the same CSV schema as the in-tracker logger, so _appres_synchrony_diag.py consumes either.

Usage:
  python _appres_replay.py --run mc_dare_cv_rerun0803 --out _appres_logs_replay
"""
import argparse
import os
import os.path as osp
import sys
import time

import cv2
import numpy as np
from scipy.spatial.distance import cdist

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

DATA_ROOT = r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone2019-MOT-val\sequences"


def build_extractor(model_name, weights, device):
    from torchreid.reid.utils import FeatureExtractor
    return FeatureExtractor(model_name=model_name, model_path=weights,
                            device=device, verbose=False)


def crop_boxes(img, boxes, shrink=0.0):
    """Same crop rule as byte_tracker._extract_features_osnet."""
    H, W = img.shape[:2]
    crops, idxs = [], []
    for i, (x, y, w, h) in enumerate(boxes):
        if shrink > 0.0:
            dx, dy = w * shrink, h * shrink
            x, y, w, h = x + dx, y + dy, w - 2 * dx, h - 2 * dy
        x, y, w, h = int(x), int(y), int(w), int(h)
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)
        c = img[y1:y2, x1:x2]
        if c.size > 0:
            if c.ndim == 3 and c.shape[2] == 3:
                c = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
            crops.append(c)
            idxs.append(i)
    return crops, idxs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='mc_dare_cv_rerun0803')
    ap.add_argument('--out', default='_scratch/_appres_logs_replay')
    ap.add_argument('--reid-model', default=os.environ.get('DARE_REID_MODEL', 'osnet_ain_x1_0'))
    ap.add_argument('--reid-weights',
                    default=os.environ.get('DARE_REID_WEIGHTS',
                                           r'reid_weights\osnet_ain_x1_0_visdrone_ft.pth'))
    ap.add_argument('--crop-shrink', type=float, default=float(os.environ.get('DARE_CROP_SHRINK', '0.0')))
    ap.add_argument('--batch', type=int, default=256)
    args = ap.parse_args()

    # Match the headline config's aggregation before STrack reads these at construction time.
    os.environ.setdefault('DARE_AGG', 'B')
    os.environ.setdefault('DARE_TAU', '0.5')
    os.environ.setdefault('DARE_AGG_ORDER', '2')
    os.environ.setdefault('DARE_STATIC_EMA', '-1')
    os.environ.setdefault('DARE_STATIC_GAMMAS', '')

    import torch
    from yolox.tracker.byte_tracker import STrack

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device=%s  reid=%s  weights=%s  agg_order=%s tau=%s'
          % (device, args.reid_model, args.reid_weights,
             os.environ['DARE_AGG_ORDER'], os.environ['DARE_TAU']))
    ex = build_extractor(args.reid_model, args.reid_weights, device)

    res_dir = osp.join('YOLOX_outputs', args.run, 'track_results')
    if not osp.isdir(res_dir):
        sys.exit('no track_results at %s' % res_dir)
    os.makedirs(args.out, exist_ok=True)

    for fn in sorted(f for f in os.listdir(res_dir) if f.endswith('.txt')):
        seq = fn[:-4]
        t0 = time.time()
        raw = np.loadtxt(osp.join(res_dir, fn), delimiter=',', ndmin=2)
        if raw.size == 0:
            print('%-22s EMPTY' % seq)
            continue
        frames = raw[:, 0].astype(int)
        order = np.argsort(frames, kind='stable')
        raw, frames = raw[order], frames[order]

        img_dir = osp.join(DATA_ROOT, seq)
        tracks = {}
        n_rows = 0
        with open(osp.join(args.out, seq + '.csv'), 'w') as fh:
            print('seq,frame,track_id,cls,resid,valid,area,tracklet_len,'
                  'was_lost,lam,n_match,n_valid,n_pool,n_det', file=fh)

            for f in np.unique(frames):
                sel = raw[frames == f]
                img_path = osp.join(img_dir, '%07d.jpg' % f)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                boxes = sel[:, 2:6]
                crops, idxs = crop_boxes(img, boxes, args.crop_shrink)
                if not crops:
                    continue
                feats = np.zeros((len(sel), 512), dtype=np.float32)
                got = np.zeros(len(sel), dtype=bool)
                for b in range(0, len(crops), args.batch):
                    with torch.no_grad():
                        out = ex(crops[b:b + args.batch]).cpu().numpy()
                    for k, i in enumerate(idxs[b:b + args.batch]):
                        feats[i] = out[k]
                        got[i] = True

                rows, n_valid = [], 0
                for i, r in enumerate(sel):
                    tid, score, cls = int(r[1]), float(r[6]), int(r[7])
                    x, y, w, h = r[2], r[3], r[4], r[5]
                    if not got[i]:
                        continue
                    fj = feats[i].astype(np.float64)
                    trk = tracks.get(tid)
                    if trk is None:
                        # first appearance == activate(): template seeded from the raw feature,
                        # no residual is defined yet (the tracker had no template to compare to)
                        tracks[tid] = STrack([x, y, w, h], score, feature=fj, cls=cls)
                        tracks[tid].tracklet_len = 1
                        continue
                    d = float(np.maximum(0.0, cdist(trk.smooth_feat[None, :],
                                                    fj[None, :], 'cosine')[0, 0]))
                    n_valid += 1
                    rows.append((seq, int(f), tid, cls, d, 1, float(w * h),
                                 int(trk.tracklet_len), 0, 0.5))
                    trk.update_features(fj, score)
                    trk.tracklet_len += 1

                n_match, n_pool, n_det = len(rows), len(tracks), len(sel)
                for rr in rows:
                    print('%s,%d,%d,%d,%.6f,%d,%.1f,%d,%d,%.4f,%d,%d,%d,%d'
                          % (rr + (n_match, n_valid, n_pool, n_det)), file=fh)
                n_rows += len(rows)

        print('%-22s %6d residual rows | %5d tracks | %.1fs'
              % (seq, n_rows, len(tracks), time.time() - t0))

    print('\nwrote replayed residual logs to %s/' % args.out)


if __name__ == '__main__':
    main()
