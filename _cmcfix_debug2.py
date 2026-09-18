"""Catch the CA-KF + CMC crash at its source (2026-09-10). Full DARE config (ReID on), dumped
detections, and a wrapped KalmanFilter.update that checks P and S before the Cholesky. On failure,
prints the offending track's P history (min eig / max entry per frame) so we can see WHERE PSD is
lost: in the warp, in predict, or in a previous update."""
import os
import sys
import csv
import types
import numpy as np
import torch
import cv2

FT = r"C:\Users\User\Desktop\projects\DARE-MOT\reid_weights\osnet_ain_x1_0_visdrone_ft.pth"
os.environ.update(DARE_REID="osnet", DARE_REID_MODEL="osnet_ain_x1_0", DARE_REID_WEIGHTS=FT,
                  DARE_LAMBDA="0.5", DARE_LAMBDA_GATE="size", DARE_GATE_LO="2500", DARE_GATE_HI="0",
                  DARE_POOL="mean", DARE_CROP_SHRINK="0.0", DARE_REASSOC_MAX="-1", DARE_AGG_ORDER="2",
                  DARE_STATIC_EMA="-1", DARE_STATIC_GAMMAS="", DARE_LOCK="0", DARE_IOU_GATE="0.95",
                  DARE_KF_MODEL="ca", DARE_KF_ACCEL_NOISE="0.0125", DARE_CMC="sparseOptFlow",
                  DARE_CMC_FIX=os.environ.get("DARE_CMC_FIX", "bug"))
from yolox.tracker import byte_tracker as bt
from yolox.tracker.kalman_filter import KalmanFilter

hist = {}          # id(covariance-owner mean array)->list of (frame, stage, mineig, maxabs)
cur = {"frame": 0}


def stat(P):
    S = (P + P.T) / 2
    return float(np.linalg.eigvalsh(S).min()), float(np.abs(P).max())


orig_gmc = bt.STrack.multi_gmc


def gmc(stracks, H=np.eye(2, 3), mode="parity"):
    pre = {s.track_id: stat(s.covariance) for s in stracks}
    orig_gmc(stracks, H, mode)
    for s in stracks:
        hist.setdefault(s.track_id, []).append((cur["frame"], "pre-warp", *pre[s.track_id]))
        hist[s.track_id].append((cur["frame"], "post-warp", *stat(s.covariance)))


bt.STrack.multi_gmc = staticmethod(gmc)
orig_update = KalmanFilter.update


def upd(self, mean, covariance, measurement):
    ev, mx = stat(covariance)
    pm, pc = self.project(mean, covariance)
    sev = np.linalg.eigvalsh((pc + pc.T) / 2).min()
    if ev < -1e-9 * mx or sev <= 0:
        print(f"\n!!! FRAME {cur['frame']}: bad covariance entering KF.update: P min eig {ev:.3e} (max {mx:.3e}), "
              f"S min eig {sev:.3e}")
        np.set_printoptions(precision=4, linewidth=220)
        print("mean:", mean)
        print("P diag:", np.diag(covariance))
        # find which track this is by matching mean
        for tid, h in hist.items():
            pass
        raise SystemExit(2)
    return orig_update(self, mean, covariance, measurement)


KalmanFilter.update = upd

# also record predict-stage stats
orig_mp = bt.STrack.multi_predict


def mp(stracks):
    orig_mp(stracks)
    for s in stracks:
        hist.setdefault(s.track_id, []).append((cur["frame"], "post-predict", *stat(s.covariance)))


bt.STrack.multi_predict = staticmethod(mp)

SEQ = sys.argv[1] if len(sys.argv) > 1 else "uav0000086_00000_v"
DUMP = rf"C:\Users\User\Desktop\projects\DARE-MOT\_scratch\detdump_val7\{SEQ}.csv"
IMG = rf"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone2019-MOT-val\sequences\{SEQ}"
dets = {}
with open(DUMP) as f:
    for d in csv.DictReader(f):
        dets.setdefault(int(d["frame"]), []).append(
            [float(d[k]) for k in ("x1", "y1", "x2", "y2", "obj", "cls_conf")] + [float(d["cls"])])
args = types.SimpleNamespace(track_thresh=0.6, track_buffer=30, match_thresh=0.9, mot20=False, min_box_area=100)
trk = bt.BYTETracker(args)
try:
    for fr in range(1, max(dets) + 1):
        cur["frame"] = fr
        im = cv2.imread(os.path.join(IMG, f"{fr:07d}.jpg"))
        h, w = im.shape[:2]
        out = torch.tensor(np.array(dets.get(fr, np.zeros((0, 7))), dtype=np.float32).reshape(-1, 7))
        trk.update(out, [im, h, w, fr, SEQ], (h, w))
        if fr % 100 == 0:
            print(f"frame {fr}", flush=True)
    print("completed without a bad covariance")
except SystemExit:
    # dump the worst recent histories (tracks whose post-warp min eig went negative)
    bad = [(tid, h) for tid, h in hist.items() if any(e[2] < -1e-9 * max(e[3], 1) for e in h)]
    print(f"tracks that ever had a non-PSD P: {len(bad)}")
    for tid, h in bad[:3]:
        print(f"-- track {tid}: first non-PSD entry and the 4 before it")
        k = next(i for i, e in enumerate(h) if e[2] < -1e-9 * max(e[3], 1))
        for e in h[max(0, k - 4):k + 1]:
            print(f"   frame {e[0]:4d} {e[1]:13s} min eig {e[2]: .3e}  max {e[3]:.3e}")
