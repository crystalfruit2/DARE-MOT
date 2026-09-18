"""Reproduce the parity-CMC covariance crash on CPU from dumped detections (2026-09-10).
Feeds _scratch/detdump_val7 into BYTETracker (ByteTrack settings, CA-KF, CMC parity) and checks
every track covariance right after the warp. Stops at the first non-PSD covariance."""
import os
import sys
import csv
import types
import numpy as np
import torch
import cv2

os.environ.update(DARE_LAMBDA="0.0", DARE_IOU_GATE="1.0", DARE_LOCK="0", DARE_KF_MODEL="ca",
                  DARE_KF_ACCEL_NOISE="0.0125", DARE_CMC="sparseOptFlow",
                  DARE_CMC_FIX=os.environ.get("DARE_CMC_FIX", "parity"))
from yolox.tracker.byte_tracker import BYTETracker, STrack

SEQ = sys.argv[1] if len(sys.argv) > 1 else "uav0000086_00000_v"
DUMP = rf"C:\Users\User\Desktop\projects\DARE-MOT\_scratch\detdump_val7\{SEQ}.csv"
IMG = rf"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone2019-MOT-val\sequences\{SEQ}"

dets = {}
with open(DUMP) as f:
    for d in csv.DictReader(f):
        dets.setdefault(int(d["frame"]), []).append(
            [float(d[k]) for k in ("x1", "y1", "x2", "y2", "obj", "cls_conf")] + [float(d["cls"])])

args = types.SimpleNamespace(track_thresh=0.6, track_buffer=30, match_thresh=0.9, mot20=False,
                             min_box_area=100)
trk = BYTETracker(args)

orig = STrack.multi_gmc
state = {"frame": 0}


def checked(stracks, H=np.eye(2, 3), mode="parity"):
    before = {id(s): (s.mean.copy(), s.covariance.copy()) for s in stracks}
    orig(stracks, H, mode)
    for s in stracks:
        P = s.covariance
        ev = np.linalg.eigvalsh((P + P.T) / 2).min()
        sc = np.abs(P).max()
        if not np.isfinite(P).all() or ev < -1e-12 * sc or not np.allclose(P, P.T, rtol=0, atol=1e-12 * sc):
            m0, P0 = before[id(s)]
            np.set_printoptions(precision=4, suppress=False, linewidth=200)
            print(f"\nFRAME {state['frame']} track {s.track_id} state {s.state}  min eig {ev:.3e}")
            print("mean before:", m0)
            print("mean after: ", s.mean)
            print("P before diag:", np.diag(P0))
            print("P before min eig:", np.linalg.eigvalsh((P0 + P0.T) / 2).min())
            print("P after diag: ", np.diag(P))
            print("H:", H)
            sys.exit(1)


STrack.multi_gmc = staticmethod(checked)
for fr in range(1, max(dets) + 1):
    state["frame"] = fr
    im = cv2.imread(os.path.join(IMG, f"{fr:07d}.jpg"))
    h, w = im.shape[:2]
    out = torch.tensor(np.array(dets.get(fr, np.zeros((0, 7))), dtype=np.float32).reshape(-1, 7))
    # also check covariances entering the frame (after last update) to see if P was already bad
    for s in trk.tracked_stracks + trk.lost_stracks:
        ev = np.linalg.eigvalsh((s.covariance + s.covariance.T) / 2).min()
        if ev < -1e-6:
            print(f"FRAME {fr}: track {s.track_id} ENTERS non-PSD (min eig {ev:.3e}) -- broken before the warp")
            print("mean:", s.mean); sys.exit(1)
    trk.update(out, [im, h, w, fr, SEQ], (h, w))
    if fr % 100 == 0:
        print(f"frame {fr}: {len(trk.tracked_stracks)} tracked, {len(trk.lost_stracks)} lost", flush=True)
print("no non-PSD covariance found")
