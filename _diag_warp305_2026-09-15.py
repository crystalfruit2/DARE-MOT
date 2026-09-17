"""Why does BoT-SORT's kron(I4, sR) warp hurt uav0000305 while the scale warp does not? (CPU, 2026-09-15)

The review argued R ~ sI on 305 (median |theta| ~ 1e-4), so the two warps should be numerically identical --
yet swapping the warp took BoT-SORT 21 -> 8 IDSw on 305 and changed nothing elsewhere. This recomputes the
per-frame GMC affine with the exact estimator every arm used (baselines/gmc.py, sparseOptFlow, downscale 2)
and reports the tail, not the median: |theta| quantiles/max, frames where RANSAC returned a large rotation,
and the per-frame and cumulative difference the two warps would make to a typical box's (w, h, vw, vh).
Sequences: 305 (the failure) and 086 / 117 (highest-rotation / camera-motion controls).
"""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from baselines.gmc import GMC  # noqa: E402

VAL = r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone2019-MOT-val\sequences"
OUT = os.path.join(ROOT, "_scratch", "diag_warp305")
os.makedirs(OUT, exist_ok=True)


def affines(seq):
    g = GMC(method="sparseOptFlow", downscale=2)
    frames = sorted(f for f in os.listdir(os.path.join(VAL, seq)) if f.endswith(".jpg"))
    H = []
    for f in frames:
        H.append(g.apply(cv2.imread(os.path.join(VAL, seq, f)), None))
    return np.array(H)


def warps(H, box_wh=(40.0, 30.0), vel_wh=(0.3, 0.2)):
    """Per-frame |difference| between kron(I4,sR) and scale warps applied to (w,h,vw,vh) of a fixed box."""
    d_size, d_vel = [], []
    for A in H:
        R = A[:2, :2]
        s = float(np.sqrt(abs(np.linalg.det(R))))
        kw = R @ np.array(box_wh)
        kv = R @ np.array(vel_wh)
        sw = s * np.array(box_wh)
        sv = s * np.array(vel_wh)
        d_size.append(np.abs(kw - sw).max())
        d_vel.append(np.abs(kv - sv).max())
    return np.array(d_size), np.array(d_vel)


for seq in ("uav0000305_00000_v", "uav0000086_00000_v", "uav0000117_02622_v"):
    H = affines(seq)
    np.save(os.path.join(OUT, f"{seq}_affines.npy"), H)
    a, b, c, dd = H[:, 0, 0], H[:, 0, 1], H[:, 1, 0], H[:, 1, 1]
    theta = np.arctan2(c, a)
    s = np.sqrt(np.abs(a * dd - b * c))
    ident = np.all(np.isclose(H, np.eye(2, 3)), axis=(1, 2))
    shear = np.abs((a - dd)) + np.abs(b + c)            # 0 for a pure similarity
    ds, dv = warps(H)
    q = lambda x: " / ".join(f"{v:.2e}" for v in np.quantile(np.abs(x), [0.5, 0.9, 0.99, 1.0]))
    print(f"\n=== {seq}: {len(H)} frames, identity (estimator fell back) {ident.sum()}")
    print(f"  |theta| median/p90/p99/max : {q(theta)}   frames |theta|>1e-2: {(np.abs(theta) > 1e-2).sum()}  >5e-2: {(np.abs(theta) > 5e-2).sum()}")
    print(f"  |s-1|   median/p90/p99/max : {q(s - 1)}   cumulative scale {np.prod(s):.3f}")
    print(f"  similarity residual (|a-d|+|b+c|) median/max: {np.median(shear):.2e} / {shear.max():.2e}")
    print(f"  kron vs scale |d size| px (40x30 box) median/p99/max: {np.median(ds):.3f} / {np.quantile(ds, 0.99):.3f} / {ds.max():.3f};"
          f" frames > 0.5 px: {(ds > 0.5).sum()}")
    worst = np.argsort(-np.abs(theta))[:5]
    print("  largest-|theta| frames (idx: theta, s, tx, ty):",
          "; ".join(f"{i + 1}: {theta[i]:+.3f}, {s[i]:.3f}, {H[i, 0, 2]:+.1f}, {H[i, 1, 2]:+.1f}" for i in worst))
