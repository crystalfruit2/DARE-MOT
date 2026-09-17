"""Scan sequences for camera-yaw segments with the exact GMC estimator every arm uses (CPU, 2026-09-15).

Motivation: on uav0000305 a sustained in-place yaw (|theta| ~ 0.13 rad/frame, frames 147-184) is where the
kron / corner CMC warps of BoT-SORT and Deep OC-SORT fail and the scale-exact warp does not. Before reading the
pre-registered out-of-val check R1 (parity vs scale on 7 train sequences) we need to know whether those
sequences contain any yaw at all - if not, "parity == scale" there is untested, not general. The same scan over
all 56 train sequences is the first step of a systematic yaw study.

A frame is "yaw" when |theta| > 0.01 rad (~0.57 deg/frame); a segment is >= 5 consecutive yaw frames.
Usage: python _diag_yaw_scan_2026-09-15.py trainsub|val|train   (writes _scratch/yaw_scan/<set>.json)
"""
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from baselines.gmc import GMC  # noqa: E402

SETS = {
    "val": (r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone2019-MOT-val\sequences", None),
    "trainsub": (r"C:\Users\User\Desktop\datasets\VisDrone2019-MOT-train\sequences",
                 ["uav0000071_03240_v", "uav0000124_00944_v", "uav0000222_03150_v", "uav0000266_03598_v",
                  "uav0000289_00001_v", "uav0000315_00000_v", "uav0000360_00001_v"]),
    "train": (r"C:\Users\User\Desktop\datasets\VisDrone2019-MOT-train\sequences", None),
}
THR, MINLEN = 0.01, 5


def segments(mask):
    segs, start = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if i - start >= MINLEN:
                segs.append((start + 1, i))          # 1-based inclusive frame ids
            start = None
    return segs


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "trainsub"
    root, seqs = SETS[which]
    seqs = seqs or sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    out_dir = os.path.join(ROOT, "_scratch", "yaw_scan")
    os.makedirs(out_dir, exist_ok=True)
    report = {}
    for seq in seqs:
        g = GMC(method="sparseOptFlow", downscale=2)
        frames = sorted(f for f in os.listdir(os.path.join(root, seq)) if f.endswith(".jpg"))
        th = []
        for f in frames:
            H = g.apply(cv2.imread(os.path.join(root, seq, f)), None)
            th.append(float(np.arctan2(H[1, 0], H[0, 0])))
        th = np.array(th)
        segs = segments(np.abs(th) > THR)
        yaw_frames = int(sum(b - a + 1 for a, b in segs))
        report[seq] = {"frames": len(th), "abs_theta_median": float(np.median(np.abs(th))),
                       "abs_theta_p99": float(np.quantile(np.abs(th), 0.99)), "abs_theta_max": float(np.abs(th).max()),
                       "yaw_segments": segs, "yaw_frames": yaw_frames,
                       "cumulative_rotation_rad": float(np.abs(th[np.abs(th) > THR]).sum())}
        print(f"{seq}: frames {len(th):4d} | |theta| med {np.median(np.abs(th)):.1e} p99 {np.quantile(np.abs(th), 0.99):.1e} "
              f"max {np.abs(th).max():.3f} | yaw segments {segs} ({yaw_frames} frames)", flush=True)
    json.dump(report, open(os.path.join(out_dir, f"{which}.json"), "w"), indent=1)
    n_yaw = sum(1 for v in report.values() if v["yaw_segments"])
    print(f"\n{which}: {n_yaw}/{len(report)} sequences contain a yaw segment (|theta|>{THR} rad for >= {MINLEN} frames)")


if __name__ == "__main__":
    main()
