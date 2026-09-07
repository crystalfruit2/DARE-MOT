"""R3-3 step 1: standalone per-frame GMC affine log (2026-09-07).

Same GMC config as the mc_dare_cmc / mc_bytetrack_cmc runs (sparseOptFlow, downscale=2,
unmasked) — see _cmc_residual_pass.py, which logs only the fit residual. Here we keep the
2x3 affine H itself, which is what a CMC-CHAINED virtual trajectory needs.

Semantics (BoT-SORT): H_t maps FULL-RES coords of frame t-1 -> frame t. Frame 1 = identity.
Needs no `lap` — pure OpenCV/CPU, so it runs under the Smart App Control blocker.
"""
import os
import cv2
import numpy as np
from baselines.gmc import GMC

VAL_DIR = r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone_MOT_Format\VisDrone2019-MOT-val"
OUT_DIR = r"C:\Users\User\Desktop\projects\DARE-MOT\_r33_affines"
SEQS = ["uav0000086_00000_v", "uav0000117_02622_v", "uav0000137_00458_v", "uav0000182_00000_v",
        "uav0000268_05773_v", "uav0000305_00000_v", "uav0000339_00001_v"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for seq in SEQS:
        imgdir = os.path.join(VAL_DIR, seq, "img1")
        frames = sorted(f for f in os.listdir(imgdir) if f.endswith(".jpg"))
        gmc = GMC(method="sparseOptFlow", downscale=2)
        rows = []
        for fname in frames:
            fid = int(os.path.splitext(fname)[0])
            img = cv2.imread(os.path.join(imgdir, fname))
            if img is None:
                continue
            H = gmc.apply(img, None)
            rows.append([fid, H[0, 0], H[0, 1], H[0, 2], H[1, 0], H[1, 1], H[1, 2],
                         -1.0 if gmc.last_inlier_ratio is None else gmc.last_inlier_ratio,
                         -1.0 if gmc.last_resid_px is None else gmc.last_resid_px])
        arr = np.array(rows, dtype=float)
        out = os.path.join(OUT_DIR, seq + ".csv")
        np.savetxt(out, arr, delimiter=",", fmt="%.8f",
                   header="frame,a,b,tx,c,d,ty,inlier_ratio,resid_px", comments="")
        print(f"{seq}: {len(arr)} frames -> {out}", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
