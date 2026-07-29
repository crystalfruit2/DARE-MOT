"""Standalone CPU residual-logging pass (2026-07-29).

Cheapest decisive experiment for the scale-conditional CMC gate design
(Projects/Dare_Mot/cmc-gate-design-2026-07-29.md): run the exact GMC config used by the
mc_dare_cmc / mc_bytetrack_cmc runs (DARE_CMC=sparseOptFlow, downscale=2, unmasked — see
byte_tracker.py on exp/cmc-move1) standalone over every val sequence's raw frames, no
tracker involved, and log the per-frame warp-fit residual instrumented into
baselines/gmc.py's applySparseOptFlow. Output feeds _cmc_idsw_join.py.

Frame indexing matches gt.txt / track_results (1-indexed, frame N's residual describes
the N-1 -> N warp; frame 1 has no prior frame so residual is None).
"""
import json
import os

import cv2

from baselines.gmc import GMC

VAL_DIR = r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone_MOT_Format\VisDrone2019-MOT-val"
OUT_DIR = r"C:\Users\User\Desktop\projects\DARE-MOT\_cmc_residual_logs"
SEQS = [
    "uav0000086_00000_v", "uav0000117_02622_v", "uav0000137_00458_v", "uav0000182_00000_v",
    "uav0000268_05773_v", "uav0000305_00000_v", "uav0000339_00001_v",
]


def process_seq(seq):
    imgdir = os.path.join(VAL_DIR, seq, "img1")
    frames = sorted(f for f in os.listdir(imgdir) if f.endswith(".jpg"))

    gmc = GMC(method="sparseOptFlow", downscale=2)
    log = []
    for fname in frames:
        frame_id = int(os.path.splitext(fname)[0])
        img = cv2.imread(os.path.join(imgdir, fname))
        if img is None:
            continue
        gmc.apply(img, None)
        log.append({
            "frame": frame_id,
            "resid_px": gmc.last_resid_px,
            "resid_p90_px": gmc.last_resid_p90_px,
            "inlier_ratio": gmc.last_inlier_ratio,
            "trans_px": gmc.last_trans_px,
        })
    return log


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for seq in SEQS:
        print(f"{seq} ...", flush=True)
        log = process_seq(seq)
        out_path = os.path.join(OUT_DIR, f"{seq}.json")
        with open(out_path, "w") as f:
            json.dump(log, f, indent=1)
        n_valid = sum(1 for r in log if r["resid_px"] is not None)
        print(f"{seq}: {len(log)} frames, {n_valid} with valid residual -> {out_path}")
    print("DONE")


if __name__ == "__main__":
    main()
