"""Where on uav0000305 do the identity switches happen, relative to the GMC rotation burst? (CPU, 2026-09-15)

_diag_warp305_2026-09-15.py found that GMC returns |theta| ~ 0.13 rad on a contiguous tail of frames of 305
(median |theta| 1e-4 hides it), and that only there do BoT-SORT's kron(I4,sR) warp and the scale warp differ by
> 0.5 px in box size. If the warp is the mechanism, BoT-SORT's extra switches must sit inside that burst.
Same inputs as the headline IDSw (voted + official-filtered tracks, official-filtered GT, per-class accumulators).
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import _score_multiclass as sm  # noqa: E402  (carries the motmetrics / NumPy shim)
import motmetrics as mm  # noqa: E402
import tempfile  # noqa: E402

SEQ = "uav0000305_00000_v"
GT = os.path.join(ROOT, "_scratch", "official", "_gt", SEQ, "gt", "gt.txt")
VOTED = os.path.join(ROOT, "_scratch", "path1_score", "{}__on_score_official", SEQ + ".txt")
H = np.load(os.path.join(ROOT, "_scratch", "diag_warp305", SEQ + "_affines.npy"))
theta = np.arctan2(H[:, 1, 0], H[:, 0, 0])
burst = np.where(np.abs(theta) > 1e-2)[0] + 1          # 1-based frame ids
b0, b1 = int(burst.min()), int(burst.max())
contiguous = len(burst) == b1 - b0 + 1
print(f"rotation burst: frames {b0}-{b1} ({len(burst)} frames, contiguous={contiguous}) of {len(H)}")

ARMS = [("p1_botsort", "BoT-SORT-ReID (kron warp)"), ("p1_botsort_scalewarp", "BoT-SORT-ReID + scale warp"),
        ("p1_botsort_noreid", "BoT-SORT no ReID"), ("p1_deepocsort", "Deep OC-SORT (BoT GMC on corners)"),
        ("p1ref_bt_cmcscale_j", "ByteTrack + scale CMC"), ("p1_ocsort", "OC-SORT (no CMC)"), ("p1ref_bt", "ByteTrack (no CMC)")]
with tempfile.TemporaryDirectory() as tmp:
    for expn, label in ARMS:
        src = VOTED.format(expn)
        if not os.path.exists(src):
            print(f"{label:36s} (no voted file)")
            continue
        frames = []
        for cls in sm.MC_NAMES:
            gt_f, n_gt = sm._filter_by_class(GT, cls, tmp, "gt")
            ts_f, n_ts = sm._filter_by_class(src, cls, tmp, "ts")
            if n_gt == 0 and n_ts == 0:
                continue
            gt = mm.io.loadtxt(gt_f, fmt="mot15-2D", min_confidence=1)
            ts = mm.io.loadtxt(ts_f, fmt="mot15-2D", min_confidence=-1)
            acc = mm.utils.compare_to_groundtruth(gt, ts, "iou", distth=0.5)
            ev = acc.mot_events
            frames += [int(f) for f in ev[ev.Type == "SWITCH"].index.get_level_values(0)]
        frames = np.array(sorted(frames))
        inside = int(((frames >= b0) & (frames <= b1)).sum())
        print(f"{label:36s} IDSw {len(frames):3d} | inside burst {inside:3d} | before {len(frames) - inside:3d} | frames {frames.tolist()}")
