"""R3-3 step 5b: the ladder again, with arm B built HONESTLY.

The first pass (_r33_ladder.py) measured v1 from raw image-space box centres and then also
applied the cumulative CMC warp — double-counting the camera motion, which made the
prior-art arm look worse than no CMC at all (B 0.128 vs A 0.079 box units). That was a
strawman, not a finding. In BoT-SORT the Kalman update runs AFTER each frame's gmc warp,
so the velocity state converges to object motion in the STABILISED frame; the warp then
supplies the camera component exactly once.

Corrected arms (all reconstructing the hidden frames of a pseudo-gap):
  A  cv_raw   : p1 + k*v_raw                        DARE-MOT / ByteTrack today (no CMC)
  B  cv_cmc   : M_t (p1 + k*v_comp)                 BoT-SORT / DARE exp/cmc-move1 today
                 with v_comp = p1 - H_t1(p_{t1-1})  (camera motion removed once)
  C  lerp     : p1 + a (p2 - p1)                    OC-SORT ORU today
  D  lerp_cmc : M_t [p1 + a (M_t2^-1 p2 - p1)]      THE CANDIDATE

A/B are forward-only (available during the gap, so they gate re-association); C/D are
retrospective (available only after re-association, so they re-update the KF). Different
roles — the shared yardstick is reconstruction accuracy against the observed boxes.
"""
import os.path as osp

import numpy as np
import pandas as pd

AFF_DIR = "_scratch/_r33_affines"
TRKDIR = osp.join("YOLOX_outputs", "mc_dare_cv_rerun0803", "track_results")
SEQS = ["uav0000086_00000_v", "uav0000117_02622_v", "uav0000137_00458_v", "uav0000182_00000_v",
        "uav0000268_05773_v", "uav0000305_00000_v", "uav0000339_00001_v"]
LENGTHS = [3, 5, 8, 12, 18, 25]
STRIDE = 5


def load_boxes(seq):
    raw = np.loadtxt(osp.join(TRKDIR, seq + ".txt"), delimiter=",", ndmin=2)
    return pd.DataFrame({"frame": raw[:, 0].astype(int), "track_id": raw[:, 1].astype(int),
                         "x": raw[:, 2], "y": raw[:, 3], "w": raw[:, 4], "h": raw[:, 5],
                         "cls": raw[:, 7].astype(int)})


def aff_dict(seq):
    df = pd.read_csv(osp.join(AFF_DIR, seq + ".csv"))
    H = {}
    for r in df.itertuples():
        M = np.eye(3)
        M[0, 0], M[0, 1], M[0, 2] = r.a, r.b, r.tx
        M[1, 0], M[1, 1], M[1, 2] = r.c, r.d, r.ty
        H[int(r.frame)] = M
    return H


def wp(M, p):
    q = M @ np.array([p[0], p[1], 1.0])
    return q[:2]


def iou_c(pc, w, h, tc, tw, th):
    iw = max(0.0, min(pc[0] + w / 2, tc[0] + tw / 2) - max(pc[0] - w / 2, tc[0] - tw / 2))
    ih = max(0.0, min(pc[1] + h / 2, tc[1] + th / 2) - max(pc[1] - h / 2, tc[1] - th / 2))
    inter = iw * ih
    return inter / max(w * h + tw * th - inter, 1e-9)


def main():
    rows = []
    for seq in SEQS:
        H = aff_dict(seq)
        b = load_boxes(seq)
        b["cx"] = b["x"] + b["w"] / 2.0
        b["cy"] = b["y"] + b["h"] / 2.0
        for tid, g in b.groupby("track_id"):
            g = g.sort_values("frame")
            f = g["frame"].to_numpy()
            cx = g["cx"].to_numpy(); cy = g["cy"].to_numpy()
            w = g["w"].to_numpy(); h = g["h"].to_numpy()
            brk = np.where(np.diff(f) != 1)[0]
            for s, e in zip(np.r_[0, brk + 1], np.r_[brk, len(f) - 1]):
                for L in LENGTHS:
                    span = L + 1
                    for i0 in range(s + 1, e - span + 1, STRIDE):
                        i1 = i0 + span
                        if i1 > e:
                            continue
                        t1, t2 = int(f[i0]), int(f[i1])
                        if t1 not in H:
                            continue
                        cur = np.eye(3); M = {t1: np.eye(3)}; ok = True
                        for t in range(t1 + 1, t2 + 1):
                            if t not in H:
                                ok = False; break
                            cur = H[t] @ cur
                            M[t] = cur
                        if not ok:
                            continue
                        try:
                            inv2 = np.linalg.inv(M[t2])
                        except np.linalg.LinAlgError:
                            continue
                        p1 = np.array([cx[i0], cy[i0]])
                        p2 = np.array([cx[i1], cy[i1]])
                        pprev = np.array([cx[i0 - 1], cy[i0 - 1]])
                        v_raw = p1 - pprev                    # apparent (image-space) velocity
                        v_comp = p1 - wp(H[t1], pprev)        # camera motion removed once
                        p2b = wp(inv2, p2)
                        size = float(np.sqrt(max(w[i0] * h[i0], 1.0)))
                        errs = {k: [] for k in "ABCD"}
                        ious = {k: [] for k in "ABCD"}
                        for k in range(1, span):
                            t = t1 + k
                            a = k / float(span)
                            truth = np.array([cx[i0 + k], cy[i0 + k]])
                            tw, th = w[i0 + k], h[i0 + k]
                            pw = w[i0] + a * (w[i1] - w[i0]); ph = h[i0] + a * (h[i1] - h[i0])
                            pred = {"A": p1 + k * v_raw,
                                    "B": wp(M[t], p1 + k * v_comp),
                                    "C": p1 + a * (p2 - p1),
                                    "D": wp(M[t], p1 + a * (p2b - p1))}
                            for kk, pv in pred.items():
                                errs[kk].append(np.linalg.norm(pv - truth))
                                ious[kk].append(iou_c(pv, pw, ph, truth, tw, th))
                        r = dict(seq=seq, tid=int(tid), L=L, t1=t1, t2=t2, size=size)
                        for kk in "ABCD":
                            r["e_" + kk] = float(np.mean(errs[kk])) / size
                            r["iou_" + kk] = float(np.mean(ious[kk]))
                        rows.append(r)
    d = pd.DataFrame(rows)
    d.to_csv("_scratch/_r33_ladder2.csv", index=False)
    pd.set_option("display.width", 240)
    print("=== corrected four-arm ladder, %d pseudo-gap windows ===" % len(d))
    print("centre error over hidden frames, BOX UNITS (median):")
    print(d.groupby("L").agg(n=("L", "size"), A=("e_A", "median"), B=("e_B", "median"),
                             C=("e_C", "median"), D=("e_D", "median")).round(4))
    print("\nIoU of reconstructed vs observed box (median):")
    print(d.groupby("L").agg(A=("iou_A", "median"), B=("iou_B", "median"),
                             C=("iou_C", "median"), D=("iou_D", "median")).round(4))
    print("\npooled  A %.4f  B %.4f  C %.4f  D %.4f" % (d.e_A.median(), d.e_B.median(),
                                                        d.e_C.median(), d.e_D.median()))
    print("B<A %.1f%% | C<B %.1f%% | D<C %.1f%% | D<B %.1f%%"
          % (100 * (d.e_B < d.e_A).mean(), 100 * (d.e_C < d.e_B).mean(),
             100 * (d.e_D < d.e_C).mean(), 100 * (d.e_D < d.e_B).mean()))
    print("IoU<0.5: A %.2f%% B %.2f%% C %.2f%% D %.2f%%"
          % tuple(100 * (d["iou_" + k] < 0.5).mean() for k in "ABCD"))
    e = d[d.seq != "uav0000305_00000_v"]
    print("\nexcl uav0000305: A %.4f B %.4f C %.4f D %.4f | D<C %.1f%% D<B %.1f%%"
          % (e.e_A.median(), e.e_B.median(), e.e_C.median(), e.e_D.median(),
             100 * (e.e_D < e.e_C).mean(), 100 * (e.e_D < e.e_B).mean()))
    print("\nper sequence:")
    t = d.groupby("seq").agg(n=("L", "size"), A=("e_A", "median"), B=("e_B", "median"),
                             C=("e_C", "median"), D=("e_D", "median")).round(4)
    t["D_beats_C"] = d.groupby("seq").apply(lambda g: float((g.e_D < g.e_C).mean()),
                                            include_groups=False).round(3)
    print(t)


if __name__ == "__main__":
    main()
