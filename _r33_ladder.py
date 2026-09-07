"""R3-3 step 5: the four-arm ladder the prior art forces, plus the assignment-margin test.

PRIOR-ART CORRECTION (2026-09-07, found while running this diagnostic). The candidate was
written on the premise that "CMC is only ever applied to the next-frame predict step,
never chained across a gap". That premise is FALSE for BoT-SORT: `bot_sort.py` builds
`strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)` (line 292) and then calls
`STrack.multi_gmc(strack_pool, warp)` (line 299) — the warp is applied to LOST tracks too,
every frame. Applied repeatedly across an N-frame gap that composes into exactly the
cumulative affine this diagnostic computes. So chaining CMC across a gap is not new.

What may still be new is chaining it inside a RETROSPECTIVE, two-endpoint interpolation:
BoT-SORT's chained warp rides a forward KF extrapolation that never sees the re-associated
observation z2; OC-SORT's ORU sees z2 but has no CMC. The ladder that separates these:

  A  cv_plain : z1 + k*v1                      -> DARE-MOT / ByteTrack today (no CMC)
  B  cv_cmc   : M_t (z1 + k*v1)                -> BoT-SORT today (chained CMC, forward)
  C  lerp     : z1 + a (z2 - z1)               -> OC-SORT ORU today (retrospective, no CMC)
  D  lerp_cmc : M_t [z1 + a (M_t2^-1 z2 - z1)] -> THE CANDIDATE

The novelty is D - C. B is the prior-art alternative a reviewer will reach for. A is the
floor. (A/B are forward-only, usable during the gap for re-association gating; C/D are
retrospective, usable only after re-association to re-update the KF. Their reconstruction
accuracy is still the right common yardstick.)

MARGIN TEST (the discipline that killed Round 3 #1): ORU's only channel into the tracker is
the post-recovery KF velocity — the box at t2 is re-anchored on the detection regardless.
So the question is whether the D-vs-C velocity difference can flip a subsequent
association. For each real re-association we propagate both velocities forward and compare
the resulting IoU shift against the actual IoU margin (own-track IoU minus best competitor
IoU) at those frames.
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
HORIZON = 5          # frames after re-association over which the velocity difference acts


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


def warp_pt(M, x, y):
    p = M @ np.array([x, y, 1.0])
    return p[:2]


def iou_c(pc, w, h, tc, tw, th):
    iw = max(0.0, min(pc[0] + w / 2, tc[0] + tw / 2) - max(pc[0] - w / 2, tc[0] - tw / 2))
    ih = max(0.0, min(pc[1] + h / 2, tc[1] + th / 2) - max(pc[1] - h / 2, tc[1] - th / 2))
    inter = iw * ih
    return inter / max(w * h + tw * th - inter, 1e-9)


def ladder():
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
                    for i0 in range(s + 1, e - span + 1, STRIDE):   # s+1: need v1
                        i1 = i0 + span
                        if i1 > e:
                            continue
                        t1, t2 = int(f[i0]), int(f[i1])
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
                        v1 = p1 - np.array([cx[i0 - 1], cy[i0 - 1]])   # velocity just before the gap
                        p2b = warp_pt(inv2, p2[0], p2[1])
                        size = float(np.sqrt(max(w[i0] * h[i0], 1.0)))
                        errs = {k: [] for k in ("A", "B", "C", "D")}
                        ious = {k: [] for k in ("A", "B", "C", "D")}
                        last = {}
                        for k in range(1, span):
                            t = t1 + k
                            a = k / float(span)
                            truth = np.array([cx[i0 + k], cy[i0 + k]])
                            tw, th = w[i0 + k], h[i0 + k]
                            pw = w[i0] + a * (w[i1] - w[i0]); ph = h[i0] + a * (h[i1] - h[i0])
                            pred = {
                                "A": p1 + k * v1,
                                "B": warp_pt(M[t], *(p1 + k * v1)),
                                "C": p1 + a * (p2 - p1),
                                "D": warp_pt(M[t], *(p1 + a * (p2b - p1))),
                            }
                            for kk, pv in pred.items():
                                errs[kk].append(np.linalg.norm(pv - truth))
                                ious[kk].append(iou_c(pv, pw, ph, truth, tw, th))
                            last = pred
                        v_true = p2 - np.array([cx[i1 - 1], cy[i1 - 1]])
                        r = dict(seq=seq, tid=int(tid), L=L, t1=t1, t2=t2, size=size)
                        for kk in ("A", "B", "C", "D"):
                            r["e_" + kk] = float(np.mean(errs[kk])) / size
                            r["iou_" + kk] = float(np.mean(ious[kk]))
                            r["ve_" + kk] = float(np.linalg.norm((p2 - last[kk]) - v_true))
                        rows.append(r)
    return pd.DataFrame(rows)


def margin_test(d):
    """How large is the D-vs-C velocity difference compared with the IoU margin the solver
    actually has in the frames right after re-association?"""
    out = []
    for seq in SEQS:
        b = load_boxes(seq)
        b["cx"] = b["x"] + b["w"] / 2.0
        b["cy"] = b["y"] + b["h"] / 2.0
        byf = {int(t): g for t, g in b.groupby("frame")}
        sub = d[d.seq == seq]
        for r in sub.itertuples():
            own = b[(b.track_id == r.tid)]
            for k in range(1, HORIZON + 1):
                t = r.t2 + k
                row = own[own.frame == t]
                fr = byf.get(t)
                if len(row) == 0 or fr is None:
                    continue
                tc = np.array([row.cx.iloc[0], row.cy.iloc[0]])
                tw, th = row.w.iloc[0], row.h.iloc[0]
                # own IoU when the prediction is exact, vs best competing box in that frame
                comp = fr[fr.track_id != r.tid]
                best = 0.0
                for c in comp.itertuples():
                    best = max(best, iou_c(tc, tw, th, np.array([c.cx, c.cy]), c.w, c.h))
                # IoU cost the velocity difference would introduce: shift of k*dv
                dv = abs(r.ve_C - r.ve_D)
                shifted = iou_c(tc + np.array([k * dv, 0.0]), tw, th, tc, tw, th)
                out.append(dict(seq=seq, tid=r.tid, L=r.L, k=k,
                                margin=1.0 - best, iou_loss=1.0 - shifted, dv=dv,
                                flippable=(1.0 - shifted) > (1.0 - best)))
    return pd.DataFrame(out)


def main():
    pd.set_option("display.width", 240)
    d = ladder()
    d.to_csv("_scratch/_r33_ladder.csv", index=False)
    print("=== four-arm ladder, %d pseudo-gap windows ===" % len(d))
    print("mean centre error over the hidden frames, BOX UNITS (median across windows)\n")
    t = d.groupby("L").agg(n=("L", "size"), A_cv=("e_A", "median"), B_cv_cmc=("e_B", "median"),
                           C_lerp=("e_C", "median"), D_lerp_cmc=("e_D", "median")).round(4)
    print(t)
    print("\nsame, mean IoU of the reconstructed box vs the observed one:")
    print(d.groupby("L").agg(A=("iou_A", "median"), B=("iou_B", "median"),
                             C=("iou_C", "median"), D=("iou_D", "median")).round(4))
    print("\nterminal velocity error vs truth, px/frame (median):")
    print(d.groupby("L").agg(A=("ve_A", "median"), B=("ve_B", "median"),
                             C=("ve_C", "median"), D=("ve_D", "median")).round(3))
    print("\npooled medians  A %.4f  B %.4f  C %.4f  D %.4f  (box units)"
          % (d.e_A.median(), d.e_B.median(), d.e_C.median(), d.e_D.median()))
    print("D beats C in %.1f%% of windows; D beats B in %.1f%%; C beats B in %.1f%%"
          % (100 * (d.e_D < d.e_C).mean(), 100 * (d.e_D < d.e_B).mean(), 100 * (d.e_C < d.e_B).mean()))
    print("IoU<0.5 rate:  A %.2f%%  B %.2f%%  C %.2f%%  D %.2f%%"
          % tuple(100 * (d["iou_" + k] < 0.5).mean() for k in "ABCD"))

    m = margin_test(d)
    m.to_csv("_scratch/_r33_margin.csv", index=False)
    print("\n=== margin test: can the D-vs-C velocity difference flip an association? ===")
    print("%d (track, frame) checks in the %d frames after re-association" % (len(m), HORIZON))
    print("IoU margin (1 - best competitor IoU): median %.4f  p10 %.4f  min %.4f"
          % (m.margin.median(), np.percentile(m.margin, 10), m.margin.min()))
    print("IoU loss from the D-vs-C velocity difference: median %.5f  p90 %.5f  max %.5f"
          % (m.iou_loss.median(), np.percentile(m.iou_loss, 90), m.iou_loss.max()))
    print("checks where the difference exceeds the margin: %.4f%% (%d of %d)"
          % (100 * m.flippable.mean(), int(m.flippable.sum()), len(m)))


if __name__ == "__main__":
    main()
