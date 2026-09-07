"""R3-3 step 6 (decisive): does the C-vs-D velocity difference reach the association?

The discipline that killed Round 3 #1: a statistic can be real and still be unable to move
anything the solver sees. ORU's only channel into the tracker is the post-recovery Kalman
state — the box at t2 is re-anchored on the detection either way (verified in
`baselines/ocsort/kalmanfilter.py::unfreeze`, whose replay loop ends with `update(box2)`),
so what carries forward is the VELOCITY.

So the question is concrete: after the track resumes at t2 with velocity v_hat, the KF
predicts p2 + k*v_hat at t2+k. Does the plain-ORU velocity put that prediction outside the
IoU window that the next association needs, where the CMC-chained one does not?

Measured on the pseudo-gap windows, which have ground truth for the frames after t2 too.
Thresholds: IoU < 0.2 fails ByteTrack's first-association gate at match_thresh 0.8;
IoU < 0.5 is a badly degraded match that can lose to a competitor.

CAVEAT recorded up front: v_hat here is the terminal difference of the replayed virtual
trajectory, not the Kalman-smoothed velocity the real ORU replay would leave in the state.
The KF averages over the whole replayed path, so it will damp both arms' errors somewhat.
This measures the input to that smoothing, and is an upper bound on the gap between them.
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
HORIZON = 3


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
        # per-frame arrays of every other track's box, for the competitor scan
        byf = {}
        for t, g in b.groupby("frame"):
            byf[int(t)] = (g.track_id.to_numpy(), g.cx.to_numpy(), g.cy.to_numpy(),
                           g.w.to_numpy(), g.h.to_numpy())
        for tid, g in b.groupby("track_id"):
            g = g.sort_values("frame")
            f = g["frame"].to_numpy()
            cx = g["cx"].to_numpy(); cy = g["cy"].to_numpy()
            w = g["w"].to_numpy(); h = g["h"].to_numpy()
            brk = np.where(np.diff(f) != 1)[0]
            for s, e in zip(np.r_[0, brk + 1], np.r_[brk, len(f) - 1]):
                for L in LENGTHS:
                    span = L + 1
                    for i0 in range(s, e - span - HORIZON + 1, STRIDE):
                        i1 = i0 + span
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
                        p2b = wp(inv2, p2)
                        a_last = (span - 1) / float(span)
                        lastC = p1 + a_last * (p2 - p1)
                        lastD = wp(M[t2 - 1], p1 + a_last * (p2b - p1))
                        vC = p2 - lastC
                        vD = p2 - lastD
                        vT = p2 - np.array([cx[i1 - 1], cy[i1 - 1]])
                        for k in range(1, HORIZON + 1):
                            j = i1 + k
                            if j > e:
                                break
                            t = int(f[j])
                            tc = np.array([cx[j], cy[j]]); tw, th = w[j], h[j]
                            iouC = iou_c(p2 + k * vC, w[i1], h[i1], tc, tw, th)
                            iouD = iou_c(p2 + k * vD, w[i1], h[i1], tc, tw, th)
                            iouT = iou_c(p2 + k * vT, w[i1], h[i1], tc, tw, th)
                            # best competing box in that frame (what a wrong prediction could grab)
                            ids, ccx, ccy, cw, ch = byf[t]
                            mask = ids != tid
                            bestC = bestD = 0.0
                            if mask.any():
                                pc = p2 + k * vC
                                pd_ = p2 + k * vD
                                for xx, yy, ww, hh in zip(ccx[mask], ccy[mask], cw[mask], ch[mask]):
                                    o = np.array([xx, yy])
                                    bestC = max(bestC, iou_c(pc, w[i1], h[i1], o, ww, hh))
                                    bestD = max(bestD, iou_c(pd_, w[i1], h[i1], o, ww, hh))
                            rows.append(dict(seq=seq, tid=int(tid), L=L, t2=t2, k=k,
                                             iouC=iouC, iouD=iouD, iouT=iouT,
                                             bestC=bestC, bestD=bestD))
    d = pd.DataFrame(rows)
    d.to_csv("_scratch/_r33_downstream.csv", index=False)
    pd.set_option("display.width", 240)
    print("=== post-recovery prediction quality, %d (window, frame) checks ===" % len(d))
    print("IoU of the KF prediction at t2+k against the box actually observed there\n")
    t = d.groupby(["L", "k"]).agg(n=("k", "size"), C=("iouC", "median"), D=("iouD", "median"),
                                  oracle=("iouT", "median")).round(4)
    print(t)
    print("\nassociation-breaking rates (pooled over k=1..%d):" % HORIZON)
    for thr in (0.5, 0.2, 0.1):
        print("  prediction IoU < %.1f :  plain-ORU %6.3f%%   CMC-ORU %6.3f%%   oracle %6.3f%%"
              % (thr, 100 * (d.iouC < thr).mean(), 100 * (d.iouD < thr).mean(),
                 100 * (d.iouT < thr).mean()))
    print("\nID-SWITCH-SHAPED events (prediction overlaps a COMPETITOR more than its own box):")
    swC = (d.bestC > d.iouC) & (d.bestC > 0.1)
    swD = (d.bestD > d.iouD) & (d.bestD > 0.1)
    print("  plain-ORU %.3f%% (%d)   CMC-ORU %.3f%% (%d)"
          % (100 * swC.mean(), int(swC.sum()), 100 * swD.mean(), int(swD.sum())))
    print("  cases fixed by the CMC composition: %d;  cases broken by it: %d"
          % (int((swC & ~swD).sum()), int((~swC & swD).sum())))
    print("\nby gap length, IoU<0.2 rate:")
    print(d.groupby("L").apply(lambda g: pd.Series({
        "n": len(g), "plain": 100 * (g.iouC < 0.2).mean(), "cmc": 100 * (g.iouD < 0.2).mean(),
        "oracle": 100 * (g.iouT < 0.2).mean()}), include_groups=False).round(3))
    e = d[d.seq != "uav0000305_00000_v"]
    print("\nexcl uav0000305 — IoU<0.2: plain %.3f%%  cmc %.3f%%  oracle %.3f%%"
          % (100 * (e.iouC < 0.2).mean(), 100 * (e.iouD < 0.2).mean(), 100 * (e.iouT < 0.2).mean()))


if __name__ == "__main__":
    main()
