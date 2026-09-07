"""R3-3 step 4: resolve the contradiction between step 2 and step 3, and measure the one
quantity that actually reaches the tracker.

Step 2 (real gaps): plain-lerp vs CMC-lerp disagree by a median 0.17 box units, 27% above
1.0 box unit.
Step 3 (pseudo-gaps on continuously tracked segments): CMC-lerp is more accurate, but both
are accurate in absolute terms (0.028 vs 0.020 box units) — 5x less disagreement than the
real gaps show at the same gap length.

Two readings, opposite verdicts:
  (A) real gaps occur exactly when the camera is doing something hard, so the mechanism
      fires where it is needed  -> selection bias in step 3, candidate looks better;
  (B) real gaps occur exactly where GMC ITSELF fails, so the composed chain is garbage
      there -> the step-2 divergence is chain blow-up, candidate is dead.

Three probes:

  P1  GMC fit quality (inlier ratio, reprojection residual) inside real-gap intervals vs
      the rest of the same sequence. (B) predicts real gaps sit on bad fits.

  P2  Pseudo-gaps restricted to GAP-CONCURRENT windows: tracks that survived across the
      same frame interval where some other track had a real gap. Same camera conditions,
      but ground truth available. Separates (A) from (B) directly.

  P3  Terminal velocity error against truth. This is what ORU actually hands the Kalman
      filter at re-association, so it is the quantity with a downstream path — the
      analogue of the assignment-margin measurement that killed Round 3 #1. Also IoU of
      the reconstructed box against the observed one, since IoU is what the next
      association step sees.
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
CLSMAP = {1: "pedestrian", 2: "car", 3: "van", 4: "truck", 5: "bus"}


def load_boxes(seq):
    raw = np.loadtxt(osp.join(TRKDIR, seq + ".txt"), delimiter=",", ndmin=2)
    return pd.DataFrame({"frame": raw[:, 0].astype(int), "track_id": raw[:, 1].astype(int),
                         "x": raw[:, 2], "y": raw[:, 3], "w": raw[:, 4], "h": raw[:, 5],
                         "cls": raw[:, 7].astype(int)})


def load_aff_df(seq):
    return pd.read_csv(osp.join(AFF_DIR, seq + ".csv"))


def aff_dict(df):
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


def iou_c(pred_c, w, h, true_c, tw, th):
    ax1, ay1 = pred_c[0] - w / 2, pred_c[1] - h / 2
    ax2, ay2 = pred_c[0] + w / 2, pred_c[1] + h / 2
    bx1, by1 = true_c[0] - tw / 2, true_c[1] - th / 2
    bx2, by2 = true_c[0] + tw / 2, true_c[1] + th / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    return inter / max(w * h + tw * th - inter, 1e-9)


def main():
    gaps = pd.read_csv("_scratch/_r33_gap_divergence.csv")

    # ---------------- P1: GMC fit quality inside real gaps vs elsewhere ----------------
    print("=== P1: GMC fit quality inside real-gap intervals vs the rest of the sequence ===")
    p1 = []
    gapframes = {}
    for seq in SEQS:
        adf = load_aff_df(seq).set_index("frame")
        gf = set()
        for r in gaps[gaps.seq == seq].itertuples():
            gf.update(range(r.t1 + 1, r.t2 + 1))
        gapframes[seq] = gf
        inside = adf.index.isin(sorted(gf))
        valid = adf["inlier_ratio"] >= 0
        a_in = adf[inside & valid]
        a_out = adf[(~inside) & valid]
        if len(a_in) == 0:
            continue
        p1.append(dict(seq=seq, n_in=len(a_in), n_out=len(a_out),
                       inlier_in=a_in.inlier_ratio.median(), inlier_out=a_out.inlier_ratio.median(),
                       resid_in=a_in.resid_px.median(), resid_out=a_out.resid_px.median(),
                       trans_in=np.hypot(a_in.tx, a_in.ty).median(),
                       trans_out=np.hypot(a_out.tx, a_out.ty).median()))
    pd.set_option("display.width", 220)
    print(pd.DataFrame(p1).round(4).to_string(index=False))

    # ---------------- P2/P3: pseudo-gaps, gap-concurrent flag, velocity + IoU ----------
    rows = []
    for seq in SEQS:
        H = aff_dict(load_aff_df(seq))
        gf = gapframes[seq]
        b = load_boxes(seq)
        b["cx"] = b["x"] + b["w"] / 2.0
        b["cy"] = b["y"] + b["h"] / 2.0
        for tid, g in b.groupby("track_id"):
            g = g.sort_values("frame")
            f = g["frame"].to_numpy()
            cx = g["cx"].to_numpy(); cy = g["cy"].to_numpy()
            w = g["w"].to_numpy(); h = g["h"].to_numpy()
            cls = int(g["cls"].iloc[0])
            brk = np.where(np.diff(f) != 1)[0]
            for s, e in zip(np.r_[0, brk + 1], np.r_[brk, len(f) - 1]):
                n = e - s + 1
                for L in LENGTHS:
                    span = L + 1
                    if n < span + 1:
                        continue
                    for i0 in range(s, e - span + 1, STRIDE):
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
                        p1v = np.array([cx[i0], cy[i0]])
                        p2v = np.array([cx[i1], cy[i1]])
                        p2b = warp_pt(inv2, p2v[0], p2v[1])
                        size = float(np.sqrt(max(w[i0] * h[i0], 1.0)))
                        ep, ec = [], []
                        ip, ic = [], []
                        last_plain = last_cmc = None
                        for k in range(1, span):
                            t = t1 + k
                            a = k / float(span)
                            truth = np.array([cx[i0 + k], cy[i0 + k]])
                            tw, th = w[i0 + k], h[i0 + k]
                            plain = p1v + a * (p2v - p1v)
                            cmcp = warp_pt(M[t], *(p1v + a * (p2b - p1v)))
                            sc = float(np.sqrt(abs(np.linalg.det(M[t][:2, :2]))))
                            pw, ph = w[i0] + a * (w[i1] - w[i0]), h[i0] + a * (h[i1] - h[i0])
                            ep.append(np.linalg.norm(plain - truth))
                            ec.append(np.linalg.norm(cmcp - truth))
                            ip.append(iou_c(plain, pw, ph, truth, tw, th))
                            ic.append(iou_c(cmcp, pw * sc / sc, ph, truth, tw, th))
                            last_plain, last_cmc = plain, cmcp
                        # terminal velocity: what ORU hands the KF at re-association
                        v_true = p2v - np.array([cx[i1 - 1], cy[i1 - 1]])
                        v_plain = p2v - last_plain
                        v_cmc = p2v - last_cmc
                        gapconc = len(gf.intersection(range(t1 + 1, t2 + 1))) > 0
                        rows.append(dict(seq=seq, tid=int(tid), L=L, t1=t1,
                                         cls=CLSMAP.get(cls, str(cls)), size=size,
                                         gapconc=gapconc,
                                         plain=float(np.mean(ep)) / size,
                                         cmc=float(np.mean(ec)) / size,
                                         iou_plain=float(np.mean(ip)), iou_cmc=float(np.mean(ic)),
                                         ve_plain=float(np.linalg.norm(v_plain - v_true)),
                                         ve_cmc=float(np.linalg.norm(v_cmc - v_true)),
                                         v_true=float(np.linalg.norm(v_true))))
    d = pd.DataFrame(rows)
    d.to_csv("_scratch/_r33_pseudogap2.csv", index=False)

    print("\n=== P2: gap-concurrent windows (same frames where real gaps happened) ===")
    print("%d gap-concurrent vs %d ordinary windows\n" % (int(d.gapconc.sum()), int((~d.gapconc).sum())))
    t = d.groupby(["gapconc", "L"]).agg(n=("L", "size"), plain=("plain", "median"),
                                        cmc=("cmc", "median"),
                                        iou_plain=("iou_plain", "median"),
                                        iou_cmc=("iou_cmc", "median")).round(4)
    t["cmc_wins"] = d.groupby(["gapconc", "L"]).apply(
        lambda g: float((g.cmc < g.plain).mean()), include_groups=False).round(3)
    print(t)

    print("\n=== P3: terminal velocity error vs truth (px/frame) — what ORU feeds the KF ===")
    v = d.groupby("L").agg(n=("L", "size"), v_true=("v_true", "median"),
                           ve_plain=("ve_plain", "median"), ve_cmc=("ve_cmc", "median")).round(3)
    v["cmc_wins"] = d.groupby("L").apply(lambda g: float((g.ve_cmc < g.ve_plain).mean()),
                                         include_groups=False).round(3)
    print(v)
    print("\npooled velocity error: plain %.3f px/f | cmc %.3f px/f | true speed %.3f px/f"
          % (d.ve_plain.median(), d.ve_cmc.median(), d.v_true.median()))
    print("IoU of reconstructed box vs observed: plain %.4f | cmc %.4f"
          % (d.iou_plain.median(), d.iou_cmc.median()))
    print("windows where plain IoU < 0.5 (would break an IoU match): %.2f%%  | cmc: %.2f%%"
          % (100 * (d.iou_plain < 0.5).mean(), 100 * (d.iou_cmc < 0.5).mean()))


if __name__ == "__main__":
    main()
