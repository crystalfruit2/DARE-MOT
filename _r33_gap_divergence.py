"""R3-3 step 2: does a CMC-CHAINED virtual trajectory differ from OC-SORT's plain
image-space lerp? (2026-09-07, offline, no tracker, no `lap`.)

Candidate (Round 3 #3): on re-association after a gap, OC-SORT's ORU builds virtual
observations by lerping in RAW IMAGE coordinates:
    z~_t = z_t1 + (t-t1)/(t2-t1) * (z_t2 - z_t1)
The proposal is to instead compose the per-frame GMC affines across the gap into one
cumulative warp and lerp in the ego-motion-compensated frame.

STRUCTURAL NOTE THAT DEFINES THE MEASUREMENT. If the camera path over the gap is affine-
linear in time (constant-velocity pan, no rotation/zoom change), the two interpolants are
*identically equal*: the endpoint-anchored lerp already absorbs any linear component.
Proof: with M_t = I + (t-t1)*V (pure translation V per frame), z2' = z2 - (t2-t1)V, so
    z_cmc = M_t[z1 + a(z2' - z1)] = z1 + a(z2-z1) - a(t2-t1)V + (t-t1)V = z_plain.
So the ONLY thing this candidate can buy is the NON-LINEAR (accelerating / curving /
zooming) part of the camera path. Reporting "cumulative camera motion during the gap"
would be a false positive; we measure deviation from the chord instead.

Frame convention: H_t maps frame t-1 -> frame t (full-res). M_t = H_t @ ... @ H_{t1+1}.
Boxes are similarity-warped: centre through the affine, w/h by the similarity scale.
"""
import os.path as osp

import numpy as np
import pandas as pd

AFF_DIR = "_scratch/_r33_affines"
TRKDIR = osp.join("YOLOX_outputs", "mc_dare_cv_rerun0803", "track_results")
SEQS = ["uav0000086_00000_v", "uav0000117_02622_v", "uav0000137_00458_v", "uav0000182_00000_v",
        "uav0000268_05773_v", "uav0000305_00000_v", "uav0000339_00001_v"]
MIN_GAP = 3
CLSMAP = {1: "pedestrian", 2: "car", 3: "van", 4: "truck", 5: "bus"}  # 1-based COCO ids


def load_boxes(seq):
    raw = np.loadtxt(osp.join(TRKDIR, seq + ".txt"), delimiter=",", ndmin=2)
    return pd.DataFrame({"frame": raw[:, 0].astype(int), "track_id": raw[:, 1].astype(int),
                         "x": raw[:, 2], "y": raw[:, 3], "w": raw[:, 4], "h": raw[:, 5],
                         "cls": raw[:, 7].astype(int)})


def load_affines(seq):
    df = pd.read_csv(osp.join(AFF_DIR, seq + ".csv"))
    H = {}
    for r in df.itertuples():
        M = np.eye(3)
        M[0, 0], M[0, 1], M[0, 2] = r.a, r.b, r.tx
        M[1, 0], M[1, 1], M[1, 2] = r.c, r.d, r.ty
        H[int(r.frame)] = M
    return H


def warp_box(M, box):
    """box = (cx, cy, w, h) -> similarity-warped box."""
    cx, cy, w, h = box
    p = M @ np.array([cx, cy, 1.0])
    s = float(np.sqrt(abs(np.linalg.det(M[:2, :2]))))
    return np.array([p[0], p[1], w * s, h * s])


def gaps_for_seq(seq):
    b = load_boxes(seq)
    b["cx"] = b["x"] + b["w"] / 2.0
    b["cy"] = b["y"] + b["h"] / 2.0
    out = []
    for tid, g in b.groupby("track_id"):
        g = g.sort_values("frame")
        f = g["frame"].to_numpy()
        d = np.diff(f)
        for i in np.where(d - 1 >= MIN_GAP)[0]:
            r1, r2 = g.iloc[i], g.iloc[i + 1]
            out.append(dict(seq=seq, tid=int(tid), t1=int(r1.frame), t2=int(r2.frame),
                            gap=int(r2.frame - r1.frame - 1), cls=int(r1.cls),
                            z1=np.array([r1.cx, r1.cy, r1.w, r1.h]),
                            z2=np.array([r2.cx, r2.cy, r2.w, r2.h])))
    return out


def analyse_gap(g, H):
    t1, t2 = g["t1"], g["t2"]
    z1, z2 = g["z1"], g["z2"]
    # cumulative warps M[t] : coords(t1) -> coords(t)
    M = {t1: np.eye(3)}
    cur = np.eye(3)
    for t in range(t1 + 1, t2 + 1):
        h = H.get(t)
        if h is None:
            return None
        cur = h @ cur
        M[t] = cur
    try:
        z2_back = warp_box(np.linalg.inv(M[t2]), z2)
    except np.linalg.LinAlgError:
        return None

    ts = np.arange(t1 + 1, t2)          # the virtual (gap) frames
    a = (ts - t1) / float(t2 - t1)
    plain = z1[None, :] + a[:, None] * (z2 - z1)[None, :]
    cmc = np.stack([warp_box(M[t], z1 + ai * (z2_back - z1)) for t, ai in zip(ts, a)])

    dcen = np.hypot(cmc[:, 0] - plain[:, 0], cmc[:, 1] - plain[:, 1])
    size = np.sqrt(np.maximum(plain[:, 2] * plain[:, 3], 1.0))
    rel = dcen / size
    mid = len(ts) // 2

    # camera-path non-linearity at the object's own location, in px: deviation of
    # M_t(z1_centre) from the straight chord between its endpoints.
    pc = np.stack([(M[t] @ np.array([z1[0], z1[1], 1.0]))[:2] for t in range(t1, t2 + 1)])
    ac = (np.arange(t1, t2 + 1) - t1) / float(t2 - t1)
    chord = pc[0][None, :] + ac[:, None] * (pc[-1] - pc[0])[None, :]
    cam_nonlin = float(np.linalg.norm(pc - chord, axis=1).max())
    cam_total = float(np.linalg.norm(pc[-1] - pc[0]))

    A = M[t2][:2, :2]
    scale = float(np.sqrt(abs(np.linalg.det(A))))
    rot = float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))

    # terminal velocity handed to the KF by ORU (last virtual step -> t2)
    v_plain = z2[:2] - plain[-1, :2]
    v_cmc = z2[:2] - cmc[-1, :2]

    return dict(seq=g["seq"], tid=g["tid"], t1=t1, t2=t2, gap=g["gap"],
                cls=CLSMAP.get(g["cls"], str(g["cls"])),
                cx1=float(z1[0]), cy1=float(z1[1]),
                obj_disp=float(np.linalg.norm(z2[:2] - z1[:2])),
                box_size=float(np.sqrt(z1[2] * z1[3])),
                cam_total=cam_total, cam_nonlin=cam_nonlin,
                cum_rot_deg=rot, cum_scale=scale,
                d_mid_px=float(dcen[mid]), d_max_px=float(dcen.max()),
                d_mid_rel=float(rel[mid]), d_max_rel=float(rel.max()),
                dv_terminal_px=float(np.linalg.norm(v_cmc - v_plain)))


def p90(s):
    return np.percentile(s, 90)


def main():
    rows = []
    for seq in SEQS:
        H = load_affines(seq)
        for g in gaps_for_seq(seq):
            r = analyse_gap(g, H)
            if r is not None:
                rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv("_scratch/_r33_gap_divergence.csv", index=False)

    pd.set_option("display.width", 200)

    def q(col, p):
        return float(np.percentile(df[col], p))

    print("\n=== %d gaps >= %d frames across %d sequences ===\n" % (len(df), MIN_GAP, df.seq.nunique()))
    print("gap length      : median %.0f  p90 %.0f  max %.0f" % (q("gap", 50), q("gap", 90), df.gap.max()))
    print("box size (sqrt) : median %.1f px" % q("box_size", 50))
    print("object displacement over gap: median %.1f px" % q("obj_disp", 50))
    print("camera path over gap: total %.1f px (median) | NON-LINEAR part %.2f px (median), p90 %.2f, max %.2f"
          % (q("cam_total", 50), q("cam_nonlin", 50), q("cam_nonlin", 90), df.cam_nonlin.max()))
    print()
    print("DIVERGENCE  plain-lerp vs CMC-chained lerp")
    for c, lab in [("d_mid_px", "midpoint, px"), ("d_max_px", "max over gap, px"),
                   ("d_mid_rel", "midpoint, box units"), ("d_max_rel", "max over gap, box units"),
                   ("dv_terminal_px", "terminal velocity diff, px/frame")]:
        print("  %-32s median %8.4f   p90 %8.4f   p99 %8.4f   max %8.4f"
              % (lab, q(c, 50), q(c, 90), q(c, 99), df[c].max()))
    print()
    print("fraction of gaps with max divergence >= X box units:")
    for thr in (0.05, 0.1, 0.25, 0.5, 1.0):
        print("   >= %4.2f : %5.2f%%  (%d gaps)" % (thr, 100 * (df.d_max_rel >= thr).mean(),
                                                    int((df.d_max_rel >= thr).sum())))
    print("\nper sequence:")
    print(df.groupby("seq").agg(n=("gap", "size"), gap_med=("gap", "median"),
                                cam_nonlin_med=("cam_nonlin", "median"),
                                d_max_rel_med=("d_max_rel", "median"),
                                d_max_rel_p90=("d_max_rel", p90),
                                frac_ge_010=("d_max_rel", lambda s: (s >= 0.1).mean())).round(4))
    print("\nstratified by gap length (the fake-win check):")
    df["gapbin"] = pd.cut(df.gap, [2, 5, 10, 20, 50, 100000],
                          labels=["3-5", "6-10", "11-20", "21-50", "50+"])
    print(df.groupby("gapbin", observed=True).agg(
        n=("gap", "size"), cam_nonlin_med=("cam_nonlin", "median"),
        d_max_px_med=("d_max_px", "median"), d_max_rel_med=("d_max_rel", "median"),
        d_max_rel_p90=("d_max_rel", p90)).round(4))
    print("\nby class:")
    print(df.groupby("cls").agg(n=("gap", "size"), box=("box_size", "median"),
                                d_max_rel_med=("d_max_rel", "median")).round(4))


if __name__ == "__main__":
    main()
