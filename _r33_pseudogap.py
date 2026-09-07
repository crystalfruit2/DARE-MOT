"""R3-3 step 3 (decisive): which virtual trajectory is actually MORE ACCURATE?

Step 2 established that a CMC-chained lerp and OC-SORT's plain image-space lerp diverge
substantially (median 0.17 box units, 27% of real gaps above 1.0 box unit). Divergence
alone is not a win — it only says the two disagree. The candidate is only alive if the
CMC-chained one is the one that is RIGHT.

Ground truth is available offline and for free: take tracks that were observed
CONTINUOUSLY, hide an interior window, rebuild both virtual trajectories from the two
surviving endpoints, and score each against the boxes that were actually observed inside
the window. Same camera motion, same GMC affines, same box source as the real gaps.

Reported per gap length L, because the real gap distribution is median 8 / p90 18 and the
step-2 divergence is strongly L-dependent.

Third arm as a sanity floor: HOLD (freeze at z1), which is what DARE-MOT does today (no
ORU at all). If plain lerp does not beat HOLD, ORU itself is not worth adding here and the
CMC variant is moot.
"""
import os.path as osp

import numpy as np
import pandas as pd

AFF_DIR = "_scratch/_r33_affines"
TRKDIR = osp.join("YOLOX_outputs", "mc_dare_cv_rerun0803", "track_results")
SEQS = ["uav0000086_00000_v", "uav0000117_02622_v", "uav0000137_00458_v", "uav0000182_00000_v",
        "uav0000268_05773_v", "uav0000305_00000_v", "uav0000339_00001_v"]
LENGTHS = [3, 5, 8, 12, 18, 25]     # brackets the real gap distribution (median 8, p90 18, max 29)
STRIDE = 5                          # windows per track are strided to limit overlap
CLSMAP = {1: "pedestrian", 2: "car", 3: "van", 4: "truck", 5: "bus"}


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


def warp_pt(M, x, y):
    p = M @ np.array([x, y, 1.0])
    return p[:2]


def main():
    rows = []
    for seq in SEQS:
        H = load_affines(seq)
        b = load_boxes(seq)
        b["cx"] = b["x"] + b["w"] / 2.0
        b["cy"] = b["y"] + b["h"] / 2.0
        for tid, g in b.groupby("track_id"):
            g = g.sort_values("frame")
            f = g["frame"].to_numpy()
            cx = g["cx"].to_numpy(); cy = g["cy"].to_numpy()
            w = g["w"].to_numpy(); h = g["h"].to_numpy()
            cls = int(g["cls"].iloc[0])
            # contiguous runs of observed frames
            brk = np.where(np.diff(f) != 1)[0]
            starts = np.r_[0, brk + 1]
            ends = np.r_[brk, len(f) - 1]
            for s, e in zip(starts, ends):
                n = e - s + 1
                for L in LENGTHS:
                    span = L + 1                      # t2 - t1
                    if n < span + 1:
                        continue
                    for i0 in range(s, e - span + 1, STRIDE):
                        i1 = i0 + span
                        t1, t2 = int(f[i0]), int(f[i1])
                        # cumulative warps
                        cur = np.eye(3); M = {t1: np.eye(3)}
                        ok = True
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
                        p2_back = warp_pt(inv2, p2[0], p2[1])
                        size = float(np.sqrt(max(w[i0] * h[i0], 1.0)))
                        e_plain = []; e_cmc = []; e_hold = []
                        for k in range(1, span):
                            t = t1 + k
                            a = k / float(span)
                            truth = np.array([cx[i0 + k], cy[i0 + k]])
                            plain = p1 + a * (p2 - p1)
                            cmcp = warp_pt(M[t], *(p1 + a * (p2_back - p1)))
                            hold = warp_pt(M[t], p1[0], p1[1])
                            e_plain.append(np.linalg.norm(plain - truth))
                            e_cmc.append(np.linalg.norm(cmcp - truth))
                            e_hold.append(np.linalg.norm(hold - truth))
                        rows.append(dict(seq=seq, tid=int(tid), L=L, t1=t1,
                                         cls=CLSMAP.get(cls, str(cls)), size=size,
                                         plain=float(np.mean(e_plain)) / size,
                                         cmc=float(np.mean(e_cmc)) / size,
                                         hold=float(np.mean(e_hold)) / size,
                                         plain_px=float(np.mean(e_plain)),
                                         cmc_px=float(np.mean(e_cmc))))
    d = pd.DataFrame(rows)
    d.to_csv("_scratch/_r33_pseudogap.csv", index=False)
    pd.set_option("display.width", 220)

    print("\n=== pseudo-gap reconstruction, %d windows ===" % len(d))
    print("mean centre error over the hidden frames, in BOX UNITS (lower is better)\n")
    agg = d.groupby("L").agg(n=("L", "size"),
                             plain=("plain", "median"), cmc=("cmc", "median"),
                             hold=("hold", "median"),
                             cmc_wins=("cmc", lambda s: np.nan))
    wins = d.groupby("L").apply(lambda g: float((g.cmc < g.plain).mean()), include_groups=False)
    agg["cmc_wins"] = wins
    print(agg.round(4))

    print("\npooled: plain %.4f | cmc %.4f | hold %.4f  (box units, median)"
          % (d.plain.median(), d.cmc.median(), d.hold.median()))
    print("cmc beats plain in %.1f%% of windows; median (cmc - plain) = %+.4f box units"
          % (100 * (d.cmc < d.plain).mean(), (d.cmc - d.plain).median()))
    print("plain beats hold in %.1f%% of windows" % (100 * (d.plain < d.hold).mean()))

    print("\nper sequence (median box units):")
    ps = d.groupby("seq").agg(n=("L", "size"), plain=("plain", "median"),
                              cmc=("cmc", "median"), hold=("hold", "median"))
    ps["cmc_wins"] = d.groupby("seq").apply(lambda g: float((g.cmc < g.plain).mean()),
                                            include_groups=False)
    print(ps.round(4))

    print("\nby class (median box units):")
    pc = d.groupby("cls").agg(n=("L", "size"), plain=("plain", "median"), cmc=("cmc", "median"))
    pc["cmc_wins"] = d.groupby("cls").apply(lambda g: float((g.cmc < g.plain).mean()),
                                            include_groups=False)
    print(pc.round(4))


if __name__ == "__main__":
    main()
