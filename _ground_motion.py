"""Ground-relative motion as a detection-evidence signal (2026-09-10).

_mid_zone.py: sub-threshold raw detections not covered by a track are 82% background, detector
score separates them weakly (AUC 0.687) and naive persistence not at all (0.55) -- clutter
(signs, rooftop shapes) is as persistent as real objects because it is fixed to the ground.
Hypothesis: after removing camera motion, clutter is ~static while real objects move. The
single-frame detector cannot see this; ego-motion compensation makes it measurable.

For every free sub-threshold det at frame t (score in [lo,0.6), max IoU with tracker boxes <0.3):
  chain backwards: warp the chain box from frame s+1 into frame s with inv(H_{s+1})
  (H_s = cached GMC affine, frame s-1 -> s, _scratch/_r33_affines), link the best-IoU raw det
  (score>=0.05) at IoU>=0.3, stop at the first frame with no link, cap K frames.
Features:
  clen      compensated chain length (0..K)
  gdisp     ground-relative displacement over the chain, oldest centre mapped into frame t, px
  gdisp_n   gdisp / sqrt(box area)  (scale-free)
  gspeed_n  gdisp_n / clen
GT side (ceiling): for each GT box the tracker MISSED that has a sub-threshold raw det, the
same ground-relative displacement of the GT track over the previous K frames -- what fraction
of the rescue population is actually moving?
"""
import os
import sys
import csv
import numpy as np
from _fn_decomp import SEQS, NAMES, MC_GT, iou, load_mot, load_dump, match
from _mid_zone import auc

HERE = os.path.dirname(os.path.abspath(__file__))
AFF = os.path.join(HERE, "_scratch", "_r33_affines")


def load_aff(seq):
    H = {}
    with open(os.path.join(AFF, seq + ".csv")) as f:
        for d in csv.DictReader(f):
            H[int(float(d["frame"]))] = np.array([[float(d["a"]), float(d["b"]), float(d["tx"])],
                                                  [float(d["c"]), float(d["d"]), float(d["ty"])],
                                                  [0, 0, 1.0]])
    return H


def warp_box(b, M):
    """warp tlbr box by 3x3 affine M (centre + isotropic scale)."""
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    w, h = b[2] - b[0], b[3] - b[1]
    p = M @ np.array([cx, cy, 1.0])
    s = np.sqrt(abs(np.linalg.det(M[:2, :2])))
    return np.array([p[0] - s * w / 2, p[1] - s * h / 2, p[0] + s * w / 2, p[1] + s * h / 2])


def run(res_dir, lo=0.1, hi=0.6, K=10):
    feats, labs, clss, scores = [], [], [], []
    gt_move = {"rescue": [], "tp": []}
    for seq in SEQS:
        H = load_aff(seq)
        gt = load_mot(MC_GT.format(seq=seq), conf_min=1)
        ts = load_mot(os.path.join(res_dir, seq + ".txt"))
        dets = load_dump(seq)
        frames = sorted(dets)
        DB = {fr: (np.array([d[0] for d in dets[fr]]).reshape(-1, 4),
                   np.array([d[1] for d in dets[fr]]), np.array([d[2] for d in dets[fr]])) for fr in frames}
        # GT tracks by id for the ceiling
        gt_by_id = {}
        for fr, L in gt.items():
            for g in L:
                gt_by_id.setdefault(int(float(g[2][1])), {})[fr] = np.array(g[0])
        for fr in frames:
            db, dc, ds = DB[fr]
            G = gt.get(fr, []); T = ts.get(fr, [])
            gb = np.array([g[0] for g in G]).reshape(-1, 4); gc = np.array([g[1] for g in G])
            gid = np.array([int(float(g[2][1])) for g in G])
            tb = np.array([t[0] for t in T]).reshape(-1, 4); tc = np.array([t[1] for t in T])
            g_hit = np.zeros(len(G), bool)
            for c in NAMES:
                gi = np.where(gc == c)[0]; ti = np.where(tc == c)[0]
                mg, _ = match(gb[gi], tb[ti]); g_hit[gi[list(mg)]] = True
            sel = np.where((ds >= lo) & (ds < hi))[0]
            if len(sel) == 0:
                continue
            near = iou(db[sel], tb).max(1) if len(T) else np.zeros(len(sel))
            sel = sel[near < 0.3]
            if len(sel) == 0:
                continue
            # label
            lab = np.array(["bg"] * len(sel), dtype=object); lab_gid = np.full(len(sel), -1)
            for c in NAMES:
                si = np.where(dc[sel] == c)[0]; gi = np.where(gc == c)[0]
                if len(si) == 0 or len(gi) == 0:
                    continue
                m = iou(db[sel[si]], gb[gi])
                from scipy.optimize import linear_sum_assignment
                cost = np.where(m >= 0.5, 1 - m, 1e6)
                r, cc = linear_sum_assignment(cost)
                for a, b in zip(r, cc):
                    if cost[a, b] < 1e5:
                        lab[si[a]] = "rescue" if not g_hit[gi[b]] else "redund"
                        lab_gid[si[a]] = gid[gi[b]]
            for k, i in enumerate(sel):
                if lab[k] == "redund":
                    continue
                box = db[i].copy(); cum = np.eye(3); clen = 0; oldest = box.copy()
                s = fr
                while clen < K and (s - 1) in DB and s in H:
                    Minv = np.linalg.inv(H[s])
                    pb = warp_box(box, Minv)            # chain box expressed in frame s-1
                    pdb, _, pds = DB[s - 1]
                    cand = pds >= 0.05
                    if not cand.any():
                        break
                    ii = iou(pb[None], pdb[cand])[0]
                    j = ii.argmax()
                    if ii[j] < 0.3:
                        break
                    box = pdb[cand][j]
                    cum = cum @ H[s]                    # maps frame s-1 coords -> frame fr coords
                    clen += 1; s -= 1
                    oldest = warp_box(box, cum)
                c0 = (db[i][:2] + db[i][2:]) / 2; c1 = (oldest[:2] + oldest[2:]) / 2
                gd = float(np.linalg.norm(c0 - c1))
                sa = float(np.sqrt((db[i][2] - db[i][0]) * (db[i][3] - db[i][1])))
                feats.append((clen, gd, gd / sa, (gd / sa) / max(clen, 1)))
                labs.append(lab[k]); clss.append(dc[i]); scores.append(ds[i])
            # ceiling: ground-relative motion of missed GT objects that have a sub-threshold det
            for k, i in enumerate(sel):
                if lab[k] != "rescue":
                    continue
                tr = gt_by_id.get(int(lab_gid[k]), {})
                if (fr - K) not in tr:
                    continue
                cum = np.eye(3)
                for s in range(fr, fr - K, -1):
                    cum = cum @ H.get(s, np.eye(3))
                ob = warp_box(tr[fr - K], cum); nb = tr[fr]
                sa = np.sqrt((nb[2] - nb[0]) * (nb[3] - nb[1]))
                gt_move["rescue"].append(np.linalg.norm((ob[:2] + ob[2:]) / 2 - (nb[:2] + nb[2:]) / 2) / sa)
    F = np.array(feats, float); lab = np.array(labs); cls = np.array(clss); sc = np.array(scores)
    pos, neg = lab == "rescue", lab == "bg"
    print(f"\nRUN {res_dir}\nfree sub-threshold dets: rescue {pos.sum()}  bg {neg.sum()}")
    names = ["clen", "gdisp", "gdisp_n", "gspeed_n"]
    print("\nAUC rescue-vs-bg:")
    print(f"  score      {auc(sc[pos], sc[neg]):.3f}")
    for j, nm in enumerate(names):
        print(f"  {nm:10s} {auc(F[pos, j], F[neg, j]):.3f}")
    for L in [3, 5, 8]:
        m = F[:, 0] >= L
        print(f"  among chains clen>={L} (rescue {int((pos & m).sum())}, bg {int((neg & m).sum())}): "
              f"gdisp_n AUC {auc(F[pos & m, 2], F[neg & m, 2]):.3f}, score AUC {auc(sc[pos & m], sc[neg & m]):.3f}")
    print("\nper class, chains clen>=5: gdisp_n AUC")
    for c, nm in NAMES.items():
        m = (cls == c) & (F[:, 0] >= 5)
        print(f"  {nm:10s} rescue {int((pos & m).sum()):5d} bg {int((neg & m).sum()):6d}  AUC {auc(F[pos & m, 2], F[neg & m, 2]):.3f}")
    print("\nprecision rescue/(rescue+bg), clen>=5, by score band x ground displacement (box widths):")
    dbins = [(0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 1e9)]
    print("  score    " + "".join(f"{f'{a}-{b if b < 1e8 else inf}':>15s}" for a, b in dbins))
    for a, b in [(0.1, 0.3), (0.3, 0.45), (0.45, 0.6)]:
        cells = []
        for da, dbb in dbins:
            m = (F[:, 0] >= 5) & (sc >= a) & (sc < b) & (F[:, 2] >= da) & (F[:, 2] < dbb)
            npz, nn = int((pos & m).sum()), int((neg & m).sum())
            cells.append(f"{100 * npz / max(npz + nn, 1):5.1f}% ({npz + nn:5d})")
        print(f"  {a:.2f}-{b:.2f} " + "".join(f"{c:>15s}" for c in cells))
    gm = np.array(gt_move["rescue"])
    print(f"\nCEILING: ground-relative motion of missed GT objects over {K} frames (box widths): "
          f"n={len(gm)}  median {np.median(gm):.2f}  share >0.25: {100*(gm>0.25).mean():.1f}%  "
          f">0.5: {100*(gm>0.5).mean():.1f}%  >1.0: {100*(gm>1.0).mean():.1f}%")


inf = "inf"
if __name__ == "__main__":
    for e in sys.argv[1:]:
        run(e if os.path.isabs(e) else os.path.join(HERE, "YOLOX_outputs", e, "track_results"))
