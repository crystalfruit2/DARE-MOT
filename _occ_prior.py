"""Track-derived scene-occupancy prior as detection evidence (2026-09-10).

Ground-relative motion failed (_ground_motion.py: AUC 0.55-0.60) because the missed objects are
mostly static. Next hypothesis: CONTEXT. Real objects occur where objects have been (roads,
sidewalks, parking), clutter often does not (roofs, vegetation). Build, causally, a map of where
confident tracks have been, carried with the camera:
    M_t = decay * warp(M_{t-1}, H_t) + splat(tracker output boxes at t)
kept on a coarse grid in current-frame coordinates (cv2.warpAffine with the cached GMC affine).
A candidate at frame t is scored with M_{t-1} warped into t (never its own frame).
Variants: class-agnostic map, same-class map (per-class maps).
"""
import os
import sys
import csv
import numpy as np
import cv2
from _fn_decomp import SEQS, NAMES, MC_GT, iou, load_mot, load_dump, match
from _mid_zone import auc
from _ground_motion import load_aff

HERE = os.path.dirname(os.path.abspath(__file__))
G = 8  # grid cell, px


def img_size(seq):
    d = os.path.join(r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone2019-MOT-val\sequences", seq)
    f = sorted(os.listdir(d))[0]
    im = cv2.imread(os.path.join(d, f))
    return im.shape[0], im.shape[1]


def run(res_dir, lo=0.1, hi=0.6, decay=0.99):
    rows = []
    for seq in SEQS:
        H = load_aff(seq)
        h, w = img_size(seq)
        gh, gw = (h + G - 1) // G, (w + G - 1) // G
        gt = load_mot(MC_GT.format(seq=seq), conf_min=1)
        ts = load_mot(os.path.join(res_dir, seq + ".txt"))
        dets = load_dump(seq)
        Mall = np.zeros((gh, gw), np.float32)
        Mc = {c: np.zeros((gh, gw), np.float32) for c in NAMES}
        age = 0
        for fr in sorted(set(dets) | set(ts)):
            if fr in H and age > 0:
                A = H[fr][:2].copy(); A[:, 2] /= G           # affine in grid units
                Mall = cv2.warpAffine(Mall, A, (gw, gh), flags=cv2.INTER_LINEAR) * decay
                for c in NAMES:
                    Mc[c] = cv2.warpAffine(Mc[c], A, (gw, gh), flags=cv2.INTER_LINEAR) * decay
            age += 1
            D = dets.get(fr, [])
            T = ts.get(fr, [])
            if D:
                db = np.array([d[0] for d in D]).reshape(-1, 4); dc = np.array([d[1] for d in D])
                ds = np.array([d[2] for d in D])
                Gt = gt.get(fr, [])
                gb = np.array([g[0] for g in Gt]).reshape(-1, 4); gc = np.array([g[1] for g in Gt])
                tb = np.array([t[0] for t in T]).reshape(-1, 4); tc = np.array([t[1] for t in T])
                g_hit = np.zeros(len(Gt), bool)
                for c in NAMES:
                    gi = np.where(gc == c)[0]; ti = np.where(tc == c)[0]
                    mg, _ = match(gb[gi], tb[ti]); g_hit[gi[list(mg)]] = True
                sel = np.where((ds >= lo) & (ds < hi))[0]
                if len(sel):
                    near = iou(db[sel], tb).max(1) if len(T) else np.zeros(len(sel))
                    sel = sel[near < 0.3]
                if len(sel):
                    lab = np.array(["bg"] * len(sel), dtype=object)
                    from scipy.optimize import linear_sum_assignment
                    for c in NAMES:
                        si = np.where(dc[sel] == c)[0]; gi = np.where(gc == c)[0]
                        if len(si) == 0 or len(gi) == 0:
                            continue
                        m = iou(db[sel[si]], gb[gi]); cost = np.where(m >= 0.5, 1 - m, 1e6)
                        r, cc = linear_sum_assignment(cost)
                        for a, b in zip(r, cc):
                            if cost[a, b] < 1e5:
                                lab[si[a]] = "rescue" if not g_hit[gi[b]] else "redund"
                    for k, i in enumerate(sel):
                        if lab[k] == "redund":
                            continue
                        x0, y0, x1, y1 = db[i]
                        gx0, gy0 = int(max(0, x0 // G)), int(max(0, y0 // G))
                        gx1, gy1 = int(min(gw - 1, x1 // G)), int(min(gh - 1, y1 // G))
                        va = float(Mall[gy0:gy1 + 1, gx0:gx1 + 1].max()) if gx1 >= gx0 and gy1 >= gy0 else 0.0
                        vc = float(Mc[dc[i]][gy0:gy1 + 1, gx0:gx1 + 1].max()) if gx1 >= gx0 and gy1 >= gy0 else 0.0
                        rows.append((lab[k], dc[i], ds[i], va, vc, age))
            # splat this frame's tracker outputs AFTER scoring candidates (causal)
            for t in T:
                x0, y0, x1, y1 = t[0]
                gx0, gy0 = int(max(0, x0 // G)), int(max(0, y0 // G))
                gx1, gy1 = int(min(gw - 1, x1 // G)), int(min(gh - 1, y1 // G))
                if gx1 < gx0 or gy1 < gy0:
                    continue
                Mall[gy0:gy1 + 1, gx0:gx1 + 1] += 1.0
                if t[1] in Mc:
                    Mc[t[1]][gy0:gy1 + 1, gx0:gx1 + 1] += 1.0
    lab = np.array([r[0] for r in rows]); cls = np.array([r[1] for r in rows])
    sc = np.array([r[2] for r in rows]); va = np.array([r[3] for r in rows]); vc = np.array([r[4] for r in rows])
    age = np.array([r[5] for r in rows])
    pos, neg = lab == "rescue", lab == "bg"
    print(f"\nRUN {res_dir}  decay={decay}\nfree sub-threshold dets: rescue {pos.sum()}  bg {neg.sum()}")
    print(f"AUC rescue-vs-bg:  score {auc(sc[pos], sc[neg]):.3f}   occ_all {auc(va[pos], va[neg]):.3f}   "
          f"occ_same_class {auc(vc[pos], vc[neg]):.3f}")
    m = age > 60
    print(f"  after 60 frames of history: occ_all {auc(va[pos & m], va[neg & m]):.3f}  occ_same {auc(vc[pos & m], vc[neg & m]):.3f}")
    # simple combination: rank-average of score and same-class occupancy
    from scipy.stats import rankdata
    comb = rankdata(sc) + rankdata(vc)
    print(f"  rank(score)+rank(occ_same): {auc(comb[pos], comb[neg]):.3f}")
    print("\nper class occ_same AUC / score AUC:")
    for c, nm in NAMES.items():
        mm = cls == c
        print(f"  {nm:10s} rescue {int((pos & mm).sum()):5d} bg {int((neg & mm).sum()):6d}  "
              f"occ {auc(vc[pos & mm], vc[neg & mm]):.3f}  score {auc(sc[pos & mm], sc[neg & mm]):.3f}")
    print("\nprecision by score band x same-class occupancy (0 / low / mid / high tertiles of >0):")
    nz = vc[vc > 0]
    q = np.quantile(nz, [1 / 3, 2 / 3]) if len(nz) else [0, 0]
    obins = [(-1, 0), (0, q[0]), (q[0], q[1]), (q[1], 1e12)]
    print("  score      " + "".join(f"{lab_:>16s}" for lab_ in ["occ=0", "low", "mid", "high"]))
    for a, b in [(0.1, 0.3), (0.3, 0.45), (0.45, 0.6)]:
        cells = []
        for oa, ob in obins:
            mm = (sc >= a) & (sc < b) & (vc > oa) & (vc <= ob)
            npz, nn = int((pos & mm).sum()), int((neg & mm).sum())
            cells.append(f"{100 * npz / max(npz + nn, 1):5.1f}% ({npz + nn:5d})")
        print(f"  {a:.2f}-{b:.2f}  " + "".join(f"{c:>16s}" for c in cells))


if __name__ == "__main__":
    for e in sys.argv[1:]:
        run(e if os.path.isabs(e) else os.path.join(HERE, "YOLOX_outputs", e, "track_results"))
