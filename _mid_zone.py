"""The 'seen but rejected' population (2026-09-10): can real objects below the tracker's
thresholds be separated from background clutter at the same score?

_fn_decomp.py found 60% of CA-DARE's 29k FN have a correct raw detection (same-location
IoU>=0.5) scoring below the tracker's 0.6 threshold. Lowering the threshold only raises MOTA if
the recovered boxes are >50% real (every background box is a new FP). This script labels every raw
detection with score in [lo, 0.6) and tests which cheap signals separate them.

Labels per det (same-class Hungarian IoU>=0.5 against GT):
  rescue   covers a GT box the tracker MISSED   -> recovering it = -1 FN
  redund   covers a GT box the tracker already HAS -> harmless if suppressed by the tracker
  bg       covers no GT box of its class           -> +1 FP if emitted
Signals:
  score          detector score
  persist_k      # of the previous k frames with ANY raw det (score>=0.05) at IoU>=0.3
                 (a track-before-detect evidence count; no ego-motion compensation yet)
  run            length of the unbroken backwards run of such frames (cap 30)
  near_trk       max IoU with any tracker output box in the same frame (already covered?)
"""
import os
import sys
from collections import defaultdict
import numpy as np
from _fn_decomp import (SEQS, NAMES, MC_GT, iou, load_mot, load_dump, match)

HERE = os.path.dirname(os.path.abspath(__file__))


def auc(pos, neg):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float) + 1
    # average ranks for ties
    order = np.argsort(allv); sv = allv[order]; r = np.empty(len(allv))
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    rp = r[:len(pos)].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def run(res_dir, lo=0.1, hi=0.6, K=10):
    rows = []  # (label, cls, score, persist5, persist10, run, near_trk, sqrt_area)
    for seq in SEQS:
        gt = load_mot(MC_GT.format(seq=seq), conf_min=1)
        ts = load_mot(os.path.join(res_dir, seq + ".txt"))
        dets = load_dump(seq)
        frames = sorted(dets)
        hist = []  # list of arrays of boxes (score>=0.05) for previous frames
        for fr in frames:
            D = dets[fr]
            db = np.array([d[0] for d in D]).reshape(-1, 4)
            dc = np.array([d[1] for d in D]); ds = np.array([d[2] for d in D])
            G = gt.get(fr, []); T = ts.get(fr, [])
            gb = np.array([g[0] for g in G]).reshape(-1, 4); gc = np.array([g[1] for g in G])
            tb = np.array([t[0] for t in T]).reshape(-1, 4); tc = np.array([t[1] for t in T])
            # which GT the tracker got (per class)
            g_hit = np.zeros(len(G), bool)
            for c in NAMES:
                gi = np.where(gc == c)[0]; ti = np.where(tc == c)[0]
                mg, _ = match(gb[gi], tb[ti]); g_hit[gi[list(mg)]] = True
            sel = np.where((ds >= lo) & (ds < hi))[0]
            if len(sel):
                lab = np.array(["bg"] * len(sel), dtype=object)
                for c in NAMES:
                    si = sel[dc[sel] == c]; gi = np.where(gc == c)[0]
                    if len(si) == 0 or len(gi) == 0:
                        continue
                    ms, mgs = [], []
                    m = iou(db[si], gb[gi])
                    cost = np.where(m >= 0.5, 1 - m, 1e6)
                    from scipy.optimize import linear_sum_assignment
                    r, cc = linear_sum_assignment(cost)
                    for a, b in zip(r, cc):
                        if cost[a, b] < 1e5:
                            k = np.where(sel == si[a])[0][0]
                            lab[k] = "rescue" if not g_hit[gi[b]] else "redund"
                p = np.zeros((len(sel), K), bool)
                for back, hb in enumerate(reversed(hist[-K:])):
                    if len(hb):
                        p[:, back] = (iou(db[sel], hb) >= 0.3).any(1)
                runlen = np.zeros(len(sel), int)
                for i in range(len(sel)):
                    n = 0
                    while n < min(K, len(hist)) and p[i, n]:
                        n += 1
                    runlen[i] = n
                near = iou(db[sel], tb).max(1) if len(T) else np.zeros(len(sel))
                sa = np.sqrt((db[sel, 2] - db[sel, 0]) * (db[sel, 3] - db[sel, 1]))
                for i in range(len(sel)):
                    rows.append((lab[i], dc[sel[i]], ds[sel[i]], p[i, :5].sum(), p[i].sum(),
                                 runlen[i], near[i], sa[i]))
            hist.append(db[ds >= 0.05])
    lab = np.array([r[0] for r in rows]); cls = np.array([r[1] for r in rows])
    X = {k: np.array([r[j] for r in rows], float) for j, k in
         enumerate(["_", "_", "score", "persist5", "persist10", "run", "near_trk", "sqrt_area"]) if k != "_"}
    n = len(rows)
    print(f"\nRUN {res_dir}\nraw dets with score in [{lo},{hi}): {n}")
    for L in ["rescue", "redund", "bg"]:
        print(f"  {L:7s} {int((lab == L).sum()):7d}  {100 * (lab == L).mean():5.1f}%")
    # Exclude dets that sit on an existing tracker box (they can't become new outputs anyway)
    free = X["near_trk"] < 0.3
    print(f"\nnot already covered by a tracker box (near_trk<0.3): {int(free.sum())}")
    for L in ["rescue", "redund", "bg"]:
        print(f"  {L:7s} {int(((lab == L) & free).sum()):7d}  {100 * ((lab == L) & free).sum() / max(free.sum(), 1):5.1f}%")
    pos = (lab == "rescue") & free; neg = (lab == "bg") & free
    print("\nAUC rescue-vs-bg (free dets only):")
    for k in ["score", "persist5", "persist10", "run", "sqrt_area"]:
        print(f"  {k:10s} {auc(X[k][pos], X[k][neg]):.3f}")
    print("\nprecision rescue/(rescue+bg) among free dets, by score band x run length:")
    bands = [(lo, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, hi)]
    runs = [(0, 0), (1, 2), (3, 5), (6, 9), (10, 10)]
    print("  score\\run  " + "".join(f"{f'{a}-{b}':>14s}" for a, b in runs) + "     all")
    for a, b in bands:
        m = free & (X["score"] >= a) & (X["score"] < b)
        cells = []
        for ra, rb in runs:
            mm = m & (X["run"] >= ra) & (X["run"] <= rb)
            npz, nn = int((pos & mm).sum()), int((neg & mm).sum())
            cells.append(f"{100 * npz / max(npz + nn, 1):5.1f}% ({npz + nn:5d})")
        npz, nn = int((pos & m).sum()), int((neg & m).sum())
        print(f"  {a:.1f}-{b:.1f}   " + "".join(f"{c:>14s}" for c in cells) + f"  {100 * npz / max(npz + nn, 1):5.1f}%")
    print("\nby class (free dets): rescue / bg / precision")
    for c, nm in NAMES.items():
        m = cls == c
        npz, nn = int((pos & m).sum()), int((neg & m).sum())
        print(f"  {nm:10s} {npz:6d} {nn:6d}  {100 * npz / max(npz + nn, 1):5.1f}%")


if __name__ == "__main__":
    lo = float(os.environ.get("MZ_LO", "0.1"))
    for e in sys.argv[1:]:
        run(e if os.path.isabs(e) else os.path.join(HERE, "YOLOX_outputs", e, "track_results"), lo=lo)
