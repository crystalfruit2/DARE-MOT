"""Is the tracker's OUTPUT BOX worse than its own detection? (2026-09-10)

_fn_decomp.py: 2,667 FN had a confident (>=0.6) same-class raw detection at IoU>=0.5, and 2,768 FP
sit at IoU 0.1-0.5 from a GT box. Hypothesis: same events -- the tracker emits the KF posterior box
(STrack.tlwh, from the filtered mean), which lags/shrinks for small fast targets, lands at IoU<0.5,
and so costs an FP AND an FN although the detector was right.

For every tracker FP row: find an unmatched same-class GT with IoU in [0.1,0.5) to the output box
('near miss'). Then check whether a raw detection (score>=0.1, same class) exists that has
IoU>=0.5 with that GT AND IoU>=0.3 with the output box (i.e. plausibly the detection that updated
this track). If so, emitting the detection box instead would convert FP+FN -> TP.
Also reports the IoU(output, GT) vs IoU(det, GT) distribution and the box-size profile.
"""
import os
import sys
import numpy as np
from collections import Counter
from _fn_decomp import SEQS, NAMES, MC_GT, iou, load_mot, load_dump, match, size_bin, SIZE_BINS

HERE = os.path.dirname(os.path.abspath(__file__))


def main(res_dir):
    n_fp = n_near = n_fix = 0
    fix_by_cls = Counter(); fix_by_size = Counter(); near_by_size = Counter()
    gains = []
    for seq in SEQS:
        gt = load_mot(MC_GT.format(seq=seq), conf_min=1)
        ts = load_mot(os.path.join(res_dir, seq + ".txt"))
        dets = load_dump(seq)
        for fr, T in ts.items():
            G = gt.get(fr, []); D = dets.get(fr, [])
            gb = np.array([g[0] for g in G]).reshape(-1, 4); gc = np.array([g[1] for g in G])
            tb = np.array([t[0] for t in T]).reshape(-1, 4); tc = np.array([t[1] for t in T])
            db = np.array([d[0] for d in D]).reshape(-1, 4); dc = np.array([d[1] for d in D])
            ds = np.array([d[2] for d in D])
            g_hit = np.zeros(len(G), bool); t_hit = np.zeros(len(T), bool)
            for c in NAMES:
                gi = np.where(gc == c)[0]; ti = np.where(tc == c)[0]
                mg, mt = match(gb[gi], tb[ti]); g_hit[gi[list(mg)]] = True; t_hit[ti[list(mt)]] = True
            if not len(G):
                n_fp += int((~t_hit).sum()); continue
            tg = iou(tb, gb)
            used_g = set()
            for j in np.where(~t_hit)[0]:
                n_fp += 1
                cand = np.where((gc == tc[j]) & ~g_hit & (tg[j] >= 0.1) & (tg[j] < 0.5))[0]
                cand = [g for g in cand if g not in used_g]
                if not cand:
                    continue
                g = cand[int(np.argmax(tg[j, cand]))]
                n_near += 1; near_by_size[size_bin(G[g][0])] += 1
                if len(D):
                    m = (dc == tc[j]) & (ds >= 0.1)
                    if m.any():
                        dg = iou(db[m], gb[g:g + 1])[:, 0]; dt = iou(db[m], tb[j:j + 1])[:, 0]
                        ok = (dg >= 0.5) & (dt >= 0.3)
                        if ok.any():
                            n_fix += 1; used_g.add(g)
                            fix_by_cls[NAMES[int(tc[j])]] += 1; fix_by_size[size_bin(G[g][0])] += 1
                            gains.append((tg[j, g], dg[ok].max()))
    print(f"\nRUN {res_dir}\nFP rows {n_fp};  near-miss FPs (same-class unmatched GT at IoU 0.1-0.5): {n_near}")
    print(f"  ...of which a same-class raw det (score>=0.1) overlaps the output (IoU>=0.3) AND hits the GT (IoU>=0.5): "
          f"{n_fix}  = FP+FN pairs a det-box output would turn into TPs")
    if gains:
        g = np.array(gains)
        print(f"  median IoU(output box, GT) {np.median(g[:, 0]):.3f}  ->  median IoU(det box, GT) {np.median(g[:, 1]):.3f}")
    print(f"  upper-bound MOTA gain if all fixed: +{2 * n_fix / 72690 * 100:.2f} points (each fix = -1 FP and -1 FN)")
    print("  by class:", dict(fix_by_cls))
    print("  by GT size sqrt(area):", {f"{SIZE_BINS[k][0]}-{SIZE_BINS[k][1] if SIZE_BINS[k][1] < 1e8 else 'inf'}":
                                       f"{fix_by_size[k]}/{near_by_size[k]}" for k in sorted(near_by_size)})


if __name__ == "__main__":
    for e in sys.argv[1:]:
        main(e if os.path.isabs(e) else os.path.join(HERE, "YOLOX_outputs", e, "track_results"))
