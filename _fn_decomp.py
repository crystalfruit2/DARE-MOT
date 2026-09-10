"""Error-mass decomposition for the novelty hunt (2026-09-10): WHERE are the misses and false alarms?

Standing rule (project_daremot_r33_dead_population_rule): count the addressable population of the
target metric before designing anything. Every novelty candidate so far attacked IDSw -- 282 of
~40k MOTA error events. This script sizes the other two buckets from real data.

Inputs
  tracker output : YOLOX_outputs/<expn>/track_results/<seq>.txt   (or an absolute dir)
  raw detections : _scratch/detdump_val7/<seq>.csv                 (_detdump_val7.py)
  GT             : MC gt.txt, conf>=1 (same filter as _score_multiclass.py)

Per class, per frame: Hungarian IoU>=0.5 between output and GT (same criterion as the scorer;
motmetrics' ID-continuity preference is ignored, so counts agree to within a fraction of a %).

FN buckets (each missed GT box, by the BEST raw detection covering it):
  hi_same   same-class det, IoU>=.5, score>=0.6  -> tracker had a confident det and still missed
  hi_other  only other-class dets IoU>=.5 score>=0.6 -> label error
  mid       best IoU>=.5 det scores in [0.1, 0.6) -> 2nd-stage zone: can extend, never births
  low       best IoU>=.5 det scores in [0.001, 0.1) -> below the tracker's floor
  loc       no IoU>=.5 det, but one at IoU in [0.3, 0.5) -> localization failure
  none      nothing at IoU>=0.3 at any score  -> detector blind
FP buckets (each output row unmatched in its own class):
  label_err  IoU>=.5 with an UNMATCHED GT of another class (paired with an FN)
  dup_tp     IoU>=.5 with a GT box that IS matched  (duplicate of a correct output)
  near_gt    best IoU with any GT in [0.1, 0.5)
  isolated   nothing
"""
import os
import sys
import csv
from collections import defaultdict, Counter
import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
DUMP = os.path.join(HERE, "_scratch", "detdump_val7")
MC_GT = (r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone_MOT_Format_MC"
         r"\VisDrone2019-MOT-val\{seq}\gt\gt.txt")
SEQS = ["uav0000086_00000_v", "uav0000117_02622_v", "uav0000137_00458_v",
        "uav0000182_00000_v", "uav0000268_05773_v", "uav0000305_00000_v",
        "uav0000339_00001_v"]
NAMES = {1: "pedestrian", 2: "car", 3: "van", 4: "truck", 5: "bus"}
SIZE_BINS = [(0, 12), (12, 20), (20, 32), (32, 64), (64, 1e9)]


def iou(a, b):
    """a: (n,4) tlbr, b: (m,4) tlbr -> (n,m)"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def load_mot(path, conf_min=None):
    """-> dict frame -> list of (tlbr, cls, extra)"""
    out = defaultdict(list)
    with open(path) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 8:
                continue
            if conf_min is not None and float(p[6]) < conf_min:
                continue
            fr = int(float(p[0])); x, y, w, h = map(float, p[2:6])
            out[fr].append(([x, y, x + w, y + h], int(float(p[7])), p))
    return out


def load_dump(seq):
    out = defaultdict(list)
    with open(os.path.join(DUMP, seq + ".csv")) as f:
        r = csv.DictReader(f)
        for d in r:
            out[int(d["frame"])].append(([float(d["x1"]), float(d["y1"]), float(d["x2"]), float(d["y2"])],
                                         int(d["cls"]) + 1, float(d["score"])))
    return out


def match(a_boxes, b_boxes):
    """Hungarian on IoU>=0.5. Returns set of matched a-idx, set of matched b-idx."""
    m = iou(a_boxes, b_boxes)
    if m.size == 0:
        return set(), set()
    cost = np.where(m >= 0.5, 1 - m, 1e6)
    r, c = linear_sum_assignment(cost)
    ok = cost[r, c] < 1e5
    return set(r[ok].tolist()), set(c[ok].tolist())


def size_bin(box):
    s = np.sqrt(max(box[2] - box[0], 0) * max(box[3] - box[1], 0))
    for i, (lo, hi) in enumerate(SIZE_BINS):
        if lo <= s < hi:
            return i
    return len(SIZE_BINS) - 1


def run(res_dir):
    fn_b = Counter(); fn_by_cls = defaultdict(Counter); fn_by_size = defaultdict(Counter)
    fp_b = Counter(); fp_by_cls = defaultdict(Counter)
    gt_size = Counter(); tp_size = Counter()
    n_gt = n_tp = 0
    for seq in SEQS:
        gt = load_mot(MC_GT.format(seq=seq), conf_min=1)
        ts = load_mot(os.path.join(res_dir, seq + ".txt"))
        dets = load_dump(seq)
        for fr in sorted(set(gt) | set(ts)):
            G = gt.get(fr, []); T = ts.get(fr, []); D = dets.get(fr, [])
            gb = np.array([g[0] for g in G]).reshape(-1, 4); gc = np.array([g[1] for g in G])
            tb = np.array([t[0] for t in T]).reshape(-1, 4); tc = np.array([t[1] for t in T])
            db = np.array([d[0] for d in D]).reshape(-1, 4); dc = np.array([d[1] for d in D])
            ds = np.array([d[2] for d in D])
            g_matched = np.zeros(len(G), bool); t_matched = np.zeros(len(T), bool)
            for c in NAMES:
                gi = np.where(gc == c)[0]; ti = np.where(tc == c)[0]
                mg, mt = match(gb[gi], tb[ti])
                g_matched[gi[list(mg)]] = True; t_matched[ti[list(mt)]] = True
            n_gt += len(G); n_tp += int(g_matched.sum())
            for i, g in enumerate(G):
                sb = size_bin(g[0]); gt_size[sb] += 1
                if g_matched[i]:
                    tp_size[sb] += 1
                    continue
                gi_iou = iou(gb[i:i + 1], db)[0] if len(D) else np.zeros(0)
                if len(gi_iou) and (gi_iou >= 0.5).any():
                    cover = gi_iou >= 0.5
                    same_hi = (cover & (dc == g[1]) & (ds >= 0.6)).any()
                    other_hi = (cover & (dc != g[1]) & (ds >= 0.6)).any()
                    best = ds[cover].max()
                    if same_hi:
                        b = "hi_same"
                    elif other_hi:
                        b = "hi_other"
                    elif best >= 0.1:
                        b = "mid"
                    else:
                        b = "low"
                elif len(gi_iou) and (gi_iou >= 0.3).any():
                    b = "loc"
                else:
                    b = "none"
                fn_b[b] += 1; fn_by_cls[NAMES[g[1]]][b] += 1; fn_by_size[sb][b] += 1
            if len(T):
                tg = iou(tb, gb) if len(G) else np.zeros((len(T), 0))
                for j, t in enumerate(T):
                    if t_matched[j]:
                        continue
                    row = tg[j] if len(G) else np.zeros(0)
                    if len(row) and ((row >= 0.5) & ~g_matched & (gc != t[1])).any():
                        b = "label_err"
                    elif len(row) and ((row >= 0.5) & g_matched).any():
                        b = "dup_tp"
                    elif len(row) and (row >= 0.1).any():
                        b = "near_gt"
                    else:
                        b = "isolated"
                    fp_b[b] += 1; fp_by_cls[NAMES.get(t[1], str(t[1]))][b] += 1

    n_fn = sum(fn_b.values()); n_fp = sum(fp_b.values())
    print(f"\nRUN {res_dir}\nGT {n_gt}  TP {n_tp}  FN {n_fn}  FP {n_fp}  "
          f"(pooled MOTA w/o IDSw = {1 - (n_fn + n_fp) / n_gt:.4f})")
    order = ["hi_same", "hi_other", "mid", "low", "loc", "none"]
    print("\nFN buckets:")
    for b in order:
        print(f"  {b:9s} {fn_b[b]:6d}  {100 * fn_b[b] / n_fn:5.1f}%")
    print("\nFN by class (% of that class's FN):")
    print("  class       " + "".join(f"{b:>9s}" for b in order) + "      FN")
    for c in NAMES.values():
        t = sum(fn_by_cls[c].values()) or 1
        print(f"  {c:11s} " + "".join(f"{100 * fn_by_cls[c][b] / t:8.1f}%" for b in order) + f"  {t:6d}")
    print("\nFN by GT size sqrt(area) px  (recall = TP/GT in that bin):")
    print("  size        " + "".join(f"{b:>9s}" for b in order) + "      FN   recall")
    for i, (lo, hi) in enumerate(SIZE_BINS):
        t = sum(fn_by_size[i].values()) or 1
        lab = f"{lo}-{int(hi) if hi < 1e8 else 'inf'}"
        print(f"  {lab:11s} " + "".join(f"{100 * fn_by_size[i][b] / t:8.1f}%" for b in order)
              + f"  {t:6d}   {100 * tp_size[i] / max(gt_size[i], 1):5.1f}%")
    print("\nFP buckets:")
    for b in ["label_err", "dup_tp", "near_gt", "isolated"]:
        print(f"  {b:9s} {fp_b[b]:6d}  {100 * fp_b[b] / max(n_fp, 1):5.1f}%")
    print("\nFP by predicted class:")
    for c, cnt in sorted(fp_by_cls.items()):
        t = sum(cnt.values())
        print(f"  {c:11s} " + "  ".join(f"{b}={cnt[b]}" for b in ["label_err", "dup_tp", "near_gt", "isolated"]) + f"  total={t}")


if __name__ == "__main__":
    for e in sys.argv[1:]:
        run(e if os.path.isabs(e) else os.path.join(HERE, "YOLOX_outputs", e, "track_results"))
