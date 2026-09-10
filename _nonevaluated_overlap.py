"""How much of the 'clutter' is actually real, non-evaluated VisDrone content? (2026-09-10)

Zoom re-inspection gave a FLAT ~35% precision at every threshold, which is the signature of a
label problem, not a signal problem. The MC ground truth keeps only VisDrone's 5 evaluated
classes (pedestrian 1, car 4, van 5, truck 6, bus 9 -> 72,690 boxes on val7). The raw annotations
also contain people (2), bicycle (3), tricycle (7), awning-tricycle (8), motor (10), others (11)
and IGNORED REGIONS (category 0 / score 0). Our scorer counts a box on any of those as an FP.

For (a) the tracker's own FPs and (b) the free sub-threshold 'bg' candidates, report the share
that overlaps (IoU>=0.5 with an object, or >=50% of the box inside an ignored/others region).
"""
import os
import sys
from collections import Counter, defaultdict
import numpy as np
from _fn_decomp import SEQS, NAMES, MC_GT, iou, load_mot, match

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone2019-MOT-val\annotations\{seq}.txt"
VNAME = {0: "ignored", 1: "pedestrian", 2: "people", 3: "bicycle", 4: "car", 5: "van", 6: "truck",
         7: "tricycle", 8: "awning-tri", 9: "bus", 10: "motor", 11: "others"}
NONEVAL = {0, 2, 3, 7, 8, 10, 11}


def load_raw(seq):
    out = defaultdict(list)
    with open(RAW.format(seq=seq)) as f:
        for line in f:
            p = line.strip().split(",")
            fr = int(p[0]); x, y, w, h = map(float, p[2:6]); cat = int(p[7])
            out[fr].append(([x, y, x + w, y + h], cat))
    return out


def ioa(a, b):
    """fraction of each box in a inside each box in b"""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    return inter / (aa[:, None] + 1e-9)


def classify(boxes, raw):
    """-> list of labels: noneval category name or 'none'"""
    if not raw or len(boxes) == 0:
        return ["none"] * len(boxes)
    rb = np.array([r[0] for r in raw]); rc = np.array([r[1] for r in raw])
    ov = iou(boxes, rb); ia = ioa(boxes, rb)
    labs = []
    for i in range(len(boxes)):
        best = "none"
        reg = np.where(np.isin(rc, [0, 11]) & (ia[i] >= 0.5))[0]
        obj = np.where(np.isin(rc, [2, 3, 7, 8, 10]) & (ov[i] >= 0.5))[0]
        if len(obj):
            best = VNAME[int(rc[obj[np.argmax(ov[i, obj])]])]
        elif len(reg):
            best = "ignored-region"
        labs.append(best)
    return labs


def main(res_dir):
    fp_lab = Counter(); fp_by_cls = defaultdict(Counter); n_fp = 0
    for seq in SEQS:
        gt = load_mot(MC_GT.format(seq=seq), conf_min=1)
        ts = load_mot(os.path.join(res_dir, seq + ".txt"))
        raw = load_raw(seq)
        for fr, T in ts.items():
            G = gt.get(fr, [])
            gb = np.array([g[0] for g in G]).reshape(-1, 4); gc = np.array([g[1] for g in G])
            tb = np.array([t[0] for t in T]).reshape(-1, 4); tc = np.array([t[1] for t in T])
            t_hit = np.zeros(len(T), bool)
            for c in NAMES:
                gi = np.where(gc == c)[0]; ti = np.where(tc == c)[0]
                _, mt = match(gb[gi], tb[ti]); t_hit[ti[list(mt)]] = True
            fp = np.where(~t_hit)[0]
            if len(fp) == 0:
                continue
            labs = classify(tb[fp], raw.get(fr, []))
            for i, l in zip(fp, labs):
                fp_lab[l] += 1; fp_by_cls[NAMES.get(int(tc[i]), "?")][l] += 1; n_fp += 1
    print(f"\nRUN {res_dir}\nTracker FPs: {n_fp}")
    for l, c in fp_lab.most_common():
        print(f"  {l:15s} {c:6d}  {100 * c / n_fp:5.1f}%")
    print("\nby predicted class:")
    for k, cnt in fp_by_cls.items():
        t = sum(cnt.values())
        print(f"  {k:10s} total {t:5d}  on non-evaluated: {100 * (t - cnt['none']) / t:5.1f}%  " +
              "  ".join(f"{l}={c}" for l, c in cnt.most_common() if l != "none"))

    # sub-threshold 'bg' candidates from the zoom CSV
    import csv
    p = os.path.join(HERE, "_scratch", "zoom_reinspect.csv")
    if os.path.exists(p):
        rows = list(csv.DictReader(open(p)))
        raws = {s: load_raw(s) for s in SEQS}
        lab = Counter(); zs = defaultdict(list)
        for r in rows:
            if r["label"] != "bg":
                continue
            b = np.array([[float(r[k]) for k in ("x1", "y1", "x2", "y2")]])
            l = classify(b, raws[r["seq"]].get(int(r["frame"]), []))[0]
            lab[l] += 1; zs[l].append(float(r["z_same"]))
        n = sum(lab.values())
        print(f"\n'bg' candidates (score 0.3-0.6, free): {n}")
        for l, c in lab.most_common():
            print(f"  {l:15s} {c:6d}  {100 * c / n:5.1f}%   median zoom score {np.median(zs[l]):.3f}")


if __name__ == "__main__":
    e = sys.argv[1]
    main(e if os.path.isabs(e) else os.path.join(HERE, "YOLOX_outputs", e, "track_results"))
