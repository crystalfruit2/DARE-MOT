"""Score under the OFFICIAL VisDrone-MOT ignored-region rule (2026-09-10).

The official toolkit (VisDrone2018-MOT-toolkit, evaluateTrackB.m -> eval/dropObjects.m) removes,
per frame, every box whose area lies >=50% inside an ignored region (raw categories 0 and 11) --
from the tracker results AND from the ground truth -- before per-class CLEAR scoring. Our
_score_multiclass.py never did this, so every DARE-MOT number to date is on a stricter protocol
than published VisDrone numbers (boxes on non-evaluated OBJECTS -- people, tricycles -- still count
as FP officially; only ignored REGIONS are excused).

This wrapper writes filtered copies (results + MC GT) under _scratch/official/ and reuses
_score_multiclass.score_run unchanged.
Usage: python _score_official.py <expn-or-absdir> [...]
"""
import os
import sys
import numpy as np
import _score_multiclass as sm

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone2019-MOT-val\annotations\{seq}.txt"
OUT = os.path.join(HERE, "_scratch", "official")


def ignored_regions(seq):
    reg = {}
    with open(RAW.format(seq=seq)) as f:
        for line in f:
            p = line.strip().split(",")
            if int(p[7]) in (0, 11):
                reg.setdefault(int(p[0]), []).append([float(v) for v in p[2:6]])
    return reg


def keep_mask(rows, reg):
    """rows: list of split lines; True where the box is <50% inside the union of ignored regions."""
    keep = []
    for p in rows:
        fr = int(float(p[0])); R = reg.get(fr)
        if not R:
            keep.append(True); continue
        x, y, w, h = map(float, p[2:6])
        if w <= 0 or h <= 0:
            keep.append(True); continue
        # union coverage on a local integer grid (matches dropObjects' pixel-map semantics)
        x0, y0, x1, y1 = int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))
        if x1 <= x0 or y1 <= y0:
            keep.append(True); continue
        m = np.zeros((y1 - y0, x1 - x0), bool)
        for rx, ry, rw, rh in R:
            a0, b0 = max(int(round(rx)) - x0, 0), max(int(round(ry)) - y0, 0)
            a1, b1 = min(int(round(rx + rw)) - x0, x1 - x0), min(int(round(ry + rh)) - y0, y1 - y0)
            if a1 > a0 and b1 > b0:
                m[b0:b1, a0:a1] = True
        keep.append(m.mean() < 0.5)
    return keep


def filter_file(src, dst, reg):
    rows = [l.rstrip("\n").split(",") for l in open(src) if l.strip()]
    k = keep_mask(rows, reg)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        for p, kk in zip(rows, k):
            if kk:
                f.write(",".join(p) + "\n")
    return len(rows), sum(k)


def prepare_gt():
    gdir = os.path.join(OUT, "_gt")
    tot = kept = 0
    for seq in sm.SEQS:
        a, b = filter_file(sm.MC_GT.format(seq=seq), os.path.join(gdir, seq, "gt", "gt.txt"), ignored_regions(seq))
        tot += a; kept += b
    print(f"GT: kept {kept}/{tot} rows ({tot - kept} inside ignored regions dropped)")
    return os.path.join(gdir, "{seq}", "gt", "gt.txt")


def prepare_res(expn):
    src_dir = expn if os.path.isabs(expn) else os.path.join(HERE, "YOLOX_outputs", expn, "track_results")
    tag = os.path.basename(expn.rstrip("\\/")) if os.path.isabs(expn) else expn
    dst_dir = os.path.join(OUT, tag)
    tot = kept = 0
    for seq in sm.SEQS:
        a, b = filter_file(os.path.join(src_dir, seq + ".txt"), os.path.join(dst_dir, seq + ".txt"), ignored_regions(seq))
        tot += a; kept += b
    return dst_dir, tot, kept


if __name__ == "__main__":
    sm.MC_GT = prepare_gt()
    for e in sys.argv[1:]:
        d, tot, kept = prepare_res(e)
        r = sm.score_run(d)
        mi, ma = r["MICRO"], r["AVG"]
        print(f"{os.path.basename(d):40s} dropped {tot - kept:5d}/{tot} rows | IDSw {r['SUM_IDSw']:4d} | "
              f"micro IDF1 {100 * mi['idf1']:.1f} MOTA {100 * mi['mota']:.1f} | macro IDF1 {100 * ma['idf1']:.1f} MOTA {100 * ma['mota']:.1f}")
