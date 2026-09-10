"""Re-score the zoom screen against TRUE background only (2026-09-10).
48% of the 'bg' candidates in _scratch/zoom_reinspect.csv sit on real, non-evaluated VisDrone
content (ignored regions, people, tricycles, ...; _nonevaluated_overlap.py). Recompute AUC and
precision for rescue vs clean background, and vs each polluting group."""
import csv
import os
import numpy as np
from _mid_zone import auc
from _nonevaluated_overlap import load_raw, classify
from _fn_decomp import SEQS

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "_scratch", "zoom_reinspect.csv"))))
raws = {s: load_raw(s) for s in SEQS}
grp, s, z = [], [], []
for r in rows:
    g = r["label"]
    if g == "bg":
        b = np.array([[float(r[k]) for k in ("x1", "y1", "x2", "y2")]])
        l = classify(b, raws[r["seq"]].get(int(r["frame"]), []))[0]
        g = "bg_clean" if l == "none" else ("bg_ignored" if l == "ignored-region" else "bg_noneval_obj")
    grp.append(g); s.append(float(r["score"])); z.append(float(r["z_same"]))
grp = np.array(grp); s = np.array(s); z = np.array(z)
pos = grp == "rescue"
for neg_name in ["bg_clean", "bg_ignored", "bg_noneval_obj"]:
    neg = grp == neg_name
    print(f"rescue ({pos.sum()}) vs {neg_name} ({neg.sum()}):  score AUC {auc(s[pos], s[neg]):.3f}   "
          f"zoom AUC {auc(z[pos], z[neg]):.3f}   max(score,zoom) {auc(np.maximum(s, z)[pos], np.maximum(s, z)[neg]):.3f}")
clean = grp == "bg_clean"
print("\nprecision rescue/(rescue+clean bg) when accepting z_same >= t  [ignored-region boxes cost nothing"
      " under the official protocol; non-evaluated objects still cost an FP]:")
noe = grp == "bg_noneval_obj"
for t in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    m = z >= t
    tp, fc, fo = int((pos & m).sum()), int((clean & m).sum()), int((noe & m).sum())
    print(f"  t={t:.1f}: rescue {tp:5d}  clean-bg {fc:5d}  noneval-obj {fo:5d}  precision(clean) {100*tp/max(tp+fc,1):5.1f}%  "
          f"net events (FN saved - FP added) {tp - fc - fo:+6d}")
