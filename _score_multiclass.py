"""Per-class multi-class MOTChallenge scorer for the DARE-MOT val7 benchmark (Phase 3).

The old harness (_score_ped.py) scored with fmt="mot15-2D", which IGNORES the category
column -- so "ped-only" was really all-class (the class-leak). This scorer splits BOTH the
predictions and the GT by class, runs the identical motmetrics recipe (IoU@0.5) per class,
then reports the per-class table plus the class-AVERAGED MOTA/IDF1 (the VisDrone-MOT
convention) and summed ID switches.

Requires predictions produced by the class-emitting tracker (byte_tracker.STrack.cls ->
mot_evaluator.write_results): result rows are  frame,id,x,y,w,h,score,category,-1,-1  with
category in {1..5}. GT is the MC MOTChallenge gt.txt (frame,id,x,y,w,h,conf,category,vis),
category in {1..5}, built by convert_visdrone_mc.py.

Usage:
  python _score_multiclass.py <expn>              # score one run
  python _score_multiclass.py <expnA> <expnB>     # score both + delta (e.g. DARE vs ByteTrack)
    <expn> = folder under YOLOX_outputs/ (uses its track_results/<seq>.txt)
             OR an absolute path to a dir containing <seq>.txt
"""
import os
import sys
import tempfile
import numpy as np

# motmetrics 1.4.0 predates NumPy 2.0, which removed np.asfarray / np.float_ (both used deep in
# motmetrics.distances). The DARE-MOT training env now has NumPy 2.x and CANNOT be downgraded
# (the detector trains in it), so restore the removed aliases here instead. Contained shim --
# affects only this process. NOTE: _score_ped.py / eval_mot.py need the same shim to run today.
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)
if not hasattr(np, "float_"):
    np.float_ = np.float64

import motmetrics as mm

HERE = os.path.dirname(os.path.abspath(__file__))

# MC class ids (1..5) as written by convert_visdrone_mc.py (category_id-1 = model head id).
MC_NAMES = {1: "pedestrian", 2: "car", 3: "van", 4: "truck", 5: "bus"}
CAT_COL = 7  # 0-based index of the category field in both GT and prediction rows

MC_GT = (r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone_MOT_Format_MC"
         r"\VisDrone2019-MOT-val\{seq}\gt\gt.txt")
SEQS = [
    "uav0000086_00000_v", "uav0000117_02622_v", "uav0000137_00458_v",
    "uav0000182_00000_v", "uav0000268_05773_v", "uav0000305_00000_v",
    "uav0000339_00001_v",
]

METRICS = ["num_switches", "idf1", "mota", "num_false_positives", "num_misses"]


def res_path(expn, seq):
    if os.path.isabs(expn):
        return os.path.join(expn, seq + ".txt")
    return os.path.join(HERE, "YOLOX_outputs", expn, "track_results", seq + ".txt")


def _filter_by_class(src, cls, tmpdir, tag):
    """Write only rows whose category == cls to a temp file; return (path, n_rows).
    Missing source file -> empty (0 rows)."""
    out = os.path.join(tmpdir, f"{tag}.txt")
    n = 0
    with open(out, "w") as f_out:
        if os.path.exists(src):
            with open(src) as f_in:
                for line in f_in:
                    parts = line.strip().split(",")
                    if len(parts) <= CAT_COL:
                        continue
                    try:
                        if int(float(parts[CAT_COL])) == cls:
                            f_out.write(line)
                            n += 1
                    except ValueError:
                        continue
    return out, n


def score_run(expn):
    """Return dict: class_id -> {idsw, idf1, mota, fp, fn}, plus 'AVG' and 'SUM_IDSw'.
    Averages MOTA/IDF1 across classes that have any GT (VisDrone-MOT convention)."""
    mh = mm.metrics.create()
    per_class = {}
    with tempfile.TemporaryDirectory() as tmp:
        for cls, name in MC_NAMES.items():
            accs, names, has_gt = [], [], False
            for seq in SEQS:
                rp = res_path(expn, seq)
                gp = MC_GT.format(seq=seq)
                gt_f, n_gt = _filter_by_class(gp, cls, tmp, "gt")
                ts_f, n_ts = _filter_by_class(rp, cls, tmp, "ts")
                if n_gt == 0 and n_ts == 0:
                    continue  # class absent from this seq for both -> nothing to score
                if n_gt > 0:
                    has_gt = True
                gt = mm.io.loadtxt(gt_f, fmt="mot15-2D", min_confidence=1)
                ts = mm.io.loadtxt(ts_f, fmt="mot15-2D", min_confidence=-1)
                acc = mm.utils.compare_to_groundtruth(gt, ts, "iou", distth=0.5)
                accs.append(acc)
                names.append(seq)
            if not accs or not has_gt:
                per_class[cls] = None  # class never appears in val7 GT -> excluded from AVG
                continue
            summ = mh.compute_many(accs, names=names, metrics=METRICS,
                                   generate_overall=True).loc["OVERALL"]
            per_class[cls] = {
                "idsw": int(summ["num_switches"]),
                "idf1": float(summ["idf1"]),
                "mota": float(summ["mota"]),
                "fp": int(summ["num_false_positives"]),
                "fn": int(summ["num_misses"]),
            }

    scored = {c: v for c, v in per_class.items() if v is not None}
    n = len(scored) or 1
    per_class["AVG"] = {
        "idf1": sum(v["idf1"] for v in scored.values()) / n,
        "mota": sum(v["mota"] for v in scored.values()) / n,
    }
    per_class["SUM_IDSw"] = sum(v["idsw"] for v in scored.values())
    return per_class


def print_table(expn, res):
    print(f"\n=== {expn}  [per-class, IoU@0.5] ===")
    print(f"{'class':12s} {'IDSw':>6s} {'IDF1':>7s} {'MOTA':>8s} {'FP':>8s} {'FN':>8s}")
    for cls, name in MC_NAMES.items():
        v = res[cls]
        if v is None:
            print(f"{name:12s} {'--':>6s} {'--':>7s} {'--':>8s}   (no GT in val7)")
        else:
            print(f"{name:12s} {v['idsw']:6d} {v['idf1']*100:6.1f}% {v['mota']*100:7.1f}% "
                  f"{v['fp']:8d} {v['fn']:8d}")
    avg = res["AVG"]
    print(f"{'-'*12}")
    print(f"{'AVG(classes)':12s} {res['SUM_IDSw']:6d} {avg['idf1']*100:6.1f}% {avg['mota']*100:7.1f}%"
          f"   (IDSw col = SUM across classes)")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    a = sys.argv[1]
    res_a = score_run(a)
    print_table(a, res_a)

    if len(sys.argv) > 2:
        b = sys.argv[2]
        res_b = score_run(b)
        print_table(b, res_b)
        # delta = A - B  (e.g. DARE - ByteTrack)
        print(f"\n=== DELTA  {a}  -  {b} ===")
        print(f"{'class':12s} {'dIDSw':>6s} {'dIDF1':>8s} {'dMOTA':>8s}")
        for cls, name in MC_NAMES.items():
            va, vb = res_a[cls], res_b[cls]
            if va is None or vb is None:
                print(f"{name:12s} {'--':>6s}")
                continue
            print(f"{name:12s} {va['idsw']-vb['idsw']:+6d} {(va['idf1']-vb['idf1'])*100:+7.1f}% "
                  f"{(va['mota']-vb['mota'])*100:+7.1f}%")
        da_idf1 = (res_a['AVG']['idf1'] - res_b['AVG']['idf1']) * 100
        da_mota = (res_a['AVG']['mota'] - res_b['AVG']['mota']) * 100
        print(f"{'-'*12}")
        print(f"{'AVG':12s} {res_a['SUM_IDSw']-res_b['SUM_IDSw']:+6d} {da_idf1:+7.1f}% {da_mota:+7.1f}%")


if __name__ == "__main__":
    main()
