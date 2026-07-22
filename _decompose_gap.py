"""Decompose the MOTA / IDF1 gap between two trackers, reusing _score_ped.py's exact
motmetrics recipe (IoU@0.5, same GT loaders). No re-tracking -- reads prediction files
already on disk. Answers: WHERE do ByteTrack's ~3-pt IDF1/MOTA lead come from --
false positives, misses (FN), identity switches, or track fragmentation?

Usage:
  python _decompose_gap.py <expnA> <expnB> [ped|allclass]
  (A = baseline e.g. bt_l0_s0_r1, B = candidate e.g. ftgate2500_lockoff)
"""
import os, sys
import motmetrics as mm

HERE = os.path.dirname(os.path.abspath(__file__))
PED_GT = os.path.join(HERE, "_ped_gt")
ALLCLASS_GT = r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone_MOT_Format\VisDrone2019-MOT-val\{seq}\gt\gt.txt"
SEQS = [
    "uav0000086_00000_v", "uav0000117_02622_v", "uav0000137_00458_v",
    "uav0000182_00000_v", "uav0000268_05773_v", "uav0000305_00000_v",
    "uav0000339_00001_v",
]

METRICS = ["num_objects", "num_false_positives", "num_misses", "num_switches",
           "num_fragmentations", "mota", "recall", "precision",
           "idtp", "idfp", "idfn", "idf1"]


def res_path(expn, seq):
    if os.path.isabs(expn):
        return os.path.join(expn, seq + ".txt")
    return os.path.join(HERE, "YOLOX_outputs", expn, "track_results", seq + ".txt")


def gt_path(mode, seq):
    if mode == "ped":
        return os.path.join(PED_GT, seq + ".txt")
    return ALLCLASS_GT.format(seq=seq)


def score(expn, mode):
    mh = mm.metrics.create()
    accs, names = [], []
    for seq in SEQS:
        rp, gp = res_path(expn, seq), gt_path(mode, seq)
        if not os.path.exists(rp):
            sys.exit(f"MISSING PREDICTIONS for {expn}: {rp}")
        gt = mm.io.loadtxt(gp, fmt="mot15-2D", min_confidence=1)
        ts = mm.io.loadtxt(rp, fmt="mot15-2D", min_confidence=-1)
        acc = mm.utils.compare_to_groundtruth(gt, ts, "iou", distth=0.5)
        accs.append(acc); names.append(seq)
    overall = mh.compute_many(accs, names=names, metrics=METRICS, generate_overall=True)
    return overall.loc["OVERALL"]


def main():
    A, B = sys.argv[1], sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else "allclass"
    a, b = score(A, mode), score(B, mode)

    print(f"\n=== GAP DECOMPOSITION  [{mode}]   A={A}  vs  B={B} ===")
    print(f"{'metric':22s} {'A(base)':>12s} {'B(cand)':>12s} {'B - A':>10s}")
    def row(label, key, pct=False, intfmt=False):
        av, bv = a[key], b[key]
        if pct:
            print(f"{label:22s} {av*100:11.1f}% {bv*100:11.1f}% {(bv-av)*100:+9.1f}")
        else:
            print(f"{label:22s} {av:12.0f} {bv:12.0f} {bv-av:+10.0f}")
    row("GT objects", "num_objects")
    row("False positives (FP)", "num_false_positives")
    row("Misses (FN)", "num_misses")
    row("ID switches", "num_switches")
    row("Fragmentations", "num_fragmentations")
    print("-" * 58)
    row("MOTA", "mota", pct=True)
    row("Recall", "recall", pct=True)
    row("Precision", "precision", pct=True)
    print("-" * 58)
    row("IDTP", "idtp")
    row("IDFP", "idfp")
    row("IDFN", "idfn")
    row("IDF1", "idf1", pct=True)


if __name__ == "__main__":
    main()
