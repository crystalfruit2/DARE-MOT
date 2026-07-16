"""Score a tracker's predictions across all 7 VisDrone val sequences, either all-class
(vs the leaky gt.txt that reproduces the published 655) or pedestrian-only (vs cat==1 GT
from _build_ped_gt.py). Sums IDSw and reports IDF1/MOTA. Uses the same motmetrics recipe
as eval_mot.py (IoU@0.5). Durable rebuild of the lost re-score harness.

Usage:
  python _score_ped.py <expn> [ped|allclass]
    <expn>  = folder under YOLOX_outputs/ (uses its track_results/<seq>.txt)
              OR an absolute path to a dir containing <seq>.txt
  default mode = ped
"""
import os
import sys
import motmetrics as mm

HERE = os.path.dirname(os.path.abspath(__file__))
PED_GT = os.path.join(HERE, "_ped_gt")
ALLCLASS_GT = r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone_MOT_Format\VisDrone2019-MOT-val\{seq}\gt\gt.txt"
SEQS = [
    "uav0000086_00000_v", "uav0000117_02622_v", "uav0000137_00458_v",
    "uav0000182_00000_v", "uav0000268_05773_v", "uav0000305_00000_v",
    "uav0000339_00001_v",
]


def res_path(expn, seq):
    if os.path.isabs(expn):
        return os.path.join(expn, seq + ".txt")
    return os.path.join(HERE, "YOLOX_outputs", expn, "track_results", seq + ".txt")


def gt_path(mode, seq):
    if mode == "ped":
        return os.path.join(PED_GT, seq + ".txt")
    return ALLCLASS_GT.format(seq=seq)


def main():
    expn = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "ped"
    assert mode in ("ped", "allclass")

    mh = mm.metrics.create()
    print(f"\n=== {expn}  [{mode}] ===")
    print(f"{'seq':28s} {'IDSw':>6s} {'IDF1':>7s} {'MOTA':>8s}")
    tot_sw = 0
    accs, names = [], []
    for seq in SEQS:
        rp, gp = res_path(expn, seq), gt_path(mode, seq)
        if not os.path.exists(rp):
            print(f"{seq:28s}  MISSING PREDICTIONS ({rp})")
            return
        gt = mm.io.loadtxt(gp, fmt="mot15-2D", min_confidence=1)
        ts = mm.io.loadtxt(rp, fmt="mot15-2D", min_confidence=-1)
        acc = mm.utils.compare_to_groundtruth(gt, ts, "iou", distth=0.5)
        s = mh.compute(acc, metrics=["num_switches", "idf1", "mota"], name=seq)
        sw = int(s["num_switches"][seq])
        tot_sw += sw
        accs.append(acc); names.append(seq)
        print(f"{seq:28s} {sw:6d} {s['idf1'][seq]*100:6.1f}% {s['mota'][seq]*100:7.1f}%")

    overall = mh.compute_many(accs, names=names, metrics=["idf1", "mota"],
                              generate_overall=True)
    ov = overall.loc["OVERALL"]
    print(f"{'TOTAL':28s} {tot_sw:6d} {ov['idf1']*100:6.1f}% {ov['mota']*100:7.1f}%")


if __name__ == "__main__":
    main()
