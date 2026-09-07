"""Round 2 #1 (track-intrinsic embedding dispersion -> per-track lambda), step 0:
COUNT THE POPULATION FIRST.

This is the standing rule that came out of the Round 3 #3 kill: before building anything,
count the events the mechanism can act on and the events it is meant to prevent. R3-3 died
because its population was 193 gaps; this candidate is per-track AND per-frame, so it
should be orders of magnitude larger — but that has to be measured, not assumed.

Two populations matter:
  (1) ACTING population  — track-frames where a per-track lambda would be applied.
  (2) TARGET population  — actual ID switches, per class, that it could plausibly prevent.

ID switches are taken from real ground truth, not a proxy: MC MOTChallenge gt.txt matched
to the cached tracker output at IoU 0.5, per class, exactly the recipe `_score_multiclass.py`
uses for the paper's numbers (CAT_COL=7, category in 1..5 on BOTH sides).

SOLVER NOTE: motmetrics' default `lap` solver is blocked by Smart App Control, so the
evaluation matching is forced onto scipy. This is the EVALUATION assignment, not the
tracker's association — the standing "never swap scipy for lap" rule is about the tracker's
own `lapjv(extend_cost=True, cost_limit=thresh)` semantics and does not apply here.
motmetrics itself falls back to scipy whenever lap is absent.
"""
import os
import os.path as osp
import tempfile

import numpy as np
import pandas as pd

if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)
if not hasattr(np, "float_"):
    np.float_ = np.float64

import motmetrics as mm

mm.lap.default_solver = "scipy"

MC_GT = (r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone_MOT_Format_MC"
         r"\VisDrone2019-MOT-val\{seq}\gt\gt.txt")
RUN = "mc_dare_cv_rerun0803"
TRK = osp.join("YOLOX_outputs", RUN, "track_results", "{seq}.txt")
SEQS = ["uav0000086_00000_v", "uav0000117_02622_v", "uav0000137_00458_v", "uav0000182_00000_v",
        "uav0000268_05773_v", "uav0000305_00000_v", "uav0000339_00001_v"]
NAMES = {1: "pedestrian", 2: "car", 3: "van", 4: "truck", 5: "bus"}
CAT_COL = 7


def filt(src, cls, out):
    n = 0
    with open(out, "w") as fo:
        if osp.exists(src):
            for line in open(src):
                p = line.strip().split(",")
                if len(p) <= CAT_COL:
                    continue
                try:
                    if int(float(p[CAT_COL])) == cls:
                        fo.write(line)
                        n += 1
                except ValueError:
                    continue
    return n


def main():
    ev_rows = []
    summary = []
    with tempfile.TemporaryDirectory() as td:
        for seq in SEQS:
            for cls in NAMES:
                gp, tp = osp.join(td, "g.txt"), osp.join(td, "t.txt")
                ng = filt(MC_GT.format(seq=seq), cls, gp)
                nt = filt(TRK.format(seq=seq), cls, tp)
                if ng == 0:
                    continue
                gt = mm.io.loadtxt(gp, fmt="mot15-2D", min_confidence=1)
                ts = mm.io.loadtxt(tp, fmt="mot15-2D")
                acc = mm.utils.compare_to_groundtruth(gt, ts, "iou", distth=0.5)
                ev = acc.mot_events.reset_index()
                sw = ev[ev.Type == "SWITCH"]
                summary.append(dict(seq=seq, cls=NAMES[cls], gt_boxes=ng, trk_boxes=nt,
                                    gt_ids=gt.index.get_level_values(1).nunique(),
                                    switches=len(sw)))
                for r in sw.itertuples():
                    ev_rows.append(dict(seq=seq, cls=NAMES[cls], frame=int(r.FrameId),
                                        gt_id=r.OId, new_hid=r.HId))
                # previous hypothesis for the same GT id (the track that lost it)
    s = pd.DataFrame(summary)
    e = pd.DataFrame(ev_rows)
    s.to_csv("_scratch/_r21_population_summary.csv", index=False)
    e.to_csv("_scratch/_r21_switch_events.csv", index=False)

    pd.set_option("display.width", 200)
    print("=== TARGET population: real ID switches (GT-matched, IoU 0.5, per class) ===\n")
    byc = s.groupby("cls").agg(gt_boxes=("gt_boxes", "sum"), gt_ids=("gt_ids", "sum"),
                               switches=("switches", "sum")).sort_values("switches",
                                                                         ascending=False)
    byc["pct_of_switches"] = (100 * byc.switches / byc.switches.sum()).round(1)
    print(byc)
    print("\nTOTAL ID switches (sum over classes) = %d" % s.switches.sum())
    print("\nper sequence:")
    print(s.groupby("seq").agg(switches=("switches", "sum"), gt_boxes=("gt_boxes", "sum")))

    print("\n=== ACTING population: track-frames a per-track lambda would touch ===")
    tot = 0
    perc = {}
    for seq in SEQS:
        raw = np.loadtxt(TRK.format(seq=seq), delimiter=",", ndmin=2)
        tot += len(raw)
        for c in np.unique(raw[:, 7]).astype(int):
            perc[c] = perc.get(c, 0) + int((raw[:, 7].astype(int) == c).sum())
    print("tracker output rows (track-frames) = %d" % tot)
    for c in sorted(perc):
        print("   %-11s %6d" % (NAMES.get(c, c), perc[c]))
    rep = pd.read_csv("_scratch/_appres_logs_replay/uav0000086_00000_v.csv")
    n_rep = sum(len(pd.read_csv(f"_scratch/_appres_logs_replay/{s_}.csv")) for s_ in SEQS)
    print("appearance-residual rows available for the dispersion signal = %d" % n_rep)

    print("\n=== ratio that decides whether this is worth building ===")
    print("acting events per target event = %.1f  (R3-3 had 193 acting events TOTAL)"
          % (tot / max(s.switches.sum(), 1)))


if __name__ == "__main__":
    main()
