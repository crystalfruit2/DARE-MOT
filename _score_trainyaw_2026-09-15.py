"""Score the yaw-generalization check (9 yaw-containing train sequences, D3) under the headline protocol,
then count identity switches inside / outside the yaw windows (segment + 5 following frames).

Pipeline = the val7 headline (on_score vote -> official ignored-region filter -> per-class motmetrics, pooled),
redirected to train. Pre-registered rules Y1-Y4: vault experiment-log §2026-09-15 "Yaw generalization check".
Absolute numbers are NOT reportable (D3 trained on these frames).
"""
import importlib.util
import json
import os
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("sc", os.path.join(HERE, "_score_cmcfix_2026-09-10.py"))
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)
sm, so = sc.sm, sc.so
import motmetrics as mm  # noqa: E402  (after sm import: shim in place)

SCAN = json.load(open(os.path.join(HERE, "_scratch", "yaw_scan", "train.json")))
SEQS = sorted(s for s, v in SCAN.items() if v["yaw_segments"])
TAIL = 5
WINDOWS = {s: [(a, b + TAIL) for a, b in SCAN[s]["yaw_segments"]] for s in SEQS}
TRAIN_MC = (r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone_MOT_Format_MC"
            r"\VisDrone2019-MOT-train\{seq}\gt\gt.txt")
sm.SEQS, sm.MC_GT, sc.RAW_GT = list(SEQS), TRAIN_MC, TRAIN_MC
so.RAW = r"C:\Users\User\Desktop\datasets\VisDrone2019-MOT-train\annotations\{seq}.txt"
so.OUT = os.path.join(HERE, "_scratch", "official_trainyaw")
sc.OUT = os.path.join(HERE, "_scratch", "trainyaw_score")
WT, MAIN = sc.WT, sc.MAIN
ARMS = [
    ("ty_bt", WT, "ByteTrack (no CMC)"),
    ("ty_bt_cmcparity_j", WT, "ByteTrack + parity CMC"),
    ("ty_bt_cmcscale_j", WT, "ByteTrack + scale CMC"),
    ("ty_botsort", MAIN, "BoT-SORT-ReID (kron warp)"),
    ("ty_botsort_scalewarp", MAIN, "BoT-SORT-ReID + scale warp"),
]


def switch_frames(voted_file, gt_file, tmp):
    frames = []
    for cls in sm.MC_NAMES:
        gt_f, n_gt = sm._filter_by_class(gt_file, cls, tmp, "gt")
        ts_f, n_ts = sm._filter_by_class(voted_file, cls, tmp, "ts")
        if n_gt == 0 and n_ts == 0:
            continue
        gt = mm.io.loadtxt(gt_f, fmt="mot15-2D", min_confidence=1)
        ts = mm.io.loadtxt(ts_f, fmt="mot15-2D", min_confidence=-1)
        ev = mm.utils.compare_to_groundtruth(gt, ts, "iou", distth=0.5).mot_events
        frames += [int(f) for f in ev[ev.Type == "SWITCH"].index.get_level_values(0)]
    return sorted(frames)


if __name__ == "__main__":
    res = {"head": {}, "windows": {}}
    present = []
    for expn, root, label in ARMS:
        d = os.path.join(root, expn, "track_results")
        if all(os.path.exists(os.path.join(d, s + ".txt")) for s in SEQS):
            present.append((expn, root, label))
        else:
            print(f"{expn:24s} INCOMPLETE -- skipped")
    off_gt = so.prepare_gt()
    with tempfile.TemporaryDirectory() as tmp:
        for expn, root, label in present:
            sm.MC_GT, sm.SEQS = off_gt, list(SEQS)
            vdir = sc.voted_filtered(expn, root)
            res["head"][expn] = sm.score_run(vdir)
            per_seq = {}
            for s in SEQS:
                fr = switch_frames(os.path.join(vdir, s + ".txt"), off_gt.format(seq=s), tmp)
                inside = sum(any(a <= f <= b for a, b in WINDOWS[s]) for f in fr)
                per_seq[s] = {"total": len(fr), "in_window": inside, "out_window": len(fr) - inside, "frames": fr}
            res["windows"][expn] = per_seq

    print(f"\nYAW CHECK: {len(SEQS)} yaw-containing TRAIN sequences, D3 (trained on them -> rankings/locations only)")
    print(f"windows = yaw segment + {TAIL} frames; {sum(b - a + 1 for s in SEQS for a, b in WINDOWS[s])} window frames of "
          f"{sum(SCAN[s]['frames'] for s in SEQS)}")
    print(f"{'arm':30s} {'IDSw':>5s} {'IDF1':>6s} {'MOTA':>6s} | {'in-win':>6s} {'out-win':>7s}")
    for expn, root, label in present:
        h, w = res["head"][expn], res["windows"][expn]
        print(f"{label:30s} {h['SUM_IDSw']:5d} {100 * h['MICRO']['idf1']:6.2f} {100 * h['MICRO']['mota']:6.2f} | "
              f"{sum(v['in_window'] for v in w.values()):6d} {sum(v['out_window'] for v in w.values()):7d}")
    print("\nin-window IDSw per sequence (max |theta| in brackets)")
    print(f"{'arm':30s}" + "".join(f"{s[3:10]:>9s}" for s in SEQS))
    print(f"{'':30s}" + "".join(f"{'(' + format(SCAN[s]['abs_theta_max'], '.3f') + ')':>9s}" for s in SEQS))
    for expn, root, label in present:
        print(f"{label:30s}" + "".join(f"{res['windows'][expn][s]['in_window']:9d}" for s in SEQS))
    print("\nout-of-window IDSw per sequence")
    for expn, root, label in present:
        print(f"{label:30s}" + "".join(f"{res['windows'][expn][s]['out_window']:9d}" for s in SEQS))
    os.makedirs(sc.OUT, exist_ok=True)
    dst = os.path.join(sc.OUT, "trainyaw_summary.json")
    json.dump(res, open(dst, "w"), indent=1, default=str)
    print(f"\nwrote {dst}")
