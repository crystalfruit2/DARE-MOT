"""Join CMC warp residual to CMC-attributable IDSw events (2026-07-29).

Second half of the cheapest decisive experiment for the scale-conditional CMC gate design
(Projects/Dare_Mot/cmc-gate-design-2026-07-29.md). Produces the ΔIDSw-vs-(warp
residual/box diagonal) figure that decides G2 (warp-SNR gate) vs G1 (hard scale veto) vs
"CMC gate isn't real."

Method (own construction, not literally scripted in the design note — _idswitch_extract.py
only ever dumped crops for 2 hardcoded sequences; this generalizes its GT<->pred IoU
matching to all 7 val sequences and adds cross-run attribution):

1. For each of the two on-disk CMC-vs-no-CMC pairs (mc_dare_cmc vs mc_dare,
   mc_bytetrack_cmc vs mc_bytetrack), run the same per-frame GT<->pred Hungarian IoU
   matching (thresh 0.5) as _idswitch_extract.py, independently on each run's track
   output, to get each run's own list of IDSw events keyed by (seq, frame, gt_id).
2. A switch key present in the CMC run but NOT in the no-CMC run is "CMC-added"
   (CMC introduced this identity switch -> ΔIDSw=+1, bad).
   A switch key present in the no-CMC run but NOT the CMC run is "CMC-removed"
   (CMC prevented this switch -> ΔIDSw=-1, good).
   Present in both = not attributable to CMC either way, dropped.
   This is an operational definition (same GT identity switching at the same frame),
   not literal event tracing across two independently-ID-spaced runs -- the honest
   caveat is that it approximates "did CMC change what happens to this GT identity at
   this instant," not a guarantee it's mechanistically the same failure.
3. For each attributable switch, look up:
     d_i   = GT box diagonal (sqrt(w^2+h^2)) for that gt_id at that frame
     r_t   = frame's CMC warp residual (median inlier reprojection error, px),
             from _cmc_residual_logs/<seq>.json (property of the warp itself,
             shared by both runs in a pair since both use DARE_CMC=sparseOptFlow).
   ratio = r_t / d_i
4. Bin by ratio, plot net ΔIDSw (removed-count minus added-count, i.e. positive =
   CMC nets helpful in that bin) per bin, for DARE-pair, ByteTrack-pair, and pooled.

Caveat carried over from _idswitch_extract.py: this Hungarian-IoU switch definition is a
diagnostic approximation of IDSw, not the official py-motmetrics count -- use it only for
relative attribution/ratio analysis, not as a replacement headline number.
"""
import json
import os
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

VAL_DIR = r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone_MOT_Format\VisDrone2019-MOT-val"
# GT for matching/class/box-size must be the multi-class conversion (convert_visdrone_mc.py),
# NOT VisDrone_MOT_Format (the legacy single-class/pedestrian-leak-era gt.txt, which has an
# identical column layout but every row hardcoded to category=1 -- caught 2026-07-29 when a
# per-class cross-check against the established official CMC finding came back 100% pedestrian).
GT_DIR = r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone_MOT_Format_MC\VisDrone2019-MOT-val"
YOLOX_OUT = r"C:\Users\User\Desktop\projects\DARE-MOT\YOLOX_outputs"
RESID_DIR = r"C:\Users\User\Desktop\projects\DARE-MOT\_cmc_residual_logs"
OUT_DIR = r"C:\Users\User\Desktop\projects\DARE-MOT\_cmc_gate_decisive"

SEQS = [
    "uav0000086_00000_v", "uav0000117_02622_v", "uav0000137_00458_v", "uav0000182_00000_v",
    "uav0000268_05773_v", "uav0000305_00000_v", "uav0000339_00001_v",
]
IOU_THRESH = 0.5

PAIRS = {
    "dare": ("mc_dare_cmc", "mc_dare"),
    "bytetrack": ("mc_bytetrack_cmc", "mc_bytetrack"),
}


def xywh_to_xyxy(b):
    x, y, w, h = b
    return np.array([x, y, x + w, y + h])


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ix1 = np.maximum(ax1, bx1); iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2); iy2 = np.minimum(ay2, by2)
    iw = np.clip(ix2 - ix1, 0, None); ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / np.clip(union, 1e-9, None)


def load_mot(path, with_class=False):
    cols = (0, 1, 2, 3, 4, 5, 7) if with_class else (0, 1, 2, 3, 4, 5)
    rows = np.loadtxt(path, delimiter=",", usecols=cols)
    if rows.ndim == 1:
        rows = rows[None, :]
    by_frame = defaultdict(list)
    for r in rows:
        if with_class:
            fr, tid, x, y, w, h, cat = r
            by_frame[int(fr)].append((int(tid), x, y, w, h, int(cat)))
        else:
            fr, tid, x, y, w, h = r
            by_frame[int(fr)].append((int(tid), x, y, w, h))
    return by_frame


def find_switches(gt_by_frame, pred_by_frame):
    """Returns dict {(frame, gt_id): (x, y, w, h, cat)} for every IDSw event.
    gt_by_frame entries are (tid, x, y, w, h, cat) -- loaded with_class=True."""
    frames = sorted(set(gt_by_frame) | set(pred_by_frame))
    last_match = {}
    switches = {}
    for fr in frames:
        gts = gt_by_frame.get(fr, [])
        preds = pred_by_frame.get(fr, [])
        if not gts or not preds:
            continue
        gt_ids = [g[0] for g in gts]
        gt_boxes = np.array([xywh_to_xyxy(g[1:5]) for g in gts])
        pred_ids = [p[0] for p in preds]
        pred_boxes = np.array([xywh_to_xyxy(p[1:5]) for p in preds])

        ious = iou_matrix(gt_boxes, pred_boxes)
        row_ind, col_ind = linear_sum_assignment(-ious)

        for r, c in zip(row_ind, col_ind):
            if ious[r, c] < IOU_THRESH:
                continue
            gid, pid = gt_ids[r], pred_ids[c]
            if gid in last_match and last_match[gid] != pid:
                switches[(fr, gid)] = gts[r][1:]  # (x, y, w, h, cat)
            last_match[gid] = pid
    return switches


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    resid_by_seq = {}
    for seq in SEQS:
        with open(os.path.join(RESID_DIR, f"{seq}.json")) as f:
            log = json.load(f)
        resid_by_seq[seq] = {r["frame"]: r for r in log}

    records = []  # one row per attributable switch
    coverage = defaultdict(lambda: {"cmc_only": 0, "nocmc_only": 0, "both": 0,
                                     "no_residual": 0})

    for pair_name, (cmc_exp, nocmc_exp) in PAIRS.items():
        for seq in SEQS:
            gt_path = os.path.join(GT_DIR, seq, "gt", "gt.txt")
            gt_by_frame = load_mot(gt_path, with_class=True)

            cmc_pred = load_mot(os.path.join(YOLOX_OUT, cmc_exp, "track_results", f"{seq}.txt"))
            nocmc_pred = load_mot(os.path.join(YOLOX_OUT, nocmc_exp, "track_results", f"{seq}.txt"))

            cmc_sw = find_switches(gt_by_frame, cmc_pred)
            nocmc_sw = find_switches(gt_by_frame, nocmc_pred)

            cmc_keys = set(cmc_sw)
            nocmc_keys = set(nocmc_sw)
            added = cmc_keys - nocmc_keys      # CMC introduced -> bad
            removed = nocmc_keys - cmc_keys    # CMC prevented -> good
            both = cmc_keys & nocmc_keys

            coverage[(pair_name, seq)]["cmc_only"] = len(added)
            coverage[(pair_name, seq)]["nocmc_only"] = len(removed)
            coverage[(pair_name, seq)]["both"] = len(both)

            resid_map = resid_by_seq[seq]

            def emit(keys, box_lookup, label):
                for (fr, gid) in keys:
                    r = resid_map.get(fr)
                    if r is None or r["resid_px"] is None:
                        coverage[(pair_name, seq)]["no_residual"] += 1
                        continue
                    x, y, w, h, cat = box_lookup[(fr, gid)]
                    diag = float(np.hypot(w, h))
                    if diag <= 0:
                        continue
                    records.append({
                        "pair": pair_name, "seq": seq, "frame": fr, "gt_id": gid,
                        "label": label, "resid_px": r["resid_px"], "diag_px": diag,
                        "ratio": r["resid_px"] / diag, "cat": int(cat),
                    })

            emit(added, cmc_sw, "added")
            emit(removed, nocmc_sw, "removed")

    with open(os.path.join(OUT_DIR, "switch_records.json"), "w") as f:
        json.dump(records, f, indent=1)

    print(f"Total attributable switch events with valid residual: {len(records)}")
    for (pair_name, seq), c in coverage.items():
        print(f"  {pair_name:10s} {seq:22s} added={c['cmc_only']:3d} removed={c['nocmc_only']:3d} "
              f"both={c['both']:3d} no_resid={c['no_residual']:3d}")

    # --- Decisive figure: net ΔIDSw (removed - added) per ratio bin ---
    if len(records) == 0:
        print("No records -- cannot build figure.")
        return

    ratios = np.array([r["ratio"] for r in records])
    labels = np.array([r["label"] for r in records])
    pairs_arr = np.array([r["pair"] for r in records])

    bin_edges = np.quantile(ratios, np.linspace(0, 1, 7))  # 6 equal-count bins
    bin_edges[0] -= 1e-9
    bin_edges[-1] += 1e-9

    def summarize(mask, tag):
        r = ratios[mask]; l = labels[mask]
        if len(r) == 0:
            print(f"  [{tag}] no data")
            return
        bin_ids = np.digitize(r, bin_edges) - 1
        print(f"  [{tag}] n={len(r)}, mean ratio added={r[l=='added'].mean() if (l=='added').any() else float('nan'):.3f}, "
              f"mean ratio removed={r[l=='removed'].mean() if (l=='removed').any() else float('nan'):.3f}")
        for b in range(len(bin_edges) - 1):
            m = bin_ids == b
            n_add = int(((l == "added") & m).sum())
            n_rem = int(((l == "removed") & m).sum())
            lo, hi = bin_edges[b], bin_edges[b + 1]
            net = n_rem - n_add
            print(f"    ratio[{lo:6.3f},{hi:6.3f}) n={m.sum():3d} added={n_add:3d} removed={n_rem:3d} net(removed-added)={net:+d}")

    print("\n=== Pooled (dare + bytetrack pairs) ===")
    summarize(np.ones(len(records), dtype=bool), "pooled")
    print("\n=== DARE pair only ===")
    summarize(pairs_arr == "dare", "dare")
    print("\n=== ByteTrack pair only ===")
    summarize(pairs_arr == "bytetrack", "bytetrack")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, mask, title in [
            (axes[0], np.ones(len(records), dtype=bool), "Pooled"),
            (axes[1], None, "By pair (added=red, removed=green)"),
        ]:
            if mask is not None:
                r = ratios[mask]; l = labels[mask]
                ax.hist(r[l == "added"], bins=20, alpha=0.6, label="added (CMC caused)", color="tab:red")
                ax.hist(r[l == "removed"], bins=20, alpha=0.6, label="removed (CMC fixed)", color="tab:green")
                ax.set_xlabel("warp residual / GT box diagonal")
                ax.set_ylabel("count")
                ax.set_title(title)
                ax.legend()
            else:
                for pname, color in [("dare", "tab:blue"), ("bytetrack", "tab:orange")]:
                    m = pairs_arr == pname
                    r = ratios[m]; l = labels[m]
                    net_by_bin = []
                    centers = []
                    for b in range(len(bin_edges) - 1):
                        bm = (r >= bin_edges[b]) & (r < bin_edges[b + 1])
                        n_add = int(((l == "added") & bm).sum())
                        n_rem = int(((l == "removed") & bm).sum())
                        net_by_bin.append(n_rem - n_add)
                        centers.append((bin_edges[b] + bin_edges[b + 1]) / 2)
                    ax.plot(centers, net_by_bin, marker="o", label=pname, color=color)
                ax.axhline(0, color="k", lw=0.8)
                ax.set_xlabel("warp residual / GT box diagonal (bin center)")
                ax.set_ylabel("net ΔIDSw (removed - added)")
                ax.set_title(title)
                ax.legend()
        fig.tight_layout()
        fig_path = os.path.join(OUT_DIR, "cmc_gate_decisive.png")
        fig.savefig(fig_path, dpi=150)
        print(f"\nFigure saved to {fig_path}")
    except ImportError:
        print("\nmatplotlib not available -- skipped figure, data is in switch_records.json")

    print("DONE")


if __name__ == "__main__":
    main()
