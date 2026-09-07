"""Round 2 #1, step 1: does track-intrinsic embedding dispersion actually forewarn an ID
switch — and does it do so beyond what DARE already knows?

The candidate: the per-track historical-embedding buffer currently feeds only its FIRST
moment (mean / EMA / softmax) into lambda, and that whole dimension already tested inert
(the aggregation ladder). Use the SECOND moment instead — the track's own embedding
dispersion — as a live per-track appearance-trust signal driving a per-track lambda.

The signal is already on disk. `_appres_logs_replay/<seq>.csv` column `resid` is
`1 - cos(F_template, f_matched_detection)` evolved through the tracker's own
`STrack.update_features()` — i.e. literally the "newest-to-centroid distance" the candidate
names as one of its two dispersion measures. Its rolling standard deviation is the other
(intra-buffer variance). So this costs nothing to test.

Ground truth for the events: real ID switches from the MC MOTChallenge gt.txt matched at
IoU 0.5 per class, the same recipe the paper's numbers use (reproduces 282 exactly).
Evaluation matching is forced onto scipy because `lap` is blocked by Smart App Control;
that is the EVALUATION assignment, not the tracker's association.

THE CONFOUND THIS MUST SURVIVE, stated before the result (the Round 3 #1 lesson): small and
crowded objects have both noisier embeddings and more ID switches. If dispersion only
predicts switches because it proxies "small object in a crowd", it adds nothing over
DARE-MOT's existing size gate or over Deep OC-SORT's Adaptive Weighting (prior art). So
every headline number here is reported again stratified by box area, and head-to-head
against area alone.
"""
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
TRK = osp.join("YOLOX_outputs", "mc_dare_cv_rerun0803", "track_results", "{seq}.txt")
LOG = "_scratch/_appres_logs_replay/{seq}.csv"
SEQS = ["uav0000086_00000_v", "uav0000117_02622_v", "uav0000137_00458_v", "uav0000182_00000_v",
        "uav0000268_05773_v", "uav0000305_00000_v", "uav0000339_00001_v"]
NAMES = {1: "pedestrian", 2: "car", 3: "van", 4: "truck", 5: "bus"}
CAT_COL = 7
WINDOWS = (5, 10, 30)
LEAD = 5          # frames before the switch that a forewarning signal would have to live in


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


def switch_events():
    """(seq, cls, frame, gt_id, new_hid, old_hid) for every real ID switch."""
    rows = []
    with tempfile.TemporaryDirectory() as td:
        for seq in SEQS:
            for cls in NAMES:
                gp, tp = osp.join(td, "g.txt"), osp.join(td, "t.txt")
                if filt(MC_GT.format(seq=seq), cls, gp) == 0:
                    continue
                filt(TRK.format(seq=seq), cls, tp)
                gt = mm.io.loadtxt(gp, fmt="mot15-2D", min_confidence=1)
                ts = mm.io.loadtxt(tp, fmt="mot15-2D")
                acc = mm.utils.compare_to_groundtruth(gt, ts, "iou", distth=0.5)
                ev = acc.mot_events.reset_index()
                matched = ev[ev.Type.isin(["MATCH", "SWITCH"])]
                for r in ev[ev.Type == "SWITCH"].itertuples():
                    prev = matched[(matched.OId == r.OId) & (matched.FrameId < r.FrameId)]
                    old = prev.iloc[-1].HId if len(prev) else np.nan
                    rows.append(dict(seq=seq, cls=NAMES[cls], frame=int(r.FrameId),
                                     gt_id=r.OId, new_hid=r.HId, old_hid=old))
    return pd.DataFrame(rows)


def features():
    """Per (seq, frame, track_id) dispersion features from the replayed residuals."""
    out = []
    for seq in SEQS:
        d = pd.read_csv(LOG.format(seq=seq))
        d = d[d["valid"] == 1].sort_values(["track_id", "frame"]).copy()
        g = d.groupby("track_id")["resid"]
        for w in WINDOWS:
            d[f"mean{w}"] = g.transform(lambda s: s.rolling(w, min_periods=3).mean())
            d[f"std{w}"] = g.transform(lambda s: s.rolling(w, min_periods=3).std())
        d["n_seen"] = d.groupby("track_id").cumcount() + 1
        out.append(d)
    return pd.concat(out, ignore_index=True)


def auc(pos, neg):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos = pos[~np.isnan(pos)]; neg = neg[~np.isnan(neg)]
    if len(pos) < 5 or len(neg) < 5:
        return np.nan, len(pos), len(neg)
    allv = np.concatenate([pos, neg])
    r = pd.Series(allv).rank().to_numpy()
    n1, n0 = len(pos), len(neg)
    a = (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return a, n1, n0


def main():
    pd.set_option("display.width", 220)
    sw = switch_events()
    f = features()
    print("=== %d switch events; %d valid track-frames with a dispersion signal ==="
          % (len(sw), len(f)))

    # label: a track-frame is POSITIVE if that track is about to be involved in a switch
    # within LEAD frames (either as the hypothesis that takes over, or the one that loses it)
    f["key"] = f.seq + "|" + f.track_id.astype(str)
    pos_keys = {}
    for r in sw.itertuples():
        for hid in (r.new_hid, r.old_hid):
            if pd.isna(hid):
                continue
            pos_keys.setdefault(f"{r.seq}|{int(hid)}", []).append(r.frame)
    def is_pos(row):
        fr = pos_keys.get(row.key)
        if not fr:
            return False
        return any(0 < row.frame - t <= 0 or 0 <= t - row.frame <= LEAD for t in fr)
    lead_frames = []
    for k, frames in pos_keys.items():
        for t in frames:
            for dt in range(1, LEAD + 1):
                lead_frames.append((k, t - dt))
    lead_set = set(lead_frames)
    f["pos"] = [(k, fr) in lead_set for k, fr in zip(f.key, f.frame)]
    f["cname"] = f.cls.map(NAMES)
    print("positive (pre-switch) track-frames: %d of %d (%.2f%%)"
          % (f.pos.sum(), len(f), 100 * f.pos.mean()))

    feats = ["resid"] + [f"mean{w}" for w in WINDOWS] + [f"std{w}" for w in WINDOWS] + ["area"]
    print("\n=== discrimination: AUC of each feature, pre-switch vs all other track-frames ===")
    print("(0.50 = no information; higher = feature is larger before a switch)\n")
    rows = []
    for c in ["pedestrian", "car", "van", "ALL"]:
        sub = f if c == "ALL" else f[f.cname == c]
        r = {"class": c, "n_pos": int(sub.pos.sum()), "n_neg": int((~sub.pos).sum())}
        for ft in feats:
            a, n1, n0 = auc(sub.loc[sub.pos, ft], sub.loc[~sub.pos, ft])
            r[ft] = round(a, 4) if a == a else np.nan
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== the confound: same AUC computed WITHIN area quintiles (ALL classes) ===")
    f["aq"] = pd.qcut(f.area, 5, labels=["Q1 smallest", "Q2", "Q3", "Q4", "Q5 largest"])
    rows = []
    for q, sub in f.groupby("aq", observed=True):
        r = {"area quintile": q, "n_pos": int(sub.pos.sum()),
             "median area": round(sub.area.median(), 0)}
        for ft in ["resid", "mean10", "std10", "std30"]:
            a, _, _ = auc(sub.loc[sub.pos, ft], sub.loc[~sub.pos, ft])
            r[ft] = round(a, 4) if a == a else np.nan
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== operating point: top-decile flag rate, ALL classes ===")
    for ft in ["resid", "mean10", "std10", "std30"]:
        v = f[ft]
        thr = np.nanpercentile(v, 90)
        flag = v >= thr
        p_sw_given_flag = f.pos[flag].mean()
        p_sw = f.pos.mean()
        print("  %-7s thr=%.4f  P(pre-switch | flagged) = %.4f%%  vs base %.4f%%  lift %.2fx"
              % (ft, thr, 100 * p_sw_given_flag, 100 * p_sw, p_sw_given_flag / max(p_sw, 1e-12)))

    f.drop(columns=["key"]).to_csv("_scratch/_r21_features.csv", index=False)
    sw.to_csv("_scratch/_r21_switch_events_full.csv", index=False)
    print("\nwrote _r21_features.csv, _r21_switch_events_full.csv")


if __name__ == "__main__":
    main()
