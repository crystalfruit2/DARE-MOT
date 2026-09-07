"""Round 4 #1 — chronic-ambiguity modulation: is the TEMPORAL INTEGRAL of the per-track
assignment margin better than the margin itself?

The candidate. Deep OC-SORT's Adaptive Weighting consumes the per-track appearance margin
(best minus second-best) AT ONE INSTANT. This consumes its history: a running low-margin
occupancy A_i(t) = fraction of this track's recent frames whose margin sat below a
separability floor, driving a continuous per-track lambda. The claim: a 1-frame ambiguity
and a 100-frame ambiguity are identical to every published margin mechanism and are
physically different situations -- the first is an occlusion that appearance recovers from,
the second means the embedding CANNOT separate these objects and never will.

Three arms, mandatory, and the third has to beat the first two on their own terms:
  AW    : instantaneous -reid_margin        (the correctly-specified prior-art proxy)
  R2-1  : std30 of the appearance residual  (the project's live candidate)
  INT   : low-margin occupancy over W       (this candidate)

Pre-registered kill conditions, from the Round 4 write-up:
  (a) INT fails to beat AW by a clear gap at lead d >= 10  -> it is AW with a delay;
  (b) INT's AUC is within noise of R2-1 AND its flagged frames overlap R2-1's by > 80%
      at a matched flag rate -> it is a re-parameterisation of the live candidate, and it
      becomes a footnote exactly as the mean-object-area gate reduced Round 3 #1;
  (c) INT collapses onto `area` or a density proxy inside strata.

Per filter 3 (suspect the prior-art arm): the AW arm is given the same tuning budget as
INT -- both are swept over their free parameter and reported at their best.

Offline throughout: `_verify_out_E_margins.csv` (54,727 track-frames) + the real GT-matched
switches in `_r21_switch_events_full.csv`. No tracker run.
"""

# DEFECT NOTE 2026-09-07: flag selection below uses np.nanpercentile + '>=', which
# under-selects any column carrying NaN (std30 has 955 NaN from min_periods=3), giving a
# ~1.8% budget mismatch between arms at a nominal 10%. Select by exact count instead:
#     idx = np.argsort(-np.nan_to_num(v, nan=-np.inf))[:int(round(p*len(v)))]
# Immaterial to the conclusions on record (all were re-run by exact count), but any new
# comparison must fix this first.
# ALSO: `reach` at a fixed frame budget is an INVALID metric -- a uniform-random flagger
# beats every real signal on it (199.9 +/- 6.5 vs AW 192, std30 165 at 10%). It is a flag-
# scatter statistic. Use AUC. See experiment-log.md 2026-09-07 CORRECTION.

import numpy as np
import pandas as pd

LEADS = (1, 3, 5, 10, 20, 40)
WINDOWS = (10, 30, 60)
FLOOR_Q = (10, 25, 40)          # percentile of the global margin distribution
NAMES = {1: "pedestrian", 2: "car", 3: "van", 4: "truck", 5: "bus"}


def auc(pos, neg):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos = pos[~np.isnan(pos)]; neg = neg[~np.isnan(neg)]
    if len(pos) < 5 or len(neg) < 5:
        return np.nan
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    n1, n0 = len(pos), len(neg)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    pd.set_option("display.width", 240)
    m = pd.read_csv("_scratch/_verify_out_E_margins.csv").rename(columns={"tid": "track_id"})
    f = pd.read_csv("_scratch/_r21_features.csv")
    sw = pd.read_csv("_scratch/_r21_switch_events_full.csv")

    d = m.merge(f[["seq", "track_id", "frame", "std30", "mean30", "area", "n_valid"]],
                on=["seq", "track_id", "frame"], how="left")
    d["cname"] = d.cls.map(NAMES)
    d = d.sort_values(["seq", "track_id", "frame"]).reset_index(drop=True)
    print("rows: %d  (margins joined to R2-1 features)" % len(d))

    # ---- build the integral arms: low-margin occupancy over W, at several floors ----
    g = d.groupby(["seq", "track_id"])["reid_margin"]
    for q in FLOOR_Q:
        floor = np.percentile(d.reid_margin, q)
        below = (d.reid_margin < floor).astype(float)
        d["_b"] = below
        gb = d.groupby(["seq", "track_id"])["_b"]
        for W in WINDOWS:
            d[f"INT_q{q}_w{W}"] = gb.transform(lambda s: s.rolling(W, min_periods=3).mean())
    d.drop(columns=["_b"], inplace=True)
    # AW arm, with its own tuning budget: instantaneous, and short smoothings of it
    d["AW"] = -d.reid_margin
    for W in (3, 5, 10):
        d[f"AW_s{W}"] = g.transform(lambda s: -s.rolling(W, min_periods=2).mean())

    # ---- label positives at a single fixed lead offset ----
    pairs = []
    for r in sw.itertuples():
        for hid in (r.new_hid, r.old_hid):
            if pd.isna(hid):
                continue
            pairs.append((r.seq, int(hid), r.frame, r.cls))
    idx = {(s, t, fr): i for i, (s, t, fr) in enumerate(zip(d.seq, d.track_id, d.frame))}

    int_cols = [c for c in d.columns if c.startswith("INT_")]
    aw_cols = ["AW"] + [f"AW_s{w}" for w in (3, 5, 10)]

    print("\n=== ARM COMPARISON: AUC at a single fixed lead offset (best variant per arm) ===")
    print("higher = the signal is larger before a switch\n")
    rows = []
    for dlead in LEADS:
        sel = [idx.get((s, t, fr - dlead)) for s, t, fr, _ in pairs]
        sel = [i for i in sel if i is not None]
        if len(sel) < 10:
            continue
        pos = d.iloc[sel]
        neg = d.drop(index=d.index[sel])
        row = {"lead d": dlead, "n_pos": len(sel)}
        best_aw = max(((auc(pos[c], neg[c]), c) for c in aw_cols), key=lambda z: (z[0] or 0))
        best_int = max(((auc(pos[c], neg[c]), c) for c in int_cols), key=lambda z: (z[0] or 0))
        row["AW (best)"] = round(best_aw[0], 4); row["AW variant"] = best_aw[1]
        row["R2-1 std30"] = round(auc(pos["std30"], neg["std30"]), 4)
        row["INT (best)"] = round(best_int[0], 4); row["INT variant"] = best_int[1].replace("INT_", "")
        row["INT - AW"] = round(best_int[0] - best_aw[0], 4)
        rows.append(row)
    tbl = pd.DataFrame(rows)
    print(tbl.to_string(index=False))

    # ---- kill condition (a) ----
    long_lead = tbl[tbl["lead d"] >= 10]
    beats = (long_lead["INT - AW"] > 0.02).all() if len(long_lead) else False
    print("\n(a) INT beats AW by >0.02 AUC at every lead >= 10 : %s" % ("YES" if beats else "NO"))
    print("    margins at d>=10: %s" % list(long_lead["INT - AW"]))

    # ---- kill condition (b): flag overlap with R2-1 at matched 10% flag rate ----
    best_int_col = tbl["INT variant"].iloc[-1]
    best_int_col = "INT_" + best_int_col
    thr_i = np.nanpercentile(d[best_int_col], 90)
    thr_s = np.nanpercentile(d["std30"], 90)
    fi = d[best_int_col] >= thr_i
    fs = d["std30"] >= thr_s
    both = (fi & fs).sum()
    ov = both / max(min(fi.sum(), fs.sum()), 1)
    print("\n(b) flag overlap with R2-1 at a matched 10%% flag rate: %.1f%% (%d of %d)"
          % (100 * ov, both, int(min(fi.sum(), fs.sum()))))
    print("    kill if > 80%% AND AUC within noise -> %s"
          % ("OVERLAP HIGH" if ov > 0.8 else "distinct enough"))

    # ---- kill condition (c): strata ----
    print("\n(c) does INT survive inside area quintiles? (lead d=5)")
    sel = [idx.get((s, t, fr - 5)) for s, t, fr, _ in pairs]
    sel = [i for i in sel if i is not None]
    pmask = np.zeros(len(d), bool); pmask[sel] = True
    d["pos5"] = pmask
    d["aq"] = pd.qcut(d.area.rank(method="first"), 5,
                      labels=["Q1 small", "Q2", "Q3", "Q4", "Q5 large"])
    out = []
    for q, sub in d.groupby("aq", observed=True):
        out.append({"area quintile": q, "n_pos": int(sub.pos5.sum()),
                    "AW": round(auc(sub.loc[sub.pos5, "AW"], sub.loc[~sub.pos5, "AW"]), 4),
                    "std30": round(auc(sub.loc[sub.pos5, "std30"], sub.loc[~sub.pos5, "std30"]), 4),
                    "INT": round(auc(sub.loc[sub.pos5, best_int_col],
                                     sub.loc[~sub.pos5, best_int_col]), 4)})
    print(pd.DataFrame(out).to_string(index=False))

    # ---- addressable share / reach on real GT, the standing rule ----
    print("\n=== REACH on real GT switches (a switch is reached if either involved track is")
    print("    flagged within 20 frames before it) ===")
    sid = {}
    sw["sid"] = sw.seq + "|" + sw.frame.astype(str) + "|" + sw.gt_id.astype(str)
    for r in sw.itertuples():
        for hid in (r.new_hid, r.old_hid):
            if pd.isna(hid):
                continue
            for dt in range(1, 21):
                sid.setdefault((r.seq, int(hid), r.frame - dt), set()).add(r.sid)
    kf = list(zip(d.seq, d.track_id, d.frame))
    for name, col in (("AW", "AW"), ("R2-1 std30", "std30"), ("INT", best_int_col)):
        v = d[col].to_numpy(float)
        for p in (0.10, 0.20):
            thr = np.nanpercentile(v, 100 * (1 - p))
            reached = set()
            for i in np.flatnonzero(v >= thr):
                s = sid.get(kf[i])
                if s:
                    reached |= s
            print("  %-11s flag %3.0f%%  reach %3d / %d (%.1f%%)"
                  % (name, 100 * p, len(reached), len(sw), 100 * len(reached) / len(sw)))

    print("\n=== per class, AUC at lead d=5 ===")
    out = []
    for c, sub in d.groupby("cname"):
        if sub.pos5.sum() < 10:
            continue
        out.append({"class": c, "n_pos": int(sub.pos5.sum()),
                    "AW": round(auc(sub.loc[sub.pos5, "AW"], sub.loc[~sub.pos5, "AW"]), 4),
                    "std30": round(auc(sub.loc[sub.pos5, "std30"], sub.loc[~sub.pos5, "std30"]), 4),
                    "INT": round(auc(sub.loc[sub.pos5, best_int_col],
                                     sub.loc[~sub.pos5, best_int_col]), 4)})
    print(pd.DataFrame(out).to_string(index=False))
    d.to_csv("_scratch/_r4c1_margin_integral.csv", index=False)
    print("\nbest INT variant: %s" % best_int_col)


if __name__ == "__main__":
    main()
