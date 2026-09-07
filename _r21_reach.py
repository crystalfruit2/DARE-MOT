"""Round 2 #1, step 3: apply the "count the population, then multiply by the per-event
rate" rule — the standing rule that came out of the Round 3 #3 kill — so this candidate is
directly comparable to the one that just died.

R3-3's ceiling was 193 events -> ~3.5 IDSw (1.2%). This asks the same question of R2-1:
at a given operating point, how many of the 282 real ID switches does the dispersion signal
actually REACH (i.e. how many have at least one flagged track-frame in the window where a
per-track lambda could still have acted)?

Reach is an upper bound, not an effect. The conversion factor phi — the fraction of reached
switches that a lower lambda would actually prevent — cannot be measured without running the
tracker, which is blocked by Smart App Control. So the output here is
    expected prevented = reach x phi
with phi left explicit. That is the honest form, and it is what the run would settle.

The cost side is stated too: how many correct, non-switching track-frames get flagged at the
same operating point (those would lose appearance weight for nothing).
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

FEATS = ["std30", "std10", "mean30", "resid"]
RATES = (0.05, 0.10, 0.20, 0.30)
ACT_WINDOW = 20          # a lambda change is useful if it fires within this many frames before


def main():
    pd.set_option("display.width", 220)
    f = pd.read_csv("_scratch/_r21_features.csv")
    sw = pd.read_csv("_scratch/_r21_switch_events_full.csv")
    f["key"] = f.seq + "|" + f.track_id.astype(str)

    pairs = []
    for r in sw.itertuples():
        for hid in (r.new_hid, r.old_hid):
            if pd.isna(hid):
                continue
            pairs.append((f"{r.seq}|{int(hid)}", r.frame, r.cls, r.gt_id))
    # a switch is REACHED if any of its two tracks has a flagged frame in [t-W, t-1]
    sw["sid"] = sw.seq + "|" + sw.frame.astype(str) + "|" + sw.gt_id.astype(str)
    sid_of = {}
    for r in sw.itertuples():
        for hid in (r.new_hid, r.old_hid):
            if pd.isna(hid):
                continue
            for dt in range(1, ACT_WINDOW + 1):
                sid_of.setdefault((f"{r.seq}|{int(hid)}", r.frame - dt), set()).add(r.sid)

    keyframe = list(zip(f.key, f.frame))
    n_sw = len(sw)
    print("=== REACH: how many of the %d real ID switches does the signal touch? ===" % n_sw)
    print("(a switch is reached if either involved track is flagged within %d frames before it)\n"
          % ACT_WINDOW)
    rows = []
    for ft in FEATS:
        v = f[ft].to_numpy(dtype=float)
        for p in RATES:
            thr = np.nanpercentile(v, 100 * (1 - p))
            flag = v >= thr
            reached = set()
            for i in np.flatnonzero(flag):
                s = sid_of.get(keyframe[i])
                if s:
                    reached |= s
            n_flag = int(flag.sum())
            # cost side: flagged frames that are not in any pre-switch window
            useful = sum(1 for i in np.flatnonzero(flag) if keyframe[i] in sid_of)
            rows.append(dict(feature=ft, flag_rate=f"{p:.0%}", threshold=round(float(thr), 4),
                             flagged_frames=n_flag,
                             switches_reached=len(reached),
                             reach_pct=round(100 * len(reached) / n_sw, 1),
                             wasted_flags=n_flag - useful,
                             wasted_per_reached=round((n_flag - useful) / max(len(reached), 1), 1)))
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== per class, best feature (std30) at a 10%% flag rate ===")
    v = f["std30"].to_numpy(dtype=float)
    thr = np.nanpercentile(v, 90)
    flag = v >= thr
    reached = set()
    for i in np.flatnonzero(flag):
        s = sid_of.get(keyframe[i])
        if s:
            reached |= s
    sw["reached"] = sw.sid.isin(reached)
    print(sw.groupby("cls").agg(switches=("sid", "size"), reached=("reached", "sum")).assign(
        reach_pct=lambda d: (100 * d.reached / d.switches).round(1)).to_string())

    print("\n=== expected prevented IDSw, with the conversion factor phi left explicit ===")
    R = sw.reached.sum()
    print("reach at 10%% flag rate = %d of %d switches (%.1f%%)" % (R, n_sw, 100 * R / n_sw))
    print("   expected prevented = %d x phi   ->" % R)
    for phi in (0.10, 0.25, 0.50):
        print("      phi=%.2f  ->  %.1f IDSw  (%.1f%% of %d)"
              % (phi, R * phi, 100 * R * phi / n_sw, n_sw))
    print("\nfor comparison, Round 3 #3 (KILLED 2026-09-07) had a hard ceiling of 3.5 IDSw (1.2%).")
    print("phi is NOT measurable offline — it needs a tracker run, blocked by Smart App Control.")

    print("\n=== structural note on the cost side ===")
    corr = f[flag]
    print("flagged frames: %d (%.1f%% of %d). By construction these are the frames where the"
          % (flag.sum(), 100 * flag.mean(), len(f)))
    print("track's own embedding is LEAST self-consistent, i.e. where the appearance cue is")
    print("already weakest — median resid on flagged %.4f vs unflagged %.4f (%.1fx)."
          % (corr.resid.median(), f[~flag].resid.median(),
             corr.resid.median() / max(f[~flag].resid.median(), 1e-9)))
    print("So down-weighting appearance there removes a cue that was contributing little,")
    print("which bounds the collateral — but this is an ARGUMENT, not a measurement. Only a")
    print("run settles it.")


if __name__ == "__main__":
    main()
