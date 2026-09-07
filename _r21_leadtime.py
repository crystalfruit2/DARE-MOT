"""Round 2 #1, step 2: is the dispersion signal a FOREWARNING, or is it the ID switch's own
onset leaking backwards?

This is the circularity that would kill the candidate, and it has to be excluded before any
of step 1's AUCs mean anything. `resid = 1 - cos(F_template, f_matched_detection)` is
computed on whatever detection the tracker matched. In the frames immediately before a
logged ID switch the track may ALREADY be drifting onto the wrong object, so a high residual
there would be a *symptom* of the switch in progress, not a signal that could have prevented
it. A per-track lambda has to act BEFORE the association goes wrong to be worth anything.

Test: recompute the discrimination at a single, fixed lead offset — the feature is read
exactly d frames before the switch and nowhere else — for d = 1, 2, 3, 5, 10, 20, 40.
  - real forewarning  -> AUC stays well above 0.5 out to d = 10-20
  - switch onset only -> AUC collapses towards 0.5 beyond d ~ 2-3

Two further confounds are controlled in the same pass, both of which could manufacture the
step-1 result on their own:
  - `tracklet_len` / frames-seen: long-lived tracks accumulate more dispersion AND have more
    opportunity to eventually switch.
  - `was_lost`: a track that was recently lost has both an elevated residual and an elevated
    switch risk, so it could carry the whole effect.
Both are reported as their own AUCs and as strata.
"""
import numpy as np
import pandas as pd

LEADS = (1, 2, 3, 5, 10, 20, 40)
FEATS = ["resid", "mean10", "mean30", "std10", "std30", "area", "tracklet_len", "was_lost"]


def auc(pos, neg):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos = pos[~np.isnan(pos)]; neg = neg[~np.isnan(neg)]
    if len(pos) < 5 or len(neg) < 5:
        return np.nan, len(pos)
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    n1, n0 = len(pos), len(neg)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0), n1


def main():
    pd.set_option("display.width", 220)
    f = pd.read_csv("_scratch/_r21_features.csv")
    sw = pd.read_csv("_scratch/_r21_switch_events_full.csv")
    f["key"] = f.seq + "|" + f.track_id.astype(str)

    # every (track, switch-frame) pair the mechanism could have acted on
    pairs = []
    for r in sw.itertuples():
        for hid in (r.new_hid, r.old_hid):
            if pd.isna(hid):
                continue
            pairs.append((f"{r.seq}|{int(hid)}", r.frame, r.cls))
    print("=== single-offset lead test: %d (track, switch) pairs ===" % len(pairs))
    print("feature read EXACTLY d frames before the switch, nowhere else\n")

    idx = {(k, fr): i for i, (k, fr) in enumerate(zip(f.key, f.frame))}
    rows = []
    for d in LEADS:
        sel = [idx.get((k, fr - d)) for k, fr, _ in pairs]
        sel = [i for i in sel if i is not None]
        pos = f.iloc[sel]
        neg = f.drop(index=f.index[sel])
        row = {"lead d": d, "n_pos": len(pos)}
        for ft in FEATS:
            a, _ = auc(pos[ft], neg[ft])
            row[ft] = round(a, 4) if a == a else np.nan
        rows.append(row)
    lead_tbl = pd.DataFrame(rows)
    print(lead_tbl.to_string(index=False))

    print("\n=== same, vehicles only (car + van + truck + bus) ===")
    fv = f[f.cname != "pedestrian"]
    idxv = {(k, fr): i for i, (k, fr) in enumerate(zip(fv.key, fv.frame))}
    rows = []
    for d in LEADS:
        sel = [idxv.get((k, fr - d)) for k, fr, c in pairs if c != "pedestrian"]
        sel = [i for i in sel if i is not None]
        if len(sel) < 5:
            continue
        pos = fv.iloc[sel]
        neg = fv.drop(index=fv.index[sel])
        row = {"lead d": d, "n_pos": len(pos)}
        for ft in ["resid", "mean30", "std10", "std30"]:
            a, _ = auc(pos[ft], neg[ft])
            row[ft] = round(a, 4) if a == a else np.nan
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== confound: was_lost ===")
    print("share of track-frames with was_lost=1: %.3f%%" % (100 * f.was_lost.mean()))
    for d in (1, 5, 10):
        sel = [idx.get((k, fr - d)) for k, fr, _ in pairs]
        sel = [i for i in sel if i is not None]
        pos = f.iloc[sel]
        print("  d=%2d  was_lost rate: pre-switch %.3f%%  baseline %.3f%%"
              % (d, 100 * pos.was_lost.mean(), 100 * f.was_lost.mean()))
        sub = f[f.was_lost == 0].reset_index(drop=True)
        idx2 = {(k, fr): i for i, (k, fr) in enumerate(zip(sub.key, sub.frame))}
        sel2 = [i for i in (idx2.get((k, fr - d)) for k, fr, _ in pairs) if i is not None]
        if len(sel2) >= 5:
            p2 = sub.iloc[sel2]; n2 = sub.drop(index=sub.index[sel2])
            a30, _ = auc(p2["std30"], n2["std30"])
            am, _ = auc(p2["mean30"], n2["mean30"])
            print("        never-lost frames only: std30 AUC %.4f | mean30 AUC %.4f (n_pos %d)"
                  % (a30, am, len(p2)))

    print("\n=== confound: tracklet length ===")
    f["tlq"] = pd.qcut(f.tracklet_len.rank(method="first"), 4,
                       labels=["T1 short", "T2", "T3", "T4 long"])
    d = 5
    sel = [idx.get((k, fr - d)) for k, fr, _ in pairs]
    sel = [i for i in sel if i is not None]
    posmask = np.zeros(len(f), bool); posmask[sel] = True
    f["pos5"] = posmask
    rows = []
    for q, sub in f.groupby("tlq", observed=True):
        r = {"tracklet quartile": q, "n_pos": int(sub.pos5.sum()),
             "median len": int(sub.tracklet_len.median())}
        for ft in ["std30", "mean30"]:
            a, _ = auc(sub.loc[sub.pos5, ft], sub.loc[~sub.pos5, ft])
            r[ft] = round(a, 4) if a == a else np.nan
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))

    lead_tbl.to_csv("_scratch/_r21_leadtime.csv", index=False)
    print("\nwrote _r21_leadtime.csv")


if __name__ == "__main__":
    main()
