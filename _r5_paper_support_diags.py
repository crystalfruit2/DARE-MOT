"""Three offline paper-support diagnostics (2026-09-07). No novelty claims here — these
size existing candidates and settle two open premises. All output to _scratch/.

  D1  FP PARTITION. Split the 11,115 false positives into "co-located with an unmatched GT
      box" (label errors, duplicates, IoU misses — fixable at zero recall cost) versus
      "isolated" (hallucinations on background). This sizes Round 4 candidate ② (per-track
      class-label churn, which can only fix the co-located kind) and ④ (object-vs-background
      margin, which targets the isolated kind) at the same time, before either is built.
      Also broken out by owning-track length, since GIAOTracker's length re-scoring makes
      track length a mandatory control for any FP mechanism.

  D2  EMBEDDING SEPARABILITY PER CLASS. The premise "vehicle appearance carries near-zero
      information" was only ever established on truck (1 IDSw) and bus (0) — classes with
      nothing to decide. The direct test is same-ID versus different-ID embedding distance,
      per class, per area quintile. It is computable with ZERO new compute: the cached
      margin file already carries `resid` = distance from a detection to its OWN track's
      template (a same-ID distance) and `d_second` = distance to the best COMPETING track's
      template (a different-ID distance, and specifically the hardest impostor). AUC between
      them is exactly "can the embedder tell this object from the one most likely to be
      confused with it". Caveat stated up front: `d_second` is the nearest impostor, not an
      average negative, so this is a harder test than a random-pair protocol — which is the
      right test, because the nearest impostor is what the assignment actually competes with.

  D3  CROWD-DENSITY PREP. The co-factor for the CMC width-corruption finding: harm should
      scale as (perturbation) / (nearest-neighbour spacing), not (perturbation) / (target
      size). Flagged as cheap on 2026-07-29 and never run. This computes the density metrics
      and lays out the regression; the full CMC-on/off fit stays behind the SAC decision
      because it needs a corrected re-run to be worth anything.
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

OUT = "_scratch"
MC_GT = (r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone_MOT_Format_MC"
         r"\VisDrone2019-MOT-val\{seq}\gt\gt.txt")
TRK = osp.join("YOLOX_outputs", "mc_dare_cv_rerun0803", "track_results", "{seq}.txt")
SEQS = ["uav0000086_00000_v", "uav0000117_02622_v", "uav0000137_00458_v", "uav0000182_00000_v",
        "uav0000268_05773_v", "uav0000305_00000_v", "uav0000339_00001_v"]
NAMES = {1: "pedestrian", 2: "car", 3: "van", 4: "truck", 5: "bus"}
CAT_COL = 7


def load_mot(path):
    if not osp.exists(path):
        return pd.DataFrame(columns=["frame", "id", "x", "y", "w", "h", "conf", "cls"])
    raw = np.loadtxt(path, delimiter=",", ndmin=2)
    return pd.DataFrame({"frame": raw[:, 0].astype(int), "id": raw[:, 1].astype(int),
                         "x": raw[:, 2], "y": raw[:, 3], "w": raw[:, 4], "h": raw[:, 5],
                         "conf": raw[:, 6], "cls": raw[:, CAT_COL].astype(int)})


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


def iou_xywh(a, B):
    """a = (x,y,w,h) top-left; B = Nx4 same. Returns N IoUs."""
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1 = B[:, 0], B[:, 1]
    bx2, by2 = B[:, 0] + B[:, 2], B[:, 1] + B[:, 3]
    iw = np.maximum(0, np.minimum(ax2, bx2) - np.maximum(ax1, bx1))
    ih = np.maximum(0, np.minimum(ay2, by2) - np.maximum(ay1, by1))
    inter = iw * ih
    return inter / np.maximum(a[2] * a[3] + B[:, 2] * B[:, 3] - inter, 1e-9)


# ======================================================================================
def d1_fp_partition():
    print("=" * 78)
    print("D1 — FP PARTITION: co-located with an unmatched GT box, or isolated?")
    print("=" * 78)
    rows = []
    with tempfile.TemporaryDirectory() as td:
        for seq in SEQS:
            gt_all = load_mot(MC_GT.format(seq=seq))
            trk_all = load_mot(TRK.format(seq=seq))
            matched_t, matched_g = set(), set()
            for cls in NAMES:
                gp, tp = osp.join(td, "g.txt"), osp.join(td, "t.txt")
                if filt(MC_GT.format(seq=seq), cls, gp) == 0:
                    continue
                filt(TRK.format(seq=seq), cls, tp)
                gt = mm.io.loadtxt(gp, fmt="mot15-2D", min_confidence=1)
                ts = mm.io.loadtxt(tp, fmt="mot15-2D")
                acc = mm.utils.compare_to_groundtruth(gt, ts, "iou", distth=0.5)
                ev = acc.mot_events.reset_index()
                for r in ev[ev.Type.isin(["MATCH", "SWITCH"])].itertuples():
                    matched_t.add((int(r.FrameId), cls, r.HId))
                    matched_g.add((int(r.FrameId), cls, r.OId))
            # every tracker row that never matched, in its own class, is an FP
            gtf = {f: g for f, g in gt_all.groupby("frame")}
            tlen = trk_all.groupby("id").size()
            for r in trk_all.itertuples():
                if (r.frame, r.cls, r.id) in matched_t:
                    continue
                g = gtf.get(r.frame)
                colo, colo_cls, best = False, -1, 0.0
                if g is not None and len(g):
                    un = g[[(r.frame, int(c), i) not in matched_g
                            for c, i in zip(g.cls, g.id)]]
                    if len(un):
                        B = un[["x", "y", "w", "h"]].to_numpy()
                        ious = iou_xywh((r.x, r.y, r.w, r.h), B)
                        j = int(np.argmax(ious))
                        best = float(ious[j])
                        if best >= 0.5:
                            colo, colo_cls = True, int(un.cls.iloc[j])
                rows.append(dict(seq=seq, frame=r.frame, tid=r.id, cls=r.cls,
                                 area=r.w * r.h, tracklen=int(tlen.get(r.id, 0)),
                                 colocated=colo, colo_cls=colo_cls, best_iou_unmatched=best))
    d = pd.DataFrame(rows)
    d.to_csv(osp.join(OUT, "_r5_fp_partition.csv"), index=False)
    n = len(d)
    print("\ntotal FP: %d" % n)
    print("  co-located with an unmatched GT box (IoU>=0.5): %5d (%.1f%%)  <- fixable at zero recall cost"
          % (d.colocated.sum(), 100 * d.colocated.mean()))
    print("  isolated (no unmatched GT overlaps it):         %5d (%.1f%%)  <- hallucination on background"
          % ((~d.colocated).sum(), 100 * (~d.colocated).mean()))
    print("\nof the co-located ones, is it a CLASS-LABEL error? (predicted cls vs the GT it covers)")
    co = d[d.colocated]
    if len(co):
        lab = (co.cls != co.colo_cls)
        print("  wrong label on a real object: %d (%.1f%% of co-located, %.1f%% of ALL FP)"
              % (lab.sum(), 100 * lab.mean(), 100 * lab.sum() / n))
        print("  same label, IoU/duplicate issue: %d" % (~lab).sum())
        print("\n  confusion of the mislabelled ones (predicted -> actual):")
        cm = co[lab].groupby([co[lab].cls.map(NAMES), co[lab].colo_cls.map(NAMES)]).size()
        print(cm.sort_values(ascending=False).head(10).to_string())
    print("\nFP by predicted class:")
    print(d.groupby(d.cls.map(NAMES)).agg(n=("cls", "size"),
                                          colocated=("colocated", "sum"),
                                          pct_colo=("colocated", lambda s: round(100 * s.mean(), 1))
                                          ).sort_values("n", ascending=False).to_string())
    print("\nFP mass by owning-track length (the mandatory GIAOTracker control):")
    d["lenq"] = pd.cut(d.tracklen, [0, 5, 15, 40, 100, 10 ** 6],
                       labels=["1-5", "6-15", "16-40", "41-100", "100+"])
    print(d.groupby("lenq", observed=True).agg(FP=("tid", "size"),
                                               pct_of_FP=("tid", lambda s: round(100 * len(s) / n, 1)),
                                               pct_colocated=("colocated", lambda s: round(100 * s.mean(), 1))
                                               ).to_string())
    return d


# ======================================================================================
def d2_embedding_separability():
    print("\n" + "=" * 78)
    print("D2 — EMBEDDING SEPARABILITY PER CLASS: same-ID vs hardest-impostor distance")
    print("=" * 78)
    m = pd.read_csv(osp.join(OUT, "_verify_out_E_margins.csv"))
    m["cname"] = m.cls.map(NAMES)

    def auc_from(pos, neg):
        pos = np.asarray(pos, float); neg = np.asarray(neg, float)
        pos = pos[~np.isnan(pos)]; neg = neg[~np.isnan(neg)]
        if len(pos) < 5 or len(neg) < 5:
            return np.nan
        r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
        n1, n0 = len(pos), len(neg)
        return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

    print("\nAUC separating `d_second` (different ID, nearest impostor) from `resid` (same ID).")
    print("1.00 = the embedder always puts the right object closer. 0.50 = no information.\n")
    rows = []
    for c, sub in m.groupby("cname"):
        rows.append({"class": c, "n": len(sub),
                     "same-ID med": round(sub.resid.median(), 4),
                     "impostor med": round(sub.d_second.median(), 4),
                     "AUC": round(auc_from(sub.d_second, sub.resid), 4)})
    t = pd.DataFrame(rows).sort_values("n", ascending=False)
    print(t.to_string(index=False))
    print("\nsame, within area quintile (does it collapse for small objects?):")
    m["aq"] = pd.qcut(m.groupby("cname").resid.rank(method="first"), 5,
                      labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    # area is not in this file; use the joined feature file if present
    fp = osp.join(OUT, "_r21_features.csv")
    if osp.exists(fp):
        f = pd.read_csv(fp)[["seq", "track_id", "frame", "area"]]
        m2 = m.merge(f, left_on=["seq", "tid", "frame"], right_on=["seq", "track_id", "frame"],
                     how="inner")
        m2["aq"] = pd.qcut(m2.area.rank(method="first"), 5,
                           labels=["Q1 small", "Q2", "Q3", "Q4", "Q5 large"])
        out = []
        for c in ["pedestrian", "car", "van"]:
            s = m2[m2.cname == c]
            row = {"class": c}
            for q, g in s.groupby("aq", observed=True):
                row[str(q)] = round(auc_from(g.d_second, g.resid), 4)
            out.append(row)
        print(pd.DataFrame(out).to_string(index=False))
    m.to_csv(osp.join(OUT, "_r5_embed_separability.csv"), index=False)


# ======================================================================================
def d3_density_prep():
    print("\n" + "=" * 78)
    print("D3 — CROWD-DENSITY PREP (design + metrics; the CMC fit stays behind SAC)")
    print("=" * 78)
    rows = []
    for seq in SEQS:
        gt = load_mot(MC_GT.format(seq=seq))
        gt["cx"] = gt.x + gt.w / 2
        gt["cy"] = gt.y + gt.h / 2
        for cls, g in gt.groupby("cls"):
            nn, h2w = [], []
            for fr, gf in g.groupby("frame"):
                if len(gf) < 2:
                    continue
                P = gf[["cx", "cy"]].to_numpy()
                D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
                np.fill_diagonal(D, np.inf)
                nn.extend((D.min(axis=1) / np.maximum(gf.w.to_numpy(), 1)).tolist())
            h2w.extend((g.h.to_numpy() ** 2 / np.maximum(g.w.to_numpy(), 1)).tolist())
            if nn:
                rows.append(dict(seq=seq, cls=NAMES.get(cls, cls), n_boxes=len(g),
                                 nn_over_w_med=float(np.median(nn)),
                                 h2w_med=float(np.median(h2w))))
    d = pd.DataFrame(rows)
    d.to_csv(osp.join(OUT, "_r5_density.csv"), index=False)
    print("\nnearest-neighbour centre distance / box width (lower = denser => more amplification)")
    piv = d.pivot(index="seq", columns="cls", values="nn_over_w_med").round(2)
    print(piv.to_string())
    print("\npedestrian-only, with the CMC width-corruption driver h^2/w:")
    p = d[d.cls == "pedestrian"].sort_values("nn_over_w_med")
    print(p[["seq", "n_boxes", "nn_over_w_med", "h2w_med"]].to_string(index=False))
    print("""
DESIGN (full fit deferred to the SAC-gated corrected re-run):
  unit          = (sequence, class); n = 7 sequences, pedestrian is the class of interest
  response      = net Delta IDSw, CMC-on minus CMC-off, from the corrected multi_gmc
  predictors    = |theta|_med * h2w_med   (the measured per-frame width corruption)
                  1 / nn_over_w_med       (crowding: how close the nearest competitor is)
  control arms  = (i) the SAME regression on the BUGGY warp, which should show a much
                      steeper corruption slope; (ii) CMC-off IDSw as an offset, so the fit
                      explains the DELTA rather than the level; (iii) box area, already
                      known to point the wrong way (AUC 0.451) and therefore a placebo.
  pre-registered kill: if the corruption term carries no weight once crowding is in the
  model, the width-corruption story is an amplification artifact of dense scenes and the
  CMC finding is demoted to 'one crowded sequence', not a mechanism.
  CAVEAT that cannot be fixed: n = 7 with one extreme. This can corroborate, never prove.""")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    d1_fp_partition()
    d2_embedding_separability()
    d3_density_prep()
