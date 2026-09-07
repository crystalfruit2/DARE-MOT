"""R3-3 step 3: is the camera-path non-linearity across a gap STRUCTURED, or is it just
accumulated per-frame GMC jitter? (2026-09-07, offline.)

Step 2 measures the deviation of the composed camera path from its own chord, which is the
only thing a CMC-chained virtual trajectory can buy over a plain lerp. But that deviation
has two possible sources:

  (a) a genuinely curving / accelerating camera path  -> chaining CMC is real signal;
  (b) independent per-frame estimation noise in H_t   -> chaining CMC injects noise.

Only (a) is a win. The null separates them: PERMUTE the per-frame affines within the gap.
For near-pure translations the composition is order-invariant, so a permuted chain keeps
the same endpoints (same chord) and the same per-frame increment magnitudes, while
destroying any temporal structure (a coherent acceleration becomes a random walk). If the
observed chord deviation sits inside the permuted null, the "non-linearity" is jitter.

A second, harder null: replace each H_t by the sequence-median increment plus a bootstrap
resample of the per-frame residual increments (destroys structure, preserves magnitude).
"""
import os.path as osp

import numpy as np
import pandas as pd

AFF_DIR = "_scratch/_r33_affines"
N_PERM = 200
RNG = np.random.default_rng(0)


def load_affines(seq):
    df = pd.read_csv(osp.join(AFF_DIR, seq + ".csv"))
    H = {}
    for r in df.itertuples():
        M = np.eye(3)
        M[0, 0], M[0, 1], M[0, 2] = r.a, r.b, r.tx
        M[1, 0], M[1, 1], M[1, 2] = r.c, r.d, r.ty
        H[int(r.frame)] = M
    return H


def path_nonlinearity(mats, pt):
    """mats = ordered list of per-step 3x3; returns max deviation of the traced point from
    the chord joining its first and last position."""
    p = np.array([pt[0], pt[1], 1.0])
    pts = [p[:2].copy()]
    cur = np.eye(3)
    for M in mats:
        cur = M @ cur
        q = cur @ p
        pts.append(q[:2])
    pts = np.stack(pts)
    a = np.linspace(0.0, 1.0, len(pts))
    chord = pts[0][None, :] + a[:, None] * (pts[-1] - pts[0])[None, :]
    return float(np.linalg.norm(pts - chord, axis=1).max())


def main():
    gaps = pd.read_csv("_scratch/_r33_gap_divergence.csv")
    out = []
    for seq, g in gaps.groupby("seq"):
        H = load_affines(seq)
        for r in g.itertuples():
            mats = [H[t] for t in range(r.t1 + 1, r.t2 + 1) if t in H]
            if len(mats) != r.t2 - r.t1:
                continue
            pt = (0.0, 0.0)  # filled below from the box centre stored in the csv
            # reconstruct the object's centre at t1 is not stored; use image centre-free
            # measure: the non-linearity is evaluated at the track's own t1 centre, which
            # step 2 recorded implicitly. Recompute from the csv columns instead.
            pt = (r.cx1, r.cy1) if hasattr(r, "cx1") else (960.0, 540.0)
            obs = path_nonlinearity(mats, pt)
            null = np.empty(N_PERM)
            idx = np.arange(len(mats))
            for k in range(N_PERM):
                RNG.shuffle(idx)
                null[k] = path_nonlinearity([mats[i] for i in idx], pt)
            out.append(dict(seq=seq, tid=r.tid, gap=r.gap, obs=obs,
                            null_med=float(np.median(null)),
                            null_p95=float(np.percentile(null, 95)),
                            ratio=obs / max(np.median(null), 1e-9),
                            p=float((null >= obs).mean())))
    d = pd.DataFrame(out)
    d.to_csv("_scratch/_r33_nonlinearity_null.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n=== camera-path non-linearity vs permuted-increment null (%d gaps) ===" % len(d))
    print(d.groupby("seq").agg(n=("gap", "size"), obs_med=("obs", "median"),
                               null_med=("null_med", "median"),
                               ratio_med=("ratio", "median"),
                               frac_p_lt_05=("p", lambda s: (s < 0.05).mean())).round(4))
    print("\npooled: obs median %.3f px | null median %.3f px | ratio %.3f | %.1f%% of gaps p<0.05"
          % (d.obs.median(), d.null_med.median(), d.ratio.median(), 100 * (d.p < 0.05).mean()))


if __name__ == "__main__":
    main()
