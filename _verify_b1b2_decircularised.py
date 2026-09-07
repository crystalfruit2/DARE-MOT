"""De-circularising the candidate's own TEST B.

B1 as run is circular: frames are flagged BECAUSE their median residual is high, so of course
residuals are high there. The unbiased version is SPLIT-HALF: flag frames using a random half of
the tracks, then measure the excursion on the HELD-OUT half. If the common mode is real, the
held-out half must move at the same frames. This also gives an honest magnitude, free of the
selection bias that inflates the +0.0153 figure.

B2 (birth lift) as run has no null at all. Two nulls here:
  * plain    -- relocate the flagged frames uniformly at random (same count)
  * matched  -- relocate them onto frames with the SAME global-acceleration profile. Because the
                flagged frames sit at the 92-96th accel percentile (attack A), this asks whether
                births are caused by the appearance event or merely by the camera motion that
                the appearance flag is standing in for.
"""
import numpy as np
import pandas as pd
from _verify_common import SEQS, load_resid, frame_median, zseries, geometry_series

RNG = np.random.default_rng(11)
NPERM = 2000

print('=' * 118)
print('B1 DE-CIRCULARISED -- split-half: flag on half the tracks, measure on the other half')
print('   (100 random splits; excursion = held-out median resid at flagged minus at unflagged)')
print('=' * 118)
print('%-22s %8s %14s %16s %16s %10s' %
      ('sequence', 'nA(half)', 'in-sample dm', 'HELD-OUT dm', 'held-out/insample', 'p(split)'))
rows = []
for seq in SEQS:
    d = load_resid(seq)
    tids = d['track_id'].unique()
    ins, hel, ps = [], [], []
    for _ in range(100):
        perm = RNG.permutation(tids)
        h1 = set(perm[:len(perm) // 2].tolist())
        A_ = d[d['track_id'].isin(h1)]
        B_ = d[~d['track_id'].isin(h1)]
        if len(A_) < 500 or len(B_) < 500:
            continue
        g, m = frame_median(A_, 3)
        z = zseries(m, 31)
        fl = set(g[(~np.isnan(z)) & (np.abs(z) > 3)].tolist())
        if len(fl) < 3:
            continue
        for src, acc in ((A_, ins), (B_, hel)):
            gg = src.groupby('frame')['resid'].median()
            n = src.groupby('frame').size()
            ok = n[n >= 3].index
            isf = np.array([f in fl for f in ok])
            if isf.sum() < 2 or (~isf).sum() < 10:
                acc.append(np.nan)
                continue
            acc.append(float(gg[ok][isf].median() - gg[ok][~isf].median()))
        ps.append(len(fl))
    ins, hel = np.array(ins, float), np.array(hel, float)
    ok = ~np.isnan(ins) & ~np.isnan(hel)
    if ok.sum() < 10:
        print('%-22s  too few valid splits' % seq)
        continue
    mi, mh = np.median(ins[ok]), np.median(hel[ok])
    p = float(np.mean(hel[ok] <= 0))
    print('%-22s %8.1f %14.4f %16.4f %16.2f %10.3f' %
          (seq, np.median(ps), mi, mh, mh / mi if mi else np.nan, p))
    rows.append(dict(seq=seq, ins=mi, held=mh, p=p))
R = pd.DataFrame(rows)
print('-' * 118)
print('%-22s %8s %14.4f %16.4f %16.2f' %
      ('MEDIAN', '', R.ins.median(), R.held.median(), (R.held / R.ins).median()))
print('')
print('READ: held-out dm close to in-sample dm => the common mode is real in the replayed data')
print('      (says nothing yet about whether it is the tracker or the replay producing it).')
print('      held-out dm near 0 => B1 was pure selection bias.')

print('')
print('=' * 118)
print('B2 WITH A NULL -- track-birth lift at flagged frames +-2')
print('=' * 118)
print('%-22s %6s %10s %12s %14s %12s %14s' %
      ('sequence', 'nA', 'obs lift', 'p(uniform)', 'lift@uniform', 'p(accel-matched)',
       'lift@matched'))
out = []
for seq in SEQS:
    d = load_resid(seq)
    grid, med = frame_median(d, 5)
    z = zseries(med, 31)
    G = geometry_series(seq, grid)
    acc = G['glob_accel'].to_numpy()
    ok = (~np.isnan(z))
    frames = grid[ok]
    A = np.abs(z[ok]) > 3
    if A.sum() < 3:
        continue
    first = d.groupby('track_id')['frame'].min()
    births = pd.Series(0, index=frames, dtype=float)
    for fr, c in first.value_counts().items():
        if fr in births.index:
            births[fr] = c
    bv = births.to_numpy()
    accf = acc[ok]

    def lift(mask_frames):
        w = np.zeros(len(frames), bool)
        idx = {f: i for i, f in enumerate(frames)}
        for f in mask_frames:
            for o in (-2, -1, 0, 1, 2):
                j = idx.get(f + o)
                if j is not None:
                    w[j] = True
        if w.sum() == 0 or (~w).sum() == 0:
            return np.nan
        b = bv[~w].mean()
        return bv[w].mean() / b if b > 0 else np.nan

    obs = lift(frames[A])
    k = int(A.sum())
    nullU, nullM = [], []
    # accel-matched sampling: draw from the same accel deciles as the flagged frames
    dec = pd.qcut(pd.Series(accf).rank(method='first'), 10, labels=False).to_numpy()
    want = pd.Series(dec[A]).value_counts().to_dict()
    pools = {q: np.where(dec == q)[0] for q in range(10)}
    for _ in range(NPERM):
        nullU.append(lift(RNG.choice(frames, k, replace=False)))
        pick = []
        for q, c in want.items():
            pool = pools[q]
            pick.extend(RNG.choice(pool, min(c, len(pool)), replace=False).tolist())
        nullM.append(lift(frames[np.array(pick)]))
    nu = np.array(nullU, float)
    nm = np.array(nullM, float)
    pu = (np.nansum(nu >= obs) + 1) / (np.sum(~np.isnan(nu)) + 1)
    pm = (np.nansum(nm >= obs) + 1) / (np.sum(~np.isnan(nm)) + 1)
    print('%-22s %6d %10.2fx %12.3f %14.2fx %12.3f %14.2fx' %
          (seq, k, obs, pu, np.nanmedian(nu), pm, np.nanmedian(nm)))
    out.append(dict(seq=seq, obs=obs, pu=pu, pm=pm))
O = pd.DataFrame(out)
print('-' * 118)
print('sequences with p(uniform)<0.05: %d/%d    p(accel-matched)<0.05: %d/%d' %
      ((O.pu < 0.05).sum(), len(O), (O.pm < 0.05).sum(), len(O)))
