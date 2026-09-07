"""ATTACK A, part 2 -- the decisive version.

A1 showed flagged frames sit at ~10x the typical global acceleration. Two readings:
  (i) real: abrupt camera motion genuinely perturbs appearance (blur / viewpoint) -- but then the
      event is CAMERA MOTION, which cheap geometry (and the tracker's own CMC) already sees;
  (ii) artifact: the replay crops the KF POSTERIOR, which lags during abrupt motion for EVERY
      track at once, so the crops are jointly misaligned -> manufactured common mode.
Either way the "independent perturbation" defence fails. This script asks the quantitative
question: how much of the synchrony is left once per-track motion is controlled for?
"""
import numpy as np
import pandas as pd
from _verify_common import *

RNG = np.random.default_rng(0)
NPERM = 200
MINTRK = 5
WIN = 31


def per_track_motion(seq):
    b = load_boxes(seq)
    b['cx'] = b['x'] + b['w'] / 2.0
    b['cy'] = b['y'] + b['h'] / 2.0
    b['ar'] = b['w'] * b['h']
    b = b.sort_values(['track_id', 'frame'])
    g = b.groupby('track_id')
    b['pf'] = g['frame'].shift(1)
    b['pcx'] = g['cx'].shift(1)
    b['pcy'] = g['cy'].shift(1)
    b['par'] = g['ar'].shift(1)
    ok = (b['frame'] - b['pf']) == 1
    dx = np.where(ok, b['cx'] - b['pcx'], np.nan)
    dy = np.where(ok, b['cy'] - b['pcy'], np.nan)
    sz = np.sqrt(np.maximum(b['ar'].to_numpy(), 1.0))
    b['dsp'] = np.hypot(dx, dy)
    b['dsp_rel'] = b['dsp'] / sz
    b['dla'] = np.where(ok, np.log(np.maximum(b['ar'], 1) / np.maximum(b['par'], 1)), np.nan)
    tmp = b.assign(dx=dx, dy=dy)
    gdx = tmp.groupby('frame')['dx'].median()
    gdy = tmp.groupby('frame')['dy'].median()
    b['ndsp_rel'] = np.hypot(dx - b['frame'].map(gdx), dy - b['frame'].map(gdy)) / sz
    return b[['frame', 'track_id', 'dsp', 'dsp_rel', 'dla', 'ndsp_rel', 'ar']]


def screen(df, valcol, min_tracks=MINTRK, nperm=NPERM, win=WIN):
    d = df.dropna(subset=[valcol])
    f0 = int(d['frame'].min())
    n_frames = int(d['frame'].max()) - f0 + 1
    fi = d['frame'].to_numpy() - f0
    v = d[valcol].to_numpy(float)
    t = d['track_id'].to_numpy()

    def med_series(fi_, v_):
        o = np.argsort(fi_, kind='stable')
        f_, x_ = fi_[o], v_[o]
        bnd = np.searchsorted(f_, np.arange(n_frames + 1))
        out = np.full(n_frames, np.nan)
        for k in range(n_frames):
            lo, hi = bnd[k], bnd[k + 1]
            if hi - lo >= min_tracks:
                out[k] = np.median(x_[lo:hi])
        return out

    def stat(m):
        okm = ~np.isnan(m)
        if okm.sum() < 10:
            return np.nan, 0
        x = m.copy()
        x[~okm] = np.nanmedian(m)
        base, sc = rolling_baseline(x, win)
        z = (x - base) / sc
        dd = (x - base)[okm]
        return 1.4826 * np.median(np.abs(dd - np.median(dd))), int((np.abs(z[okm]) > 3).sum())

    obs_sd, obs_nz = stat(med_series(fi, v))
    o = np.argsort(t, kind='stable')
    ts, fs, vs = t[o], fi[o], v[o]
    uq, st = np.unique(ts, return_index=True)
    en = np.append(st[1:], len(ts))
    nulls = []
    for _ in range(nperm):
        vv = vs.copy()
        for lo, hi in zip(st, en):
            n = hi - lo
            if n >= 2:
                vv[lo:hi] = np.roll(vs[lo:hi], int(RNG.integers(1, n)))
        nulls.append(stat(med_series(fs, vv))[0])
    nulls = np.array(nulls, float)
    nm = np.nanmedian(nulls)
    return dict(obs_sd=obs_sd, null_sd=nm, ratio=obs_sd / nm if nm > 0 else np.nan,
                p=(np.sum(nulls >= obs_sd) + 1) / (len(nulls) + 1), nz3=obs_nz)


def bin_residualize(df, cols, nbin=10):
    """Subtract the median residual of the joint covariate bin (non-parametric control)."""
    key = None
    for c in cols:
        v = df[c].to_numpy(float)
        r = pd.Series(v).rank(pct=True, na_option='keep').to_numpy()
        b = np.floor(np.clip(r, 0, 0.9999) * nbin)
        b = np.where(np.isnan(v), -1.0, b)
        key = b if key is None else key * (nbin + 1) + (b + 1)
    out = df.assign(_k=key)
    med = out.groupby('_k')['resid'].transform('median')
    return (out['resid'] - med).to_numpy()


print('=' * 122)
print('STEP 1 -- permutation test: do flagged frames really sit at high global acceleration?')
print('=' * 122)
print('%-22s %5s %14s %14s %8s %8s %10s' %
      ('sequence', 'nA', 'accel@flag', 'accel@all', 'lift', 'p(perm)', 'pctile'))
step1 = []
for seq in SEQS:
    df = load_resid(seq)
    grid, med = frame_median(df, MINTRK)
    z = zseries(med, WIN)
    G = geometry_series(seq, grid)
    acc = G['glob_accel'].to_numpy()
    ok = (~np.isnan(z)) & (~np.isnan(acc))
    A = ok & (np.abs(z) > 3)
    if A.sum() < 3:
        continue
    obs = np.median(acc[A])
    allm = np.median(acc[ok])
    pool = acc[ok]
    k = int(A.sum())
    draws = np.array([np.median(RNG.choice(pool, k, replace=False)) for _ in range(2000)])
    p = (np.sum(draws >= obs) + 1) / 2001
    pct = 100 * np.mean(pool < obs)
    print('%-22s %5d %14.4f %14.4f %8.2f %8.4f %9.1f%%' % (seq, k, obs, allm, obs / allm, p, pct))
    step1.append(dict(seq=seq, nA=k, obs=obs, allm=allm, p=p, pct=pct))
s1 = pd.DataFrame(step1)
print('  --> sequences with p<0.05: %d/%d' % (int((s1.p < 0.05).sum()), len(s1)))

print('')
print('=' * 122)
print('STEP 2 -- synchrony screen BEFORE vs AFTER controlling per-track motion / size')
print('=' * 122)
print('%-22s %27s %27s %27s' % ('', '--- raw resid ---', '--- motion-controlled ---',
                                '--- +size-controlled ---'))
print('%-22s %9s %8s %8s %9s %8s %8s %9s %8s %8s' %
      ('sequence', 'ratio', 'p', 'n|z|>3', 'ratio', 'p', 'n|z|>3', 'ratio', 'p', 'n|z|>3'))
out = []
for seq in SEQS:
    df = load_resid(seq)
    M = per_track_motion(seq)
    d = df.merge(M, on=['frame', 'track_id'], how='left')
    r0 = screen(d, 'resid')
    d['r_mot'] = bin_residualize(d, ['dsp_rel'])
    r1 = screen(d, 'r_mot')
    d['r_full'] = bin_residualize(d, ['dsp_rel', 'area'])
    r2 = screen(d, 'r_full')
    print('%-22s %9.2f %8.3f %8d %9.2f %8.3f %8d %9.2f %8.3f %8d' %
          (seq, r0['ratio'], r0['p'], r0['nz3'], r1['ratio'], r1['p'], r1['nz3'],
           r2['ratio'], r2['p'], r2['nz3']))
    out.append(dict(seq=seq, raw_ratio=r0['ratio'], raw_p=r0['p'], raw_nz=r0['nz3'],
                    mot_ratio=r1['ratio'], mot_p=r1['p'], mot_nz=r1['nz3'],
                    full_ratio=r2['ratio'], full_p=r2['p'], full_nz=r2['nz3']))
O = pd.DataFrame(out)
print('%-22s %9.2f %8s %8d %9.2f %8s %8d %9.2f %8s %8d' %
      ('MEDIAN/SUM', O.raw_ratio.median(), '', O.raw_nz.sum(), O.mot_ratio.median(), '',
       O.mot_nz.sum(), O.full_ratio.median(), '', O.full_nz.sum()))
O.to_csv('_scratch/_verify_out_A2.csv', index=False)
print('')
print('wrote _verify_out_A2.csv')
