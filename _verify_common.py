"""Shared loaders for the adversarial verification of the frame-level appearance-shift gate."""
import os, os.path as osp
import numpy as np, pandas as pd

LOGDIR = '_scratch/_appres_logs_replay'
TRKDIR = osp.join('YOLOX_outputs', 'mc_dare_cv_rerun0803', 'track_results')
SEQS = sorted(f[:-4] for f in os.listdir(LOGDIR) if f.endswith('.csv'))


def rolling_baseline(x, win=31):
    s = pd.Series(x, dtype=float)
    if len(s) < win:
        med = pd.Series(np.full(len(s), np.nanmedian(x)))
        mad = np.nanmedian(np.abs(x - np.nanmedian(x)))
        scale = pd.Series(np.full(len(s), mad))
    else:
        med = s.rolling(win, center=True, min_periods=win // 3).median().bfill().ffill()
        dev = (s - med).abs()
        scale = dev.rolling(win, center=True, min_periods=win // 3).median().bfill().ffill()
    scale = scale.replace(0.0, np.nan)
    gl = np.nanmedian(np.abs(x - np.nanmedian(x)))
    scale = scale.fillna(gl if gl > 0 else 1e-6)
    return med.to_numpy(), scale.to_numpy() * 1.4826


def zseries(x, win=31):
    """z of a series that may contain NaN; NaN slots are filled with the global median
    for the purpose of detrending (exactly what excursion_stats does), then re-NaN'd."""
    x = np.asarray(x, dtype=float)
    ok = ~np.isnan(x)
    if ok.sum() < 10:
        return np.full(len(x), np.nan)
    xf = x.copy(); xf[~ok] = np.nanmedian(x)
    base, scale = rolling_baseline(xf, win)
    z = (xf - base) / scale
    z[~ok] = np.nan
    return z


def robust_sd(x):
    d = x[~np.isnan(x)]
    if len(d) < 5:
        return np.nan
    return 1.4826 * np.median(np.abs(d - np.median(d)))


def load_resid(seq):
    df = pd.read_csv(osp.join(LOGDIR, seq + '.csv'))
    return df[df['valid'] == 1].copy()


def frame_median(df, min_tracks=5, col='resid'):
    """Return (frames_grid, median_series) on the full contiguous frame grid."""
    g = df.groupby('frame')[col]
    med = g.median(); n = df.groupby('frame').size()
    f0, f1 = int(df['frame'].min()), int(df['frame'].max())
    grid = np.arange(f0, f1 + 1)
    out = np.full(len(grid), np.nan)
    idx = med.index.to_numpy() - f0
    vals = med.to_numpy(); nn = n.to_numpy()
    keep = nn >= min_tracks
    out[idx[keep]] = vals[keep]
    return grid, out


def load_boxes(seq):
    raw = np.loadtxt(osp.join(TRKDIR, seq + '.txt'), delimiter=',', ndmin=2)
    return pd.DataFrame({
        'frame': raw[:, 0].astype(int), 'track_id': raw[:, 1].astype(int),
        'x': raw[:, 2], 'y': raw[:, 3], 'w': raw[:, 4], 'h': raw[:, 5],
        'score': raw[:, 6], 'cls': raw[:, 7].astype(int)})


def geometry_series(seq, grid):
    """Per-frame geometry channels computed from the CACHED (KF-posterior) boxes, over the
    tracks present in both frame t-1 and t. Aligned onto `grid`."""
    b = load_boxes(seq)
    b['cx'] = b['x'] + b['w'] / 2.0
    b['cy'] = b['y'] + b['h'] / 2.0
    b['ar'] = b['w'] * b['h']
    prev = {int(f): g for f, g in b.groupby('frame')}
    keys = ['glob_dx', 'glob_dy', 'glob_speed', 'med_disp', 'med_disp_rel',
            'med_dlogarea', 'med_abs_dlogarea', 'resid_motion', 'n_common', 'mean_area']
    out = {k: np.full(len(grid), np.nan) for k in keys}
    for i, f in enumerate(grid):
        a, c = prev.get(int(f) - 1), prev.get(int(f))
        if c is not None:
            out['mean_area'][i] = c['ar'].mean()
        if a is None or c is None:
            continue
        m = a.merge(c, on='track_id', suffixes=('_p', '_c'))
        if len(m) < 3:
            continue
        dx = (m['cx_c'] - m['cx_p']).to_numpy()
        dy = (m['cy_c'] - m['cy_p']).to_numpy()
        gdx, gdy = np.median(dx), np.median(dy)
        sz = np.sqrt(m['ar_c'].to_numpy())
        dla = np.log(np.maximum(m['ar_c'].to_numpy(), 1.0) / np.maximum(m['ar_p'].to_numpy(), 1.0))
        out['glob_dx'][i] = gdx; out['glob_dy'][i] = gdy
        out['glob_speed'][i] = np.hypot(gdx, gdy)
        out['med_disp'][i] = np.median(np.hypot(dx, dy))
        out['med_disp_rel'][i] = np.median(np.hypot(dx, dy) / np.maximum(sz, 1.0))
        out['med_dlogarea'][i] = np.median(dla)
        out['med_abs_dlogarea'][i] = np.median(np.abs(dla))
        out['resid_motion'][i] = np.median(np.hypot(dx - gdx, dy - gdy) / np.maximum(sz, 1.0))
        out['n_common'][i] = len(m)
    # global acceleration = second difference of the global velocity (what makes a KF posterior lag)
    gv = np.stack([out['glob_dx'], out['glob_dy']])
    acc = np.full(len(grid), np.nan)
    acc[1:] = np.hypot(gv[0, 1:] - gv[0, :-1], gv[1, 1:] - gv[1, :-1])
    out['glob_accel'] = acc
    return pd.DataFrame(out, index=grid)
