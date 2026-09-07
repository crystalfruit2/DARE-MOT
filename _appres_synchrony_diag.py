"""Round-3 candidate #1 -- cheap offline diagnostic: is there a COMMON-MODE appearance shift?

The candidate assumes that a drone zoom / altitude / gimbal event shifts ground-sample-distance
for the whole scene at once, so the appearance-cost residuals of many simultaneously-tracked
objects spike *together* in the same frame -- a signal no per-class / per-instance / per-track
gate can see, because none of them aggregate across tracks within a frame.

The null hypothesis that has to be beaten is NOT "residuals are flat". It is:

    residuals move, but each track moves on its own schedule.

Under that null the cross-track median of a frame is an average of many independent series and
is therefore much steadier than any individual track. So the test is a comparison of the OBSERVED
frame-median series against a null that keeps every single-track property intact and destroys
only the cross-track alignment:

  * shift null (decisive) -- circularly shift each track's residual series by a random lag within
    its own frames. Each track keeps its marginal distribution AND its temporal autocorrelation
    (slow individual drift, its own bursts); only the alignment between tracks is broken. The
    number of tracks contributing to each frame is preserved exactly.
  * shuffle null (secondary) -- permute each track's values across its own frames. Also destroys
    per-track autocorrelation, so it is a weaker/looser null; reported for contrast. If observed
    beats shuffle but not shift, the "synchrony" was just individual drift.

Statistics on the detrended frame-median series (baseline = centred rolling median, scale = MAD):
  sd      -- robust spread of the detrended median series
  n_z3    -- frames with |z| > 3
  max_z   -- largest excursion
  n_major -- frames that are BOTH a z>3 median excursion AND shared by a majority of tracks
             (> 60% of that frame's tracks above their own per-track median). This is the
             candidate's actual claim; a spike carried by 2 tracks is not a scene-wide event.

Every statistic is reported per sequence, never pooled -- this project has killed two candidates
that were one sequence wearing a pooled number as a disguise.

Usage:  python _appres_synchrony_diag.py [log_dir] [--nperm 200] [--min-tracks 5]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

# 1-BASED COCO category ids, as written into track_results and val7_mc.json --
# NOT the 0-based YOLOX head id the byte_tracker docstrings refer to. There is no
# cls=0 anywhere in the cached output; an earlier 0-based map here was wrong.
CLS_NAMES = {1: 'pedestrian', 2: 'car', 3: 'van', 4: 'truck', 5: 'bus'}


def rolling_baseline(x, win=31):
    """Centred rolling median and MAD-based scale. Short series fall back to global."""
    s = pd.Series(x)
    if len(s) < win:
        med = pd.Series(np.full(len(s), np.nanmedian(x)))
        mad = np.nanmedian(np.abs(x - np.nanmedian(x)))
        scale = pd.Series(np.full(len(s), mad))
    else:
        med = s.rolling(win, center=True, min_periods=win // 3).median()
        med = med.bfill().ffill()
        dev = (s - med).abs()
        scale = dev.rolling(win, center=True, min_periods=win // 3).median()
        scale = scale.bfill().ffill()
    scale = scale.replace(0.0, np.nan)
    gl = np.nanmedian(np.abs(x - np.nanmedian(x)))
    scale = scale.fillna(gl if gl > 0 else 1e-6)
    return med.to_numpy(), (scale.to_numpy() * 1.4826)


def median_series(frame_idx, vals, n_frames, min_tracks):
    """Cross-track median per frame; NaN where too few tracks to be meaningful."""
    order = np.argsort(frame_idx, kind='stable')
    fi, vv = frame_idx[order], vals[order]
    bounds = np.searchsorted(fi, np.arange(n_frames + 1))
    out = np.full(n_frames, np.nan)
    for f in range(n_frames):
        lo, hi = bounds[f], bounds[f + 1]
        if hi - lo >= min_tracks:
            out[f] = np.median(vv[lo:hi])
    return out


def excursion_stats(med, win=31):
    """Robust spread + excursion counts of the detrended cross-track median series."""
    ok = ~np.isnan(med)
    if ok.sum() < 10:
        return dict(sd=np.nan, n_z3=0, max_z=np.nan, z=np.full(len(med), np.nan))
    x = med.copy()
    x[~ok] = np.nanmedian(med)
    base, scale = rolling_baseline(x, win)
    z = (x - base) / scale
    z[~ok] = np.nan
    d = (x - base)[ok]
    sd = 1.4826 * np.median(np.abs(d - np.median(d)))
    za = np.abs(z[ok])
    return dict(sd=float(sd), n_z3=int((za > 3).sum()), max_z=float(za.max()), z=z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log_dir', nargs='?', default='_scratch/_appres_logs')
    ap.add_argument('--nperm', type=int, default=200)
    ap.add_argument('--min-tracks', type=int, default=5)
    ap.add_argument('--win', type=int, default=31)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='_appres_diag_out')
    args = ap.parse_args()

    if not os.path.isdir(args.log_dir):
        sys.exit('no such log dir: %s' % args.log_dir)
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    files = sorted(f for f in os.listdir(args.log_dir) if f.endswith('.csv'))
    if not files:
        sys.exit('no CSVs in %s' % args.log_dir)

    print('=' * 108)
    print('ROUND-3 #1 DIAGNOSTIC -- cross-track appearance-residual synchrony')
    print('log dir: %s   |   %d sequences   |   nperm=%d   min-tracks/frame=%d   win=%d'
          % (args.log_dir, len(files), args.nperm, args.min_tracks, args.win))
    print('=' * 108)

    summary = []
    spike_rows = []

    for fn in files:
        df = pd.read_csv(os.path.join(args.log_dir, fn))
        seq = fn[:-4]
        tot = len(df)
        df = df[df['valid'] == 1]
        if df.empty:
            print('\n%-22s NO VALID RESIDUALS (%d rows, all invalid)' % (seq, tot))
            continue

        frames = df['frame'].to_numpy()
        f0 = frames.min()
        fi = frames - f0
        n_frames = int(fi.max()) + 1
        vals = df['resid'].to_numpy(dtype=np.float64)
        tids = df['track_id'].to_numpy()

        obs_med = median_series(fi, vals, n_frames, args.min_tracks)
        obs = excursion_stats(obs_med, args.win)
        n_used = int((~np.isnan(obs_med)).sum())

        # ---- per-track structures for the nulls (frame sets preserved exactly) ----
        order = np.argsort(tids, kind='stable')
        t_sorted, fi_s, v_s = tids[order], fi[order], vals[order]
        uniq, starts = np.unique(t_sorted, return_index=True)
        ends = np.append(starts[1:], len(t_sorted))

        null = {'shift': [], 'shuffle': []}
        for _ in range(args.nperm):
            v_shift = v_s.copy()
            v_shuf = v_s.copy()
            for lo, hi in zip(starts, ends):
                n = hi - lo
                if n < 2:
                    continue
                v_shift[lo:hi] = np.roll(v_s[lo:hi], int(rng.integers(1, n)))
                v_shuf[lo:hi] = rng.permutation(v_s[lo:hi])
            for key, vv in (('shift', v_shift), ('shuffle', v_shuf)):
                m = median_series(fi_s, vv, n_frames, args.min_tracks)
                st = excursion_stats(m, args.win)
                null[key].append((st['sd'], st['n_z3'], st['max_z']))

        res = {'seq': seq, 'rows': tot, 'valid': len(df), 'frames_used': n_used,
               'median_tracks_per_frame': float(df.groupby('frame').size().median()),
               'obs_sd': obs['sd'], 'obs_nz3': obs['n_z3'], 'obs_maxz': obs['max_z']}

        for key in ('shift', 'shuffle'):
            arr = np.array(null[key], dtype=np.float64)
            sd_n, nz_n = arr[:, 0], arr[:, 1]
            res['%s_sd_med' % key] = float(np.nanmedian(sd_n))
            res['%s_sd_p95' % key] = float(np.nanpercentile(sd_n, 95))
            res['%s_sd_ratio' % key] = float(obs['sd'] / np.nanmedian(sd_n)) if np.nanmedian(sd_n) > 0 else np.nan
            res['%s_sd_p' % key] = float((np.sum(sd_n >= obs['sd']) + 1) / (len(sd_n) + 1))
            res['%s_nz3_med' % key] = float(np.nanmedian(nz_n))
            res['%s_nz3_p' % key] = float((np.sum(nz_n >= obs['n_z3']) + 1) / (len(nz_n) + 1))

        # ---- shared-majority test on the observed excursions ----
        z = obs['z']
        cand = np.where(~np.isnan(z) & (np.abs(z) > 3))[0]
        n_major = 0
        if len(cand):
            tmed = df.groupby('track_id')['resid'].median()
            df = df.assign(_above=(df['resid'].to_numpy()
                                   > df['track_id'].map(tmed).to_numpy()).astype(float))
            frac = df.groupby('frame')['_above'].mean()
            cnt = df.groupby('frame').size()
            for c in cand:
                fr = int(c + f0)
                if fr in frac.index and frac[fr] > 0.60 and z[c] > 0:
                    n_major += 1
                    spike_rows.append(dict(seq=seq, frame=fr, z=float(z[c]),
                                           frac_above=float(frac[fr]),
                                           n_tracks=int(cnt[fr]),
                                           median_resid=float(obs_med[c])))
        res['n_major'] = n_major
        summary.append(res)

        print('\n%-22s valid %6d / %6d rows | %4d usable frames | median %4.1f tracks/frame'
              % (seq, len(df), tot, n_used, res['median_tracks_per_frame']))
        print('   observed   sd=%.4f   n(|z|>3)=%3d   max|z|=%5.2f   shared-majority spikes=%d'
              % (res['obs_sd'], res['obs_nz3'], res['obs_maxz'], n_major))
        for key in ('shift', 'shuffle'):
            print('   null %-7s sd=%.4f (p95 %.4f)  ratio=%5.2fx  p=%.3f | n(|z|>3)=%5.1f  p=%.3f'
                  % (key, res['%s_sd_med' % key], res['%s_sd_p95' % key],
                     res['%s_sd_ratio' % key], res['%s_sd_p' % key],
                     res['%s_nz3_med' % key], res['%s_nz3_p' % key]))

    sm = pd.DataFrame(summary)
    sm.to_csv(os.path.join(args.out, 'synchrony_summary.csv'), index=False)
    sp = pd.DataFrame(spike_rows)
    sp.to_csv(os.path.join(args.out, 'shared_majority_spikes.csv'), index=False)

    print('\n' + '=' * 108)
    print('VERDICT TABLE (decisive null = shift; it preserves each track\'s own drift)')
    print('=' * 108)
    print('%-22s %8s %10s %8s %8s %10s %8s' %
          ('sequence', 'obs_sd', 'shift_sd', 'ratio', 'p(sd)', 'maj.spikes', 'n(|z|>3)'))
    for _, r in sm.iterrows():
        print('%-22s %8.4f %10.4f %8.2f %8.3f %10d %8d' %
              (r['seq'], r['obs_sd'], r['shift_sd_med'], r['shift_sd_ratio'],
               r['shift_sd_p'], r['n_major'], r['obs_nz3']))
    print('\nwrote %s/synchrony_summary.csv and %s/shared_majority_spikes.csv'
          % (args.out, args.out))
    print('\nREAD: a candidate needs BOTH (a) observed spread materially above the shift null')
    print('      (ratio >> 1 with small p) AND (b) shared-majority spikes in MORE THAN ONE')
    print('      sequence. Ratio ~1.0 everywhere means the frame median is just averaging out')
    print('      independent per-track noise exactly as the null predicts -- no common mode,')
    print('      nothing for a frame-level gate to fire on.')


if __name__ == '__main__':
    main()
