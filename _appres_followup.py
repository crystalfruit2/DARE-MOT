"""Round-3 #1 follow-up: the two tests that decide whether the synchrony finding is worth a
section, a footnote, or nothing.

The synchrony screen (_appres_synchrony_diag.py) found the cross-track median residual moves
more than a per-track-shift null predicts in 6/7 sequences. That is necessary, not sufficient.
Two things can still kill it, and both are answerable offline from the same CSVs:

TEST A -- HDST-GNN C1 co-ablation (novelty).
    HDST-GNN (arXiv 2606.05587) already pools a per-frame scene-wide statistic on VisDrone to
    detect zoom/altitude: MEAN OBJECT AREA, z = -log(a_bar / a_ref). If that cheaper geometric
    proxy flags the same frames, candidate #1 is C1 with a different sensor bolted on, and a
    reviewer will say so. The appearance channel only earns a section if it fires on frames the
    area channel cannot see. Reported as: overlap of the two flag sets, and the correlation of
    their z-series.

TEST B -- operational magnitude (does it matter at all).
    A statistically detectable common mode can still be far too small to change any association.
    Two measurements:
      B1  the actual residual distribution at flagged vs unflagged frames -- if the typical
          track's appearance cost moves from ~0.15 to ~0.16, discounting lambda there buys
          nothing regardless of how many sigma the median moved.
      B2  track-birth rate around flagged frames. A new track id appearing is what an identity
          break looks like in the output; if scene-wide appearance events actually cause
          fragmentation, births should concentrate at and just after flagged frames. This is a
          GT-free proxy for ID switches, so it is directional evidence, not a substitute for
          scoring against ground truth.

Usage:  python _appres_followup.py _appres_logs_replay --win 31
"""
import argparse
import os

import numpy as np
import pandas as pd


def rolling_z(x, win=31):
    s = pd.Series(x, dtype=float)
    med = s.rolling(win, center=True, min_periods=win // 3).median().bfill().ffill()
    dev = (s - med).abs()
    sc = dev.rolling(win, center=True, min_periods=win // 3).median().bfill().ffill()
    sc = sc.replace(0.0, np.nan)
    gl = np.nanmedian(np.abs(x - np.nanmedian(x)))
    sc = sc.fillna(gl if gl > 0 else 1e-6) * 1.4826
    return ((s - med) / sc).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log_dir', nargs='?', default='_scratch/_appres_logs_replay')
    ap.add_argument('--win', type=int, default=31)
    ap.add_argument('--zthr', type=float, default=3.0)
    ap.add_argument('--min-tracks', type=int, default=5)
    args = ap.parse_args()

    print('=' * 104)
    print('TEST A -- appearance-residual gate vs HDST-GNN C1 mean-object-area gate')
    print('=' * 104)
    print('%-22s %7s %7s %7s %8s %9s %9s' %
          ('sequence', 'nA(app)', 'nB(area)', 'both', 'app-only', 'corr(z)', 'jaccard'))

    rowsA, per_seq = [], {}
    for fn in sorted(f for f in os.listdir(args.log_dir) if f.endswith('.csv')):
        seq = fn[:-4]
        df = pd.read_csv(os.path.join(args.log_dir, fn))
        df = df[df['valid'] == 1]
        g = df.groupby('frame')
        n = g.size()
        keep = n[n >= args.min_tracks].index
        d = pd.DataFrame({
            'resid': g['resid'].median(),
            'area': g['area'].mean(),
            'n': n,
        }).loc[keep].sort_index()
        if len(d) < 40:
            print('%-22s too short (%d frames)' % (seq, len(d)))
            continue

        zr = rolling_z(d['resid'].to_numpy(), args.win)
        # HDST-GNN C1's altitude proxy, computed the way the paper defines it
        za = rolling_z(-np.log(d['area'].to_numpy() / np.median(d['area'].to_numpy())), args.win)

        A = np.abs(zr) > args.zthr
        B = np.abs(za) > args.zthr
        both = int((A & B).sum())
        corr = float(np.corrcoef(np.abs(zr), np.abs(za))[0, 1])
        jac = both / max(1, int((A | B).sum()))
        print('%-22s %7d %8d %7d %8d %9.3f %9.3f'
              % (seq, int(A.sum()), int(B.sum()), both, int((A & ~B).sum()), corr, jac))
        rowsA.append(dict(seq=seq, nA=int(A.sum()), nB=int(B.sum()), both=both,
                          app_only=int((A & ~B).sum()), corr=corr, jaccard=jac))
        per_seq[seq] = (d, zr, A)

    ra = pd.DataFrame(rowsA)
    if len(ra):
        print('-' * 104)
        print('%-22s %7d %8d %7d %8d %9.3f %9.3f'
              % ('TOTAL / mean', ra['nA'].sum(), ra['nB'].sum(), ra['both'].sum(),
                 ra['app_only'].sum(), ra['corr'].mean(), ra['jaccard'].mean()))
        print('\nREAD: high overlap / high corr => the appearance channel is re-deriving the')
        print('      geometric altitude proxy HDST-GNN C1 already publishes -> footnote.')
        print('      Low overlap => the appearance channel sees events geometry cannot -> a')
        print('      distinct sensor, and the co-ablation becomes an argument FOR the candidate.')

    print('\n' + '=' * 104)
    print('TEST B -- operational magnitude at flagged frames')
    print('=' * 104)
    print('%-22s %26s %26s %14s' %
          ('sequence', '--- resid median (p75) ---', '--- track births/frame ---', 'lift'))
    print('%-22s %12s %13s %12s %13s %14s' %
          ('', 'flagged', 'unflagged', 'flagged+-2', 'baseline', 'births'))

    rowsB = []
    for fn in sorted(f for f in os.listdir(args.log_dir) if f.endswith('.csv')):
        seq = fn[:-4]
        if seq not in per_seq:
            continue
        d, zr, A = per_seq[seq]
        df = pd.read_csv(os.path.join(args.log_dir, fn))
        df = df[df['valid'] == 1]

        frames = d.index.to_numpy()
        flagged = set(frames[A].tolist())
        sub = df[df['frame'].isin(frames)]
        inF = sub['frame'].isin(flagged)
        m_f = float(sub.loc[inF, 'resid'].median()) if inF.any() else np.nan
        p_f = float(sub.loc[inF, 'resid'].quantile(0.75)) if inF.any() else np.nan
        m_u = float(sub.loc[~inF, 'resid'].median())
        p_u = float(sub.loc[~inF, 'resid'].quantile(0.75))

        # track births: first frame each id is seen (replay logs a row only from a track's
        # SECOND appearance onward, so this is "first residual frame" -- a consistent proxy)
        first = df.groupby('track_id')['frame'].min()
        births = pd.Series(0, index=frames)
        vc = first.value_counts()
        for fr, c in vc.items():
            if fr in births.index:
                births[fr] = c
        win = set()
        for fr in flagged:
            win.update([fr - 2, fr - 1, fr, fr + 1, fr + 2])
        inW = np.array([f in win for f in frames])
        b_f = float(births[inW].mean()) if inW.any() else np.nan
        b_u = float(births[~inW].mean()) if (~inW).any() else np.nan
        lift = (b_f / b_u) if (b_u and b_u > 0) else np.nan

        print('%-22s %12s %13s %12.3f %13.3f %14s'
              % (seq, '%.4f (%.3f)' % (m_f, p_f), '%.4f (%.3f)' % (m_u, p_u),
                 b_f, b_u, '%.2fx' % lift if lift == lift else 'n/a'))
        rowsB.append(dict(seq=seq, resid_flagged=m_f, resid_unflagged=m_u,
                          p75_flagged=p_f, p75_unflagged=p_u,
                          births_flagged=b_f, births_base=b_u, birth_lift=lift,
                          n_flagged=int(A.sum())))

    rb = pd.DataFrame(rowsB)
    if len(rb):
        print('-' * 104)
        print('pooled resid shift at flagged frames: %+.4f (%.1f%% of the unflagged median)'
              % (rb.resid_flagged.mean() - rb.resid_unflagged.mean(),
                 100 * (rb.resid_flagged.mean() - rb.resid_unflagged.mean()) / rb.resid_unflagged.mean()))
        print('median birth lift across sequences: %.2fx   (sequences with lift > 1.2: %d/%d)'
              % (rb.birth_lift.median(), int((rb.birth_lift > 1.2).sum()), len(rb)))
        print('\nREAD: a shift of a couple of percent in the appearance cost, with no birth lift,')
        print('      is a real but inert common mode -- detectable, not actionable. A large shift')
        print('      plus a birth lift concentrated at flagged frames is what would justify the gate.')
        os.makedirs('_scratch/_appres_diag_out_replay', exist_ok=True)
        ra.to_csv('_scratch/_appres_diag_out_replay/testA_area_coablation.csv', index=False)
        rb.to_csv('_scratch/_appres_diag_out_replay/testB_magnitude.csv', index=False)
        print('\nwrote _appres_diag_out_replay/testA_area_coablation.csv, testB_magnitude.csv')


if __name__ == '__main__':
    main()
