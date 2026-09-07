"""ATTACK D -- is the 'scene-wide' spike actually one CLASS?
     and the common-mode invariance check that decides ATTACK E.

D. If flagged frames are carried by pedestrians only (or cars only), it is a class effect wearing
   a frame-level disguise -- the exact failure mode that killed the class-leak and the CMC
   candidates. Probes:
     * class composition per sequence
     * per-class frame-median residual series; do the flagged frames show a z>2 excursion in
       MORE THAN ONE class (only frames where >=2 classes have >=3 tracks are testable)
     * leave-one-class-out: recompute the frame median after dropping each class in turn and see
       whether the flags survive

E-prelude. A linear assignment problem is INVARIANT to adding a constant to the whole cost
   matrix. A pure common-mode inflation of appearance cost therefore cannot flip any assignment;
   only the DISPERSION across candidate pairs can. So: at flagged frames, does the cross-track
   spread of residuals grow by more than the median does? If the spike is median-only, the
   candidate has detected the one perturbation the solver is structurally immune to.
"""
import numpy as np
import pandas as pd
from _verify_common import SEQS, load_resid, frame_median, zseries

CLS = {0: 'pedestrian', 1: 'car', 2: 'van', 3: 'truck', 4: 'bus'}

print('=' * 120)
print('ATTACK D1 -- class composition of the residual rows')
print('=' * 120)
print('%-22s %8s' % ('sequence', 'rows') + ''.join('%14s' % c for c in CLS.values()))
for seq in SEQS:
    d = load_resid(seq)
    n = len(d)
    vc = d['cls'].value_counts(normalize=True)
    print('%-22s %8d' % (seq, n) + ''.join('%13.1f%%' % (100 * vc.get(k, 0.0)) for k in CLS))

print('')
print('=' * 120)
print('ATTACK D2 -- do flagged frames excite MORE THAN ONE class?')
print('   testable frame = flagged AND >=2 classes with >=3 tracks;  per-class |z|>2 counts as excited')
print('=' * 120)
print('%-22s %6s %10s %12s %12s %12s %10s' %
      ('sequence', 'nA', 'testable', '0 cls exc.', '1 cls exc.', '>=2 cls exc.', 'multi%'))
tot = np.zeros(3)
for seq in SEQS:
    d = load_resid(seq)
    grid, med = frame_median(d, 5)
    z = zseries(med, 31)
    A = (~np.isnan(z)) & (np.abs(z) > 3)
    flagged = set(grid[A].tolist())
    zc = {}
    for c in sorted(d['cls'].unique()):
        dc = d[d['cls'] == c]
        if len(dc) < 200:
            continue
        gc, mc = frame_median(dc, 3)
        s = pd.Series(zseries(mc, 31), index=gc)
        if s.notna().sum() < 40:
            continue
        zc[c] = s
    cnt = np.zeros(3)
    n_test = 0
    for f in sorted(flagged):
        avail = [c for c, s in zc.items() if f in s.index and not np.isnan(s[f])]
        if len(avail) < 2:
            continue
        n_test += 1
        k = sum(abs(zc[c][f]) > 2 for c in avail)
        cnt[min(k, 2)] += 1
    tot += cnt
    pct = 100 * cnt[2] / n_test if n_test else np.nan
    print('%-22s %6d %10d %12d %12d %12d %9.1f%%' %
          (seq, int(A.sum()), n_test, cnt[0], cnt[1], cnt[2], pct))
print('%-22s %6s %10d %12d %12d %12d %9.1f%%' %
      ('TOTAL', '', int(tot.sum()), tot[0], tot[1], tot[2],
       100 * tot[2] / max(1, tot.sum())))

print('')
print('=' * 120)
print('ATTACK D3 -- leave-one-class-out: fraction of flags that survive dropping each class')
print('=' * 120)
print('%-22s %6s' % ('sequence', 'nA') + ''.join('%16s' % ('drop ' + c) for c in CLS.values()))
for seq in SEQS:
    d = load_resid(seq)
    grid, med = frame_median(d, 5)
    z = zseries(med, 31)
    A = (~np.isnan(z)) & (np.abs(z) > 3)
    base = set(grid[A].tolist())
    cells = []
    for c in CLS:
        dd = d[d['cls'] != c]
        if len(dd) < 200 or not len(base):
            cells.append('%16s' % '-')
            continue
        g2, m2 = frame_median(dd, 5)
        z2 = pd.Series(zseries(m2, 31), index=g2)
        keep = sum(1 for f in base if f in z2.index and not np.isnan(z2[f]) and abs(z2[f]) > 3)
        cells.append('%16s' % ('%d/%d (%.0f%%)' % (keep, len(base), 100 * keep / len(base))))
    print('%-22s %6d' % (seq, len(base)) + ''.join(cells))

print('')
print('=' * 120)
print('E-PRELUDE -- common mode vs dispersion at flagged frames  (LAP is invariant to a pure')
print('             common mode; only spread across candidate pairs can flip an assignment)')
print('=' * 120)
print('%-22s %6s %12s %12s %9s %12s %12s %9s %10s' %
      ('sequence', 'nA', 'med@flag', 'med@unflag', 'ratio', 'MAD@flag', 'MAD@unflag',
       'ratio', 'MAD/med'))
rows = []
for seq in SEQS:
    d = load_resid(seq)
    grid, med = frame_median(d, 5)
    z = zseries(med, 31)
    A = (~np.isnan(z)) & (np.abs(z) > 3)
    flagged = set(grid[A].tolist())
    g = d.groupby('frame')['resid']
    fm = g.median()
    fmad = g.apply(lambda s: 1.4826 * np.median(np.abs(s - np.median(s))))
    n = d.groupby('frame').size()
    ok = n[n >= 5].index
    isf = np.array([f in flagged for f in ok])
    mf, mu = fm[ok][isf].median(), fm[ok][~isf].median()
    af, au = fmad[ok][isf].median(), fmad[ok][~isf].median()
    print('%-22s %6d %12.4f %12.4f %9.2f %12.4f %12.4f %9.2f %10s' %
          (seq, int(isf.sum()), mf, mu, mf / mu, af, au, af / au,
           '%.2f/%.2f' % (af / mf, au / mu)))
    rows.append(dict(seq=seq, med_f=mf, med_u=mu, mad_f=af, mad_u=au))
R = pd.DataFrame(rows)
print('-' * 120)
print('%-22s %6s %12.4f %12.4f %9.2f %12.4f %12.4f %9.2f' %
      ('MEDIAN', '', R.med_f.median(), R.med_u.median(),
       (R.med_f / R.med_u).median(), R.mad_f.median(), R.mad_u.median(),
       (R.mad_f / R.mad_u).median()))
print('')
print('READ: MAD-ratio ~= median-ratio means the whole distribution translates AND widens')
print('      proportionally; MAD-ratio ~1 with median-ratio >1 would mean a pure common mode')
print('      -- invisible to the solver.')
