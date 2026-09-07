"""ATTACK A, part 5 -- temporal signature: instantaneous (blur) or decaying (KF lag)?

Motion blur is an instantaneous property of the exposure: the residual excursion should sit at
lag 0 relative to the global-acceleration peak and vanish immediately.
A KF-posterior registration error builds when the velocity changes and decays over the few frames
the filter needs to re-converge, and the aggregated template carries the contamination forward,
so it should show a peak at lag 0-2 with a tail.

Cross-correlation of the detrended |z_resid| series against the global-acceleration series.
"""
import numpy as np
import pandas as pd
from _verify_common import SEQS, load_resid, frame_median, zseries, geometry_series

LAGS = range(-4, 7)
print('=' * 112)
print('Cross-correlation: corr( |z_resid|[t] , glob_accel[t - lag] )')
print('  lag>0 means the appearance excursion FOLLOWS the acceleration')
print('=' * 112)
print('%-22s' % 'sequence' + ''.join('%8s' % ('L%+d' % L) for L in LAGS) + '%10s' % 'argmax')
rows = []
for seq in SEQS:
    d = load_resid(seq)
    grid, med = frame_median(d, 5)
    z = np.abs(zseries(med, 31))
    G = geometry_series(seq, grid)
    a = G['glob_accel'].to_numpy()
    a = pd.Series(a).rank(pct=True).to_numpy()
    zz = pd.Series(z).rank(pct=True).to_numpy().copy()
    zz[np.isnan(z)] = np.nan
    cs = []
    for L in LAGS:
        x = zz[max(0, L):len(zz) + min(0, L)]
        y = a[max(0, -L):len(a) + min(0, -L)]
        m = ~np.isnan(x) & ~np.isnan(y)
        cs.append(float(np.corrcoef(x[m], y[m])[0, 1]) if m.sum() > 30 else np.nan)
    am = list(LAGS)[int(np.nanargmax(cs))]
    print('%-22s' % seq + ''.join('%8.3f' % c for c in cs) + '%10d' % am)
    rows.append(cs)
R = np.array(rows, float)
print('-' * 112)
print('%-22s' % 'MEAN' + ''.join('%8.3f' % c for c in np.nanmean(R, 0)) +
      '%10d' % list(LAGS)[int(np.nanargmax(np.nanmean(R, 0)))])
print('')
print('Same, on the 5 sequences whose synchrony survives its null (086 and 268 excluded):')
keep = [i for i, s in enumerate(SEQS) if s not in ('uav0000086_00000_v', 'uav0000268_05773_v')]
print('%-22s' % 'MEAN(5)' + ''.join('%8.3f' % c for c in np.nanmean(R[keep], 0)) +
      '%10d' % list(LAGS)[int(np.nanargmax(np.nanmean(R[keep], 0)))])
