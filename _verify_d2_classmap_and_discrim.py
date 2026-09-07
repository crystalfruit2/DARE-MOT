"""(1) sanity-check the class-id mapping -- the residual logs contain ZERO class-0 rows, which
       either means pedestrians are absent from the tracker output or the id map is off by one.
       Median box area separates them unambiguously: pedestrians are ~1-2 orders smaller.

   (2) BLUR vs KF-LAG discriminator.
       Both hypotheses predict appearance excursions during camera motion, but they predict
       DIFFERENT regressors:
         motion blur      scales with the instantaneous SPEED of the scene across the sensor
         KF-posterior lag scales with ACCELERATION -- a constant-velocity pan is tracked exactly
                          by a constant-velocity KF, so a steady lag cancels between template and
                          current crop; only a CHANGE of velocity de-registers the crop.
       Partial (rank) correlation of |z_resid| with each, controlling for the other, therefore
       separates a real appearance event from the replay artifact.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from _verify_common import SEQS, load_resid, load_boxes, frame_median, zseries, geometry_series

print('=' * 112)
print('CLASS-ID SANITY -- median box area (px^2) per class id in the cached tracker output')
print('=' * 112)
print('%-22s' % 'sequence' + ''.join('%18s' % ('cls=%d' % c) for c in range(5)))
for seq in SEQS:
    b = load_boxes(seq)
    b['ar'] = b['w'] * b['h']
    cells = []
    for c in range(5):
        s = b[b['cls'] == c]['ar']
        cells.append('%18s' % ('%.0f (n=%d)' % (s.median(), len(s)) if len(s) else '-'))
    print('%-22s' % seq + ''.join(cells))
b_all = pd.concat([load_boxes(s).assign(ar=lambda d: d.w * d.h) for s in SEQS])
print('-' * 112)
print('%-22s' % 'ALL SEQUENCES' + ''.join(
    '%18s' % ('%.0f (n=%d)' % (b_all[b_all.cls == c].ar.median(), (b_all.cls == c).sum())
              if (b_all.cls == c).any() else '-') for c in range(5)))
print('')
print('VisDrone pedestrians are typically 200-2000 px^2; cars 2000-20000.')

print('')
print('=' * 112)
print('BLUR (speed) vs KF-LAG (acceleration) -- partial Spearman of |z_resid|')
print('=' * 112)
print('%-22s %8s %12s %12s %14s %14s' %
      ('sequence', 'nfr', 'rho_speed', 'rho_accel', 'rho_speed|acc', 'rho_accel|spd'))


def pcorr(x, y, z):
    """Spearman partial correlation of x,y controlling z (rank-linear residualisation)."""
    R = np.vstack([rankdata(x), rankdata(y), rankdata(z)]).T
    R = (R - R.mean(0)) / R.std(0)
    bx = np.polyfit(R[:, 2], R[:, 0], 1)
    by = np.polyfit(R[:, 2], R[:, 1], 1)
    rx = R[:, 0] - np.polyval(bx, R[:, 2])
    ry = R[:, 1] - np.polyval(by, R[:, 2])
    return float(np.corrcoef(rx, ry)[0, 1])


rows = []
for seq in SEQS:
    d = load_resid(seq)
    grid, med = frame_median(d, 5)
    z = zseries(med, 31)
    G = geometry_series(seq, grid)
    sp = G['glob_speed'].to_numpy()
    ac = G['glob_accel'].to_numpy()
    m = (~np.isnan(z)) & (~np.isnan(sp)) & (~np.isnan(ac))
    az = np.abs(z[m])
    rs = float(np.corrcoef(rankdata(az), rankdata(sp[m]))[0, 1])
    ra = float(np.corrcoef(rankdata(az), rankdata(ac[m]))[0, 1])
    ps = pcorr(az, sp[m], ac[m])
    pa = pcorr(az, ac[m], sp[m])
    print('%-22s %8d %12.3f %12.3f %14.3f %14.3f' % (seq, m.sum(), rs, ra, ps, pa))
    rows.append(dict(seq=seq, rs=rs, ra=ra, ps=ps, pa=pa))
R = pd.DataFrame(rows)
print('-' * 112)
print('%-22s %8s %12.3f %12.3f %14.3f %14.3f' %
      ('MEAN', '', R.rs.mean(), R.ra.mean(), R.ps.mean(), R.pa.mean()))
print('')
print('Same, restricted to the TAIL that actually produces flags (top-decile |z|):')
print('%-22s %8s %12s %12s' % ('sequence', 'n_tail', 'accel pctile', 'speed pctile'))
for seq in SEQS:
    d = load_resid(seq)
    grid, med = frame_median(d, 5)
    z = zseries(med, 31)
    G = geometry_series(seq, grid)
    sp, ac = G['glob_speed'].to_numpy(), G['glob_accel'].to_numpy()
    m = (~np.isnan(z)) & (~np.isnan(sp)) & (~np.isnan(ac))
    az = np.abs(z[m])
    thr = np.quantile(az, 0.9)
    t = az >= thr
    pa = 100 * np.mean(ac[m][:, None] < np.median(ac[m][t]))
    ps = 100 * np.mean(sp[m][:, None] < np.median(sp[m][t]))
    print('%-22s %8d %11.1f%% %11.1f%%' % (seq, t.sum(), pa, ps))
