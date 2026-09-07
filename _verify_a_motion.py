"""ATTACK A -- is the cross-track synchrony an artifact of the replay's crop source?

The replay crops the KF-POSTERIOR box; the tracker crops the DETECTION box. The claim under test
is that this perturbation is INDEPENDENT across tracks. It is not obviously so: when the whole
scene moves (drone pan / zoom / gimbal), every track's KF posterior lags its detection in the SAME
direction, so every crop is misaligned at the same time -> a manufactured common mode.

Probes, all computed from the cached boxes themselves:
  glob_speed   median per-frame center displacement of the tracks (rigid scene motion)
  glob_accel   |change in global velocity| -- what actually makes a KF posterior lag
  med_disp_rel median per-track |displacement| / sqrt(area)   (misalignment relative to box size)
  resid_motion median |displacement - global displacement| / sqrt(area)  (non-rigid motion)
  med_abs_dlogarea  median per-track |log area ratio| -- the HONEST zoom/GSD detector
  mean_area    HDST-GNN C1's own statistic, for reference
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from _verify_common import *

CH = ['glob_speed', 'glob_accel', 'med_disp', 'med_disp_rel', 'resid_motion',
      'med_abs_dlogarea', 'med_dlogarea', 'mean_area']

print('=' * 118)
print('ATTACK A -- does the appearance-residual excursion track scene GEOMETRY (replay crop artifact)?')
print('=' * 118)

rows = []
for seq in SEQS:
    df = load_resid(seq)
    grid, med = frame_median(df, 5)
    z = zseries(med, 31)
    G = geometry_series(seq, grid)
    A = (~np.isnan(z)) & (np.abs(z) > 3)
    ok = ~np.isnan(z)
    r = dict(seq=seq, nA=int(A.sum()), nframes=int(ok.sum()))
    for c in CH:
        v = G[c].to_numpy()
        m = ok & ~np.isnan(v)
        if m.sum() < 20:
            r['rho_' + c] = np.nan; r['lift_' + c] = np.nan; continue
        rho = spearmanr(np.abs(z[m]), v[m]).statistic
        r['rho_' + c] = rho
        af = A & m
        if af.sum() >= 3:
            hi, lo = np.median(v[af]), np.median(v[m & ~A])
            r['lift_' + c] = hi / lo if lo not in (0,) and abs(lo) > 1e-12 else np.nan
        else:
            r['lift_' + c] = np.nan
    # z-flag overlap between appearance and each geometry channel
    for c in ['glob_speed', 'glob_accel', 'med_disp_rel', 'med_abs_dlogarea', 'mean_area']:
        zg = zseries(G[c].to_numpy(), 31)
        B = (~np.isnan(zg)) & (np.abs(zg) > 3)
        both = int((A & B).sum())
        r['nB_' + c] = int(B.sum()); r['both_' + c] = both
        r['jac_' + c] = both / max(1, int((A | B).sum()))
    rows.append(r)

R = pd.DataFrame(rows)
pd.set_option('display.width', 250)

print('\n-- Spearman rho( |z_resid| , geometry channel )  over all usable frames --')
print('%-22s %7s' % ('sequence', 'frames') + ''.join('%18s' % c for c in CH))
for _, r in R.iterrows():
    print('%-22s %7d' % (r.seq, r.nframes) + ''.join('%18.3f' % r['rho_' + c] for c in CH))
print('%-22s %7s' % ('MEAN', '') + ''.join('%18.3f' % R['rho_' + c].mean() for c in CH))

print('\n-- ratio of channel median at FLAGGED frames vs unflagged (1.0 = no relation) --')
print('%-22s %7s' % ('sequence', 'nA') + ''.join('%18s' % c for c in CH))
for _, r in R.iterrows():
    print('%-22s %7d' % (r.seq, r.nA) + ''.join('%18.2f' % r['lift_' + c] for c in CH))
print('%-22s %7s' % ('MEDIAN', '') + ''.join('%18.2f' % R['lift_' + c].median() for c in CH))

print('\n-- flag-set overlap (z>3 on both channels), jaccard --')
cs = ['glob_speed', 'glob_accel', 'med_disp_rel', 'med_abs_dlogarea', 'mean_area']
print('%-22s %5s' % ('sequence', 'nA') + ''.join('%26s' % c for c in cs))
for _, r in R.iterrows():
    print('%-22s %5d' % (r.seq, r.nA) + ''.join(
        '%26s' % ('nB=%d both=%d J=%.3f' % (r['nB_' + c], r['both_' + c], r['jac_' + c])) for c in cs))
tot = {c: (int(R['nB_' + c].sum()), int(R['both_' + c].sum())) for c in cs}
print('%-22s %5d' % ('TOTAL', R.nA.sum()) + ''.join(
    '%26s' % ('nB=%d both=%d' % tot[c]) for c in cs))
R.to_csv('_scratch/_verify_out_A.csv', index=False)
print('\nwrote _verify_out_A.csv')
