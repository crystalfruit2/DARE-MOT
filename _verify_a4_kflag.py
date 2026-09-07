"""ATTACK A, part 4 -- how many pixels does the KF posterior actually lag the detection?

A3 measured that a 2-3 px crop misalignment already produces a cosine shift of 0.011-0.016,
i.e. the whole observed excursion. This closes the loop by simulating the tracker's OWN
KalmanFilter (motion_model from the run name: 'cv') and measuring |detection - posterior| for a
target whose image-plane velocity changes by the acceleration actually observed at flagged frames.

If that lag lands in the 1-3 px band, the replay can manufacture the entire finding, because the
lag is driven by a GLOBAL velocity change and is therefore in the SAME direction for every track
in the frame -- the cross-track correlation the 'independent perturbation' defence denied.

Also reports the cross-sequence relationship: does per-sequence synchrony strength track how
strongly that sequence's flagged frames coincide with acceleration?
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from yolox.tracker.kalman_filter import KalmanFilter

print('=' * 110)
print('KF posterior lag behind the detection, for a step change in image-plane velocity')
print('  (tracker KF, motion_model=cv, std_weight_position=1/20, box height h)')
print('=' * 110)
ACCS = [0.5, 1.0, 1.5, 2.0, 3.5, 6.2, 7.2]
HS = [40, 70, 120]
print('%-16s' % 'box height h' + ''.join('%16s' % ('dv=%.1f px/f' % a) for a in ACCS))
for h in HS:
    cells = []
    for a in ACCS:
        kf = KalmanFilter(motion_model='cv')
        m, c = kf.initiate(np.array([500.0, 500.0, 0.5, float(h)]))
        v = 2.0
        x = 500.0
        # burn in at constant velocity so the filter is in steady state
        for _ in range(40):
            x += v
            m, c = kf.predict(m, c)
            m, c = kf.update(m, c, np.array([x, 500.0, 0.5, float(h)]))
        # step change in velocity of magnitude a (== the measured global acceleration)
        v += a
        worst = 0.0
        for _ in range(6):
            x += v
            m, c = kf.predict(m, c)
            m, c = kf.update(m, c, np.array([x, 500.0, 0.5, float(h)]))
            worst = max(worst, abs(x - m[0]))
        cells.append('%16.2f' % worst)
    print('%-16d' % h + ''.join(cells))
print('')
print('(peak |detection - posterior| over the 6 frames after the velocity step, in pixels)')

print('')
print('=' * 110)
print('CROSS-SEQUENCE: synchrony strength vs acceleration-coincidence of the flagged frames')
print('=' * 110)
D = pd.DataFrame([
    # seq, ratio(win31 shift null), ratio@min-trk15, accel pctile of top-decile |z| frames,
    # accel pctile of the z>3 flagged frames, dominant class
    ('uav0000086_00000_v', 1.05, 1.03, 37.9, 62.3, 'pedestrian 100%'),
    ('uav0000117_02622_v', 1.25, 1.26, 87.3, 94.2, 'ped 46 / car 41'),
    ('uav0000137_00458_v', 1.77, 1.77, 89.2, 93.5, 'car 66 / ped 27'),
    ('uav0000182_00000_v', 1.24, 1.41, 72.7, 96.1, 'car 73 / van 15'),
    ('uav0000268_05773_v', 1.10, np.nan, 48.1, 66.0, 'car 86 / van 12'),
    ('uav0000305_00000_v', 1.38, 1.17, 67.6, 83.5, 'car 86'),
    ('uav0000339_00001_v', 1.13, 1.13, 85.0, 91.9, 'car 54 / ped 39'),
], columns=['seq', 'ratio', 'ratio15', 'acc_pct_tail', 'acc_pct_flag', 'dominant'])
print('%-22s %8s %9s %14s %14s   %s' %
      ('sequence', 'ratio', 'ratio@15', 'accel pct(tail)', 'accel pct(flag)', 'dominant class'))
for _, r in D.iterrows():
    print('%-22s %8.2f %9s %14.1f %14.1f   %s' %
          (r.seq, r.ratio, '%.2f' % r.ratio15 if r.ratio15 == r.ratio15 else 'n/a',
           r.acc_pct_tail, r.acc_pct_flag, r.dominant))
print('-' * 110)
print('Spearman(synchrony ratio, accel-percentile of top-decile |z| frames) = %.3f  (p=%.3f)'
      % spearmanr(D.ratio, D.acc_pct_tail))
print('Spearman(synchrony ratio, accel-percentile of z>3 flagged frames)   = %.3f  (p=%.3f)'
      % spearmanr(D.ratio, D.acc_pct_flag))
