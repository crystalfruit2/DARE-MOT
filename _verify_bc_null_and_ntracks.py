"""ATTACK B -- is the null too weak?  and  ATTACK C -- is it just thin frames?

B. Circular shift destroys any SHARED SLOW SCENE TREND as well as frame-locked synchrony. If a
   slow common drift (gradual altitude change, lighting) leaks past the 31-frame detrender, the
   observed series keeps it and the null does not -> inflated ratio with no frame-level event.
   Two probes:
     * detrend window sweep (15 / 31 / 61): a trend-leak effect shrinks as the window shrinks.
     * LOCAL-JITTER null: shift each track by a SMALL random lag only (+-5, +-15 frames). That
       preserves shared slow trend but still destroys tight frame-level alignment. A genuinely
       frame-locked common mode must beat this harsher null too.

C. With 5-6 tracks a cross-track median is unstable, and uav0000268 (median 6 tracks/frame)
   contributes ~1/3 of all flags. Re-run at min-tracks 5 / 10 / 15.
"""
import numpy as np
import pandas as pd
from _verify_common import SEQS, load_resid, rolling_baseline

RNG = np.random.default_rng(0)
NPERM = 200
WINS = [15, 31, 61]


def prep(seq):
    d = load_resid(seq)
    f0 = int(d['frame'].min())
    nF = int(d['frame'].max()) - f0 + 1
    fi = d['frame'].to_numpy() - f0
    v = d['resid'].to_numpy(float)
    t = d['track_id'].to_numpy()
    o = np.argsort(t, kind='stable')
    return fi, v, t, nF, (t[o], fi[o], v[o])


def med_series(fi, v, nF, mt):
    o = np.argsort(fi, kind='stable')
    f_, x_ = fi[o], v[o]
    bnd = np.searchsorted(f_, np.arange(nF + 1))
    out = np.full(nF, np.nan)
    for k in range(nF):
        lo, hi = bnd[k], bnd[k + 1]
        if hi - lo >= mt:
            out[k] = np.median(x_[lo:hi])
    return out


def stat(m, win):
    ok = ~np.isnan(m)
    if ok.sum() < 10:
        return np.nan, 0
    x = m.copy()
    x[~ok] = np.nanmedian(m)
    base, sc = rolling_baseline(x, win)
    z = (x - base) / sc
    dd = (x - base)[ok]
    return 1.4826 * np.median(np.abs(dd - np.median(dd))), int((np.abs(z[ok]) > 3).sum())


def run(seq, mt, nulls, wins=WINS, nperm=NPERM):
    fi, v, t, nF, (ts, fs, vs) = prep(seq)
    uq, st = np.unique(ts, return_index=True)
    en = np.append(st[1:], len(ts))
    obs = {w: stat(med_series(fi, v, nF, mt), w) for w in wins}
    acc = {(k, w): [] for k in nulls for w in wins}
    for _ in range(nperm):
        for k, maxlag in nulls.items():
            vv = vs.copy()
            for lo, hi in zip(st, en):
                n = hi - lo
                if n < 2:
                    continue
                if maxlag is None:
                    lag = int(RNG.integers(1, n))
                else:
                    ml = min(maxlag, n - 1)
                    lag = int(RNG.integers(1, ml + 1)) * (1 if RNG.random() < 0.5 else -1)
                vv[lo:hi] = np.roll(vs[lo:hi], lag)
            for w in wins:
                acc[(k, w)].append(stat(med_series(fs, vv, nF, mt), w)[0])
    res = {}
    for k in nulls:
        for w in wins:
            a = np.array(acc[(k, w)], float)
            nm = np.nanmedian(a)
            res[(k, w)] = dict(ratio=obs[w][0] / nm if nm > 0 else np.nan,
                               p=(np.sum(a >= obs[w][0]) + 1) / (len(a) + 1),
                               nz3=obs[w][1], obs_sd=obs[w][0], null_sd=nm)
    return res


NULLS = {'shift(any)': None, 'jitter+-15': 15, 'jitter+-5': 5}

print('=' * 126)
print('ATTACK B -- detrend-window sweep x null strength   (min-tracks 5)')
print('=' * 126)
print('%-22s %14s' % ('sequence', 'null') + ''.join('%22s' % ('win=%d' % w) for w in WINS))
allres = {}
for seq in SEQS:
    r = run(seq, 5, NULLS)
    allres[seq] = r
    for i, k in enumerate(NULLS):
        lbl = seq if i == 0 else ''
        print('%-22s %14s' % (lbl, k) +
              ''.join('%22s' % ('%.2fx p=%.3f n=%d' % (r[(k, w)]['ratio'], r[(k, w)]['p'],
                                                       r[(k, w)]['nz3'])) for w in WINS))
print('-' * 126)
for k in NULLS:
    print('%-22s %14s' % ('MEDIAN ratio', k) +
          ''.join('%22s' % ('%.2fx  (p<.05: %d/7)' % (
              np.median([allres[s][(k, w)]['ratio'] for s in SEQS]),
              sum(allres[s][(k, w)]['p'] < 0.05 for s in SEQS))) for w in WINS))

print('')
print('=' * 126)
print('ATTACK C -- minimum tracks per frame   (win=31, shift(any) null)')
print('=' * 126)
print('%-22s %10s' % ('sequence', 'med trk/f') +
      ''.join('%26s' % ('min-tracks=%d' % m) for m in (5, 10, 15)))
rowsC = []
for seq in SEQS:
    d = load_resid(seq)
    mtf = d.groupby('frame').size().median()
    cells, rec = [], dict(seq=seq, med_trk=mtf)
    for m in (5, 10, 15):
        n_ok = int((d.groupby('frame').size() >= m).sum())
        if n_ok < 40:
            cells.append('%26s' % ('only %d frames' % n_ok))
            rec['ratio%d' % m] = np.nan
            rec['nz%d' % m] = 0
            rec['nfr%d' % m] = n_ok
            continue
        r = run(seq, m, {'shift(any)': None}, wins=[31])[('shift(any)', 31)]
        cells.append('%26s' % ('%.2fx p=%.3f n=%d/%d' % (r['ratio'], r['p'], r['nz3'], n_ok)))
        rec['ratio%d' % m] = r['ratio']
        rec['nz%d' % m] = r['nz3']
        rec['nfr%d' % m] = n_ok
        rec['p%d' % m] = r['p']
    print('%-22s %10.1f' % (seq, mtf) + ''.join(cells))
    rowsC.append(rec)
C = pd.DataFrame(rowsC)
print('-' * 126)
print('%-22s %10s' % ('MEDIAN ratio / total flags', '') +
      ''.join('%26s' % ('%.2fx   flags=%d' % (C['ratio%d' % m].median(), C['nz%d' % m].sum()))
              for m in (5, 10, 15)))
C.to_csv('_scratch/_verify_out_C.csv', index=False)
print('')
print('wrote _verify_out_C.csv')
