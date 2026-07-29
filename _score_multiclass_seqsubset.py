"""Decisive check (2026-07-29, per independent Opus review of the seq-x-class
stratification): does the OFFICIAL py-motmetrics per-class result confirm that
uav0000086_00000_v alone flips the sign of the CMC headline, or is that a
proxy-metric (Hungarian-IoU switch diagnostic) artifact?

Re-scores the existing cached track_results (no re-tracking) via the same
recipe as _score_multiclass.py, once over all 7 val sequences and once
excluding uav0000086_00000_v, for both the DARE and ByteTrack CMC-vs-no-CMC
pairs. Reports pedestrian, car, SUM_IDSw (macro) and MICRO deltas for each.
"""
import _score_multiclass as sm

ALL_SEQS = sm.SEQS
EXCL_086 = [s for s in ALL_SEQS if s != "uav0000086_00000_v"]

PAIRS = {
    "dare": ("mc_dare_cmc", "mc_dare"),
    "bytetrack": ("mc_bytetrack_cmc", "mc_bytetrack"),
}


def delta_table(cmc_exp, nocmc_exp, seqs, tag):
    sm.SEQS = seqs
    res_cmc = sm.score_run(cmc_exp)
    res_nocmc = sm.score_run(nocmc_exp)
    print(f"\n--- {tag}: {cmc_exp} - {nocmc_exp}  (seqs={len(seqs)}) ---")
    for cls, name in sm.MC_NAMES.items():
        vc, vn = res_cmc[cls], res_nocmc[cls]
        if vc is None or vn is None:
            print(f"  {name:12s} --")
            continue
        print(f"  {name:12s} IDSw {vn['idsw']:4d} -> {vc['idsw']:4d}  "
              f"(d={vc['idsw']-vn['idsw']:+4d})   IDF1 d={ (vc['idf1']-vn['idf1'])*100:+.1f}%  "
              f"MOTA d={ (vc['mota']-vn['mota'])*100:+.1f}%")
    dsw = res_cmc['SUM_IDSw'] - res_nocmc['SUM_IDSw']
    da_mota = (res_cmc['AVG']['mota'] - res_nocmc['AVG']['mota']) * 100
    dm_mota = (res_cmc['MICRO']['mota'] - res_nocmc['MICRO']['mota']) * 100
    dm_idf1 = (res_cmc['MICRO']['idf1'] - res_nocmc['MICRO']['idf1']) * 100
    print(f"  {'SUM_IDSw (macro)':12s} d={dsw:+4d}   AVG-MOTA d={da_mota:+.1f}%   "
          f"MICRO-MOTA d={dm_mota:+.1f}%   MICRO-IDF1 d={dm_idf1:+.1f}%")
    return dsw, da_mota, dm_mota, dm_idf1


for pair_name, (cmc_exp, nocmc_exp) in PAIRS.items():
    print(f"\n{'='*70}\n{pair_name.upper()} pair\n{'='*70}")
    delta_table(cmc_exp, nocmc_exp, ALL_SEQS, "ALL 7 SEQS")
    delta_table(cmc_exp, nocmc_exp, EXCL_086, "EXCL uav0000086")
