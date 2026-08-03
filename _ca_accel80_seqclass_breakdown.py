"""CA tuned-config (accel_noise=1/80) per-sequence/class breakdown (2026-08-03 follow-up).

Same rigor already applied to the untuned CA reference (_ca_seqclass_breakdown.py,
2026-07-31: distributed effect across 5/7 sequences, best seq = 76% of net win, one
real outlier uav0000268/car-only/n=1) -- repeated here for the tuned point
(mc_ca_accel_80, DARE_KF_ACCEL_NOISE=0.0125) since it's a different (better-calibrated)
operating point on the same mechanism, not assumed to inherit the untuned check by
construction. Closes the item flagged to Prof. Farzad 2026-07-31, confirmed 2026-08-03.
"""
import _score_multiclass as sm

ALL_SEQS = list(sm.SEQS)
CA, CV = "mc_ca_accel_80", "mc_dare"


def class_table(res_ca, res_cv, label):
    print(f"\n--- {label} ---")
    for cls, name in sm.MC_NAMES.items():
        vc, vv = res_ca.get(cls), res_cv.get(cls)
        if vc is None or vv is None:
            print(f"  {name:12s} --")
            continue
        d_idsw = vc["idsw"] - vv["idsw"]
        d_idf1 = (vc["idf1"] - vv["idf1"]) * 100
        d_mota = (vc["mota"] - vv["mota"]) * 100
        print(f"  {name:12s} IDSw {vv['idsw']:4d} -> {vc['idsw']:4d}  (d={d_idsw:+4d})   "
              f"IDF1 d={d_idf1:+.1f}%   MOTA d={d_mota:+.1f}%   GT={vc['gt']}")
    if "SUM_IDSw" in res_ca:
        dsw = res_ca["SUM_IDSw"] - res_cv["SUM_IDSw"]
        d_avg_mota = (res_ca["AVG"]["mota"] - res_cv["AVG"]["mota"]) * 100
        d_avg_idf1 = (res_ca["AVG"]["idf1"] - res_cv["AVG"]["idf1"]) * 100
        print(f"  {'SUM_IDSw':12s} d={dsw:+4d}   AVG-MOTA d={d_avg_mota:+.1f}%   "
              f"AVG-IDF1 d={d_avg_idf1:+.1f}%")


print("=" * 70)
print("POOLED (all 7 seqs) -- should match the 07-31 tuned-sweep result: -39 IDSw / +1.1% IDF1 / +1.9% MOTA")
print("=" * 70)
sm.SEQS = ALL_SEQS
res_ca_all = sm.score_run(CA)
res_cv_all = sm.score_run(CV)
class_table(res_ca_all, res_cv_all, "ALL 7 SEQS")

print("\n" + "=" * 70)
print("PER-SEQUENCE breakdown (SUM_IDSw delta per sequence, all classes pooled)")
print("=" * 70)
per_seq_dsw = {}
for seq in ALL_SEQS:
    sm.SEQS = [seq]
    res_ca = sm.score_run(CA)
    res_cv = sm.score_run(CV)
    dsw = res_ca["SUM_IDSw"] - res_cv["SUM_IDSw"]
    per_seq_dsw[seq] = dsw
    d_avg_mota = (res_ca["AVG"]["mota"] - res_cv["AVG"]["mota"]) * 100
    print(f"  {seq:24s} SUM_IDSw d={dsw:+5d}   AVG-MOTA d={d_avg_mota:+.1f}%")

total = sum(per_seq_dsw.values())
worst_seq = max(per_seq_dsw, key=lambda s: per_seq_dsw[s])
best_seq = min(per_seq_dsw, key=lambda s: per_seq_dsw[s])
print(f"\n  Sum of per-seq deltas: {total:+d}  (sanity check vs pooled dsw above)")
print(f"  Best (most CA-favoring) sequence: {best_seq} ({per_seq_dsw[best_seq]:+d})")
print(f"  Worst (least CA-favoring) sequence: {worst_seq} ({per_seq_dsw[worst_seq]:+d})")
if total != 0:
    share = per_seq_dsw[best_seq] / total
    print(f"  Best sequence's share of total net win: {share*100:.0f}%")

print("\n" + "=" * 70)
print("PER-(SEQUENCE, CLASS) breakdown -- pedestrian & car only (volume classes)")
print("=" * 70)
for seq in ALL_SEQS:
    sm.SEQS = [seq]
    res_ca = sm.score_run(CA)
    res_cv = sm.score_run(CV)
    row = f"  {seq:24s}"
    for cls in (1, 2):  # pedestrian, car
        vc, vv = res_ca.get(cls), res_cv.get(cls)
        if vc is None or vv is None:
            row += f"  {sm.MC_NAMES[cls]}: --"
            continue
        d = vc["idsw"] - vv["idsw"]
        row += f"  {sm.MC_NAMES[cls]}: {vv['idsw']:3d}->{vc['idsw']:3d} (d={d:+3d})"
    print(row)
