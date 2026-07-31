"""Density-gate boost result, stratified (2026-07-31 follow-up to the same-day first test).

The first test found mc_dare_density_boost beats the CV headline (mc_dare) by -10 IDSw /
+0.9%/+0.6% IDF1 (macro/micro), pooled across all 7 val7 sequences. That's a pooled
aggregate -- exactly the shape of number this project has been burned by twice already
(the MOTA class-averaging artifact, the CMC single-sequence artifact, and most recently
the CA x CMC "safer together" interaction that collapsed to one sequence). Before trusting
this result enough to spend GPU time tuning it, check whether it's a genuine distributed
effect or concentrated in one sequence/class.

Re-scores the existing cached track_results (no re-tracking) for mc_dare_density_boost vs
mc_dare, once pooled (matches the same-day headline) and once broken out per sequence and
per (sequence, class).
"""
import _score_multiclass as sm

ALL_SEQS = list(sm.SEQS)
BOOST, CV = "mc_dare_density_boost", "mc_dare"


def class_table(res_b, res_cv, label):
    print(f"\n--- {label} ---")
    for cls, name in sm.MC_NAMES.items():
        vb, vv = res_b.get(cls), res_cv.get(cls)
        if vb is None or vv is None:
            print(f"  {name:12s} --")
            continue
        d_idsw = vb["idsw"] - vv["idsw"]
        d_idf1 = (vb["idf1"] - vv["idf1"]) * 100
        d_mota = (vb["mota"] - vv["mota"]) * 100
        print(f"  {name:12s} IDSw {vv['idsw']:4d} -> {vb['idsw']:4d}  (d={d_idsw:+4d})   "
              f"IDF1 d={d_idf1:+.1f}%   MOTA d={d_mota:+.1f}%   GT={vb['gt']}")
    if "SUM_IDSw" in res_b:
        dsw = res_b["SUM_IDSw"] - res_cv["SUM_IDSw"]
        d_avg_mota = (res_b["AVG"]["mota"] - res_cv["AVG"]["mota"]) * 100
        d_avg_idf1 = (res_b["AVG"]["idf1"] - res_cv["AVG"]["idf1"]) * 100
        print(f"  {'SUM_IDSw':12s} d={dsw:+4d}   AVG-MOTA d={d_avg_mota:+.1f}%   "
              f"AVG-IDF1 d={d_avg_idf1:+.1f}%")


print("=" * 70)
print("POOLED (all 7 seqs) -- should match the 07-31 headline: -10 IDSw / +0.9%/+0.6% IDF1")
print("=" * 70)
sm.SEQS = ALL_SEQS
res_b_all = sm.score_run(BOOST)
res_cv_all = sm.score_run(CV)
class_table(res_b_all, res_cv_all, "ALL 7 SEQS")

print("\n" + "=" * 70)
print("PER-SEQUENCE breakdown (SUM_IDSw delta per sequence, all classes pooled)")
print("=" * 70)
per_seq_dsw = {}
for seq in ALL_SEQS:
    sm.SEQS = [seq]
    res_b = sm.score_run(BOOST)
    res_cv = sm.score_run(CV)
    dsw = res_b["SUM_IDSw"] - res_cv["SUM_IDSw"]
    per_seq_dsw[seq] = dsw
    d_avg_mota = (res_b["AVG"]["mota"] - res_cv["AVG"]["mota"]) * 100
    print(f"  {seq:24s} SUM_IDSw d={dsw:+5d}   AVG-MOTA d={d_avg_mota:+.1f}%")

total = sum(per_seq_dsw.values())
worst_seq = max(per_seq_dsw, key=lambda s: per_seq_dsw[s])
best_seq = min(per_seq_dsw, key=lambda s: per_seq_dsw[s])
print(f"\n  Sum of per-seq deltas: {total:+d}  (sanity check vs pooled dsw above)")
print(f"  Best (most boost-favoring) sequence: {best_seq} ({per_seq_dsw[best_seq]:+d})")
print(f"  Worst (least boost-favoring) sequence: {worst_seq} ({per_seq_dsw[worst_seq]:+d})")
if total != 0:
    share = per_seq_dsw[best_seq] / total
    print(f"  Best sequence's share of total net win: {share*100:.0f}%")

print("\n" + "=" * 70)
print("PER-(SEQUENCE, CLASS) breakdown -- pedestrian & car only (volume classes)")
print("=" * 70)
for seq in ALL_SEQS:
    sm.SEQS = [seq]
    res_b = sm.score_run(BOOST)
    res_cv = sm.score_run(CV)
    row = f"  {seq:24s}"
    for cls in (1, 2):  # pedestrian, car
        vb, vv = res_b.get(cls), res_cv.get(cls)
        if vb is None or vv is None:
            row += f"  {sm.MC_NAMES[cls]}: --"
            continue
        d = vb["idsw"] - vv["idsw"]
        row += f"  {sm.MC_NAMES[cls]}: {vv['idsw']:3d}->{vb['idsw']:3d} (d={d:+3d})"
    print(row)
