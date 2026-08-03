"""Score the 2026-08-03 CA tuned-config (accel_noise=1/80) repeat-seed check.

Closes the item flagged open since 2026-07-31 and confirmed with Prof. Farzad on
2026-08-03 ("onu da tamamla" -- complete the tuned-value seed/sequence validation the
same way the untuned 1/160 reference already was, see _run_ca_seedcheck.ps1 /
_ca_seqclass_breakdown.py). Reports seed 0 (mc_ca_accel_80, already run 2026-07-31)
against seeds 1-3 (mc_ca_accel_80_s1/s2/s3, run today), plus both against the CV
headline (mc_dare) and the untuned CA reference (mc_ca_baseline) for context.
"""
import sys
import _score_multiclass as sm

sm.SEQS = list(sm.SEQS)  # all 7, pooled
REF_CV = "mc_dare"
REF_CA_UNTUNED = "mc_ca_baseline"
REF_CA_TUNED = "mc_ca_accel_80"

SEED_RUNS = [("mc_ca_accel_80_s1", "seed 1"), ("mc_ca_accel_80_s2", "seed 2"),
             ("mc_ca_accel_80_s3", "seed 3")]


def summarize(expn):
    res = sm.score_run(expn)
    return res["SUM_IDSw"], res["AVG"]["idf1"] * 100, res["AVG"]["mota"] * 100


def main():
    print(f"\n{'='*72}\nCA TUNED (1/80) REPEAT-SEED CHECK\n{'='*72}")
    cv_sw, cv_idf1, cv_mota = summarize(REF_CV)
    untuned_sw, untuned_idf1, untuned_mota = summarize(REF_CA_UNTUNED)
    tuned_sw, tuned_idf1, tuned_mota = summarize(REF_CA_TUNED)
    print(f"  {'CV headline (mc_dare)':28s} IDSw={cv_sw:4d}  IDF1={cv_idf1:5.1f}%  MOTA={cv_mota:5.1f}%")
    print(f"  {'CA untuned ref (1/160)':28s} IDSw={untuned_sw:4d}  IDF1={untuned_idf1:5.1f}%  MOTA={untuned_mota:5.1f}%")
    print(f"  {'CA tuned ref (1/80, seed 0)':28s} IDSw={tuned_sw:4d}  IDF1={tuned_idf1:5.1f}%  MOTA={tuned_mota:5.1f}%   "
          f"(d vs CV: {tuned_sw-cv_sw:+d} / {tuned_idf1-cv_idf1:+.1f}% / {tuned_mota-cv_mota:+.1f}%)")
    print(f"  {'-'*66}")

    all_ids_match = True
    for expn, label in SEED_RUNS:
        try:
            sw, idf1, mota = summarize(expn)
        except Exception as e:
            print(f"  {label:28s} FAILED ({e})")
            all_ids_match = False
            continue
        match = (sw == tuned_sw and abs(idf1 - tuned_idf1) < 1e-9 and abs(mota - tuned_mota) < 1e-9)
        all_ids_match = all_ids_match and match
        flag = "MATCH (byte-identical to seed 0)" if match else "MISMATCH"
        print(f"  {label:28s} IDSw={sw:4d}  IDF1={idf1:5.1f}%  MOTA={mota:5.1f}%   "
              f"(d vs CV: {sw-cv_sw:+d} / {idf1-cv_idf1:+.1f}% / {mota-cv_mota:+.1f}%)   [{flag}]")

    print(f"\n  Verdict: {'DETERMINISTIC -- all 4 seeds byte-identical.' if all_ids_match else 'NOT byte-identical -- investigate before adopting.'}")


if __name__ == "__main__":
    main()
