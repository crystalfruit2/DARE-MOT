"""Score the 2026-07-31 CA follow-up runs (seed check + accel-noise sweep) against the
CV headline (mc_dare) and the reference CA baseline (mc_ca_baseline, accel=1/160, seed=0).
"""
import sys
import _score_multiclass as sm

sm.SEQS = list(sm.SEQS)  # all 7, pooled
REF_CV = "mc_dare"
REF_CA = "mc_ca_baseline"

SEED_RUNS = [("mc_ca_baseline_s1", "seed 1"), ("mc_ca_baseline_s2", "seed 2"),
             ("mc_ca_baseline_s3", "seed 3")]
ACCEL_RUNS = [("mc_ca_accel_640", "1/640"), ("mc_ca_accel_320", "1/320"),
              ("mc_ca_baseline", "1/160 (reference)"), ("mc_ca_accel_80", "1/80"),
              ("mc_ca_accel_40", "1/40"), ("mc_ca_accel_20", "1/20")]


def summarize(expn):
    res = sm.score_run(expn)
    return res["SUM_IDSw"], res["AVG"]["idf1"] * 100, res["AVG"]["mota"] * 100


def report(runs, title):
    print(f"\n{'='*72}\n{title}\n{'='*72}")
    cv_sw, cv_idf1, cv_mota = summarize(REF_CV)
    ca_sw, ca_idf1, ca_mota = summarize(REF_CA)
    print(f"  {'CV headline (mc_dare)':28s} IDSw={cv_sw:4d}  IDF1={cv_idf1:5.1f}%  MOTA={cv_mota:5.1f}%")
    print(f"  {'CA reference (seed 0)':28s} IDSw={ca_sw:4d}  IDF1={ca_idf1:5.1f}%  MOTA={ca_mota:5.1f}%   "
          f"(d vs CV: {ca_sw-cv_sw:+d} / {ca_idf1-cv_idf1:+.1f}% / {ca_mota-cv_mota:+.1f}%)")
    print(f"  {'-'*66}")
    for expn, label in runs:
        try:
            sw, idf1, mota = summarize(expn)
        except Exception as e:
            print(f"  {label:28s} FAILED ({e})")
            continue
        print(f"  {label:28s} IDSw={sw:4d}  IDF1={idf1:5.1f}%  MOTA={mota:5.1f}%   "
              f"(d vs CV: {sw-cv_sw:+d} / {idf1-cv_idf1:+.1f}% / {mota-cv_mota:+.1f}%)   "
              f"(d vs CA-ref: {sw-ca_sw:+d} / {idf1-ca_idf1:+.1f}% / {mota-ca_mota:+.1f}%)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    if mode in ("seed", "both"):
        report(SEED_RUNS, "REPEAT-SEED CHECK (CA baseline, seeds 1-3 vs seed-0 reference)")
    if mode in ("accel", "both"):
        report(ACCEL_RUNS, "ACCEL-NOISE HYPERPARAMETER SWEEP")
