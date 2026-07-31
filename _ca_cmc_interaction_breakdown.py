"""CA x CMC interaction, stratified (2026-07-31).

Pooled result: CMC's net damage is +375 IDSw under CV (282->657) but only +148 IDSw under CA
(253->401) -- the CA-vs-CV gap WIDENS under CMC rather than shrinking, which is the opposite of
what was being checked for (whether CA's gain survives CMC). This is a real, unhypothesized
finding, or it's another single-sequence aggregate artifact -- this project's own established
pattern (the CMC scale-inversion story, the MOTA class-averaging artifact). Same rigor as those
checks: break down by sequence and by class before trusting the pooled number.
"""
import _score_multiclass as sm

ALL_SEQS = list(sm.SEQS)
PAIRS = {
    "no_CMC": ("mc_ca_baseline", "mc_dare"),
    "with_CMC": ("mc_ca_cmc", "mc_dare_cmc"),
}


def per_seq_delta(ca_exp, cv_exp, seqs):
    sm.SEQS = seqs
    ca = sm.score_run(ca_exp)
    cv = sm.score_run(cv_exp)
    return ca["SUM_IDSw"] - cv["SUM_IDSw"], (ca["AVG"]["mota"] - cv["AVG"]["mota"]) * 100


print("=" * 78)
print("POOLED (sanity check vs the known pooled numbers)")
print("=" * 78)
for tag, (ca_exp, cv_exp) in PAIRS.items():
    sm.SEQS = ALL_SEQS
    dsw, dmota = per_seq_delta(ca_exp, cv_exp, ALL_SEQS)
    print(f"  {tag:12s} CA-CV delta: IDSw={dsw:+5d}  AVG-MOTA={dmota:+.1f}%")

print("\n" + "=" * 78)
print("PER-SEQUENCE: (CA-CV delta under CMC) minus (CA-CV delta without CMC)")
print("  Positive = CMC makes CA's advantage OVER CV bigger in this sequence")
print("  Negative = CMC makes CA's advantage OVER CV smaller (or reverses it)")
print("=" * 78)
widening_total = 0
per_seq_widening = {}
for seq in ALL_SEQS:
    dsw_no_cmc, dmota_no_cmc = per_seq_delta("mc_ca_baseline", "mc_dare", [seq])
    dsw_cmc, dmota_cmc = per_seq_delta("mc_ca_cmc", "mc_dare_cmc", [seq])
    widening = dsw_no_cmc - dsw_cmc  # more negative dsw_cmc (bigger CA win) = more negative here = "widening"
    per_seq_widening[seq] = widening
    print(f"  {seq:24s} no-CMC d={dsw_no_cmc:+5d}  +CMC d={dsw_cmc:+5d}  "
          f"widening={widening:+5d}  (MOTA no-CMC={dmota_no_cmc:+.1f}% +CMC={dmota_cmc:+.1f}%)")

total_widening = sum(per_seq_widening.values())
worst = max(per_seq_widening, key=lambda s: per_seq_widening[s])  # least (or negatively) widening
best = min(per_seq_widening, key=lambda s: per_seq_widening[s])   # most widening (most negative)
print(f"\n  Sum of per-seq widening: {total_widening:+d} (sanity check vs pooled: "
      f"no-CMC dsw - with-CMC dsw = {-29 - (-256)})")
print(f"  Most-widening sequence: {best} ({per_seq_widening[best]:+d})")
print(f"  Least-widening (or narrowing) sequence: {worst} ({per_seq_widening[worst]:+d})")
if total_widening != 0:
    share = per_seq_widening[best] / total_widening
    print(f"  Most-widening sequence's share of total widening: {share*100:.0f}%")
