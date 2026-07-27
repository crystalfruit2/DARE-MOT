"""Move 0b aggregator: turn the multi-seed noise-band runs into mean+/-std deltas.

Reads the per-seed run dirs produced by _run_mc_noiseband.ps1:
    YOLOX_outputs/mc_nb_bt_s{seed}      (ByteTrack baseline, seed)
    YOLOX_outputs/mc_nb_dare_s{seed}    (DARE headline, seed)
    YOLOX_outputs/mc_nb_dare_s0_rep     (DARE seed 0, second run -> determinism check)

For each seed it scores both arms with _score_multiclass.score_run (which now emits BOTH the
class-average 'AVG' (macro) and the count-pooled 'MICRO'), forms the DARE-minus-ByteTrack delta,
then reports per-seed deltas plus mean+/-std across seeds. The whole point: put an error band on
the -80 IDSw / +1.4 IDF1 / ~0 MOTA headline so the truck/bus single-run wobble can be told apart
from a real effect.

Usage:  python _score_noiseband.py [seed ...]     (default seeds: 0 1 2 3 4)
"""
import os
import sys
import statistics as st

from _score_multiclass import score_run, res_path, SEQS

HERE = os.path.dirname(os.path.abspath(__file__))


def _metrics(res):
    """(idsw_sum, idf1_macro, mota_macro, idf1_micro, mota_micro) in %."""
    return (
        res["SUM_IDSw"],
        res["AVG"]["idf1"] * 100, res["AVG"]["mota"] * 100,
        res["MICRO"]["idf1"] * 100, res["MICRO"]["mota"] * 100,
    )


def _determinism_check():
    a, b = "mc_nb_dare_s0", "mc_nb_dare_s0_rep"
    if not os.path.isdir(os.path.join(HERE, "YOLOX_outputs", b)):
        print("  (no rep run found -- skipping determinism check)")
        return
    diff = 0
    for seq in SEQS:
        pa, pb = res_path(a, seq), res_path(b, seq)
        la = open(pa).read().splitlines() if os.path.exists(pa) else []
        lb = open(pb).read().splitlines() if os.path.exists(pb) else []
        diff += sum(1 for x, y in zip(la, lb) if x != y) + abs(len(la) - len(lb))
    verdict = "DETERMINISTIC (byte-identical)" if diff == 0 else f"NON-DETERMINISTIC ({diff} differing lines)"
    print(f"  seed-0 reproducibility: {verdict}")


def main():
    seeds = [int(s) for s in sys.argv[1:]] or [0, 1, 2, 3, 4]
    cols = ["dIDSw", "dIDF1(macro)", "dMOTA(macro)", "dIDF1(micro)", "dMOTA(micro)"]
    rows = {c: [] for c in cols}

    print(f"\n=== Noise band over seeds {seeds}  (DARE - ByteTrack) ===")
    print(f"{'seed':>4s} " + " ".join(f"{c:>13s}" for c in cols))
    for s in seeds:
        bt = score_run(f"mc_nb_bt_s{s}")
        dr = score_run(f"mc_nb_dare_s{s}")
        mb, md = _metrics(bt), _metrics(dr)
        d = [md[i] - mb[i] for i in range(5)]
        for c, v in zip(cols, d):
            rows[c].append(v)
        print(f"{s:>4d} {d[0]:>+13.0f} {d[1]:>+13.2f} {d[2]:>+13.2f} {d[3]:>+13.2f} {d[4]:>+13.2f}")

    print("-" * 74)
    means = [st.mean(rows[c]) for c in cols]
    stds  = [st.pstdev(rows[c]) if len(rows[c]) > 1 else 0.0 for c in cols]
    print(f"{'mean':>4s} " + " ".join(f"{m:>+13.2f}" for m in means))
    print(f"{'std':>4s} " + " ".join(f"{sd:>13.2f}" for sd in stds))
    print("\nRead: a delta whose |mean| >> std is a real effect; |mean| within ~1 std of 0 is noise.")
    print("\n=== Determinism check ===")
    _determinism_check()


if __name__ == "__main__":
    main()
