"""Score every Path-1 arm + the 2026-09-15 night sensitivity arms under the headline protocol.

Same pipeline as _score_path1_2026-09-15.py (on_score vote -> official ignore filter -> pooled micro),
plus per-sequence IDSw/IDF1/MOTA and pooled FP/FN. Writes voted dirs to _scratch/path1_score (read by
_score_hota_path1_2026-09-15.py) and _scratch/path1_score/night_summary.json.
"""
import importlib.util
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("sc", os.path.join(HERE, "_score_cmcfix_2026-09-10.py"))
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)
sm, so = sc.sm, sc.so
WT, MAIN = sc.WT, sc.MAIN
sc.OUT = os.path.join(HERE, "_scratch", "path1_score")
ALL_SEQS = list(sm.SEQS)
ARMS = [
    ("p1ref_bt", WT, "ByteTrack"),
    ("p1_ocsort", MAIN, "OC-SORT"),
    ("p1_botsort", MAIN, "BoT-SORT-ReID"),
    ("p1_botsort_noreid", MAIN, "BoT-SORT no ReID (GMC only)"),
    ("p1_botsort_m08", MAIN, "BoT-SORT-ReID match 0.8"),
    ("p1_botsort_scalewarp", MAIN, "BoT-SORT-ReID + scale warp"),
    ("p1_deepocsort", MAIN, "Deep OC-SORT"),
    ("p1_deepocsort_mh1", MAIN, "Deep OC-SORT min_hits 1"),
    ("p1ref_bt_cmcscale_j", WT, "ByteTrack + scale CMC (ds2)"),
    ("lat10_bt_cmcscale_j_ds4", WT, "ByteTrack + scale CMC (ds4)"),
    ("lat10_bt_cmcscale_j_ds8", WT, "ByteTrack + scale CMC (ds8)"),
    ("p1ref_dare_cv_cmcscale_j", WT, "CV-DARE + scale CMC"),
    # evening: parity (= BoT-SORT's warp recipe on the xyah state) vs scale on the REPORTED detector D3
    ("mc10_bt_cmcparity_j", WT, "ByteTrack + parity CMC"),
    ("mc10_dare_cv_cmcparity_j", WT, "CV-DARE + parity CMC"),
]


def fpfn(r):
    cls = [v for k, v in r.items() if isinstance(k, int) and v is not None]
    return sum(v["fp"] for v in cls), sum(v["fn"] for v in cls)


if __name__ == "__main__":
    res = {"raw": {}, "head": {}, "head_seq": {}, "per_class": {}}
    present = []
    for expn, root, label in ARMS:
        d = os.path.join(root, expn, "track_results")
        if not all(os.path.exists(os.path.join(d, s + ".txt")) for s in ALL_SEQS):
            print(f"{expn:28s} INCOMPLETE -- skipped")
            continue
        present.append((expn, root, label))
        sm.MC_GT, sm.SEQS = sc.RAW_GT, ALL_SEQS
        res["raw"][expn] = sm.score_run(d)
    off_gt = so.prepare_gt()
    for expn, root, label in present:
        sm.MC_GT, sm.SEQS = off_gt, ALL_SEQS
        vdir = sc.voted_filtered(expn, root)
        r = sm.score_run(vdir)
        res["head"][expn] = r
        res["per_class"][expn] = {sm.MC_NAMES[c]: r[c] for c in sm.MC_NAMES if r.get(c)}
        res["head_seq"][expn] = {}
        for s in ALL_SEQS:
            sm.SEQS = [s]
            res["head_seq"][expn][s] = sm.score_run(vdir)
        sm.SEQS = ALL_SEQS

    print("\nNight scoring, D3, val7, seed 0, pooled (micro), HEADLINE protocol")
    print(f"{'arm':32s} {'IDSw':>5s} {'IDF1':>6s} {'MOTA':>6s} {'FP':>6s} {'FN':>6s}   raw IDSw/IDF1/MOTA")
    for expn, root, label in present:
        h, rw = res["head"][expn], res["raw"][expn]
        fp, fn = fpfn(h)
        print(f"{label:32s} {h['SUM_IDSw']:5d} {100 * h['MICRO']['idf1']:6.2f} {100 * h['MICRO']['mota']:6.2f} {fp:6d} {fn:6d}"
              f"   {rw['SUM_IDSw']}/{100 * rw['MICRO']['idf1']:.2f}/{100 * rw['MICRO']['mota']:.2f}")
    print("\nper sequence IDSw / IDF1 / MOTA")
    print(f"{'arm':32s}" + "".join(f"{s[3:10]:>18s}" for s in ALL_SEQS))
    for expn, root, label in present:
        print(f"{label:32s}" + "".join(
            f"{res['head_seq'][expn][s]['SUM_IDSw']:3d}/{100 * res['head_seq'][expn][s]['MICRO']['idf1']:5.1f}/{100 * res['head_seq'][expn][s]['MICRO']['mota']:5.1f}".rjust(18)
            for s in ALL_SEQS))
    print("\nper class (headline): IDSw / IDF1 / MOTA / FP / FN")
    for expn, root, label in present:
        pc = res["per_class"][expn]
        print(f"  {label:30s} " + " | ".join(f"{k[:4]} {v['idsw']}/{100 * v['idf1']:.1f}/{100 * v['mota']:.1f}/{v['fp']}/{v['fn']}" for k, v in pc.items()))
    dst = os.path.join(sc.OUT, "night_summary.json")
    json.dump(res, open(dst, "w"), indent=1, default=str)
    print(f"\nwrote {dst}")
