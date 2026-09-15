"""Score the Path-1 controlled baseline comparison (_run_path1_2026-09-15.ps1) -- 2026-09-15.

Same headline protocol as every paper number: online score-weighted class voting (on_score) ->
official VisDrone ignored-region filter -> pooled (micro) CLEAR/IDF1 over val7. Also prints the raw
protocol (no vote, no filter), pooled FP/FN, and per-sequence headline IDSw/IDF1/MOTA, so a
single-sequence artifact is visible before any row is quoted. Reuses _score_cmcfix_2026-09-10.py's
pipeline verbatim (per-sequence numbers = the same score_run restricted to one sequence).
Usage: python _score_path1_2026-09-15.py
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
ARMS = [
    ("p1ref_bt", WT, "ByteTrack"),
    ("p1_ocsort", MAIN, "OC-SORT"),
    ("p1_botsort", MAIN, "BoT-SORT-ReID (shared OSNet)"),
    ("p1_botsort_noreid", MAIN, "BoT-SORT (GMC only, no ReID)"),
    ("p1_deepocsort", MAIN, "Deep OC-SORT (shared OSNet)"),
    ("p1ref_bt_cmcscale_j", WT, "ByteTrack + scale CMC"),
    ("p1ref_dare_cv_cmcscale_j", WT, "CV-DARE + scale CMC"),
]
ALL_SEQS = list(sm.SEQS)


def fpfn(r):
    cls = [v for k, v in r.items() if isinstance(k, int) and v is not None]
    return sum(v["fp"] for v in cls), sum(v["fn"] for v in cls)


if __name__ == "__main__":
    res = {"raw": {}, "head": {}, "head_seq": {}}
    present = []
    for expn, root, label in ARMS:
        d = os.path.join(root, expn, "track_results")
        if not all(os.path.exists(os.path.join(d, s + ".txt")) for s in ALL_SEQS):
            print(f"{expn:26s} INCOMPLETE -- skipped")
            continue
        present.append((expn, root, label))
        sm.MC_GT, sm.SEQS = sc.RAW_GT, ALL_SEQS
        res["raw"][expn] = sm.score_run(d)
    off_gt = so.prepare_gt()
    for expn, root, label in present:
        sm.MC_GT, sm.SEQS = off_gt, ALL_SEQS
        vdir = sc.voted_filtered(expn, root)
        res["head"][expn] = sm.score_run(vdir)
        res["head_seq"][expn] = {}
        for s in ALL_SEQS:
            sm.SEQS = [s]
            res["head_seq"][expn][s] = sm.score_run(vdir)
        sm.SEQS = ALL_SEQS

    print("\nPath 1, detector D3 (clean 10-class, DARE_MAX_CLASS=4), val7, seed 0, pooled (micro)")
    print(f"{'arm':32s} {'RAW':38s}  HEADLINE (on_score vote + official)")
    for expn, root, label in present:
        fp, fn = fpfn(res["head"][expn])
        print(f"{label:32s} {sc.fmt(res['raw'][expn])}  {sc.fmt(res['head'][expn])} | FP {fp:5d} | FN {fn:5d}")

    if present:
        print("\nHEADLINE per sequence: IDSw / IDF1 / MOTA (micro)")
        print(f"{'arm':32s}" + "".join(f"{s[3:10]:>20s}" for s in ALL_SEQS))
        for expn, root, label in present:
            cells = []
            for s in ALL_SEQS:
                r = res["head_seq"][expn][s]
                cells.append(f"{r['SUM_IDSw']:4d}/{100 * r['MICRO']['idf1']:5.1f}/{100 * r['MICRO']['mota']:5.1f}")
            print(f"{label:32s}" + "".join(f"{c:>20s}" for c in cells))
    os.makedirs(sc.OUT, exist_ok=True)
    with open(os.path.join(sc.OUT, "path1_summary.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(f"\nwrote {os.path.join(sc.OUT, 'path1_summary.json')}")
