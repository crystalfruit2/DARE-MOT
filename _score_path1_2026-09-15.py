"""Score the Path-1 controlled baseline comparison (_run_path1_2026-09-15.ps1) -- 2026-09-15.

Same headline protocol as every paper number: online score-weighted class voting (on_score) ->
official VisDrone ignored-region filter -> pooled (micro) CLEAR/IDF1 over val7. Also prints the raw
protocol (no vote, no filter) and per-sequence IDSw for the headline, so a single-sequence artifact
is visible before any row is quoted. Reuses _score_cmcfix_2026-09-10.py's pipeline verbatim.
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
    ("p1_deepocsort", MAIN, "Deep OC-SORT (shared OSNet)"),
    ("p1ref_bt_cmcscale_j", WT, "ByteTrack + scale CMC"),
    ("p1ref_dare_cv_cmcscale_j", WT, "CV-DARE + scale CMC"),
]


def extra(r):
    mi = r["MICRO"]
    keys = [("num_false_positives", "FP"), ("num_misses", "FN"), ("num_switches", "IDSw_mm")]
    return " | ".join(f"{lab} {int(mi[k])}" for k, lab in keys if k in mi)


if __name__ == "__main__":
    res = {"raw": {}, "head": {}}
    present = []
    for expn, root, label in ARMS:
        d = os.path.join(root, expn, "track_results")
        if not all(os.path.exists(os.path.join(d, s + ".txt")) for s in sm.SEQS):
            print(f"{expn:26s} INCOMPLETE -- skipped")
            continue
        present.append((expn, root, label))
        sm.MC_GT = sc.RAW_GT
        res["raw"][expn] = sm.score_run(d)
    off_gt = so.prepare_gt()
    for expn, root, label in present:
        sm.MC_GT = off_gt
        res["head"][expn] = sm.score_run(sc.voted_filtered(expn, root))

    print("\nPath 1, detector D3 (clean 10-class, DARE_MAX_CLASS=4), val7, seed 0, pooled (micro)")
    print(f"{'arm':32s} {'RAW':38s}  HEADLINE (on_score vote + official)")
    for expn, root, label in present:
        print(f"{label:32s} {sc.fmt(res['raw'][expn])}  {sc.fmt(res['head'][expn])}   {extra(res['head'][expn])}")

    seq_keys = [k for k in res["head"][present[0][0]] if k in sm.SEQS] if present else []
    if seq_keys:
        print("\nHEADLINE IDSw per sequence")
        print(f"{'arm':32s}" + "".join(f"{s[:10]:>12s}" for s in seq_keys))
        for expn, root, label in present:
            r = res["head"][expn]
            print(f"{label:32s}" + "".join(f"{int(r[s].get('num_switches', r[s].get('IDSw', -1))):>12d}"
                                           if isinstance(r[s], dict) else f"{'?':>12s}" for s in seq_keys))
    os.makedirs(sc.OUT, exist_ok=True)
    with open(os.path.join(sc.OUT, "path1_summary.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(f"\nwrote {os.path.join(sc.OUT, 'path1_summary.json')}")
