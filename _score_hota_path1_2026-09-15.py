"""HOTA / DetA / AssA for the Path-1 arms under the HEADLINE protocol (2026-09-15).

Reuses _trackeval/_score_hota_mc.py (per-class TrackEval HOTA + CLEAR + Identity, macro-averaged over the
classes with GT) but feeds it the SAME inputs as the headline CLEAR numbers:
  * tracker rows = the on_score-voted + official-ignore-filtered files written by the Path-1 scorers
    (_scratch/path1_score/<expn>__on_score_official/<seq>.txt);
  * GT = the official-ignore-filtered MC GT (_scratch/official/_gt/<seq>/gt/gt.txt).
TrackEval's CLEAR IDSw (summed over classes) is printed next to py-motmetrics' headline SUM_IDSw from
path1_summary.json as a cross-check: they should agree (small differences can come from matching ties).
Usage: python _score_hota_path1_2026-09-15.py      (after the night scorer has written the voted dirs)
"""
import importlib.util
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("hm", os.path.join(HERE, "_trackeval", "_score_hota_mc.py"))
hm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hm)

OFF_GT = os.path.join(HERE, "_scratch", "official", "_gt", "{seq}", "gt", "gt.txt")
VOTED = os.path.join(HERE, "_scratch", "path1_score", "{expn}__on_score_official", "{seq}.txt")
ARMS = [
    ("ByteTrack", "p1ref_bt"),
    ("OC-SORT", "p1_ocsort"),
    ("BoT-SORT-ReID", "p1_botsort"),
    ("BoT-SORT_noReID", "p1_botsort_noreid"),
    ("BoT-SORT_match0.8", "p1_botsort_m08"),
    ("BoT-SORT_scalewarp", "p1_botsort_scalewarp"),
    ("DeepOC-SORT", "p1_deepocsort"),
    ("DeepOC-SORT_minhits1", "p1_deepocsort_mh1"),
    ("BT+scaleCMC", "p1ref_bt_cmcscale_j"),
    ("CV-DARE+scaleCMC", "p1ref_dare_cv_cmcscale_j"),
    ("BT+scaleCMC_ds4", "lat10_bt_cmcscale_j_ds4"),
    ("BT+scaleCMC_ds8", "lat10_bt_cmcscale_j_ds8"),
    ("BT+parityCMC", "mc10_bt_cmcparity_j"),
    ("CV-DARE+parityCMC", "mc10_dare_cv_cmcparity_j"),
]


def build_ws(cls, trackers):
    if os.path.isdir(hm.WS):
        shutil.rmtree(hm.WS)
    gt_root = os.path.join(hm.WS, "gt", "mot_challenge")
    os.makedirs(os.path.join(gt_root, "seqmaps"), exist_ok=True)
    with open(os.path.join(gt_root, "seqmaps", f"{hm.FOL}.txt"), "w", newline="") as f:
        f.write("name\n" + "".join(s + "\n" for s in hm.SEQS))
    n_gt = 0
    for s in hm.SEQS:
        dst = os.path.join(gt_root, hm.FOL, s)
        os.makedirs(os.path.join(dst, "gt"), exist_ok=True)
        n_gt += hm._filter_gt(OFF_GT.format(seq=s), cls, os.path.join(dst, "gt", "gt.txt"))
        shutil.copyfile(os.path.join(hm.MC_GT, s, "seqinfo.ini"), os.path.join(dst, "seqinfo.ini"))
    for tname, expn in trackers.items():
        data = os.path.join(hm.WS, "trackers", "mot_challenge", hm.FOL, tname, "data")
        os.makedirs(data, exist_ok=True)
        for s in hm.SEQS:
            hm._filter_pred(VOTED.format(expn=expn, seq=s), cls, os.path.join(data, s + ".txt"))
    return n_gt


def main():
    present = {t: e for t, e in ARMS
               if all(os.path.exists(VOTED.format(expn=e, seq=s)) for s in hm.SEQS)}
    missing = [e for t, e in ARMS if t not in present]
    if missing:
        print("skipped (no voted dirs yet):", ", ".join(missing))
    per_class = {}
    for cls, name in hm.MC_NAMES.items():
        if build_ws(cls, present) == 0:
            per_class[cls] = None
            continue
        print(f"TrackEval class {name} ...", flush=True)
        per_class[cls] = hm.run_te(present)
    scored = {c: v for c, v in per_class.items() if v is not None}

    summ_path = os.path.join(HERE, "_scratch", "path1_score", "path1_summary.json")
    head = {}
    for p in (summ_path, os.path.join(HERE, "_scratch", "path1_score", "night_summary.json")):
        if os.path.exists(p):
            head.update(json.load(open(p)).get("head", {}))

    out = {}
    print("\nHEADLINE-PROTOCOL HOTA (TrackEval, per class, macro over classes with GT)")
    print(f"{'arm':22s} {'HOTA':>6s} {'DetA':>6s} {'AssA':>6s} | {'MOTA*':>6s} {'IDF1*':>6s} {'IDSw_TE':>7s} {'IDSw_mm':>7s}")
    for t, e in present.items():
        mac = {k: sum(scored[c][t][k] for c in scored) / len(scored) for k in ("HOTA", "DetA", "AssA", "MOTA", "IDF1")}
        idsw = sum(scored[c][t]["IDSw"] for c in scored)
        mm = head.get(e, {}).get("SUM_IDSw", "?")
        out[e] = {"macro": mac, "IDSw_TE": idsw, "IDSw_motmetrics": mm,
                  "per_class": {hm.MC_NAMES[c]: scored[c][t] for c in scored}}
        print(f"{t:22s} {mac['HOTA']:6.2f} {mac['DetA']:6.2f} {mac['AssA']:6.2f} | {mac['MOTA']:6.2f} {mac['IDF1']:6.2f} {idsw:7d} {str(mm):>7s}")
    print("(* MOTA/IDF1 here are class-MACRO TrackEval values; the headline table uses pooled micro.)")
    print("\nPer-class HOTA / AssA")
    for t, e in present.items():
        print(f"  {t:22s} " + "  ".join(f"{hm.MC_NAMES[c][:4]} {scored[c][t]['HOTA']:5.1f}/{scored[c][t]['AssA']:5.1f}" for c in scored))
    dst = os.path.join(HERE, "_scratch", "path1_score", "hota_summary.json")
    json.dump(out, open(dst, "w"), indent=1)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
