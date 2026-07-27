"""Per-class HOTA/DetA/AssA scorer for the canonical 5-class VisDrone-MOT val7 headline.

This is the HOTA analogue of ../_score_multiclass.py. The old _build_and_run_hota.py
collapsed every object to class=1 (the class-leak recipe, since invalidated). Here we
instead score each of the 5 canonical classes INDEPENDENTLY -- exactly mirroring
_score_multiclass.py's per-class split -- then macro-average HOTA/DetA/AssA across the
classes that have GT in val7 (the VisDrone-MOT / KITTI multi-class convention). No
cross-class ID matching is possible, so there is no class-leak.

Method, per class c in {1..5}:
  * filter GT rows to category==c, rewrite the class column to 1 (pedestrian) and
    zero_marked/conf to 1 so TrackEval scores them as a clean single-class problem;
  * filter tracker rows to category==c (TrackEval reads only frame,id,bbox,conf);
  * run TrackEval HOTA + CLEAR + Identity on the 7 val sequences (COMBINED_SEQ).

TrackEval's CLEAR MOTA/IDF1/IDSw are printed alongside HOTA as a CROSS-CHECK: they must
match the py-motmetrics per-class numbers in prof-farzad-briefing-2026-07-27.md
(ped IDSw 127/180, car 138/164, ...). If they do, the HOTA/AssA from the same run are
trustworthy.

Usage:  python _score_hota_mc.py            # scores mc_dare vs mc_bytetrack
        python _score_hota_mc.py <dareExpn> <btExpn>
"""
import os, sys, shutil, math

DARE = r"C:\Users\User\Desktop\projects\DARE-MOT"
TE   = os.path.join(DARE, "_trackeval")
WS   = os.path.join(TE, "_mc_hota_ws")            # rebuilt per class
MC_GT = (r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone_MOT_Format_MC"
         r"\VisDrone2019-MOT-val")
BENCH, SPLIT = "VisDroneMC", "val"
FOL = f"{BENCH}-{SPLIT}"
CAT_COL = 7                                        # 0-based category index in GT and preds
MC_NAMES = {1: "pedestrian", 2: "car", 3: "van", 4: "truck", 5: "bus"}
SEQS = ["uav0000086_00000_v", "uav0000117_02622_v", "uav0000137_00458_v",
        "uav0000182_00000_v", "uav0000268_05773_v", "uav0000305_00000_v",
        "uav0000339_00001_v"]


def _filter_gt(src, cls, dst):
    """GT rows of category==cls, rewritten class->1 conf->1. Returns n_rows."""
    n = 0
    with open(dst, "w", newline="") as fo:
        if os.path.exists(src):
            with open(src) as fi:
                for line in fi:
                    p = line.strip().split(",")
                    if len(p) <= CAT_COL:
                        continue
                    try:
                        if int(float(p[CAT_COL])) != cls:
                            continue
                    except ValueError:
                        continue
                    # frame,id,x,y,w,h,conf=1,class=1,vis(keep or 1)
                    vis = p[8] if len(p) > 8 else "1"
                    fo.write(f"{p[0]},{p[1]},{p[2]},{p[3]},{p[4]},{p[5]},1,1,{vis}\n")
                    n += 1
    return n


def _filter_pred(src, cls, dst):
    """Tracker rows of category==cls, class col rewritten to 1 so TrackEval's tracker
    loader (which validates the class column against 'pedestrian') accepts them.
    Score (conf) is preserved; bbox/id unchanged."""
    n = 0
    with open(dst, "w", newline="") as fo:
        if os.path.exists(src):
            with open(src) as fi:
                for line in fi:
                    p = line.strip().split(",")
                    if len(p) <= CAT_COL:
                        continue
                    try:
                        if int(float(p[CAT_COL])) != cls:
                            continue
                    except ValueError:
                        continue
                    # frame,id,x,y,w,h,score,class=1,-1,-1
                    fo.write(f"{p[0]},{p[1]},{p[2]},{p[3]},{p[4]},{p[5]},{p[6]},1,-1,-1\n")
                    n += 1
    return n


def build_ws(cls, trackers):
    if os.path.isdir(WS):
        shutil.rmtree(WS)
    gt_root = os.path.join(WS, "gt", "mot_challenge")
    seqmap_dir = os.path.join(gt_root, "seqmaps")
    os.makedirs(seqmap_dir, exist_ok=True)
    with open(os.path.join(seqmap_dir, f"{FOL}.txt"), "w", newline="") as f:
        f.write("name\n")
        for s in SEQS:
            f.write(s + "\n")
    n_gt_total = 0
    for s in SEQS:
        dst = os.path.join(gt_root, FOL, s)
        os.makedirs(os.path.join(dst, "gt"), exist_ok=True)
        n_gt_total += _filter_gt(os.path.join(MC_GT, s, "gt", "gt.txt"), cls,
                                 os.path.join(dst, "gt", "gt.txt"))
        shutil.copyfile(os.path.join(MC_GT, s, "seqinfo.ini"),
                        os.path.join(dst, "seqinfo.ini"))
    for tname, expn in trackers.items():
        data = os.path.join(WS, "trackers", "mot_challenge", FOL, tname, "data")
        os.makedirs(data, exist_ok=True)
        src = os.path.join(DARE, "YOLOX_outputs", expn, "track_results")
        for s in SEQS:
            _filter_pred(os.path.join(src, s + ".txt"), cls,
                         os.path.join(data, s + ".txt"))
    return n_gt_total


def run_te(trackers):
    import trackeval
    dcfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dcfg.update({
        'GT_FOLDER': os.path.join(WS, 'gt', 'mot_challenge'),
        'TRACKERS_FOLDER': os.path.join(WS, 'trackers', 'mot_challenge'),
        'BENCHMARK': BENCH, 'SPLIT_TO_EVAL': SPLIT,
        'TRACKERS_TO_EVAL': list(trackers.keys()),
        'CLASSES_TO_EVAL': ['pedestrian'],
        'PRINT_CONFIG': False, 'DO_PREPROC': False,   # keep all GT (no MOT distractor preproc)
    })
    ecfg = trackeval.Evaluator.get_default_eval_config()
    ecfg.update({'PRINT_RESULTS': False, 'PRINT_CONFIG': False, 'TIME_PROGRESS': False,
                 'DISPLAY_LESS_PROGRESS': True, 'USE_PARALLEL': False, 'OUTPUT_SUMMARY': False,
                 'OUTPUT_DETAILED': False, 'PLOT_CURVES': False})
    ev = trackeval.Evaluator(ecfg)
    ds = trackeval.datasets.MotChallenge2DBox(dcfg)
    mets = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    res, _ = ev.evaluate([ds], mets)
    r = res['MotChallenge2DBox']
    out = {}
    for t in trackers:
        c = r[t]['COMBINED_SEQ']['pedestrian']
        H, C, I = c['HOTA'], c['CLEAR'], c['Identity']
        out[t] = {
            'HOTA': float(H['HOTA'].mean()) * 100,
            'DetA': float(H['DetA'].mean()) * 100,
            'AssA': float(H['AssA'].mean()) * 100,
            'MOTA': float(C['MOTA']) * 100,
            'IDF1': float(I['IDF1']) * 100,
            'IDSw': int(C['IDSW']),
        }
    return out


def main():
    dare = sys.argv[1] if len(sys.argv) > 1 else "mc_dare"
    bt   = sys.argv[2] if len(sys.argv) > 2 else "mc_bytetrack"
    trackers = {"dare": dare, "bytetrack": bt}

    per_class = {}      # cls -> {tname -> metrics}
    for cls, name in MC_NAMES.items():
        n_gt = build_ws(cls, trackers)
        if n_gt == 0:
            per_class[cls] = None
            continue
        per_class[cls] = run_te(trackers)

    scored = {c: v for c, v in per_class.items() if v is not None}

    def row(tname, m):
        return (f"{m['HOTA']:6.1f} {m['DetA']:6.1f} {m['AssA']:6.1f}  |  "
                f"{m['MOTA']:6.1f} {m['IDF1']:6.1f} {m['IDSw']:5d}")

    print("\n" + "=" * 78)
    print("  PER-CLASS HOTA  (VisDrone-MOT val7, 5-class canonical; TrackEval)")
    print("  cols:  HOTA  DetA  AssA  |  MOTA  IDF1  IDSw   (last three = CLEAR/Identity x-check)")
    print("=" * 78)
    for tname in ("dare", "bytetrack"):
        print(f"\n--- {tname}  ({trackers[tname]}) ---")
        print(f"{'class':11s} {'HOTA':>6s} {'DetA':>6s} {'AssA':>6s}     {'MOTA':>6s} {'IDF1':>6s} {'IDSw':>5s}")
        for cls, name in MC_NAMES.items():
            v = per_class[cls]
            if v is None:
                print(f"{name:11s}  (no GT in val7)")
            else:
                print(f"{name:11s} {row(tname, v[tname])}")
        # macro average over classes with GT
        mac = {k: sum(scored[c][tname][k] for c in scored) / len(scored)
               for k in ('HOTA', 'DetA', 'AssA', 'MOTA', 'IDF1')}
        mac_idsw = sum(scored[c][tname]['IDSw'] for c in scored)
        print(f"{'-'*11}")
        print(f"{'MACRO avg':11s} {mac['HOTA']:6.1f} {mac['DetA']:6.1f} {mac['AssA']:6.1f}     "
              f"{mac['MOTA']:6.1f} {mac['IDF1']:6.1f} {mac_idsw:5d}")

    # delta (dare - bytetrack), macro
    print("\n" + "=" * 78)
    print("  DELTA  dare - bytetrack  (macro-averaged; + = DARE better for HOTA/DetA/AssA)")
    print("=" * 78)
    print(f"{'class':11s} {'dHOTA':>6s} {'dDetA':>6s} {'dAssA':>6s}")
    for cls, name in MC_NAMES.items():
        v = per_class[cls]
        if v is None:
            print(f"{name:11s}  --")
            continue
        d = v['dare']; b = v['bytetrack']
        print(f"{name:11s} {d['HOTA']-b['HOTA']:+6.1f} {d['DetA']-b['DetA']:+6.1f} {d['AssA']-b['AssA']:+6.1f}")
    dmac = {k: (sum(scored[c]['dare'][k] for c in scored) - sum(scored[c]['bytetrack'][k] for c in scored)) / len(scored)
            for k in ('HOTA', 'DetA', 'AssA')}
    print(f"{'-'*11}")
    print(f"{'MACRO':11s} {dmac['HOTA']:+6.1f} {dmac['DetA']:+6.1f} {dmac['AssA']:+6.1f}")
    print("\nNote: DetA depends only on detection (identical for both trackers) -> dDetA ~ 0 is")
    print("expected and is a sanity check. All HOTA movement should land in AssA (association).")


if __name__ == "__main__":
    main()
