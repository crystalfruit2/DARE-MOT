"""Build a MOTChallenge-format TrackEval workspace for the all-class VisDrone-MOT
val headline comparison, then run HOTA + CLEAR + Identity.

Trackers scored:
  bytetrack   <- YOLOX_outputs/bt_l0_s0_r1        (lambda=0 appearance-off bar)
  dare_iou95  <- YOLOX_outputs/ftg2500_lockoff_iou95 (DARE_IOU_GATE=0.95 headline)

GT is already all-class collapsed to class=1/conf=1 (114132 rows), so it is copied
verbatim -- this reproduces the exact all-class recipe of _decompose_gap.py.
CLEAR+Identity are included so TrackEval's MOTA/IDF1 cross-check the decomposition
(expected ~54.5/65.0 for bytetrack, ~54.6/66.0 for dare_iou95). If those match, the
HOTA/AssA numbers from the same run are trustworthy.
"""
import os, shutil, csv

DARE = r"C:\Users\User\Desktop\projects\DARE-MOT"
GT_SRC = r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone_MOT_Format\VisDrone2019-MOT-val"
TE = os.path.join(DARE, "_trackeval")
BENCH, SPLIT = "VisDroneAll", "val"
FOL = f"{BENCH}-{SPLIT}"

SEQS = ["uav0000086_00000_v","uav0000117_02622_v","uav0000137_00458_v",
        "uav0000182_00000_v","uav0000268_05773_v","uav0000305_00000_v",
        "uav0000339_00001_v"]
TRACKERS = {"bytetrack": "bt_l0_s0_r1", "dare_iou95": "ftg2500_lockoff_iou95"}

def build():
    gt_root = os.path.join(TE, "gt", "mot_challenge")
    seqmap_dir = os.path.join(gt_root, "seqmaps")
    os.makedirs(seqmap_dir, exist_ok=True)
    # seqmap
    with open(os.path.join(seqmap_dir, f"{FOL}.txt"), "w", newline="") as f:
        f.write("name\n")
        for s in SEQS:
            f.write(s + "\n")
    # gt + seqinfo per seq
    for s in SEQS:
        dst = os.path.join(gt_root, FOL, s)
        os.makedirs(os.path.join(dst, "gt"), exist_ok=True)
        shutil.copyfile(os.path.join(GT_SRC, s, "gt", "gt.txt"),
                        os.path.join(dst, "gt", "gt.txt"))
        shutil.copyfile(os.path.join(GT_SRC, s, "seqinfo.ini"),
                        os.path.join(dst, "seqinfo.ini"))
    # tracker predictions
    for tname, expn in TRACKERS.items():
        data = os.path.join(TE, "trackers", "mot_challenge", FOL, tname, "data")
        os.makedirs(data, exist_ok=True)
        src = os.path.join(DARE, "YOLOX_outputs", expn, "track_results")
        for s in SEQS:
            shutil.copyfile(os.path.join(src, s + ".txt"),
                            os.path.join(data, s + ".txt"))
    print("built workspace at", os.path.join(gt_root, FOL))

def run():
    import trackeval
    dcfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dcfg.update({
        'GT_FOLDER': os.path.join(TE, 'gt', 'mot_challenge'),
        'TRACKERS_FOLDER': os.path.join(TE, 'trackers', 'mot_challenge'),
        'BENCHMARK': BENCH, 'SPLIT_TO_EVAL': SPLIT,
        'TRACKERS_TO_EVAL': list(TRACKERS.keys()),
        'CLASSES_TO_EVAL': ['pedestrian'],   # everything is class 1 -> all-class
        'PRINT_CONFIG': False,
    })
    ecfg = trackeval.Evaluator.get_default_eval_config()
    ecfg.update({'PRINT_RESULTS': False, 'PRINT_CONFIG': False,
                 'DISPLAY_LESS_PROGRESS': True, 'USE_PARALLEL': False})
    ev = trackeval.Evaluator(ecfg)
    ds = trackeval.datasets.MotChallenge2DBox(dcfg)
    mets = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    res, _ = ev.evaluate([ds], mets)
    r = res['MotChallenge2DBox']
    print("\n==================  ALL-CLASS  VisDrone-MOT val (7 seq)  ==================")
    hdr = f"{'tracker':12s} {'HOTA':>7s} {'DetA':>7s} {'AssA':>7s} {'MOTA':>7s} {'IDF1':>7s} {'IDSw':>6s} {'FP':>7s} {'FN':>7s}"
    print(hdr); print("-"*len(hdr))
    for t in TRACKERS:
        c = r[t]['COMBINED_SEQ']['pedestrian']
        H, C, I = c['HOTA'], c['CLEAR'], c['Identity']
        print(f"{t:12s} {H['HOTA'].mean()*100:7.1f} {H['DetA'].mean()*100:7.1f} "
              f"{H['AssA'].mean()*100:7.1f} {C['MOTA']*100:7.1f} {I['IDF1']*100:7.1f} "
              f"{int(C['IDSW']):6d} {int(C['CLR_FP']):7d} {int(C['CLR_FN']):7d}")
    print("==========================================================================")

if __name__ == "__main__":
    build()
    run()
