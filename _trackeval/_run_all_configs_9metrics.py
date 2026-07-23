"""Score all six discussed configs on 9 metrics (all-class, 7-seq VisDrone-MOT val)
through a single TrackEval pass so every column shares one recipe.

Metrics: MOTA IDF1 IDSw Recall Prec. FP FN HOTA AssA  (+ DetA for context)
  CLEAR    -> MOTA, CLR_Re (Recall), CLR_Pr (Prec.), CLR_FP (FP), CLR_FN (FN), IDSW
  Identity -> IDF1
  HOTA     -> HOTA, DetA, AssA   (arrays over alpha -> .mean())

GT is already all-class collapsed to class=1/conf=1, copied verbatim (matches the
_decompose_gap.py all-class recipe). Sanity: bytetrack/iou95 MOTA-IDF1 must reproduce
54.5/65.0 and 54.6/66.0 (already validated once for those two).
"""
import os, shutil

DARE = r"C:\Users\User\Desktop\projects\DARE-MOT"
GT_SRC = r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone_MOT_Format\VisDrone2019-MOT-val"
TE = os.path.join(DARE, "_trackeval")
BENCH, SPLIT = "VisDroneAll", "val"
FOL = f"{BENCH}-{SPLIT}"

SEQS = ["uav0000086_00000_v","uav0000117_02622_v","uav0000137_00458_v",
        "uav0000182_00000_v","uav0000268_05773_v","uav0000305_00000_v",
        "uav0000339_00001_v"]

# ordered: (tracker label, prediction dir, human config name)
CONFIGS = [
    ("r0_bytetrack",   "bt_l0_s0_r1",            "ByteTrack (lambda=0, appearance OFF)"),
    ("r1_dare_raw",    "dare_l05_s0_r1",         "DARE raw (lambda=0.5, ungated)"),
    ("r2_sizegate5000","gate5000",               "DARE + size-gate @5000 (AIN-DG)"),
    ("r3_ft_lockon",   "ftgate2500",             "DARE + FT OSNet-AIN + size-gate, lock ON @2500"),
    ("r4_ft_lockoff",  "ftgate2500_lockoff",     "DARE + lock OFF (ftgate2500_lockoff)"),
    ("r5_iou95",       "ftg2500_lockoff_iou95",  "DARE + IoU-gate 0.95 (FINAL)"),
]

def build():
    gt_root = os.path.join(TE, "gt", "mot_challenge")
    seqmap_dir = os.path.join(gt_root, "seqmaps")
    os.makedirs(seqmap_dir, exist_ok=True)
    with open(os.path.join(seqmap_dir, f"{FOL}.txt"), "w", newline="") as f:
        f.write("name\n" + "\n".join(SEQS) + "\n")
    for s in SEQS:
        dst = os.path.join(gt_root, FOL, s)
        os.makedirs(os.path.join(dst, "gt"), exist_ok=True)
        shutil.copyfile(os.path.join(GT_SRC, s, "gt", "gt.txt"), os.path.join(dst, "gt", "gt.txt"))
        shutil.copyfile(os.path.join(GT_SRC, s, "seqinfo.ini"), os.path.join(dst, "seqinfo.ini"))
    for label, expn, _ in CONFIGS:
        data = os.path.join(TE, "trackers", "mot_challenge", FOL, label, "data")
        os.makedirs(data, exist_ok=True)
        src = os.path.join(DARE, "YOLOX_outputs", expn, "track_results")
        for s in SEQS:
            shutil.copyfile(os.path.join(src, s + ".txt"), os.path.join(data, s + ".txt"))
    print("workspace built for", len(CONFIGS), "configs")

def run():
    import trackeval
    dcfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dcfg.update({
        'GT_FOLDER': os.path.join(TE, 'gt', 'mot_challenge'),
        'TRACKERS_FOLDER': os.path.join(TE, 'trackers', 'mot_challenge'),
        'BENCHMARK': BENCH, 'SPLIT_TO_EVAL': SPLIT,
        'TRACKERS_TO_EVAL': [c[0] for c in CONFIGS],
        'CLASSES_TO_EVAL': ['pedestrian'],   # all boxes are class 1 -> all-class
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

    print("\n=========================  ALL-CLASS 9-METRIC TABLE  (VisDrone-MOT val, 7 seq, seed 0)  =========================")
    hdr = (f"{'config':52s} {'MOTA':>6s} {'IDF1':>6s} {'IDSw':>6s} {'Rcll':>6s} "
           f"{'Prec':>6s} {'FP':>7s} {'FN':>7s} {'HOTA':>6s} {'DetA':>6s} {'AssA':>6s}")
    print(hdr); print("-"*len(hdr))
    rows=[]
    for label, expn, name in CONFIGS:
        c = r[label]['COMBINED_SEQ']['pedestrian']
        H, C, I = c['HOTA'], c['CLEAR'], c['Identity']
        row = dict(name=name, MOTA=C['MOTA']*100, IDF1=I['IDF1']*100, IDSw=int(C['IDSW']),
                   Rcll=C['CLR_Re']*100, Prec=C['CLR_Pr']*100, FP=int(C['CLR_FP']),
                   FN=int(C['CLR_FN']), HOTA=H['HOTA'].mean()*100, DetA=H['DetA'].mean()*100,
                   AssA=H['AssA'].mean()*100)
        rows.append(row)
        print(f"{name:52s} {row['MOTA']:6.1f} {row['IDF1']:6.1f} {row['IDSw']:6d} {row['Rcll']:6.1f} "
              f"{row['Prec']:6.1f} {row['FP']:7d} {row['FN']:7d} {row['HOTA']:6.1f} {row['DetA']:6.1f} {row['AssA']:6.1f}")
    print("="*len(hdr))
    # markdown for the vault
    print("\n--- MARKDOWN ---")
    print("| # | Config | MOTA ↑ | IDF1 ↑ | IDSw ↓ | Recall ↑ | Prec. ↑ | FP ↓ | FN ↓ | HOTA ↑ | DetA | AssA ↑ |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i,row in enumerate(rows):
        print(f"| {i} | {row['name']} | {row['MOTA']:.1f} | {row['IDF1']:.1f} | {row['IDSw']} | "
              f"{row['Rcll']:.1f} | {row['Prec']:.1f} | {row['FP']} | {row['FN']} | "
              f"{row['HOTA']:.1f} | {row['DetA']:.1f} | {row['AssA']:.1f} |")

if __name__ == "__main__":
    build()
    run()
