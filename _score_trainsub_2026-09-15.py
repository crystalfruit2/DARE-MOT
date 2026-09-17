"""Score the out-of-val tracker-ranking check (7 train sequences, D3) under the headline protocol.

Same pipeline as the val7 scorers (on_score vote -> official ignored-region filter -> pooled micro), redirected
to train: MC GT from VisDrone_MOT_Format_MC/VisDrone2019-MOT-train, ignored regions from the raw train
annotations, filtered copies under _scratch/official_trainsub and _scratch/trainsub_score.
Absolute numbers are NOT reportable (D3 trained on these frames); read rankings only.
"""
import importlib.util
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("sc", os.path.join(HERE, "_score_cmcfix_2026-09-10.py"))
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)
sm, so = sc.sm, sc.so

SEQS = ["uav0000071_03240_v", "uav0000124_00944_v", "uav0000222_03150_v", "uav0000266_03598_v",
        "uav0000289_00001_v", "uav0000315_00000_v", "uav0000360_00001_v"]
TRAIN_MC = (r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone_MOT_Format_MC"
            r"\VisDrone2019-MOT-train\{seq}\gt\gt.txt")
sm.SEQS = list(SEQS)
sm.MC_GT = TRAIN_MC
sc.RAW_GT = TRAIN_MC
so.RAW = r"C:\Users\User\Desktop\datasets\VisDrone2019-MOT-train\annotations\{seq}.txt"
so.OUT = os.path.join(HERE, "_scratch", "official_trainsub")
sc.OUT = os.path.join(HERE, "_scratch", "trainsub_score")
WT, MAIN = sc.WT, sc.MAIN
ARMS = [
    ("ts_bt", WT, "ByteTrack"),
    ("ts_bt_cmcparity_j", WT, "ByteTrack + parity CMC"),
    ("ts_bt_cmcscale_j", WT, "ByteTrack + scale CMC"),
    ("ts_dare_cv_cmcscale_j", WT, "CV-DARE + scale CMC"),
    ("ts_botsort", MAIN, "BoT-SORT-ReID"),
    ("ts_botsort_noreid", MAIN, "BoT-SORT no ReID"),
]


def fpfn(r):
    cls = [v for k, v in r.items() if isinstance(k, int) and v is not None]
    return sum(v["fp"] for v in cls), sum(v["fn"] for v in cls)


if __name__ == "__main__":
    res = {"raw": {}, "head": {}, "head_seq": {}}
    present = []
    for expn, root, label in ARMS:
        d = os.path.join(root, expn, "track_results")
        if not all(os.path.exists(os.path.join(d, s + ".txt")) for s in SEQS):
            print(f"{expn:26s} INCOMPLETE -- skipped")
            continue
        present.append((expn, root, label))
        sm.MC_GT, sm.SEQS = TRAIN_MC, list(SEQS)
        res["raw"][expn] = sm.score_run(d)
    off_gt = so.prepare_gt()
    for expn, root, label in present:
        sm.MC_GT, sm.SEQS = off_gt, list(SEQS)
        vdir = sc.voted_filtered(expn, root)
        res["head"][expn] = sm.score_run(vdir)
        res["head_seq"][expn] = {}
        for s in SEQS:
            sm.SEQS = [s]
            res["head_seq"][expn][s] = sm.score_run(vdir)
        sm.SEQS = list(SEQS)
    print("\nOUT-OF-VAL ranking check: 7 TRAIN sequences, D3 (trained on them -> rankings only), headline protocol, pooled")
    print(f"{'arm':28s} {'IDSw':>5s} {'IDF1':>6s} {'MOTA':>6s} {'FP':>6s} {'FN':>6s}")
    for expn, root, label in present:
        h = res["head"][expn]
        fp, fn = fpfn(h)
        print(f"{label:28s} {h['SUM_IDSw']:5d} {100 * h['MICRO']['idf1']:6.2f} {100 * h['MICRO']['mota']:6.2f} {fp:6d} {fn:6d}")
    print("\nper sequence IDSw / IDF1 / MOTA")
    print(f"{'arm':28s}" + "".join(f"{s[3:10]:>18s}" for s in SEQS))
    for expn, root, label in present:
        print(f"{label:28s}" + "".join(
            f"{res['head_seq'][expn][s]['SUM_IDSw']:3d}/{100 * res['head_seq'][expn][s]['MICRO']['idf1']:5.1f}/{100 * res['head_seq'][expn][s]['MICRO']['mota']:5.1f}".rjust(18)
            for s in SEQS))
    os.makedirs(sc.OUT, exist_ok=True)
    dst = os.path.join(sc.OUT, "trainsub_summary.json")
    json.dump(res, open(dst, "w"), indent=1, default=str)
    print(f"\nwrote {dst}")
