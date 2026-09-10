"""Score the corrected-CMC queue (_run_cmcfix_2026-09-10.ps1) under the HEADLINE protocol (2026-09-10).

Protocol = online score-weighted class voting (_class_vote.vote, mode on_score) -> official VisDrone
ignored-region filter (_score_official) -> _score_multiclass.score_run. Also prints the raw protocol
(no vote, no filter) so the Joseph-control check is visible on the numbers the runs were designed on.

Fresh same-day baselines = the lap0512 repro (mc_bytetrack_lap0512, mc_dare_ca_lap0512, main repo);
CMC arms live in the worktree ..\\DARE-MOT-cmcfix\\YOLOX_outputs.
Usage: python _score_cmcfix_2026-09-10.py
"""
import os
import _score_multiclass as sm
import _score_official as so
import _class_vote as cv

HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.join(os.path.dirname(HERE), "DARE-MOT-cmcfix", "YOLOX_outputs")
MAIN = os.path.join(HERE, "YOLOX_outputs")
OUT = os.path.join(HERE, "_scratch", "cmcfix_score")
RAW_GT = sm.MC_GT

ARMS = [
    ("mc_bytetrack_lap0512", MAIN, "ByteTrack (fresh baseline)"),
    ("mc_bt_ca", WT, "ByteTrack + CA-KF"),
    ("mc_bt_cmcfix_j", WT, "ByteTrack + CMC fixed (Joseph)"),
    ("mc_dare_ca_lap0512", MAIN, "CA-DARE (fresh baseline)"),
    ("mc_dare_ca_joseph", WT, "CA-DARE + Joseph only (control)"),
    ("mc_dare_ca_cmcbug_j", WT, "CA-DARE + CMC July bug (Joseph)"),
    ("mc_dare_ca_cmcfix_j", WT, "CA-DARE + CMC fixed/parity (Joseph)"),
    ("mc_dare_ca_cmcscale_j", WT, "CA-DARE + CMC fixed/scale (Joseph)"),
    # factorial fill-ins (_run_cmc_factorial_2026-09-10.ps1)
    ("mc_dare_cv_lap0512", MAIN, "CV-DARE (fresh baseline)"),
    ("mc_dare_cv_cmcfix_j", WT, "CV-DARE + CMC fixed/parity (Joseph)"),
    ("mc_bt_ca_cmcfix_j", WT, "ByteTrack + CA-KF + CMC fixed (Joseph)"),
    # scale-mode factorial (_run_cmc_factorial_scale_2026-09-10.ps1)
    ("mc_bt_cmcscale_j", WT, "ByteTrack + CMC scale (Joseph)"),
    ("mc_bt_ca_cmcscale_j", WT, "ByteTrack + CA-KF + CMC scale (Joseph)"),
    ("mc_dare_cv_cmcscale_j", WT, "CV-DARE + CMC scale (Joseph)"),
]


def read_rows(path):
    return [l.rstrip("\n").split(",") for l in open(path) if l.strip() and len(l.split(",")) >= 8]


def voted_filtered(expn, root):
    dst = os.path.join(OUT, expn + "__on_score_official")
    os.makedirs(dst, exist_ok=True)
    for seq in sm.SEQS:
        rows = read_rows(os.path.join(root, expn, "track_results", seq + ".txt"))
        cats = cv.vote(rows, "on_score")
        tmp = os.path.join(dst, seq + ".voted")
        with open(tmp, "w") as f:
            for p, c in zip(rows, cats):
                q = list(p); q[7] = c
                f.write(",".join(q) + "\n")
        so.filter_file(tmp, os.path.join(dst, seq + ".txt"), so.ignored_regions(seq))
        os.remove(tmp)
    return dst


def fmt(r):
    mi = r["MICRO"]
    return f"IDSw {r['SUM_IDSw']:4d} | IDF1 {100 * mi['idf1']:5.2f} | MOTA {100 * mi['mota']:5.2f}"


if __name__ == "__main__":
    res = {}
    for expn, root, label in ARMS:
        d = os.path.join(root, expn, "track_results")
        if not all(os.path.exists(os.path.join(d, s + ".txt")) for s in sm.SEQS):
            print(f"{expn:24s} INCOMPLETE -- skipped"); continue
        sm.MC_GT = RAW_GT
        raw = sm.score_run(d)
        res.setdefault("raw", {})[expn] = raw
    off_gt = so.prepare_gt()
    for expn, root, label in ARMS:
        if expn not in res["raw"]:
            continue
        sm.MC_GT = off_gt
        res.setdefault("head", {})[expn] = sm.score_run(voted_filtered(expn, root))
    print("\npooled (micro), val7, seed 0")
    print(f"{'arm':40s} {'RAW':38s}  HEADLINE (on_score vote + official)")
    for expn, root, label in ARMS:
        if expn in res["raw"]:
            print(f"{label:40s} {fmt(res['raw'][expn])}  {fmt(res['head'][expn])}")
