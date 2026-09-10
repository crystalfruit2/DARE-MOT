"""Trajectory-level class voting, applied as a post-process to cached track_results (2026-09-10).

Motivation (experiment-log 2026-09-07 D1): 3,016 of 11,115 FP on val7 overlap an unmatched GT
box and every one of them is a class-label error (car<->van = 2,630). DARE-MOT's association is
class-agnostic, so a single track can carry different per-frame labels; the scorer splits by
class, so each wrong-label frame costs one FP (wrong class) AND one FN (right class).
Voting assigns ONE label per track. Standard VisDrone practice (GIAOTracker, Rough2Fine) --
adopted as a correctness fix, applied to EVERY arm, never claimed as a contribution.

Modes
  off_count  -- offline: majority label over the whole track            (uses future frames)
  off_score  -- offline: label with the largest summed detection score  (uses future frames)
  on_count   -- causal: majority over frames <= t, ties -> current label (real-time legal)
  on_score   -- causal: summed score over frames <= t, ties -> current label

Rows: frame,id,x,y,w,h,score,category,-1,-1 (category 1..5). Only the category column changes;
row order, boxes, ids and scores are preserved byte-for-byte otherwise.

Usage:
  python _class_vote.py <expn> [<expn> ...]    -> _scratch/class_vote/<expn>__<mode>/<seq>.txt
  python _class_vote.py --stats <expn>         -> label-churn statistics only
"""
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(HERE, "_scratch", "class_vote")
SEQS = [
    "uav0000086_00000_v", "uav0000117_02622_v", "uav0000137_00458_v",
    "uav0000182_00000_v", "uav0000268_05773_v", "uav0000305_00000_v",
    "uav0000339_00001_v",
]
MODES = ["off_count", "off_score", "on_count", "on_score"]


def load(expn, seq):
    path = os.path.join(HERE, "YOLOX_outputs", expn, "track_results", seq + ".txt")
    rows = []
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split(",")
            if len(p) < 8:
                continue
            rows.append(p)
    return rows


def vote(rows, mode):
    """Return a list of new category strings, aligned with rows."""
    # Process in (frame, original index) order so the causal modes only see the past.
    order = sorted(range(len(rows)), key=lambda i: (int(float(rows[i][0])), i))
    new = [None] * len(rows)
    if mode.startswith("off_"):
        tally = defaultdict(lambda: defaultdict(float))
        for i in order:
            tid, cat, sc = rows[i][1], rows[i][7], float(rows[i][6])
            tally[tid][cat] += sc if mode == "off_score" else 1.0
        # deterministic tie-break: highest tally, then lowest class id
        best = {tid: max(t.items(), key=lambda kv: (kv[1], -int(float(kv[0]))))[0]
                for tid, t in tally.items()}
        for i in order:
            new[i] = best[rows[i][1]]
        return new
    tally = defaultdict(lambda: defaultdict(float))
    for i in order:
        tid, cat, sc = rows[i][1], rows[i][7], float(rows[i][6])
        tally[tid][cat] += sc if mode == "on_score" else 1.0
        t = tally[tid]
        top = max(t.values())
        leaders = [c for c, v in t.items() if v == top]
        new[i] = cat if cat in leaders else min(leaders, key=lambda c: int(float(c)))
    return new


def write(expn, mode):
    out_dir = os.path.join(OUT_ROOT, f"{expn}__{mode}")
    os.makedirs(out_dir, exist_ok=True)
    changed = total = 0
    for seq in SEQS:
        rows = load(expn, seq)
        cats = vote(rows, mode)
        with open(os.path.join(out_dir, seq + ".txt"), "w") as f:
            for p, c in zip(rows, cats):
                changed += int(c != p[7])
                total += 1
                q = list(p)
                q[7] = c
                f.write(",".join(q) + "\n")
    print(f"{expn:28s} {mode:10s} relabelled {changed:6d} / {total} rows ({100*changed/total:.2f}%) -> {out_dir}")
    return out_dir


def stats(expn):
    n_tracks = n_mixed = rows_mixed = rows_all = 0
    for seq in SEQS:
        by_tid = defaultdict(list)
        for p in load(expn, seq):
            by_tid[p[1]].append(p[7])
        for cats in by_tid.values():
            n_tracks += 1
            rows_all += len(cats)
            if len(set(cats)) > 1:
                n_mixed += 1
                rows_mixed += len(cats)
    print(f"{expn}: {n_tracks} tracks, {n_mixed} carry >1 label ({100*n_mixed/n_tracks:.1f}%), "
          f"covering {rows_mixed}/{rows_all} rows ({100*rows_mixed/rows_all:.1f}%)")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--stats":
        for e in args[1:]:
            stats(e)
        sys.exit(0)
    for e in args:
        for m in MODES:
            write(e, m)
