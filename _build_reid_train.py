"""Build a torchreid-style pedestrian ReID training set from VisDrone2019-MOT-train.

Plan A, Task #2 (see vault: Projects/Dare_Mot/RESUME-finetune-2026-07-16).

INTEGRITY: source = the DISJOINT official *train* split, NEVER the val7 sequences
we evaluate on. This script refuses to run if any val eval sequence appears in train.

Filtering mirrors _build_ped_gt.py: raw VisDrone-MOT annotations are 10 cols
(frame,id,x,y,w,h,score,category,trunc,occ). Keep rows with score==1 (drop ignored
regions) AND category==1 (pedestrian only). Each GT box is cropped and labelled by
GLOBAL identity = (seq, track_id).

Two-pass, so the training set is clean:
  pass 1  count ped boxes per identity that pass --min-h
  keep    identities with >= --min-inst boxes; assign CONTIGUOUS pid 0..C-1 (softmax)
  pass 2  crop + save only kept identities

Output: Market-1501-style flat dir (torchreid-consumable):
    reid_train/<pid07>_c<cam03>s1_<frame06>_00.jpg   (sequences act as cameras)

Usage:
    python _build_reid_train.py [--min-h 32] [--min-inst 4] [--limit-seqs N] [--clean]
"""
import os
import glob
import argparse
import shutil

TRAIN_ROOT = r"C:\Users\User\Desktop\datasets\VisDrone2019-MOT-train"
VAL_SEQS = {  # guardrail: EVAL sequences must never appear in the training source
    "uav0000086_00000_v", "uav0000117_02622_v", "uav0000137_00458_v",
    "uav0000182_00000_v", "uav0000268_05773_v", "uav0000305_00000_v",
    "uav0000339_00001_v",
}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reid_train")


def find_frame(seq_img_dir, frame):
    for w in (7, 6, 5, 4, 3):
        p = os.path.join(seq_img_dir, f"{frame:0{w}d}.jpg")
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join(seq_img_dir, f"*{frame}.jpg"))
    return hits[0] if hits else None


def read_ped_rows(ann_path, min_h):
    """Yield (frame, tid, x, y, w, h) for pedestrian GT boxes passing min_h."""
    with open(ann_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = line.split(",")
            if c[6] != "1" or c[7] != "1":            # score==1 & category==1
                continue
            frame, tid = int(c[0]), int(c[1])
            x, y, w, h = (int(float(c[2])), int(float(c[3])),
                          int(float(c[4])), int(float(c[5])))
            if w <= 0 or h <= 0 or h < min_h:
                continue
            yield frame, tid, x, y, w, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-h", type=int, default=32,
                    help="drop boxes shorter than this many px")
    ap.add_argument("--min-inst", type=int, default=4,
                    help="drop identities with fewer than this many boxes")
    ap.add_argument("--limit-seqs", type=int, default=0)
    ap.add_argument("--clean", action="store_true", help="wipe OUT before building")
    args = ap.parse_args()

    ann_dir = os.path.join(TRAIN_ROOT, "annotations")
    seq_root = os.path.join(TRAIN_ROOT, "sequences")
    assert os.path.isdir(ann_dir) and os.path.isdir(seq_root), f"bad TRAIN_ROOT: {TRAIN_ROOT}"

    seqs = sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(ann_dir, "*.txt")))
    leaked = VAL_SEQS.intersection(seqs)
    assert not leaked, f"DATA LEAK: val eval sequences present in train split: {leaked}"
    if args.limit_seqs:
        seqs = seqs[:args.limit_seqs]

    from PIL import Image

    # ---- pass 1: count boxes per identity ----
    counts = {}   # (seq,tid) -> n
    for seq in seqs:
        for frame, tid, *_ in read_ped_rows(os.path.join(ann_dir, seq + ".txt"), args.min_h):
            counts[(seq, tid)] = counts.get((seq, tid), 0) + 1
    kept_ids = {k for k, n in counts.items() if n >= args.min_inst}
    pid_map = {k: i for i, k in enumerate(sorted(kept_ids))}   # contiguous 0..C-1
    n_boxes_kept = sum(counts[k] for k in kept_ids)
    print(f"pass1: identities total={len(counts)}  kept(>= {args.min_inst} boxes)={len(kept_ids)}  "
          f"dropped={len(counts)-len(kept_ids)}  boxes_to_write~{n_boxes_kept}")

    # ---- pass 2: crop + save kept identities ----
    if args.clean and os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT, exist_ok=True)
    n_crops = 0
    print(f"{'seq':30s} {'cam':>4s} {'crops':>7s} {'ids':>5s}")
    for cam, seq in enumerate(seqs):
        seq_img_dir = os.path.join(seq_root, seq)
        by_frame = {}
        for frame, tid, x, y, w, h in read_ped_rows(os.path.join(ann_dir, seq + ".txt"), args.min_h):
            if (seq, tid) in pid_map:
                by_frame.setdefault(frame, []).append((tid, x, y, w, h))
        seq_ids, seq_crops = set(), 0
        for frame, boxes in by_frame.items():
            img_path = find_frame(seq_img_dir, frame)
            if img_path is None:
                continue
            im = Image.open(img_path).convert("RGB")
            W, H = im.size
            for tid, x, y, w, h in boxes:
                x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
                if x1 <= x0 or y1 <= y0:
                    continue
                pid = pid_map[(seq, tid)]
                im.crop((x0, y0, x1, y1)).save(
                    os.path.join(OUT, f"{pid:07d}_c{cam:03d}s1_{frame:06d}_00.jpg"), quality=95)
                seq_ids.add(pid); seq_crops += 1; n_crops += 1
        print(f"{seq:30s} {cam:4d} {seq_crops:7d} {len(seq_ids):5d}")

    print(f"\nDONE  crops={n_crops}  identities(num_classes)={len(pid_map)}  -> {OUT}")
    print(f"knobs: min_h={args.min_h}  min_inst={args.min_inst}")


if __name__ == "__main__":
    main()
