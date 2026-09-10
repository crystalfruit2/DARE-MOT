"""VisDrone-MOT raw annotations -> 10-class COCO json with ignored regions (2026-09-10).

Why: the 5-class detector (convert_visdrone_mc.py) was trained with people, bicycle, tricycle,
awning-tricycle, motor AND the ignored regions all dropped, i.e. labelled BACKGROUND, although they look
like pedestrians/cars. 26% of CA-DARE's FPs sit on that content (experiment-log 2026-09-10). Standard
VisDrone practice: train all 10 classes, evaluate the 5.

Category ids are ORDERED so the 5 evaluated classes keep model ids 0..4 (MOTDataset maps
category_id -> sorted(cat_ids).index), i.e. the tracker and every scorer stay unchanged; the 5 extra
classes are model ids 5..9 and are dropped at tracking time.

Ignored regions = raw category 0 ("ignored region") and 11 ("others") -- the same rule as
_score_official.py. They are stored per image as "ignore_regions": [[x,y,w,h],...] (MOTDataset paints
them with the mean colour when ignore_fill is set), and any GT box >= 50% inside their union is dropped
(the official dropObjects.m rule), so the loss never sees a box whose pixels were painted over.

Usage:
  python tools/convert_visdrone_10c.py --raw C:/Users/User/Desktop/datasets/VisDrone2019-MOT-train \
      --out C:/Users/User/Desktop/datasets/VisDrone2019-MOT-train/annotations/train_10c.json
  python tools/convert_visdrone_10c.py --val-from <annotations>/val7_mc.json --out <annotations>/val7_10c.json
"""
import os
import json
import argparse
from collections import defaultdict

import numpy as np

# raw VisDrone category -> COCO category id (1..5 = evaluated, unchanged from the MC pipeline)
RAW_TO_10C = {1: 1, 4: 2, 5: 3, 6: 4, 9: 5,          # pedestrian, car, van, truck, bus
              2: 6, 3: 7, 7: 8, 8: 9, 10: 10}         # people, bicycle, tricycle, awning-tricycle, motor
CATEGORIES = [{"id": i, "name": n} for i, n in enumerate(
    ["pedestrian", "car", "van", "truck", "bus",
     "people", "bicycle", "tricycle", "awning-tricycle", "motor"], start=1)]
IGNORE_RAW = (0, 11)


def covered_frac(box, regions):
    """Fraction of box (x,y,w,h) inside the union of regions, on an integer pixel grid (dropObjects)."""
    x, y, w, h = box
    x0, y0, x1, y1 = int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    m = np.zeros((y1 - y0, x1 - x0), bool)
    for rx, ry, rw, rh in regions:
        a0, b0 = max(int(round(rx)) - x0, 0), max(int(round(ry)) - y0, 0)
        a1, b1 = min(int(round(rx + rw)) - x0, x1 - x0), min(int(round(ry + rh)) - y0, y1 - y0)
        if a1 > a0 and b1 > b0:
            m[b0:b1, a0:a1] = True
    return float(m.mean())


def image_size(path):
    import cv2
    im = cv2.imread(path)
    return im.shape[1], im.shape[0]


def convert_train(raw, out):
    seq_dir, ann_dir = os.path.join(raw, "sequences"), os.path.join(raw, "annotations")
    seqs = sorted(s for s in os.listdir(seq_dir) if os.path.isdir(os.path.join(seq_dir, s)))
    d = {"videos": [], "images": [], "annotations": [], "categories": CATEGORIES}
    img_id = ann_id = 1
    n_cls = defaultdict(int)
    n_drop_cov = n_regions = n_skip = 0
    per_img = []
    for vid, seq in enumerate(seqs, start=1):
        imgs = sorted(f for f in os.listdir(os.path.join(seq_dir, seq)) if f.endswith(".jpg"))
        w, h = image_size(os.path.join(seq_dir, seq, imgs[0]))
        boxes, regions = defaultdict(list), defaultdict(list)
        for line in open(os.path.join(ann_dir, seq + ".txt")):
            p = line.strip().split(",")
            if len(p) < 8:
                continue
            fr, tid, cat = int(p[0]), int(p[1]), int(p[7])
            bb = [float(v) for v in p[2:6]]
            if cat in IGNORE_RAW:
                regions[fr].append(bb)
            elif cat in RAW_TO_10C and bb[2] > 0 and bb[3] > 0:
                boxes[fr].append((tid, RAW_TO_10C[cat], bb))
            else:
                n_skip += 1
        d["videos"].append({"id": vid, "file_name": seq})
        first = img_id
        for f in imgs:
            fr = int(os.path.splitext(f)[0])
            R = regions.get(fr, [])
            n_regions += len(R)
            d["images"].append({"id": img_id, "video_id": vid, "file_name": f"{seq}/{f}",
                                "width": w, "height": h, "frame_id": fr, "seq_length": len(imgs),
                                "first_frame_image_id": first, "ignore_regions": R})
            kept = 0
            for tid, c, bb in boxes.get(fr, []):
                if R and covered_frac(bb, R) >= 0.5:
                    n_drop_cov += 1
                    continue
                d["annotations"].append({"id": ann_id, "image_id": img_id, "video_id": vid,
                                         "category_id": c, "bbox": bb, "area": bb[2] * bb[3],
                                         "iscrowd": 0, "track_id": tid, "visibility": 1.0})
                n_cls[c] += 1; ann_id += 1; kept += 1
            per_img.append(kept)
            img_id += 1
        print(f"  {seq}: {len(imgs)} imgs", flush=True)
    json.dump(d, open(out, "w"))
    pi = np.array(per_img)
    print(f"\n{len(d['videos'])} videos, {len(d['images'])} images, {len(d['annotations'])} boxes -> {out}")
    for c in CATEGORIES:
        print(f"  {c['id']:2d} {c['name']:<16s} {n_cls[c['id']]:8d}")
    print(f"  ignore regions stored: {n_regions}   GT dropped (>=50% inside a region): {n_drop_cov}"
          f"   rows skipped (bad class/size): {n_skip}")
    print(f"  boxes per image: mean {pi.mean():.1f}, p99 {np.percentile(pi, 99):.0f}, max {pi.max()}")


def convert_val(src, out):
    """Val stays the 5 evaluated classes (so AP is comparable with the 5-class detector's), but the
    category list must hold all 10 so the evaluator can map model ids 5..9."""
    d = json.load(open(src))
    assert [c["id"] for c in d["categories"]] == [1, 2, 3, 4, 5]
    d["categories"] = CATEGORIES
    json.dump(d, open(out, "w"))
    print(f"{src} -> {out}: {len(d['images'])} images, {len(d['annotations'])} boxes, 10 categories")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw")
    ap.add_argument("--val-from")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    convert_val(a.val_from, a.out) if a.val_from else convert_train(a.raw, a.out)
