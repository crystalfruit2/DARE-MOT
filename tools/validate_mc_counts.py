"""
Phase 1 validation gate: per-class box counts in the generated COCO json must match a direct
count of the RAW VisDrone annotations under the exact same filters (score==1, class in the 5
scored categories, w>0, h>0). Any mismatch means the conversion dropped or mislabeled boxes.

Exit code 0 = all classes match; 1 = mismatch.
"""
import os
import sys
import json
import argparse

VISDRONE_TO_MC = {1: 1, 4: 2, 5: 3, 6: 4, 9: 5}
MC_NAMES = {1: "pedestrian", 2: "car", 3: "van", 4: "truck", 5: "bus"}


def raw_counts(raw_dir):
    ann_dir = os.path.join(raw_dir, 'annotations')
    counts = {mc: 0 for mc in MC_NAMES}
    for fn in sorted(os.listdir(ann_dir)):
        if not fn.endswith('.txt'):
            continue
        with open(os.path.join(ann_dir, fn)) as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 8:
                    continue
                score = int(parts[6])
                cat = int(parts[7])
                w, h = float(parts[4]), float(parts[5])
                if score == 0 or cat not in VISDRONE_TO_MC or w <= 0 or h <= 0:
                    continue
                counts[VISDRONE_TO_MC[cat]] += 1
    return counts


def coco_counts(coco_path):
    with open(coco_path) as f:
        data = json.load(f)
    counts = {mc: 0 for mc in MC_NAMES}
    for a in data['annotations']:
        counts[a['category_id']] = counts.get(a['category_id'], 0) + 1
    return counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw', required=True, help='raw VisDrone dir (sequences/ + annotations/)')
    p.add_argument('--coco', required=True, help='generated COCO json')
    args = p.parse_args()

    raw = raw_counts(args.raw)
    coco = coco_counts(args.coco)

    print(f"{'class':<12} {'raw':>10} {'coco':>10}  match")
    ok = True
    for mc, name in MC_NAMES.items():
        m = raw[mc] == coco[mc]
        ok = ok and m
        print(f"{name:<12} {raw[mc]:>10} {coco[mc]:>10}  {'OK' if m else 'MISMATCH'}")
    print(f"{'TOTAL':<12} {sum(raw.values()):>10} {sum(coco.values()):>10}  "
          f"{'OK' if sum(raw.values()) == sum(coco.values()) else 'MISMATCH'}")

    if ok:
        print("\nVALIDATION PASSED")
        sys.exit(0)
    else:
        print("\nVALIDATION FAILED")
        sys.exit(1)


if __name__ == '__main__':
    main()
