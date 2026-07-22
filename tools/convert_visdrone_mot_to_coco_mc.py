"""
Multi-class MC MOT -> COCO converter (Phase 1, DARE-MOT multiclass migration).

NON-DESTRUCTIVE: emits NEW json (train_mc.json / val7_mc.json) alongside; does not overwrite
the leaky single-class train.json/train6.json/val1.json.

Reads boxes from the MC MOT gt.txt produced by convert_visdrone_mc.py (real class in col 8,
0-indexed field 7). Reads the image list + dims for each sequence from the RAW sequences dir
(--img-root) so no imagery is duplicated. YOLOX data_dir should point at --img-root; file_name
is written as "<seq>/<img>".

Differences vs the leaky single-class convert_visdrone_mot_to_coco.py:
  1. categories = the real 5 classes (was hardcoded [pedestrian]).
  2. category_id read from the gt class column (was hardcoded 1).
  3. requires len(parts) >= 8 so the class column must be present.
"""
import os
import json
import argparse

CATEGORIES = [
    {"id": 1, "name": "pedestrian"},
    {"id": 2, "name": "car"},
    {"id": 3, "name": "van"},
    {"id": 4, "name": "truck"},
    {"id": 5, "name": "bus"},
]


def _read_seqinfo(seq_path):
    info = {}
    ini_path = os.path.join(seq_path, 'seqinfo.ini')
    if os.path.exists(ini_path):
        with open(ini_path, 'r') as f:
            for line in f:
                if '=' in line:
                    k, v = line.strip().split('=', 1)
                    info[k.strip()] = v.strip()
    return info


def convert(mot_root, img_root, out_path):
    out = {"videos": [], "images": [], "annotations": [], "categories": CATEGORIES}

    image_id = 1
    ann_id = 1
    video_id = 1

    sequences = sorted(
        s for s in os.listdir(mot_root)
        if os.path.isdir(os.path.join(mot_root, s)) and s != 'annotations'
    )

    per_class = {c["id"]: 0 for c in CATEGORIES}

    for seq in sequences:
        mot_seq = os.path.join(mot_root, seq)
        gt_path = os.path.join(mot_seq, 'gt', 'gt.txt')
        raw_img_dir = os.path.join(img_root, seq)
        if not os.path.isdir(raw_img_dir):
            print(f"Warning: no raw images for {seq} at {raw_img_dir}. Skipping.")
            continue

        info = _read_seqinfo(mot_seq)
        width = int(info.get('imWidth', 1920))
        height = int(info.get('imHeight', 1080))

        img_files = sorted(f for f in os.listdir(raw_img_dir) if f.endswith('.jpg'))
        seq_length = len(img_files)

        out['videos'].append({"id": video_id, "file_name": seq})

        first_image_id = image_id
        frame_to_image_id = {}
        for img_file in img_files:
            try:
                frame_num = int(os.path.splitext(img_file)[0])
            except ValueError:
                continue
            frame_to_image_id[frame_num] = image_id
            out['images'].append({
                "id": image_id,
                "video_id": video_id,
                "file_name": f"{seq}/{img_file}",
                "width": width,
                "height": height,
                "frame_id": frame_num,
                "seq_length": seq_length,
                "first_frame_image_id": first_image_id,
            })
            image_id += 1

        if os.path.exists(gt_path):
            with open(gt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 8:
                        continue
                    frame = int(float(parts[0]))
                    tid = int(float(parts[1]))
                    x, y, w, h = (float(parts[2]), float(parts[3]),
                                  float(parts[4]), float(parts[5]))
                    cat = int(parts[7])          # real class 1..5
                    if w <= 0 or h <= 0:
                        continue
                    if frame not in frame_to_image_id:
                        continue
                    out['annotations'].append({
                        "id": ann_id,
                        "image_id": frame_to_image_id[frame],
                        "video_id": video_id,
                        "category_id": cat,
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                        "track_id": tid,
                        "visibility": 1.0,
                    })
                    per_class[cat] = per_class.get(cat, 0) + 1
                    ann_id += 1

        video_id += 1

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f)

    print(f"\nDone! {len(out['videos'])} videos, {len(out['images'])} images, "
          f"{len(out['annotations'])} annotations -> {out_path}")
    print("  per class:")
    for c in CATEGORIES:
        print(f"    {c['id']} {c['name']:<11}: {per_class[c['id']]}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mot-root', required=True, help='MC MOT dir from convert_visdrone_mc.py')
    p.add_argument('--img-root', required=True, help='raw sequences dir (images referenced in place)')
    p.add_argument('--out', required=True, help='output COCO json path (NEW)')
    args = p.parse_args()
    convert(args.mot_root, args.img_root, args.out)
