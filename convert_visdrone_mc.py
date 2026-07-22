"""
Multi-class VisDrone -> MOTChallenge GT converter (Phase 1, DARE-MOT multiclass migration).

NON-DESTRUCTIVE: writes to NEW output dirs (VisDrone_MOT_Format_MC/...). Does NOT touch the
existing single-class VisDrone_MOT_Format/ pipeline, which the current benchmark still depends on.

Differences vs the leaky single-class convert_visdrone.py:
  1. Keeps the REAL object class (the old script hardcoded class 1 -> the class-leak).
  2. Drops ignored-region rows (score==0).
  3. Filters to the 5 official scored VisDrone classes and remaps them to contiguous 1..5.
  4. Does NOT copy images (GT-only). TrackEval needs gt.txt + seqinfo.ini; the COCO converter
     references the raw images in place. Saves ~9GB of duplicated imagery.

MOTChallenge gt line written: frame,id,x,y,w,h,conf(=1),class(1..5),visibility(=1)
"""
import os
import argparse

# Official 5 scored VisDrone categories -> contiguous class ids (1..5).
# Model class ids will be 0..4 (category_id - 1) at detector-training time.
VISDRONE_TO_MC = {1: 1, 4: 2, 5: 3, 6: 4, 9: 5}   # pedestrian, car, van, truck, bus
MC_NAMES = {1: "pedestrian", 2: "car", 3: "van", 4: "truck", 5: "bus"}


def _image_size(img_path):
    """Return (width, height) of an image, trying cv2 then PIL."""
    try:
        import cv2
        im = cv2.imread(img_path)
        if im is not None:
            h, w = im.shape[:2]
            return w, h
    except Exception:
        pass
    from PIL import Image
    with Image.open(img_path) as im:
        return im.size  # (w, h)


def convert_visdrone_to_mot_mc(visdrone_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    seq_dir = os.path.join(visdrone_dir, 'sequences')
    ann_dir = os.path.join(visdrone_dir, 'annotations')

    sequences = sorted(
        s for s in os.listdir(seq_dir) if os.path.isdir(os.path.join(seq_dir, s))
    )

    totals = {mc: 0 for mc in MC_NAMES}
    dropped_score0 = 0
    dropped_class = 0

    for seq in sequences:
        seq_output_dir = os.path.join(output_dir, seq)
        gt_dir = os.path.join(seq_output_dir, 'gt')
        os.makedirs(gt_dir, exist_ok=True)

        src_images = os.path.join(seq_dir, seq)
        visdrone_anno_path = os.path.join(ann_dir, seq + '.txt')
        mot_anno_path = os.path.join(gt_dir, 'gt.txt')

        kept = 0
        with open(visdrone_anno_path, 'r') as f_in, open(mot_anno_path, 'w') as f_out:
            for line in f_in:
                parts = line.strip().split(',')
                if len(parts) < 8:
                    continue
                frame, tid, x, y, w, h, score, cat = parts[:8]

                if int(score) == 0:            # ignored-region row
                    dropped_score0 += 1
                    continue
                c = int(cat)
                if c not in VISDRONE_TO_MC:     # not one of the 5 scored classes
                    dropped_class += 1
                    continue
                mc = VISDRONE_TO_MC[c]
                totals[mc] += 1
                kept += 1
                # MOTChallenge gt: frame,id,x,y,w,h,conf,class,visibility
                f_out.write(f"{frame},{tid},{x},{y},{w},{h},1,{mc},1\n")

        # seqinfo.ini (dims from first image; needed by TrackEval + COCO converter)
        img_list = sorted(f for f in os.listdir(src_images) if f.endswith('.jpg'))
        if img_list:
            width, height = _image_size(os.path.join(src_images, img_list[0]))
            seq_length = len(img_list)
            ini_content = (
                "[Sequence]\n"
                f"name={seq}\n"
                "imDir=img1\n"
                "frameRate=30\n"
                f"seqLength={seq_length}\n"
                f"imWidth={width}\n"
                f"imHeight={height}\n"
                "imExt=.jpg\n"
            )
            with open(os.path.join(seq_output_dir, 'seqinfo.ini'), 'w') as f_ini:
                f_ini.write(ini_content)
        print(f"  {seq}: kept {kept} boxes, {seq_length} frames")

    print(f"\nDone -> {output_dir}")
    print(f"  dropped score==0 (ignored): {dropped_score0}")
    print(f"  dropped non-scored class:   {dropped_class}")
    print("  kept per class:")
    grand = 0
    for mc, name in MC_NAMES.items():
        print(f"    {mc} {name:<11}: {totals[mc]}")
        grand += totals[mc]
    print(f"    TOTAL       : {grand}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='in_dir', required=True,
                   help='raw VisDrone dir (has sequences/ and annotations/)')
    p.add_argument('--out', dest='out_dir', required=True,
                   help='output MC MOT dir (NEW path)')
    args = p.parse_args()
    convert_visdrone_to_mot_mc(args.in_dir, args.out_dir)
