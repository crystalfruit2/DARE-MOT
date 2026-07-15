"""Per-sequence occlusion/size/density audit for VisDrone2019-MOT.
Tests the hypothesis: does the appearance delta correlate with occlusion,
or with target size / density? (results-improvement-plan-2026-07-13)"""
import os, glob
from collections import defaultdict

ANN_DIR = r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone2019-MOT-val\annotations"
# If the val annotations live elsewhere, find them with:
#   Get-ChildItem -Recurse -Filter "uav0000339*" C:\Users\User\Desktop\projects\ByteTrack\datasets | Select FullName

# Per-sequence tracker deltas from benchmark-ablation-findings-2026-07-01
# (IDSw: appearance-on DARE minus ByteTrack; negative = appearance helped)
APPEARANCE_DELTA = {
    "uav0000086_00000_v": 0, "uav0000117_02622_v": 31, "uav0000137_00458_v": 56,
    "uav0000182_00000_v": 3, "uav0000268_05773_v": 0, "uav0000305_00000_v": -5,
    "uav0000339_00001_v": 7,
}

def audit(path):
    frames = defaultdict(int); n = occ_any = occ_heavy = 0; area_sum = 0.0
    with open(path) as f:
        for line in f:
            c = line.strip().split(',')
            if len(c) < 10: continue
            frame, w, h, cat, occ = int(c[0]), float(c[4]), float(c[5]), int(c[7]), int(c[9])
            if cat != 1:  # pedestrian only, matches the benchmark protocol
                continue
            n += 1; frames[frame] += 1; area_sum += w * h
            if occ >= 1: occ_any += 1
            if occ == 2: occ_heavy += 1
    if n == 0: return None
    return dict(boxes=n, occ_any=100*occ_any/n, occ_heavy=100*occ_heavy/n,
                mean_area=area_sum/n, density=n/max(len(frames),1))

print(f"{'sequence':<24}{'boxes':>8}{'occ%':>7}{'heavy%':>8}{'area px2':>10}{'box/frm':>9}{'IDSw delta':>11}")
for ann in sorted(glob.glob(os.path.join(ANN_DIR, "*.txt"))):
    seq = os.path.splitext(os.path.basename(ann))[0]
    r = audit(ann)
    if r is None: continue
    d = APPEARANCE_DELTA.get(seq, "?")
    print(f"{seq:<24}{r['boxes']:>8}{r['occ_any']:>7.1f}{r['occ_heavy']:>8.1f}"
          f"{r['mean_area']:>10.0f}{r['density']:>9.1f}{str(d):>11}")
print("\nRead: if IDSw delta tracks mean_area/density (not occ%), the dataset move"
      "\nmust target appearance resolvability, not occlusion alone.")
