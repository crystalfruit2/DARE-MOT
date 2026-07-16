"""Build pedestrian-only (VisDrone category==1) ground-truth per sequence, in MOT15
format, for the class-leak-corrected evaluation. Durable rebuild of the harness that
was lost with a session scratchpad (see class-leak-analysis-2026-07-15 note).

Source: raw VisDrone-MOT val annotations (10 cols: frame,id,x,y,w,h,score,category,
trunc,occ). We keep rows with score==1 (drop ignored regions, score==0) AND category==1
(pedestrian only). Output: _ped_gt/<seq>.txt as frame,id,x,y,w,h,1,-1,-1,-1.

Usage: python _build_ped_gt.py
"""
import os

RAW = r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone2019-MOT-val\annotations"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_ped_gt")
SEQS = [
    "uav0000086_00000_v", "uav0000117_02622_v", "uav0000137_00458_v",
    "uav0000182_00000_v", "uav0000268_05773_v", "uav0000305_00000_v",
    "uav0000339_00001_v",
]

os.makedirs(OUT, exist_ok=True)
print(f"{'seq':28s} {'raw':>8s} {'ped(cat1,score1)':>18s}")
total_raw = total_ped = 0
for seq in SEQS:
    src = os.path.join(RAW, seq + ".txt")
    kept, raw = [], 0
    with open(src) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw += 1
            c = line.split(",")
            score, cat = c[6], c[7]
            if score == "1" and cat == "1":
                # frame,id,x,y,w,h -> MOT15 with conf=1
                kept.append(f"{c[0]},{c[1]},{c[2]},{c[3]},{c[4]},{c[5]},1,-1,-1,-1")
    with open(os.path.join(OUT, seq + ".txt"), "w") as f:
        f.write("\n".join(kept) + ("\n" if kept else ""))
    total_raw += raw
    total_ped += len(kept)
    print(f"{seq:28s} {raw:8d} {len(kept):18d}")
print(f"{'TOTAL':28s} {total_raw:8d} {total_ped:18d}")
print(f"\nWrote pedestrian-only GT for {len(SEQS)} sequences to {OUT}")
