"""Sanity check (2026-07-29): does _cmc_idsw_join.py's own switch-attribution, broken out
by GT class, reproduce the DIRECTION of the already-established official-py-motmetrics
per-class CMC finding (pedestrian IDSw massively worse with CMC, car IDSw better with CMC
-- novelty-triage-2026-07-28 / project_daremot_novelty_pivot)? If not, the new ratio-based
result is not yet trustworthy and the discrepancy needs to be understood before it's used
for anything.
"""
import json
from collections import defaultdict

import numpy as np

VAL_DIR = r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone_MOT_Format_MC\VisDrone2019-MOT-val"
CLASS_NAMES = {1: "pedestrian", 2: "car", 3: "van", 4: "truck", 5: "bus"}

with open(r"C:\Users\User\Desktop\projects\DARE-MOT\_cmc_gate_decisive\switch_records.json") as f:
    records = json.load(f)

# gt class lookup: (seq, frame, gid) -> category
gt_class = {}
SEQS = ["uav0000086_00000_v", "uav0000117_02622_v", "uav0000137_00458_v", "uav0000182_00000_v",
        "uav0000268_05773_v", "uav0000305_00000_v", "uav0000339_00001_v"]
for seq in SEQS:
    rows = np.loadtxt(f"{VAL_DIR}\\{seq}\\gt\\gt.txt", delimiter=",")
    for r in rows:
        fr, gid, cat = int(r[0]), int(r[1]), int(r[7])
        gt_class[(seq, fr, gid)] = cat

net = defaultdict(lambda: {"added": 0, "removed": 0})
ratios_by_cat = defaultdict(list)
n_no_class = 0
all_cats_seen = defaultdict(int)
for rec in records:
    cat = gt_class.get((rec["seq"], rec["frame"], rec["gt_id"]))
    if cat is not None:
        all_cats_seen[cat] += 1
    if rec["pair"] != "dare":
        continue
    if cat is None:
        n_no_class += 1
        continue
    net[cat][rec["label"]] += 1
    ratios_by_cat[cat].append(rec["ratio"])

print(f"(no-class-match count across ALL records: {n_no_class}; category value distribution across ALL records: {dict(all_cats_seen)})\n")

print("DARE pair (mc_dare_cmc vs mc_dare), by GT class -- from THIS diagnostic's own switch definition:")
print(f"{'class':12s} {'added(CMC worse)':18s} {'removed(CMC better)':20s} {'net(removed-added)':18s} {'mean ratio':10s} {'n':5s}")
for cat in sorted(net):
    a, r = net[cat]["added"], net[cat]["removed"]
    rr = ratios_by_cat[cat]
    print(f"{CLASS_NAMES.get(cat, cat):12s} {a:18d} {r:20d} {r - a:+18d} {np.mean(rr):10.4f} {len(rr):5d}")

print("\nEstablished official py-motmetrics finding (novelty-triage-2026-07-28, no-CMC -> CMC IDSw):")
print("  pedestrian: 127 -> 564  (CMC much WORSE, +437)")
print("  car:        138 -> 75   (CMC BETTER, -63)")
