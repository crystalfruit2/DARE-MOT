"""Sequence-AND-class stratified check (2026-07-29, next-decisive-step #1 from
experiment-log's G2-falsification entry).

G2 (warp-SNR gate) is dead, but the underlying CMC scale-inversion HEADLINE
(pedestrian net-harmed, car net-helped by CMC) was only ever validated two ways:
  (a) summed across all 7 sequences per class (_cmc_verify_by_class.py) -- confirms
      direction but can't tell if it's driven by one class-skewed sequence.
  (b) pooled across all classes per within-sequence ratio rank -- found the SIZE
      effect (diagonal alone) goes backwards, a real confound, but never broke that
      down by actual GT class label, only by continuous diagonal-as-size-proxy.

This script closes that gap directly: for each (sequence, class) cell, is CMC's
net effect (removed-added) consistent with the established class direction, or is
the headline actually just "uav0000086 is bad for CMC and happens to be almost
all pedestrian"? Uses the already-logged switch_records.json -- zero new compute.
"""
import json
from collections import defaultdict

CLASS_NAMES = {1: "pedestrian", 2: "car", 3: "van", 4: "truck", 5: "bus"}

with open(r"C:\Users\User\Desktop\projects\DARE-MOT\_cmc_gate_decisive\switch_records.json") as f:
    records = json.load(f)

SEQS = ["uav0000086_00000_v", "uav0000117_02622_v", "uav0000137_00458_v", "uav0000182_00000_v",
        "uav0000268_05773_v", "uav0000305_00000_v", "uav0000339_00001_v"]


def stratify(pair_name):
    cell = defaultdict(lambda: {"added": 0, "removed": 0})
    for rec in records:
        if rec["pair"] != pair_name:
            continue
        cell[(rec["seq"], rec["cat"])][rec["label"]] += 1

    print(f"\n=== {pair_name} pair: net(removed-added) by (sequence x class) ===")
    header = f"{'sequence':22s}" + "".join(f"{CLASS_NAMES.get(c,c):>14s}" for c in range(1, 6))
    print(header)
    seq_totals = defaultdict(lambda: {1: None, 2: None, 3: None, 4: None, 5: None})
    for seq in SEQS:
        row = f"{seq:22s}"
        for cat in range(1, 6):
            c = cell.get((seq, cat))
            if c is None or (c["added"] + c["removed"]) == 0:
                row += f"{'--':>14s}"
            else:
                net = c["removed"] - c["added"]
                n = c["added"] + c["removed"]
                row += f"{f'{net:+d} (n={n})':>14s}"
                seq_totals[seq][cat] = net
        print(row)

    # sign-consistency check: for the two volume classes, how many sequences agree
    # with the established direction (pedestrian net<0, car net>0)?
    for cat, expect_sign, label in [(1, -1, "pedestrian (expect net<0, CMC harmful)"),
                                      (2, +1, "car (expect net>0, CMC helpful)")]:
        agree, disagree, no_data = 0, 0, 0
        detail = []
        for seq in SEQS:
            net = seq_totals[seq][cat]
            if net is None:
                no_data += 1
                continue
            sign = 1 if net > 0 else (-1 if net < 0 else 0)
            tag = "AGREE" if sign == expect_sign else ("FLAT" if sign == 0 else "DISAGREE")
            if tag == "AGREE":
                agree += 1
            elif tag == "DISAGREE":
                disagree += 1
            detail.append(f"{seq}={net:+d}[{tag}]")
        print(f"\n  {label}: {agree} agree / {disagree} disagree / {no_data} no-data (of {len(SEQS)} seqs)")
        print("    " + ", ".join(detail))

    # totals excluding uav0000086 (the ~100% pedestrian outlier flagged earlier)
    for cat in (1, 2):
        total_all = sum(v[cat] for v in seq_totals.values() if v[cat] is not None)
        total_excl = sum(v[cat] for s, v in seq_totals.items()
                          if v[cat] is not None and s != "uav0000086_00000_v")
        print(f"  {CLASS_NAMES[cat]}: net summed all-7={total_all:+d}, "
              f"excl-uav0000086={total_excl:+d}")


stratify("dare")
stratify("bytetrack")
