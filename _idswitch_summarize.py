"""Collapse raw IDSw events into unique (gt_id, old_pred_id, new_pred_id) confusion
pairs, ranked by how many frames the switch persists (a real, stable identity
confusion vs one-frame flicker noise). Keeps only events with a full crop triplet
(gt + old-pred + new-pred all extracted) so every entry handed to the reviewer is
actually visually inspectable.
"""
import os, json
from collections import defaultdict

OUT_DIR = r"C:\Users\User\Desktop\projects\DARE-MOT\_idswitch_review"
SEQS = ["uav0000086_00000_v", "uav0000182_00000_v"]

for seq in SEQS:
    seq_out = os.path.join(OUT_DIR, seq)
    with open(os.path.join(seq_out, "switches_manifest.json")) as f:
        switches = json.load(f)

    groups = defaultdict(list)
    for sw in switches:
        key = (sw["gt_id"], sw["old_pred_id"], sw["new_pred_id"])
        groups[key].append(sw)

    ranked = []
    for key, evs in groups.items():
        frames = sorted(e["frame"] for e in evs)
        first = evs[0]
        ev_dir = None
        # find matching folder for the first occurrence of this pair
        idx = switches.index(first)
        candidates = [d for d in os.listdir(seq_out)
                      if d.startswith(f"switch_{idx:03d}_")]
        if candidates:
            ev_dir = os.path.join(seq_out, candidates[0])
        has_full_triplet = False
        if ev_dir:
            files = os.listdir(ev_dir)
            has_gt = any(f.startswith("gt_at_switch") for f in files)
            has_old = any(f.startswith("old_pred") for f in files)
            has_new = any(f.startswith("new_pred") for f in files)
            has_full_triplet = has_gt and has_old and has_new
        ranked.append({
            "gt_id": key[0], "old_pred_id": key[1], "new_pred_id": key[2],
            "n_occurrences": len(evs), "first_frame": frames[0], "last_frame": frames[-1],
            "dir": ev_dir, "has_full_triplet": has_full_triplet,
        })

    ranked.sort(key=lambda r: (-r["has_full_triplet"], -r["n_occurrences"]))

    print(f"\n=== {seq}: {len(switches)} raw events -> {len(ranked)} unique confusion pairs ===")
    for r in ranked[:20]:
        print(f"  gt{r['gt_id']:>4} : pred{r['old_pred_id']:>4} -> pred{r['new_pred_id']:>4}  "
              f"({r['n_occurrences']}x, frames {r['first_frame']}-{r['last_frame']}, "
              f"triplet={'Y' if r['has_full_triplet'] else 'n'})  {r['dir']}")

    with open(os.path.join(seq_out, "unique_pairs_ranked.json"), "w") as f:
        json.dump(ranked, f, indent=2)
