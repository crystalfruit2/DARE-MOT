"""train_10c.json -> train_5c.json: the class-set control for the clean detector (2026-09-14).

Why: the clean 10-class run (yolox_x_visdrone_10c_mot17init) changed the init AND the class set at
once, and recall fell 14-17% (experiment-log 2026-09-11 cont. 4, read #4). The control that separates
them is a 5-class detector trained from the SAME MOT17 init on the SAME schedule, so the class set is
the only difference.

It has to be derived from train_10c.json, not from train_mc.json: the 10c pipeline also paints the
ignored regions and drops GT boxes >= 50% inside them (official dropObjects). Reusing train_mc.json
would change the ignore handling too, i.e. two things again. So: keep categories 1..5, keep every
image and its ignore_regions untouched, drop the annotations of categories 6..10.

Expected: 817,593 boxes over 24,201 images (= the 5 evaluated classes in train_10c.json).
Usage: python _make_5c_train_json.py [<in train_10c.json>] [<out train_5c.json>]
"""
import json
import os
import sys
from collections import Counter

ANN = r"C:\Users\User\Desktop\datasets\VisDrone2019-MOT-train\annotations"
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ANN, "train_10c.json")
dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ANN, "train_5c.json")

d = json.load(open(src))
keep = {c["id"] for c in d["categories"] if c["id"] <= 5}
names = {c["id"]: c["name"] for c in d["categories"]}
before = Counter(a["category_id"] for a in d["annotations"])

d["categories"] = [c for c in d["categories"] if c["id"] in keep]
d["annotations"] = [a for a in d["annotations"] if a["category_id"] in keep]

n_ign = sum(len(im.get("ignore_regions", [])) for im in d["images"])
print(f"images {len(d['images'])} | ignore regions {n_ign} (untouched)")
print("kept  : " + " · ".join(f"{names[i]} {before[i]}" for i in sorted(keep)))
print("dropped: " + " · ".join(f"{names[i]} {before[i]}" for i in sorted(before) if i not in keep))
print(f"boxes {sum(before.values())} -> {len(d['annotations'])}")
json.dump(d, open(dst, "w"))
print(f"saved {dst}")
