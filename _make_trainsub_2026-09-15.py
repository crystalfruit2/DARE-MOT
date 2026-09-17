"""Build the out-of-val tracker-ranking subset of VisDrone-MOT train (2026-09-15).

Selection rule, fixed before any tracker ran on it: sort the 56 train sequences by name and take every 8th,
starting at index 4 -> 7 sequences (3,101 frames; val7 has 2,846). No sequence was looked at before the rule.
The detector D3 was TRAINED on these frames, so detection is inflated and absolute numbers are not
reportable; the subset only tests whether tracker RANKINGS chosen on val7 hold on sequences never used for
tracker/CMC selection.

Writes <train>/annotations/trainsub7_10c.json (from train_10c.json): video ids remapped to 1..7 in name order
(mot_evaluator indexes video_names[video_id - 1]), image ids kept (ascending, contiguous per video),
ignore_regions dropped (eval never paints them).
"""
import json
import os

ANN = r"C:\Users\User\Desktop\datasets\VisDrone2019-MOT-train\annotations"
SRC, DST = os.path.join(ANN, "train_10c.json"), os.path.join(ANN, "trainsub7_10c.json")

js = json.load(open(SRC))
vids = sorted(js["videos"], key=lambda v: v["file_name"])
sel = [vids[i] for i in range(4, len(vids), 8)]
remap = {v["id"]: k + 1 for k, v in enumerate(sel)}
imgs = []
for im in sorted((i for i in js["images"] if i["video_id"] in remap), key=lambda i: i["id"]):
    im = {k: v for k, v in im.items() if k != "ignore_regions"}
    im["video_id"] = remap[im["video_id"]]
    imgs.append(im)
order = [i["video_id"] for i in imgs]
assert order == sorted(order), "images of one video must be contiguous and in video order"
img_ids = {i["id"] for i in imgs}
anns = [dict(a, video_id=remap.get(a.get("video_id"), a.get("video_id"))) for a in js["annotations"] if a["image_id"] in img_ids]
out = {"videos": [{"id": remap[v["id"]], "file_name": v["file_name"]} for v in sel],
       "images": imgs, "annotations": anns, "categories": js["categories"]}
json.dump(out, open(DST, "w"))
print("wrote", DST, "| videos:", [v["file_name"] for v in sel], "| images", len(imgs), "| annotations", len(anns))
