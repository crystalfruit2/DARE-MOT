"""Build the yaw-generalization subset of VisDrone-MOT train (2026-09-15).

Selection rule, fixed before any tracker ran on it: every train sequence in which _diag_yaw_scan_2026-09-15.py
(exact GMC estimator, |theta| > 0.01 rad on >= 5 consecutive frames) found at least one yaw segment -> 9 sequences.
D3 was trained on these frames: rankings and switch locations only.
Writes <train>/annotations/trainyaw9_10c.json (video ids 1..9 in name order, ignore_regions dropped).
"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
ANN = r"C:\Users\User\Desktop\datasets\VisDrone2019-MOT-train\annotations"
SRC, DST = os.path.join(ANN, "train_10c.json"), os.path.join(ANN, "trainyaw9_10c.json")
scan = json.load(open(os.path.join(ROOT, "_scratch", "yaw_scan", "train.json")))
names = sorted(s for s, v in scan.items() if v["yaw_segments"])

js = json.load(open(SRC))
by_name = {v["file_name"]: v for v in js["videos"]}
sel = [by_name[n] for n in names]
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
print("wrote", DST, "| videos:", names, "| images", len(imgs), "| annotations", len(anns))
