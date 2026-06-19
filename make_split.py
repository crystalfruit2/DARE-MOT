# Build a clean pedestrian train/val split from the existing all-7 train.json,
# holding out one sequence. Both output JSONs reference the same image files
# (relative paths unchanged) so they share data_dir = VisDrone2019-MOT-val.
import json, os
from collections import Counter

ANN_DIR = r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone_MOT_Format\VisDrone2019-MOT-val\annotations"
SRC = os.path.join(ANN_DIR, "train.json")
HELDOUT = "uav0000339_00001_v"

d = json.load(open(SRC))
vid_by_name = {v["file_name"]: v["id"] for v in d["videos"]}
heldout_vid = vid_by_name[HELDOUT]

def subset(keep_heldout):
    imgs = [im for im in d["images"] if (im["video_id"] == heldout_vid) == keep_heldout]
    keep_img_ids = {im["id"] for im in imgs}
    anns = [a for a in d["annotations"] if a["image_id"] in keep_img_ids]
    return {"images": imgs, "annotations": anns,
            "videos": d["videos"], "categories": d["categories"]}

val = subset(True)       # held-out sequence only
train = subset(False)    # the other 6
json.dump(train, open(os.path.join(ANN_DIR, "train6.json"), "w"))
json.dump(val,   open(os.path.join(ANN_DIR, "val1.json"),  "w"))
print(f"held out: {HELDOUT} (video_id={heldout_vid})")
print(f"train6.json: {len(train['images'])} imgs, {len(train['annotations'])} anns")
print(f"val1.json:   {len(val['images'])} imgs, {len(val['annotations'])} anns")
# sanity: no image overlap
assert not ({im['id'] for im in train['images']} & {im['id'] for im in val['images']})
print("OK: no image overlap between train6 and val1")
