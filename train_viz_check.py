# Check the TRAINING annotations: do train.json boxes land on real objects,
# and do the actual image pixel sizes match the JSON width/height?
import os, cv2, json, numpy as np

TRAIN_DIR = r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone_MOT_Format\VisDrone2019-MOT-val"
JSON = os.path.join(TRAIN_DIR, "annotations", "train.json")
OUT = "viz_out_train"; os.makedirs(OUT, exist_ok=True)

d = json.load(open(JSON))
imgs = {im["id"]: im for im in d["images"]}
from collections import defaultdict
anns_by_img = defaultdict(list)
for a in d["annotations"]:
    anns_by_img[a["image_id"]].append(a)

# look at the first few images that have annotations
ids = [im["id"] for im in d["images"]][:4]
for k, img_id in enumerate(ids):
    im = imgs[img_id]
    fn = im["file_name"]
    path = os.path.join(TRAIN_DIR, fn)
    arr = cv2.imread(path)
    real_h, real_w = (arr.shape[0], arr.shape[1]) if arr is not None else (None, None)
    print(f"[{k}] img_id={img_id} {fn}")
    print(f"     JSON says w={im['width']} h={im['height']}  |  ACTUAL file w={real_w} h={real_h}")
    anns = anns_by_img[img_id]
    if anns:
        b = np.array([a["bbox"] for a in anns])  # x,y,w,h
        print(f"     {len(anns)} boxes  x[{b[:,0].min():.0f},{b[:,0].max():.0f}] "
              f"y[{b[:,1].min():.0f},{b[:,1].max():.0f}] w[mean {b[:,2].mean():.0f}] h[mean {b[:,3].mean():.0f}]")
    if arr is not None:
        for a in anns:
            x,y,w,h = a["bbox"]
            cv2.rectangle(arr, (int(x),int(y)), (int(x+w),int(y+h)), (0,255,0), 2)
        p = os.path.join(OUT, f"train_gt_{k}_img{img_id}.jpg")
        cv2.imwrite(p, arr); print(f"     saved {p}")
