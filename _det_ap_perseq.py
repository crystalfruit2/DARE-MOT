"""Per-sequence detection AP on val7, per checkpoint (2026-09-10) -- the leak-washout test.

Why: yolox_x_visdrone_mc was warm-started from the 1-class detector trained on train6.json = 6 of the
7 val sequences (only uav0000339_00001_v held out, make_split.py). Its val7 AP50 then fell EVERY epoch
(64.5 -> 60.7 -> ... -> 54.1). best_ckpt (the detector behind every tracker number) is epoch 2.
If the fall is leak washout, the 6 leaked sequences lose far more AP than 339 does; if it is ordinary
drift/overfit, all 7 fall alike.
Pre-registered: LEAK if mean AP50 drop (ep2 -> ep8) on the 6 leaked seqs >= 2x the drop on 339 AND
339 drops < 2 AP points.

Same detection path as MOTEvaluator/_detdump_val7.py (fp16, fused, test_conf 0.001, class-wise NMS 0.7).
Usage: python _det_ap_perseq.py <ckpt> [<ckpt> ...]   -> _scratch/det_ap_perseq/<tag>.json + table
"""
import os
import sys
import json
import contextlib
import io
import torch
from pycocotools.cocoeval import COCOeval
from yolox.exp import get_exp
from yolox.utils import fuse_model, postprocess

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = "exps/example/mot/yolox_x_visdrone_mc_val7.py"
OUT = os.path.join(HERE, "_scratch", "det_ap_perseq")
HELDOUT = "uav0000339_00001_v"


@torch.no_grad()
def detect(ckpt_path, exp, loader):
    exp.model = None                       # Exp.get_model() caches; a 2nd ckpt would load into the fused fp16 model
    model = exp.get_model().cuda().eval()
    ckpt = torch.load(ckpt_path, map_location="cuda:0")
    model.load_state_dict(ckpt["model"], strict=False)
    model = fuse_model(model).half()
    ds = loader.dataset
    th, tw = exp.test_size
    dets = []
    for imgs, _, info_imgs, ids in loader:
        out = model(imgs.cuda().half())
        out = postprocess(out, exp.num_classes, exp.test_conf, exp.nmsthre)[0]
        if out is None:
            continue
        img_h, img_w = int(info_imgs[0][0]), int(info_imgs[1][0])
        scale = min(th / float(img_h), tw / float(img_w))
        out = out.float().cpu()
        b = out[:, :4] / scale
        for k in range(out.shape[0]):
            x1, y1, x2, y2 = b[k].tolist()
            dets.append({"image_id": int(ids[0]),
                         "category_id": int(ds.class_ids[int(out[k, 6])]),
                         "bbox": [x1, y1, x2 - x1, y2 - y1],
                         "score": float(out[k, 4] * out[k, 5])})
    del model
    torch.cuda.empty_cache()
    return dets


def ap_by_seq(coco, dets):
    vids = {v["id"]: v["file_name"] for v in coco.dataset["videos"]}
    by_vid = {}
    for im in coco.dataset["images"]:
        by_vid.setdefault(im["video_id"], []).append(im["id"])
    dt = coco.loadRes(dets)
    res = {}
    for vid, img_ids in sorted(by_vid.items()):
        e = COCOeval(coco, dt, "bbox")
        e.params.imgIds = img_ids
        with contextlib.redirect_stdout(io.StringIO()):
            e.evaluate(); e.accumulate(); e.summarize()
        res[vids[vid]] = {"AP": 100 * e.stats[0], "AP50": 100 * e.stats[1]}
    e = COCOeval(coco, dt, "bbox")
    with contextlib.redirect_stdout(io.StringIO()):
        e.evaluate(); e.accumulate(); e.summarize()
    res["ALL"] = {"AP": 100 * e.stats[0], "AP50": 100 * e.stats[1]}
    return res


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    exp = get_exp(EXP, None)
    exp.data_num_workers = 0
    loader = exp.get_eval_loader(batch_size=1, is_distributed=False)
    coco = loader.dataset.coco
    os.makedirs(OUT, exist_ok=True)
    table = {}
    for ck in sys.argv[1:]:
        tag = os.path.basename(os.path.dirname(ck)) + "__" + os.path.basename(ck).replace(".pth.tar", "")
        cache = os.path.join(OUT, tag + ".json")
        if os.path.exists(cache):
            table[tag] = json.load(open(cache))
        else:
            table[tag] = ap_by_seq(coco, detect(ck, exp, loader))
            json.dump(table[tag], open(cache, "w"), indent=1)
        print(f"done {tag}", flush=True)
    tags = list(table)
    seqs = [s for s in table[tags[0]] if s != "ALL"] + ["ALL"]
    print("\nAP50 per sequence" + ("   (* = held out of the leaky train6 split)"))
    print(f"{'sequence':22s}" + "".join(f"{t.split('__')[-1]:>22s}" for t in tags))
    for s in seqs:
        mark = "*" if s == HELDOUT else " "
        print(f"{s:21s}{mark}" + "".join(f"{table[t][s]['AP50']:22.2f}" for t in tags))
