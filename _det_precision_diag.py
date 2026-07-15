"""Per-sequence detection-box precision diagnostic (2026-07-15).
Question: does raw YOLOX detection-box precision (IoU vs GT) vary by sequence
in a way that explains the real per-sequence IDSw damage pattern, when the
GT-box ReID feature-quality diagnostic (_reid_diag.py) does not?
Uses the same checkpoint/exp as the real-appearance ablation re-run.
"""
import os
import numpy as np
import torch
from collections import defaultdict

from yolox.exp import get_exp
from yolox.utils import postprocess, fuse_model

EXP_FILE = "exps/example/mot/yolox_x_visdrone_finetune_val7.py"
CKPT = "YOLOX_outputs/yolox_x_visdrone_finetune/best_ckpt.pth.tar"

def iou_xyxy(a, b):
    # a: (N,4) xyxy, b: (M,4) xyxy -> (N,M) IoU matrix
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ix1 = np.maximum(ax1, bx1); iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2); iy2 = np.minimum(ay2, by2)
    iw = np.clip(ix2 - ix1, 0, None); ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / np.clip(union, 1e-9, None)

def main():
    exp = get_exp(EXP_FILE, None)
    model = exp.get_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    ckpt = torch.load(CKPT, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model = fuse_model(model)
    model = model.half()

    val_loader = exp.get_eval_loader(1, False)
    dataset = val_loader.dataset

    best_ious = defaultdict(list)   # seq -> list of best-match IoU per GT box
    matched = defaultdict(list)     # seq -> list of bool (IoU > 0.5)

    tensor_type = torch.cuda.HalfTensor if torch.cuda.is_available() else torch.HalfTensor

    with torch.no_grad():
        for idx in range(len(dataset)):
            img, target, img_info, img_id = dataset[idx]
            res, raw_info, file_name = dataset.annotations[idx]
            seq = file_name.split('/')[0]

            gt_boxes = res[:, 0:4]  # x1,y1,x2,y2 in original coords
            if gt_boxes.shape[0] == 0:
                continue

            imgs = torch.from_numpy(img).unsqueeze(0).type(tensor_type).to(device)
            outputs = model(imgs)
            outputs = postprocess(outputs, exp.num_classes, exp.test_conf, exp.nmsthre)
            output = outputs[0]

            img_h, img_w = raw_info[0], raw_info[1]
            scale = min(exp.test_size[0] / float(img_h), exp.test_size[1] / float(img_w))

            if output is None:
                for _ in range(gt_boxes.shape[0]):
                    best_ious[seq].append(0.0)
                    matched[seq].append(False)
                continue

            det_boxes = (output[:, 0:4] / scale).cpu().numpy()
            ious = iou_xyxy(gt_boxes.astype(np.float64), det_boxes.astype(np.float64))
            best = ious.max(axis=1) if ious.shape[1] > 0 else np.zeros(gt_boxes.shape[0])
            for b in best:
                best_ious[seq].append(float(b))
                matched[seq].append(bool(b > 0.5))

            if idx % 300 == 0:
                print(f"...{idx}/{len(dataset)}")

    print("\nseq | n_gt | mean_best_IoU | match_rate(IoU>0.5)")
    for seq in sorted(best_ious):
        bi = np.array(best_ious[seq])
        mr = np.mean(matched[seq])
        print(f"{seq:22s} | {len(bi):5d} | {bi.mean():.3f} | {mr:.3f}")
    print("DONE")

if __name__ == "__main__":
    main()
