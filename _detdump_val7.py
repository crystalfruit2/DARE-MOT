"""Full val7 raw-detection dump (2026-09-10) -- the input population for the FN/FP decomposition.

Replicates the MOTEvaluator detection path exactly (fp16, fused, 5-class MC detector, test_conf
0.001, class-wise batched NMS 0.7 -- same code as yolox.utils.postprocess) but additionally keeps
the full 5-way class-probability vector, which postprocess() throws away. Boxes are rescaled to
original-image pixels the same way BYTETracker.update() does (divide by
min(test_h/img_h, test_w/img_w)).

Out: _scratch/detdump_val7/<seq>.csv
     frame,x1,y1,x2,y2,obj,cls_conf,cls(0..4),score(=obj*cls_conf),p0,p1,p2,p3,p4
No tracker, no lap solver -- runs under Smart App Control.
"""
import os
import csv
import torch
import torchvision
from yolox.exp import get_exp
from yolox.utils import fuse_model

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = "exps/example/mot/yolox_x_visdrone_mc_val7.py"
CKPT = os.path.join(HERE, "YOLOX_outputs", "yolox_x_visdrone_mc", "best_ckpt.pth.tar")
OUT = os.path.join(HERE, "_scratch", "detdump_val7")


def postprocess_keep_probs(pred, num_classes, conf_thre, nms_thre):
    """yolox.utils.postprocess, verbatim logic, plus the class-probability columns."""
    box = pred.new(pred.shape)
    box[:, :, 0] = pred[:, :, 0] - pred[:, :, 2] / 2
    box[:, :, 1] = pred[:, :, 1] - pred[:, :, 3] / 2
    box[:, :, 2] = pred[:, :, 0] + pred[:, :, 2] / 2
    box[:, :, 3] = pred[:, :, 1] + pred[:, :, 3] / 2
    pred[:, :, :4] = box[:, :, :4]
    image_pred = pred[0]
    class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)
    conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
    det = torch.cat((image_pred[:, :5], class_conf, class_pred.float(),
                     image_pred[:, 5:5 + num_classes]), 1)[conf_mask]
    if not det.size(0):
        return None
    keep = torchvision.ops.batched_nms(det[:, :4], det[:, 4] * det[:, 5], det[:, 6], nms_thre)
    return det[keep]


def main():
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    exp = get_exp(EXP, None)
    exp.data_num_workers = 0
    model = exp.get_model().cuda().eval()
    ckpt = torch.load(CKPT, map_location="cuda:0")
    model.load_state_dict(ckpt["model"], strict=False)
    model = fuse_model(model).half()
    loader = exp.get_eval_loader(batch_size=1, is_distributed=False)
    th, tw = exp.test_size
    os.makedirs(OUT, exist_ok=True)

    writers, files = {}, {}
    n = 0
    with torch.no_grad():
        for imgs, _, info_imgs, ids in loader:
            img_h, img_w = float(info_imgs[0]), float(info_imgs[1])
            frame_id = int(info_imgs[2].item())
            seq = info_imgs[4][0].split("/")[0]
            if seq not in writers:
                files[seq] = open(os.path.join(OUT, seq + ".csv"), "w", newline="")
                writers[seq] = csv.writer(files[seq])
                writers[seq].writerow(["frame", "x1", "y1", "x2", "y2", "obj", "cls_conf", "cls",
                                       "score", "p0", "p1", "p2", "p3", "p4"])
            out = model(imgs.type(torch.cuda.HalfTensor))
            det = postprocess_keep_probs(out, exp.num_classes, exp.test_conf, exp.nmsthre)
            n += 1
            if det is None:
                continue
            det = det.float().cpu().numpy()
            scale = min(th / img_h, tw / img_w)
            det[:, :4] /= scale
            w = writers[seq]
            for r in det:
                w.writerow([frame_id, f"{r[0]:.2f}", f"{r[1]:.2f}", f"{r[2]:.2f}", f"{r[3]:.2f}",
                            f"{r[4]:.5f}", f"{r[5]:.5f}", int(r[6]), f"{r[4]*r[5]:.5f}"]
                           + [f"{p:.5f}" for p in r[7:12]])
            if n % 500 == 0:
                print(f"{n} frames", flush=True)
    for f in files.values():
        f.close()
    print(f"done: {n} frames, {len(files)} sequences -> {OUT}")


if __name__ == "__main__":
    main()
