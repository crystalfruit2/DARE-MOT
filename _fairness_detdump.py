# Detector-only fairness dump. Replicates the exact MOTEvaluator detection path:
#   imgs.half() -> model(imgs) -> postprocess(num_classes, test_conf, nmsthre)
# Dumps outputs[0] (the per-frame detections handed to the tracker) for the first N frames.
# Run from a repo root so `import yolox` resolves to THAT repo's tracker stack.
import sys, torch, numpy as np
from yolox.exp import get_exp
from yolox.utils import postprocess, fuse_model

EXP  = r"exps/example/mot/yolox_x_visdrone_finetune.py"
CKPT = r"C:\Users\User\Desktop\projects\DARE-MOT\YOLOX_outputs\yolox_x_visdrone_finetune\best_ckpt.pth.tar"
N_FRAMES = 3


def main():
    # Mirror tools/track.py runtime flags exactly.
    torch.backends.cudnn.benchmark = True

    exp = get_exp(EXP, None)
    exp.data_num_workers = 0   # 3-frame dump: avoid Windows dataloader spawn (no __main__ re-import deadlock)
    model = exp.get_model().cuda().eval()
    ckpt = torch.load(CKPT, map_location="cuda:0")
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    model = fuse_model(model)          # --fuse
    model = model.half()               # --fp16

    loader = exp.get_eval_loader(batch_size=1, is_distributed=False)

    lines = [f"missing={len(missing)} unexpected={len(unexpected)}",
             f"num_classes={exp.num_classes} test_conf={exp.test_conf} nms={exp.nmsthre} test_size={exp.test_size}"]
    with torch.no_grad():
        for i, (imgs, _, info_imgs, ids) in enumerate(loader):
            if i >= N_FRAMES:
                break
            frame_id = int(info_imgs[2].item())
            imgs = imgs.type(torch.cuda.HalfTensor)
            outputs = model(imgs)
            outputs = postprocess(outputs, exp.num_classes, exp.test_conf, exp.nmsthre)
            det = outputs[0]
            if det is None:
                lines.append(f"frame {frame_id}: 0 dets")
                continue
            det = det.cpu().float().numpy()
            lines.append(f"frame {frame_id}: {det.shape[0]} dets")
            for r in det:
                x1, y1, x2, y2 = r[0], r[1], r[2], r[3]
                score = r[4] * r[5]   # obj_conf * class_conf (what BYTETracker reads)
                lines.append(f"  {x1:.3f} {y1:.3f} {x2:.3f} {y2:.3f} {score:.5f}")

    txt = "\n".join(lines)
    print(txt)
    with open("_fairness_detdump.out", "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
