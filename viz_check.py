# Diagnostic: overlay predicted (red) vs GT (green) boxes for the VisDrone fine-tune,
# using the repo's own eval dataloader + postprocess so coords match COCOEvaluator exactly.
import os, cv2, numpy as np, torch
from yolox.exp import get_exp
from yolox.utils import postprocess

EXP = r"exps/example/mot/yolox_x_visdrone_finetune.py"
CKPT = r"YOLOX_outputs/yolox_x_visdrone_finetune/latest_ckpt.pth.tar"
OUT = "viz_out"
N_IMAGES = 4
CONF = 0.01   # well above eval's 0.001 but low enough to show the model's best guesses


def stats(arr, label):
    if len(arr) == 0:
        print(f"   {label}: NONE"); return
    a = np.array(arr)
    cx = (a[:,0]+a[:,2])/2; cy = (a[:,1]+a[:,3])/2; w = a[:,2]-a[:,0]; h = a[:,3]-a[:,1]
    print(f"   {label}: n={len(a)}  cx[{cx.min():.0f},{cx.max():.0f}] cy[{cy.min():.0f},{cy.max():.0f}] "
          f"w[mean {w.mean():.0f}] h[mean {h.mean():.0f}]")


def main():
    os.makedirs(OUT, exist_ok=True)
    exp = get_exp(EXP, None)
    exp.data_num_workers = 0   # avoid Windows spawn (single-process loader is fine for a few images)
    model = exp.get_model().cuda().eval()
    ckpt = torch.load(CKPT, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    print(f"loaded {CKPT} (epoch in ckpt: {ckpt.get('start_epoch')})")

    loader = exp.get_eval_loader(1, False)
    ds = loader.dataset
    coco = ds.coco
    ts = exp.test_size  # (h, w)

    for i, (imgs, _, info_imgs, ids) in enumerate(loader):
        if i >= N_IMAGES:
            break
        img_h, img_w = int(info_imgs[0]), int(info_imgs[1])
        img_id = int(ids[0])
        fn = coco.loadImgs(img_id)[0]["file_name"]
        orig = cv2.imread(os.path.join(ds.data_dir, ds.name, fn))
        scale = min(ts[0]/img_h, ts[1]/img_w)

        with torch.no_grad():
            out = postprocess(model(imgs.float().cuda()), exp.num_classes, CONF, exp.nmsthre)[0]

        print(f"\n[{i}] img_id={img_id} file={fn} orig=({img_w}x{img_h}) scale={scale:.4f}")
        pred_xyxy = []
        if out is not None:
            b = (out[:, :4] / scale).cpu().numpy()
            sc = (out[:, 4] * out[:, 5]).cpu().numpy()
            for (x1,y1,x2,y2), s in zip(b, sc):
                pred_xyxy.append([x1,y1,x2,y2])
                cv2.rectangle(orig, (int(x1),int(y1)), (int(x2),int(y2)), (0,0,255), 2)
            print(f"   pred score range: [{sc.min():.3f}, {sc.max():.3f}]")
        stats(pred_xyxy, "PRED")

        gt_xyxy = []
        for a in coco.loadAnns(coco.getAnnIds(imgIds=[img_id])):
            x,y,w,h = a["bbox"]; gt_xyxy.append([x,y,x+w,y+h])
            cv2.rectangle(orig, (int(x),int(y)), (int(x+w),int(y+h)), (0,255,0), 2)
        stats(gt_xyxy, "GT  ")

        p = os.path.join(OUT, f"overlay_{i}_img{img_id}.jpg")
        cv2.imwrite(p, orig); print(f"   saved {p}")

    print("\nDONE. Red=pred, Green=GT.")


if __name__ == "__main__":
    main()
