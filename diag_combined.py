# (A) Quantify the raw VisDrone GT we are (mis)ingesting for training:
#     distribution of the 'consider' flag (col 6) and object category (col 7),
#     which the converter currently ignores.
# (B) Run the trained model on TRAINING-domain images and overlay pred(red) vs GT(green),
#     to see if the model works on the domain it was trained on.
import os, cv2, glob, numpy as np, torch
from collections import Counter
from yolox.exp import get_exp
from yolox.utils import postprocess

TRAIN_DIR = r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone_MOT_Format\VisDrone2019-MOT-val"
EXP = r"exps/example/mot/yolox_x_visdrone_finetune.py"
CKPT = r"YOLOX_outputs/yolox_x_visdrone_finetune/latest_ckpt.pth.tar"
OUT = "viz_out_train_pred";
CONF = 0.01

def raw_gt_stats():
    gts = glob.glob(os.path.join(TRAIN_DIR, "*", "gt", "gt.txt"))
    print(f"--- raw GT scan ({len(gts)} sequences) ---")
    consider = Counter(); cats = Counter(); n=0
    for g in gts:
        for line in open(g):
            p = line.strip().split(',')
            if len(p) < 8: continue
            n += 1
            consider[p[6]] += 1
            cats[p[7]] += 1
    print(f"total GT rows: {n}")
    print(f"col6 'consider' flag counts: {dict(consider)}")
    vis = {0:'ignored',1:'pedestrian',2:'people',3:'bicycle',4:'car',5:'van',6:'truck',
           7:'tricycle',8:'awn-tri',9:'bus',10:'motor',11:'others'}
    print("col7 object-category counts:")
    for k,v in sorted(cats.items(), key=lambda kv:-kv[1]):
        print(f"    cat {k:>2} ({vis.get(int(k),'?'):11}): {v}")

def model_on_train():
    os.makedirs(OUT, exist_ok=True)
    exp = get_exp(EXP, None)
    exp.data_num_workers = 0
    # Re-point the eval loader at the TRAINING data so preprocessing matches eval exactly
    exp.data_dir = TRAIN_DIR
    exp.val_ann = "train.json"
    model = exp.get_model().cuda().eval()
    model.load_state_dict(torch.load(CKPT, map_location="cpu")["model"])
    loader = exp.get_eval_loader(1, False)
    ds = loader.dataset; coco = ds.coco; ts = exp.test_size
    for i,(imgs,_,info,ids) in enumerate(loader):
        if i>=3: break
        h,w = int(info[0]), int(info[1]); img_id=int(ids[0])
        fn = coco.loadImgs(img_id)[0]["file_name"]
        orig = cv2.imread(os.path.join(ds.data_dir, ds.name, fn))
        scale = min(ts[0]/h, ts[1]/w)
        with torch.no_grad():
            out = postprocess(model(imgs.float().cuda()), exp.num_classes, CONF, exp.nmsthre)[0]
        npred = 0
        if out is not None:
            b=(out[:,:4]/scale).cpu().numpy(); npred=len(b)
            for x1,y1,x2,y2 in b: cv2.rectangle(orig,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),2)
        gts = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
        for a in gts:
            x,y,bw,bh=a["bbox"]; cv2.rectangle(orig,(int(x),int(y)),(int(x+bw),int(y+bh)),(0,255,0),2)
        p=os.path.join(OUT,f"train_pred_{i}_img{img_id}.jpg"); cv2.imwrite(p,orig)
        print(f"[{i}] {fn}: pred={npred} gt={len(gts)} -> {p}")

if __name__ == "__main__":
    raw_gt_stats()
    print()
    model_on_train()
