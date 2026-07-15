# ReID-feature separability diagnostic.
# Question: do the tracker's MobileNetV2 appearance features separate same-ID from
# different-ID crops on VisDrone aerial pedestrians? This decides whether training a
# ReID head has any hope (Path A).
#   AUC(same<diff) ~0.5 => no identity signal (features hopeless).
#   AUC ~0.7-0.9      => real signal a trained head can amplify.
import os, cv2, numpy as np, torch
import torchvision.transforms as T
from torchvision.models import mobilenet_v2
from collections import defaultdict

VAL = r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone_MOT_Format\VisDrone2019-MOT-val"
SEQS = ["uav0000086_00000_v", "uav0000117_02622_v", "uav0000137_00458_v",
        "uav0000182_00000_v", "uav0000268_05773_v", "uav0000305_00000_v",
        "uav0000339_00001_v"]
MAX_IDS, MAX_CROPS = 60, 8

device = "cuda" if torch.cuda.is_available() else "cpu"
ext = mobilenet_v2(pretrained=True).features.to(device).eval()
tf = T.Compose([T.ToTensor(), T.Resize((128, 64)),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
rng = np.random.default_rng(0)

def feat(crop):
    if crop.ndim == 3 and crop.shape[2] == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    t = tf(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        f = ext(t).mean([2, 3]).squeeze().cpu().numpy()
    return f / (np.linalg.norm(f) + 1e-6)

print("seq | ids | sameN diffN | same_dist | diff_dist | AUC(same<diff) | d'")
for seq in SEQS:
    rows = np.loadtxt(os.path.join(VAL, seq, "gt", "gt.txt"), delimiter=",", usecols=(0,1,2,3,4,5))
    byid = defaultdict(list)
    for r in rows:
        byid[int(r[1])].append((int(r[0]), r[2], r[3], r[4], r[5]))
    ids = [i for i in byid if len(byid[i]) >= 2][:MAX_IDS]
    feats_by_id = {}
    imgdir = os.path.join(VAL, seq, "img1")
    for tid in ids:
        dets = byid[tid]
        idxs = np.unique(np.linspace(0, len(dets)-1, min(MAX_CROPS, len(dets))).astype(int))
        fl = []
        for k in idxs:
            fr, x, y, w, h = dets[k]
            img = cv2.imread(os.path.join(imgdir, f"{fr:07d}.jpg"))
            if img is None:
                continue
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(img.shape[1], int(x+w)), min(img.shape[0], int(y+h))
            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                fl.append(feat(crop))
        if len(fl) >= 2:
            feats_by_id[tid] = np.array(fl)

    same = []
    for F in feats_by_id.values():
        for a in range(len(F)):
            for b in range(a+1, len(F)):
                same.append(1 - float(np.dot(F[a], F[b])))
    idl = list(feats_by_id.keys())
    diff = []
    for _ in range(4000):
        i, j = rng.choice(len(idl), 2, replace=False)
        Fi, Fj = feats_by_id[idl[i]], feats_by_id[idl[j]]
        diff.append(1 - float(np.dot(Fi[rng.integers(len(Fi))], Fj[rng.integers(len(Fj))])))
    same, diff = np.array(same), np.array(diff)
    ss = rng.choice(same, min(4000, len(same))); dd = rng.choice(diff, min(4000, len(diff)))
    auc = float(np.mean(ss[:, None] < dd[None, :]))
    dprime = (diff.mean() - same.mean()) / np.sqrt(0.5*(same.var()+diff.var()) + 1e-9)
    print(f"{seq} | {len(feats_by_id)} | {len(same)} {len(diff)} | "
          f"{same.mean():.3f}±{same.std():.3f} | {diff.mean():.3f}±{diff.std():.3f} | {auc:.3f} | {dprime:.2f}")
print("DONE")
