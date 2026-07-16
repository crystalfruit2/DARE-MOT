"""Fine-tune OSNet-AIN on VisDrone pedestrian identities (Plan A, Task #3).

Starts from the verified-clean domain-generalized checkpoint
(reid_weights/osnet_ain_x1_0_dg_clean.pth), trains softmax+triplet on the crops
built by _build_reid_train.py (reid_train/), and saves a drop-in checkpoint that
the tracker loads via DARE_REID=osnet DARE_REID_MODEL=osnet_ain_x1_0
DARE_REID_WEIGHTS=reid_weights/osnet_ain_x1_0_visdrone_ft.pth.

Only the backbone matters at inference (FeatureExtractor discards the classifier),
so we save the full state_dict; load_pretrained_weights matches by shape.

Deterministic by default (seed 0, cudnn.deterministic) to match the project's
noise-control ethos.

Usage:
    python _finetune_reid.py [--epochs 25] [--batch 64] [--num-inst 4]
                             [--lr 3e-4] [--seed 0] [--amp]
"""
import os
import glob
import random
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image

from torchreid.reid.models import build_model
from torchreid.reid.utils import load_pretrained_weights
from torchreid.reid.data.sampler import RandomIdentitySampler
from torchreid.reid.losses import CrossEntropyLoss, TripletLoss

HERE = os.path.dirname(os.path.abspath(__file__))
CROP_DIR = os.path.join(HERE, "reid_train")
INIT_W = os.path.join(HERE, "reid_weights", "osnet_ain_x1_0_dg_clean.pth")
OUT_W = os.path.join(HERE, "reid_weights", "osnet_ain_x1_0_visdrone_ft.pth")


def parse_name(fn):
    # {pid:07d}_c{cam:03d}s1_{frame:06d}_00.jpg
    base = os.path.basename(fn)
    pid = int(base[:7])
    cam = int(base.split("_c")[1][:3])
    return pid, cam


class CropDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples          # list of (path, pid, cam)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, pid, cam = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), pid, cam


def build_samples():
    files = sorted(glob.glob(os.path.join(CROP_DIR, "*.jpg")))
    if not files:
        raise SystemExit(f"no crops in {CROP_DIR} — run _build_reid_train.py first")
    raw = [(f, *parse_name(f)) for f in files]
    pids = sorted({p for _, p, _ in raw})
    # crops are already contiguous, but remap defensively so num_classes == max+1
    remap = {p: i for i, p in enumerate(pids)}
    samples = [(f, remap[p], c) for f, p, c in raw]
    return samples, len(pids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--num-inst", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=0, help="smoke test: stop after N steps")
    ap.add_argument("--out", default=OUT_W)
    args = ap.parse_args()

    # determinism
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples, num_classes = build_samples()
    print(f"crops={len(samples)}  num_classes={num_classes}  device={device}")

    train_tf = T.Compose([
        T.Resize((256, 128)),
        T.RandomHorizontalFlip(0.5),
        T.Pad(10),
        T.RandomCrop((256, 128)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.5, value=0),
    ])
    ds = CropDataset(samples, train_tf)
    sampler = RandomIdentitySampler(samples, args.batch, args.num_inst)
    loader = DataLoader(ds, batch_size=args.batch, sampler=sampler,
                        num_workers=args.workers, pin_memory=True, drop_last=True)

    model = build_model("osnet_ain_x1_0", num_classes=num_classes,
                        loss="triplet", pretrained=False)
    load_pretrained_weights(model, INIT_W)   # backbone from DG-clean; classifier fresh
    model = model.to(device)

    xent = CrossEntropyLoss(num_classes=num_classes, use_gpu=(device == "cuda"),
                            label_smooth=True)
    tri = TripletLoss(margin=args.margin)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    steps = len(loader)
    for ep in range(args.epochs):
        model.train()
        run_loss = run_acc = seen = 0
        for it, (imgs, pids, _) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            pids = pids.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits, feats = model(imgs)          # loss='triplet' -> (y, v)
            # compute losses in fp32: torchreid's TripletLoss allocates its distance
            # matrix in float and addmm_ rejects mixed Float/Half under AMP
            loss = xent(logits.float(), pids) + tri(feats.float(), pids)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            run_loss += loss.item() * imgs.size(0)
            run_acc += (logits.argmax(1) == pids).sum().item()
            seen += imgs.size(0)
            if (it + 1) % 50 == 0 or (args.max_steps and it + 1 <= 3):
                print(f"  ep{ep+1:02d} [{it+1:4d}/{steps}] "
                      f"loss={run_loss/seen:.3f} acc={run_acc/seen:.3f}", flush=True)
            if args.max_steps and it + 1 >= args.max_steps:
                print(f"SMOKE OK: {args.max_steps} steps ran, loss={run_loss/seen:.3f}", flush=True)
                return
        sched.step()
        print(f"epoch {ep+1:02d}/{args.epochs}  loss={run_loss/seen:.3f} "
              f"acc={run_acc/seen:.3f}  lr={sched.get_last_lr()[0]:.2e}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
