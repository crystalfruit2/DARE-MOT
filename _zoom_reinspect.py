"""Zoom re-inspection of borderline detections (2026-09-10).

Four cheap signals failed to separate the ~9.5k missed-but-detected objects from ~42k clutter
boxes below the tracker threshold (score AUC 0.687; persistence 0.55; ground motion 0.60;
track-derived occupancy 0.625, 0.698 combined). None of them changes RESOLUTION. Test: crop a
window around each free candidate with score in [LO, 0.6), upscale ~3x, re-run the SAME detector,
and score the candidate by the zoomed detection that overlaps it.

Crop: side S = max(192, 3*max(w,h)) px centred on the candidate (clamped to the image), fed at
576x576 through yolox preproc (same normalisation as ValTransform), fp16, fused, conf 0.001,
class-wise NMS 0.7. Zoomed boxes are mapped back to frame pixels.
Features: z_any  = best zoomed score among boxes with IoU>=0.5 to the candidate (any class)
          z_same = same, restricted to the candidate's class
Out: _scratch/zoom_reinspect.csv + AUC / precision table.
"""
import os
import sys
import csv
import numpy as np
import cv2
import torch
from scipy.optimize import linear_sum_assignment
from yolox.exp import get_exp
from yolox.utils import fuse_model, postprocess
from yolox.data.data_augment import preproc
from _fn_decomp import SEQS, NAMES, MC_GT, iou, load_mot, load_dump, match
from _mid_zone import auc

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = "exps/example/mot/yolox_x_visdrone_mc_val7.py"
CKPT = os.path.join(HERE, "YOLOX_outputs", "yolox_x_visdrone_mc", "best_ckpt.pth.tar")
IMG_ROOT = r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone2019-MOT-val\sequences"
LO, HI, Z, BS = float(os.environ.get("ZR_LO", "0.3")), 0.6, 576, 32
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def candidates(res_dir):
    out = []  # (seq, frame, box, cls, score, label)
    for seq in SEQS:
        gt = load_mot(MC_GT.format(seq=seq), conf_min=1)
        ts = load_mot(os.path.join(res_dir, seq + ".txt"))
        dets = load_dump(seq)
        for fr in sorted(dets):
            D = dets[fr]
            db = np.array([d[0] for d in D]).reshape(-1, 4); dc = np.array([d[1] for d in D])
            ds = np.array([d[2] for d in D])
            G = gt.get(fr, []); T = ts.get(fr, [])
            gb = np.array([g[0] for g in G]).reshape(-1, 4); gc = np.array([g[1] for g in G])
            tb = np.array([t[0] for t in T]).reshape(-1, 4); tc = np.array([t[1] for t in T])
            g_hit = np.zeros(len(G), bool)
            for c in NAMES:
                gi = np.where(gc == c)[0]; ti = np.where(tc == c)[0]
                mg, _ = match(gb[gi], tb[ti]); g_hit[gi[list(mg)]] = True
            sel = np.where((ds >= LO) & (ds < HI))[0]
            if len(sel) and len(T):
                sel = sel[iou(db[sel], tb).max(1) < 0.3]
            if not len(sel):
                continue
            lab = np.array(["bg"] * len(sel), dtype=object)
            for c in NAMES:
                si = np.where(dc[sel] == c)[0]; gi = np.where(gc == c)[0]
                if len(si) == 0 or len(gi) == 0:
                    continue
                m = iou(db[sel[si]], gb[gi]); cost = np.where(m >= 0.5, 1 - m, 1e6)
                r, cc = linear_sum_assignment(cost)
                for a, b in zip(r, cc):
                    if cost[a, b] < 1e5:
                        lab[si[a]] = "rescue" if not g_hit[gi[b]] else "redund"
            for k, i in enumerate(sel):
                if lab[k] != "redund":
                    out.append((seq, fr, db[i], int(dc[i]), float(ds[i]), lab[k]))
    return out


def img_path(seq, fr):
    return os.path.join(IMG_ROOT, seq, f"{fr:07d}.jpg")


def main(res_dir):
    cands = candidates(res_dir)
    print(f"{len(cands)} candidates in [{LO},{HI}) "
          f"(rescue {sum(c[5] == 'rescue' for c in cands)}, bg {sum(c[5] == 'bg' for c in cands)})", flush=True)
    torch.backends.cudnn.benchmark = False
    exp = get_exp(EXP, None)
    model = exp.get_model().cuda().eval()
    model.load_state_dict(torch.load(CKPT, map_location="cuda:0")["model"], strict=False)
    model = fuse_model(model).half()

    by_frame = {}
    for j, c in enumerate(cands):
        by_frame.setdefault((c[0], c[1]), []).append(j)
    zany = np.zeros(len(cands)); zsame = np.zeros(len(cands))
    batch, meta = [], []

    def flush():
        if not batch:
            return
        x = torch.from_numpy(np.stack(batch)).cuda().half()
        with torch.no_grad():
            outs = postprocess(model(x), exp.num_classes, 0.001, 0.7)
        for (j, x0, y0, r), o in zip(meta, outs):
            if o is None:
                continue
            o = o.float().cpu().numpy()
            bx = o[:, :4] / r; bx[:, [0, 2]] += x0; bx[:, [1, 3]] += y0
            sc = o[:, 4] * o[:, 5]; cl = o[:, 6].astype(int) + 1
            ov = iou(cands[j][2][None], bx)[0]
            m = ov >= 0.5
            if m.any():
                zany[j] = sc[m].max()
                ms = m & (cl == cands[j][3])
                if ms.any():
                    zsame[j] = sc[ms].max()
        batch.clear(); meta.clear()

    done = 0
    for (seq, fr), idx in by_frame.items():
        im = cv2.imread(img_path(seq, fr))
        H, W = im.shape[:2]
        for j in idx:
            b = cands[j][2]
            S = int(max(192, 3 * max(b[2] - b[0], b[3] - b[1])))
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            x0 = int(np.clip(cx - S / 2, 0, max(0, W - S))); y0 = int(np.clip(cy - S / 2, 0, max(0, H - S)))
            crop = im[y0:y0 + S, x0:x0 + S]
            t, r = preproc(crop, (Z, Z), MEAN, STD)
            batch.append(t); meta.append((j, x0, y0, r))
            if len(batch) == BS:
                flush()
        done += len(idx)
        if done % 2000 < len(idx):
            print(f"  {done}/{len(cands)}", flush=True)
    flush()

    lab = np.array([c[5] for c in cands]); s = np.array([c[4] for c in cands])
    cls = np.array([c[3] for c in cands])
    with open(os.path.join(HERE, "_scratch", "zoom_reinspect.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["seq", "frame", "x1", "y1", "x2", "y2", "cls", "score", "label", "z_any", "z_same"])
        for c, a, b in zip(cands, zany, zsame):
            w.writerow([c[0], c[1], *[f"{v:.2f}" for v in c[2]], c[3], f"{c[4]:.5f}", c[5], f"{a:.5f}", f"{b:.5f}"])
    pos, neg = lab == "rescue", lab == "bg"
    print(f"\nAUC rescue-vs-bg  (candidates score in [{LO},{HI})):")
    print(f"  full-frame score {auc(s[pos], s[neg]):.3f}")
    print(f"  zoom z_any       {auc(zany[pos], zany[neg]):.3f}")
    print(f"  zoom z_same      {auc(zsame[pos], zsame[neg]):.3f}")
    print(f"  max(score, z_same) {auc(np.maximum(s, zsame)[pos], np.maximum(s, zsame)[neg]):.3f}")
    print("\nper class z_same AUC (score AUC):")
    for c, nm in NAMES.items():
        m = cls == c
        print(f"  {nm:10s} rescue {int((pos & m).sum()):5d} bg {int((neg & m).sum()):6d}  "
              f"{auc(zsame[pos & m], zsame[neg & m]):.3f} ({auc(s[pos & m], s[neg & m]):.3f})")
    print("\nprecision rescue/(rescue+bg) if we ACCEPT candidates with z_same >= t:")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        m = zsame >= t
        npz, nn = int((pos & m).sum()), int((neg & m).sum())
        print(f"  t={t:.1f}: accept {npz + nn:6d}  rescue {npz:5d}  bg {nn:5d}  precision {100 * npz / max(npz + nn, 1):5.1f}%  "
              f"net MOTA-events {npz - nn:+6d}")


if __name__ == "__main__":
    e = sys.argv[1]
    main(e if os.path.isabs(e) else os.path.join(HERE, "YOLOX_outputs", e, "track_results"))
