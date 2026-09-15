"""Detector operating-point and look-alike check, D2 / D3 / D4 on val7 (2026-09-15). Eval only, no training.

Why (experiment-log 2026-09-15, independent Opus review of the D4 control): before the paper's detector is
locked to D3, (1) separate objectness recall from class confusion, (2) look at the tracker's actual operating
point (score > 0.6 high band, > 0.1 low band), which AP ignores -- D3's tracking FN rose 14-17% -- and (3) test
the two open mechanisms for "10-class beats 5-class at matched init":
  (a) hard-negative suppression: in a 5-class run the look-alikes are unlabelled background, so objectness
      learns to suppress things that look like pedestrians/cars;
  (b) reject bin: D3's model ids 5..9 absorb look-alike predictions that would otherwise be FPs -- and may
      also absorb REAL evaluated objects, which would be recall lost at tracking time.

Protocol
- Detection path identical to _det_ap_perseq.py / MOTEvaluator: fp16, fused, test_conf 0.001, class-wise
  NMS 0.7, score = obj * cls. ALL model classes are kept and cached, so D3's ids 5..9 can be audited.
- AP (COCOeval on val7_mc.json, no ignore filter, i.e. the protocol of the AP50 of record): per class and
  class-agnostic (useCats=0), at maxDets 500 for both so the class-agnostic number is not capped harder
  than the per-class one. Class-aware AP50 at the default maxDets 100 is printed as a tie-back to the
  60.72 / 56.03 / 50.44 of record -- if it does not reproduce, nothing else here should be trusted.
- Operating point (raw VisDrone GT, official ignore rule): GT boxes and detections >= 50% inside the union
  of ignored regions (raw categories 0 and 11) are removed first, as _score_official.py does. Greedy
  score-ordered matching at IoU >= 0.5, class-aware and class-agnostic, at score > 0.6 and > 0.1.
- Look-alike audit: share of class-agnostic FPs (model ids 0..4) sitting on a look-alike GT box (raw
  people / bicycle / tricycle / awning-tricycle / motor) at IoU >= 0.5; for a 10-class head, ids 5..9
  detections that sit on an evaluated GT box no id-0..4 detection recovered ("swallowed").

Usage:
  python _det_opcheck_2026-09-15.py --gt-check          # CPU only: GT parsing / mapping sanity
  python _det_opcheck_2026-09-15.py [<ckpt> ...]        # default = D2 D3 D4
Outputs: _scratch/det_opcheck/<run>__<ckpt>.npz (detections), opcheck_summary.json, per-image npz for bootstrap.
"""
import argparse
import contextlib
import io
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ANN = r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone2019-MOT-val\annotations"
OUT = os.path.join(HERE, "_scratch", "det_opcheck")
DEFAULT_CKPTS = [os.path.join(HERE, "YOLOX_outputs", r, "best_ckpt.pth.tar") for r in (
    "yolox_x_visdrone_mc", "yolox_x_visdrone_10c_mot17init", "yolox_x_visdrone_5c_ctrl_mot17init")]
LABEL = {"yolox_x_visdrone_mc": "D2 old-5c", "yolox_x_visdrone_10c_mot17init": "D3 clean-10c",
         "yolox_x_visdrone_5c_ctrl_mot17init": "D4 clean-5c"}
RECORD_AP50 = {"yolox_x_visdrone_mc": 60.72, "yolox_x_visdrone_10c_mot17init": 56.03,
               "yolox_x_visdrone_5c_ctrl_mot17init": 50.44}
RAW_TO_10C = {1: 1, 4: 2, 5: 3, 6: 4, 9: 5, 2: 6, 3: 7, 7: 8, 8: 9, 10: 10}   # = tools/convert_visdrone_10c.py
IGNORE_RAW = (0, 11)
EVAL_NAMES = {1: "pedestrian", 2: "car", 3: "van", 4: "truck", 5: "bus"}
LOOK_NAMES = {6: "people", 7: "bicycle", 8: "tricycle", 9: "awning-tricycle", 10: "motor"}
THRS = (0.6, 0.1)
HELDOUT = "uav0000339_00001_v"


# ----------------------------------------------------------------------------------------------- GT
def load_gt():
    js = json.load(open(os.path.join(ANN, "val7_mc.json")))
    vname = {v["id"]: v["file_name"] for v in js["videos"]}
    key2img, img_seq, img_hw = {}, {}, {}
    for im in js["images"]:
        s = vname[im["video_id"]]
        key2img[(s, im["frame_id"])] = im["id"]
        img_seq[im["id"]] = s
        img_hw[im["id"]] = (im["height"], im["width"])
    json_count = defaultdict(int)
    for a in js["annotations"]:
        json_count[a["image_id"]] += 1
    ev, look, ign = defaultdict(list), defaultdict(list), defaultdict(list)
    unmapped = 0
    for s in sorted(set(img_seq.values())):
        for ln in open(os.path.join(ANN, s + ".txt")):
            v = ln.strip().split(",")
            if len(v) < 8:
                continue
            fr, c = int(v[0]), int(v[7])
            x, y, w, h = map(float, v[2:6])
            iid = key2img.get((s, fr))
            if iid is None:
                unmapped += 1
                continue
            box = [x, y, x + w, y + h]
            if c in IGNORE_RAW:
                ign[iid].append(box)
            elif RAW_TO_10C[c] <= 5:
                ev[iid].append(box + [RAW_TO_10C[c]])
            else:
                look[iid].append(box + [RAW_TO_10C[c]])
    mism = sum(1 for i in img_seq if len(ev[i]) != json_count[i])
    as_arr = lambda d, k: {i: (np.asarray(d[i], float) if d[i] else np.zeros((0, k))) for i in img_seq}
    return dict(js=js, img_seq=img_seq, img_hw=img_hw, ev=as_arr(ev, 5), look=as_arr(look, 5),
                ign=ign, unmapped=unmapped, mism=mism)


def cover_fn(regions, hw):
    """Fraction of each xyxy box inside the union of regions, same integer-grid rule as covered_frac()."""
    if not regions:
        return lambda boxes: np.zeros(len(boxes))
    H, W = hw
    mask = np.zeros((H, W), np.uint8)
    for x0, y0, x1, y1 in regions:
        a0, b0 = max(int(round(x0)), 0), max(int(round(y0)), 0)
        a1, b1 = min(int(round(x1)), W), min(int(round(y1)), H)
        if a1 > a0 and b1 > b0:
            mask[b0:b1, a0:a1] = 1
    S = np.zeros((H + 1, W + 1), np.int64)
    S[1:, 1:] = mask.cumsum(0).cumsum(1)

    def frac(boxes):
        if len(boxes) == 0:
            return np.zeros(0)
        r = np.round(boxes[:, :4])
        full = (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
        x0 = np.clip(r[:, 0], 0, W).astype(int); x1 = np.clip(r[:, 2], 0, W).astype(int)
        y0 = np.clip(r[:, 1], 0, H).astype(int); y1 = np.clip(r[:, 3], 0, H).astype(int)
        inside = S[y1, x1] - S[y0, x1] - S[y1, x0] + S[y0, x0]
        return np.where(full > 0, inside / np.maximum(full, 1), 0.0)
    return frac


def iou_xyxy(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix0 = np.maximum(a[:, None, 0], b[None, :, 0]); iy0 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix1 = np.minimum(a[:, None, 2], b[None, :, 2]); iy1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)


def greedy(det, gt, same_cls=False, thr=0.5):
    """det sorted by score desc. Returns (gt index per det or -1, taken mask over gt)."""
    md = -np.ones(len(det), int)
    taken = np.zeros(len(gt), bool)
    if len(det) == 0 or len(gt) == 0:
        return md, taken
    iou = iou_xyxy(det, gt)
    if same_cls:
        iou = np.where(det[:, 5:6] == gt[None, :, 4], iou, 0.0)
    for i in range(len(det)):
        row = np.where(taken, -1.0, iou[i])
        j = int(row.argmax())
        if row[j] >= thr:
            md[i] = j
            taken[j] = True
    return md, taken


# ----------------------------------------------------------------------------------------- detection
def detect_all(ckpt):
    import torch
    from yolox.exp import get_exp
    from yolox.utils import fuse_model, postprocess
    from _det_ap_perseq import head_ncls, EXP_BY_NCLS
    torch.backends.cudnn.benchmark = False
    ncls = head_ncls(ckpt)
    exp = get_exp(EXP_BY_NCLS[ncls], None)
    exp.data_num_workers = 0
    loader = exp.get_eval_loader(batch_size=1, is_distributed=False)
    assert list(loader.dataset.class_ids) == list(range(1, ncls + 1)), loader.dataset.class_ids
    with torch.no_grad():
        model = exp.get_model().cuda().eval()
        model.load_state_dict(torch.load(ckpt, map_location="cuda:0")["model"], strict=False)
        model = fuse_model(model).half()
        th, tw = exp.test_size
        rows = []
        for imgs, _, info_imgs, ids in loader:
            out = postprocess(model(imgs.cuda().half()), exp.num_classes, exp.test_conf, exp.nmsthre)[0]
            if out is None:
                continue
            scale = min(th / float(int(info_imgs[0][0])), tw / float(int(info_imgs[1][0])))
            out = out.float().cpu().numpy()
            n = len(out)
            rows.append(np.column_stack([np.full(n, int(ids[0])), out[:, :4] / scale,
                                         out[:, 4] * out[:, 5], out[:, 6] + 1]))
        del model
        torch.cuda.empty_cache()
    return np.concatenate(rows), ncls           # cols: image_id, x0, y0, x1, y1, score, category_id (1..ncls)


# ------------------------------------------------------------------------------------------------ AP
def ap_part(dets, G):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco = COCO()
    coco.dataset = G["js"]
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    e5 = dets[dets[:, 6] <= 5]
    arr = np.column_stack([e5[:, 0], e5[:, 1], e5[:, 2], e5[:, 3] - e5[:, 1], e5[:, 4] - e5[:, 2], e5[:, 5], e5[:, 6]])
    with contextlib.redirect_stdout(io.StringIO()):
        dt = coco.loadRes(arr)

    def run(use_cats, img_ids=None):
        e = COCOeval(coco, dt, "bbox")
        e.params.catIds = [1, 2, 3, 4, 5]
        e.params.maxDets = [1, 100, 500]
        e.params.useCats = use_cats
        if img_ids is not None:
            e.params.imgIds = img_ids
        with contextlib.redirect_stdout(io.StringIO()):
            e.evaluate(); e.accumulate()
        return e.eval["precision"]              # [T, R, K, A, M]

    def ap50(P, k=slice(None), m=2):
        p = P[0, :, k, 0, m]
        p = p[p > -1]
        return float(100 * p.mean()) if p.size else float("nan")

    Pa, Pg = run(1), run(0)
    res = {"aware_ap50_md100_ALL": ap50(Pa, m=1), "aware_ap50_md500_ALL": ap50(Pa),
           "agnostic_ap50_md500_ALL": ap50(Pg),
           "aware_ap50_md500_per_class": {EVAL_NAMES[c]: ap50(Pa, k=c - 1) for c in EVAL_NAMES},
           "agnostic_ap50_md500_per_seq": {}}
    by_seq = defaultdict(list)
    for iid, s in G["img_seq"].items():
        by_seq[s].append(iid)
    for s, ids in sorted(by_seq.items()):
        res["agnostic_ap50_md500_per_seq"][s] = ap50(run(0, ids))
    return res


# ----------------------------------------------------------------------------------- operating point
def op_part(dets, G):
    acc = defaultdict(lambda: defaultdict(float))       # (scope, thr) -> counter
    cm = {t: np.zeros((5, 5), int) for t in THRS}      # [gt class, det class], class-agnostic matches, ALL
    per_img = {t: [] for t in THRS}                     # (image_id, tp_agn, fp_agn, tp_aware, n_gt)
    dets = dets[dets[:, 5] > min(THRS)]
    order = np.argsort(dets[:, 0], kind="stable")
    dets = dets[order]
    uid, start = np.unique(dets[:, 0], return_index=True)
    bounds = dict(zip(uid.astype(int), zip(start, list(start[1:]) + [len(dets)])))
    for iid, seq in G["img_seq"].items():
        a, b = bounds.get(iid, (0, 0))
        d = dets[a:b, 1:]                               # x0 y0 x1 y1 score cat
        d = d[np.argsort(-d[:, 4], kind="stable")]
        frac = cover_fn(G["ign"][iid], G["img_hw"][iid])
        g = G["ev"][iid]
        g = g[frac(g) < 0.5] if len(g) else g
        d = d[frac(d) < 0.5] if len(d) else d
        L = G["look"][iid]
        for thr in THRS:
            dd = d[d[:, 4] > thr]
            de, dl = dd[dd[:, 5] <= 5], dd[dd[:, 5] >= 6]
            mA, tA = greedy(de, g)
            mC, tC = greedy(de, g, same_cls=True)
            tpa, tpc = int((mA >= 0).sum()), int((mC >= 0).sum())
            per_img[thr].append((iid, tpa, len(de) - tpa, tpc, len(g)))
            for scope in (seq, "ALL"):
                c = acc[(scope, thr)]
                c["n_gt"] += len(g); c["n_det"] += len(de)
                c["tp_agn"] += tpa; c["tp_aware"] += tpc
                c["confused"] += int(((mA >= 0) & (de[:, 5] != (g[np.maximum(mA, 0), 4] if len(g) else -1))).sum())
                for k in EVAL_NAMES:
                    gk = g[:, 4] == k if len(g) else np.zeros(0, bool)
                    c[f"gt_{k}"] += int(gk.sum()); c[f"rec_agn_{k}"] += int(tA[gk].sum()); c[f"rec_aware_{k}"] += int(tC[gk].sum())
                fp = de[mA < 0]
                c["fp_agn"] += len(fp)
                if len(fp) and len(L):
                    iL = iou_xyxy(fp, L)
                    on = iL.max(1) >= 0.5
                    c["fp_on_look"] += int(on.sum())
                    for lc in L[iL.argmax(1)[on], 4].astype(int):
                        c[f"fp_on_look_{lc}"] += 1
                c["fn_agn"] += int((~tA).sum())
                if len(dl):
                    c["n_reject"] += len(dl)
                    if len(L):
                        c["reject_on_look"] += int((iou_xyxy(dl, L).max(1) >= 0.5).sum())
                    if len(g):
                        ig = iou_xyxy(dl, g)
                        on_ev = ig.max(1) >= 0.5
                        c["reject_on_eval"] += int(on_ev.sum())
                        fn = ~tA
                        cov = (ig >= 0.5).any(0) & fn
                        c["fn_covered_by_reject"] += int(cov.sum())
            if len(g):
                ok = mA >= 0
                for dc, gc in zip(de[ok, 5].astype(int), g[mA[ok], 4].astype(int)):
                    cm[thr][gc - 1, dc - 1] += 1
    return acc, cm, per_img


def summarise(acc, cm):
    out = {}
    for (scope, thr), c in acc.items():
        f = lambda a, b: (100.0 * a / b) if b else float("nan")
        out.setdefault(scope, {})[str(thr)] = {
            "n_gt": int(c["n_gt"]), "n_det": int(c["n_det"]),
            "P_agn": f(c["tp_agn"], c["n_det"]), "R_agn": f(c["tp_agn"], c["n_gt"]),
            "P_aware": f(c["tp_aware"], c["n_det"]), "R_aware": f(c["tp_aware"], c["n_gt"]),
            "confused_pct_of_agn_tp": f(c["confused"], c["tp_agn"]),
            "recall_agn_by_class": {EVAL_NAMES[k]: f(c[f"rec_agn_{k}"], c[f"gt_{k}"]) for k in EVAL_NAMES},
            "recall_aware_by_class": {EVAL_NAMES[k]: f(c[f"rec_aware_{k}"], c[f"gt_{k}"]) for k in EVAL_NAMES},
            "fp_agn": int(c["fp_agn"]), "fp_on_lookalike_pct": f(c["fp_on_look"], c["fp_agn"]),
            "fp_on_lookalike_by_cat": {LOOK_NAMES[k]: int(c[f"fp_on_look_{k}"]) for k in LOOK_NAMES},
            "fn_agn": int(c["fn_agn"]),
            "reject_bin": {"n": int(c["n_reject"]), "on_lookalike_pct": f(c["reject_on_look"], c["n_reject"]),
                           "on_eval_gt_pct": f(c["reject_on_eval"], c["n_reject"]),
                           "fn_covered_by_reject": int(c["fn_covered_by_reject"]),
                           "fn_covered_pct_of_fn": f(c["fn_covered_by_reject"], c["fn_agn"])},
        }
    out["confusion_ALL"] = {str(t): cm[t].tolist() for t in THRS}
    return out


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-check", action="store_true")
    ap.add_argument("ckpts", nargs="*")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    G = load_gt()
    n_ev = sum(len(v) for v in G["ev"].values())
    n_lk = sum(len(v) for v in G["look"].values())
    n_ig = sum(len(v) for v in G["ign"].values())
    in_ign = sum(int((cover_fn(G["ign"][i], G["img_hw"][i])(G["ev"][i]) >= 0.5).sum()) for i in G["img_seq"] if len(G["ev"][i]))
    print(f"GT: images {len(G['img_seq'])}, eval boxes {n_ev}, look-alike boxes {n_lk}, ignore regions {n_ig}, "
          f"eval boxes >=50% inside ignore {in_ign}, unmapped raw rows {G['unmapped']}, "
          f"images whose raw eval count != val7_mc.json count {G['mism']}", flush=True)
    if args.gt_check:
        return

    summary = {}
    for ck in (args.ckpts or DEFAULT_CKPTS):
        run = os.path.basename(os.path.dirname(ck))
        tag = run + "__" + os.path.basename(ck).replace(".pth.tar", "")
        cache = os.path.join(OUT, tag + ".npz")
        if os.path.exists(cache):
            z = np.load(cache)
            dets, ncls = z["dets"], int(z["ncls"])
        else:
            print(f"detect {tag} ...", flush=True)
            dets, ncls = detect_all(ck)
            np.savez_compressed(cache, dets=dets, ncls=ncls)
        print(f"{tag}: {ncls}-class head, {len(dets)} raw detections; AP ...", flush=True)
        res = {"label": LABEL.get(run, run), "ncls": ncls, "ap": ap_part(dets, G)}
        rec = RECORD_AP50.get(run)
        res["ap"]["tieback_record_ap50"] = rec
        res["ap"]["tieback_ok"] = (rec is None) or abs(res["ap"]["aware_ap50_md100_ALL"] - rec) < 0.05
        print(f"  tie-back: aware AP50@100 = {res['ap']['aware_ap50_md100_ALL']:.2f} vs record {rec} "
              f"-> {'OK' if res['ap']['tieback_ok'] else '!!! MISMATCH'}", flush=True)
        print("  operating point ...", flush=True)
        acc, cm, per_img = op_part(dets, G)
        res["op"] = summarise(acc, cm)
        np.savez_compressed(os.path.join(OUT, tag + "__per_image.npz"),
                            **{f"thr{t}": np.asarray(per_img[t]) for t in THRS})
        summary[tag] = res
    json.dump(summary, open(os.path.join(OUT, "opcheck_summary.json"), "w"), indent=1)

    tags = list(summary)
    col = lambda t: summary[t]["label"]
    w = 16
    line = lambda name, vals: print(f"{name:34s}" + "".join(f"{v:>{w}}" for v in vals))
    print("\n================ DETECTOR OPCHECK (val7) ================")
    line("", [col(t) for t in tags])
    line("AP50 aware @100 (record tie-back)", [f"{summary[t]['ap']['aware_ap50_md100_ALL']:.2f}" for t in tags])
    line("AP50 aware @500", [f"{summary[t]['ap']['aware_ap50_md500_ALL']:.2f}" for t in tags])
    line("AP50 class-agnostic @500", [f"{summary[t]['ap']['agnostic_ap50_md500_ALL']:.2f}" for t in tags])
    for c in EVAL_NAMES.values():
        line(f"  AP50 aware {c}", [f"{summary[t]['ap']['aware_ap50_md500_per_class'][c]:.2f}" for t in tags])
    print("-- class-agnostic AP50 per sequence (* held out)")
    for s in sorted(summary[tags[0]]["ap"]["agnostic_ap50_md500_per_seq"]):
        line(f"  {s}{' *' if s == HELDOUT else ''}", [f"{summary[t]['ap']['agnostic_ap50_md500_per_seq'][s]:.2f}" for t in tags])
    for thr in THRS:
        k = str(thr)
        o = lambda t: summary[t]["op"]["ALL"][k]
        print(f"-- operating point, score > {thr}, official ignore filter, IoU >= 0.5")
        line("  precision class-agnostic %", [f"{o(t)['P_agn']:.2f}" for t in tags])
        line("  recall    class-agnostic %", [f"{o(t)['R_agn']:.2f}" for t in tags])
        line("  precision class-aware %", [f"{o(t)['P_aware']:.2f}" for t in tags])
        line("  recall    class-aware %", [f"{o(t)['R_aware']:.2f}" for t in tags])
        line("  class-confused % of agn TP", [f"{o(t)['confused_pct_of_agn_tp']:.2f}" for t in tags])
        for c in EVAL_NAMES.values():
            line(f"    recall agn {c}", [f"{o(t)['recall_agn_by_class'][c]:.2f}" for t in tags])
        line("  FP (agn) count", [f"{o(t)['fp_agn']}" for t in tags])
        line("  FP on look-alike GT %", [f"{o(t)['fp_on_lookalike_pct']:.2f}" for t in tags])
        line("  FN (agn) count", [f"{o(t)['fn_agn']}" for t in tags])
        line("  reject-bin dets (ids 5..9)", [f"{o(t)['reject_bin']['n']}" for t in tags])
        line("    on look-alike GT %", [f"{o(t)['reject_bin']['on_lookalike_pct']:.2f}" for t in tags])
        line("    on evaluated GT %", [f"{o(t)['reject_bin']['on_eval_gt_pct']:.2f}" for t in tags])
        line("    FN covered by reject % of FN", [f"{o(t)['reject_bin']['fn_covered_pct_of_fn']:.2f}" for t in tags])
        print("   recall agn per sequence (* held out)")
        for s in sorted(x for x in summary[tags[0]]["op"] if x.startswith("uav")):
            line(f"    {s}{' *' if s == HELDOUT else ''}", [f"{summary[t]['op'][s][k]['R_agn']:.2f}" for t in tags])
    print(f"\nwrote {os.path.join(OUT, 'opcheck_summary.json')}")


if __name__ == "__main__":
    main()
