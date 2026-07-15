"""Extract identity-switch events + crop images for qualitative review (2026-07-15).
Question: on uav0000086 and uav0000182 (the two sequences where real appearance made
things *worse* vs ByteTrack, +51 and +61 IDSw), what do the actual confused track-ID
pairs look like? All 10 quantitative scene-level factors were ruled out (see
Decision-Log) — this is the last, qualitative step.

For each GT trajectory, match it to a predicted track ID per-frame via Hungarian
IoU assignment (thresh 0.5). When the matched pred ID changes from the previous
frame's match, that's an IDSw event. Dump:
  - the GT-box crop at the switch frame (who it actually is)
  - the crop of the pred ID that was tracking it BEFORE the switch (last few frames)
  - the crop of the pred ID that WRONGLY took over AFTER the switch (next few frames)
so a reviewer can see whether the two identities are visually similar (real ReID
confusion) or something else (occlusion, missed detection, etc).
"""
import os, cv2, json
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

VAL_DIR = r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone_MOT_Format\VisDrone2019-MOT-val"
PRED_DIR = r"C:\Users\User\Desktop\projects\DARE-MOT\YOLOX_outputs\_ablation_multiseq_fixed\track_results"
OUT_DIR = r"C:\Users\User\Desktop\projects\DARE-MOT\_idswitch_review"
SEQS = ["uav0000086_00000_v", "uav0000182_00000_v"]
IOU_THRESH = 0.5
CONTEXT_FRAMES = 3  # how many frames of context to dump before/after a switch


def xywh_to_xyxy(b):
    x, y, w, h = b
    return np.array([x, y, x + w, y + h])


def iou_matrix(a, b):
    # a: (N,4) xyxy, b: (M,4) xyxy
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
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


def load_mot(path, cols=(0, 1, 2, 3, 4, 5)):
    rows = np.loadtxt(path, delimiter=",", usecols=cols)
    if rows.ndim == 1:
        rows = rows[None, :]
    by_frame = defaultdict(list)  # frame -> list of (id, x, y, w, h)
    for r in rows:
        fr, tid, x, y, w, h = r
        by_frame[int(fr)].append((int(tid), x, y, w, h))
    return by_frame


def crop_and_save(imgdir, frame, box_xywh, out_path, pad=0.15):
    img = cv2.imread(os.path.join(imgdir, f"{frame:07d}.jpg"))
    if img is None:
        return False
    x, y, w, h = box_xywh
    px, py = w * pad, h * pad
    x1 = max(0, int(x - px)); y1 = max(0, int(y - py))
    x2 = min(img.shape[1], int(x + w + px)); y2 = min(img.shape[0], int(y + h + py))
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    cv2.imwrite(out_path, crop)
    return True


def process_seq(seq):
    gt_path = os.path.join(VAL_DIR, seq, "gt", "gt.txt")
    pred_path = os.path.join(PRED_DIR, f"{seq}.txt")
    imgdir = os.path.join(VAL_DIR, seq, "img1")

    gt_by_frame = load_mot(gt_path)
    pred_by_frame = load_mot(pred_path)

    frames = sorted(set(gt_by_frame) | set(pred_by_frame))
    last_match = {}          # gt_id -> pred_id (matched last frame it was seen)
    last_seen_frame = {}     # pred_id -> last frame we saw it (for context crops)
    switches = []

    for fr in frames:
        gts = gt_by_frame.get(fr, [])
        preds = pred_by_frame.get(fr, [])
        if not gts or not preds:
            continue
        gt_ids = [g[0] for g in gts]
        gt_boxes = np.array([xywh_to_xyxy(g[1:]) for g in gts])
        pred_ids = [p[0] for p in preds]
        pred_boxes = np.array([xywh_to_xyxy(p[1:]) for p in preds])

        ious = iou_matrix(gt_boxes, pred_boxes)
        row_ind, col_ind = linear_sum_assignment(-ious)

        for r, c in zip(row_ind, col_ind):
            if ious[r, c] < IOU_THRESH:
                continue
            gid, pid = gt_ids[r], pred_ids[c]
            if gid in last_match and last_match[gid] != pid:
                switches.append({
                    "seq": seq,
                    "frame": fr,
                    "gt_id": gid,
                    "old_pred_id": last_match[gid],
                    "new_pred_id": pid,
                    "old_pred_last_frame": last_seen_frame.get(last_match[gid], None),
                })
            last_match[gid] = pid
            last_seen_frame[pid] = fr

    print(f"{seq}: {len(switches)} IDSw events")

    seq_out = os.path.join(OUT_DIR, seq)
    os.makedirs(seq_out, exist_ok=True)

    for i, sw in enumerate(switches):
        ev_dir = os.path.join(seq_out, f"switch_{i:03d}_gt{sw['gt_id']}_f{sw['frame']}")
        os.makedirs(ev_dir, exist_ok=True)

        # GT crop at switch frame (ground truth identity)
        gt_box = next((g[1:] for g in gt_by_frame[sw["frame"]] if g[0] == sw["gt_id"]), None)
        if gt_box:
            crop_and_save(imgdir, sw["frame"], gt_box, os.path.join(ev_dir, "gt_at_switch.jpg"))

        # old pred id: crops from the last few frames it was tracked (before switch)
        old_pid = sw["old_pred_id"]
        old_last = sw["old_pred_last_frame"]
        if old_last is not None:
            for k, fr2 in enumerate(range(max(1, old_last - CONTEXT_FRAMES + 1), old_last + 1)):
                box = next((p[1:] for p in pred_by_frame.get(fr2, []) if p[0] == old_pid), None)
                if box:
                    crop_and_save(imgdir, fr2, box, os.path.join(ev_dir, f"old_pred{old_pid}_f{fr2}.jpg"))

        # new pred id: crops from switch frame forward a few frames (who it switched to)
        new_pid = sw["new_pred_id"]
        for fr2 in range(sw["frame"], sw["frame"] + CONTEXT_FRAMES):
            box = next((p[1:] for p in pred_by_frame.get(fr2, []) if p[0] == new_pid), None)
            if box:
                crop_and_save(imgdir, fr2, box, os.path.join(ev_dir, f"new_pred{new_pid}_f{fr2}.jpg"))

    with open(os.path.join(seq_out, "switches_manifest.json"), "w") as f:
        json.dump(switches, f, indent=2)

    return switches


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_switches = {}
    for seq in SEQS:
        all_switches[seq] = process_seq(seq)
    total_manifest = os.path.join(OUT_DIR, "all_switches_manifest.json")
    with open(total_manifest, "w") as f:
        json.dump(all_switches, f, indent=2)
    print(f"\nManifest written to {total_manifest}")
    print("DONE")


if __name__ == "__main__":
    main()
