"""Adapters that let third-party trackers run inside DARE-MOT's own eval harness for a
CONTROLLED Path-1 comparison (same detector, same preprocessing, same scoring; only the
association logic changes). See [[Projects/Dare_Mot/path1-baseline-integration-2026-07-23]].

Every adapter exposes the harness contract used by mot_evaluator.evaluate():
    .update(output_results, img_info, img_size) -> list of objects with
        .tlwh (x,y,w,h)  .track_id (int)  .score (float)  .cls (int, model head id 0..N-1)

Class is assigned POST-HOC and uniformly (IoU-match each output box back to the frame's input
detections, take that detection's class) so the upstream tracker code stays untouched — a fair
"faithful public implementation" and class never influences association (tracking stays
class-agnostic; only scoring splits by class, same convention as DARE's own STrack.cls).

Vendored sources live under baselines/<tracker>/ with attribution; the large reference clones
are in _baselines/ (gitignored).
"""
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


class _Trk:
    """Lightweight track object matching the attributes mot_evaluator reads."""
    __slots__ = ("tlwh", "track_id", "score", "cls")

    def __init__(self, tlwh, track_id, score, cls):
        self.tlwh = np.asarray(tlwh, dtype=np.float64)
        self.track_id = int(track_id)
        self.score = float(score)
        self.cls = int(cls)


def _iou_matrix(a, b):
    """IoU between two sets of tlbr boxes. a:[N,4], b:[M,4] -> [N,M]."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area_a[:, None] + area_b[None, :] - inter + 1e-6
    return inter / union


def assign_classes_by_iou(out_tlbr, det_tlbr, det_cls, det_scores, iou_thr=0.5):
    """For each output box, return (cls, score) from the best-IoU input detection.
    No match above iou_thr -> (-1, 1.0). Uniform across all baselines."""
    out_tlbr = np.asarray(out_tlbr, dtype=np.float64).reshape(-1, 4)
    ious = _iou_matrix(out_tlbr, det_tlbr)
    classes, scores = [], []
    for i in range(len(out_tlbr)):
        if ious.shape[1] == 0:
            classes.append(-1); scores.append(1.0); continue
        j = int(np.argmax(ious[i]))
        if ious[i, j] >= iou_thr:
            classes.append(int(det_cls[j])); scores.append(float(det_scores[j]))
        else:
            classes.append(-1); scores.append(1.0)
    return classes, scores


def _parse_dets_original_coords(output_results, img_info, img_size):
    """Detections in ORIGINAL image coords (tlbr) + class + score, matching the space the
    baselines emit their output boxes in (they all do bboxes /= scale). Returns (tlbr, cls, score)."""
    o = output_results
    if hasattr(o, "cpu"):
        o = o.cpu().numpy()
    o = np.asarray(o, dtype=np.float64)
    if o.shape[1] == 5:
        scores = o[:, 4]; classes = np.full(len(o), -1)
    else:
        scores = o[:, 4] * o[:, 5]
        classes = o[:, 6] if o.shape[1] > 6 else np.full(len(o), -1)
    bboxes = o[:, :4].copy()
    img_h, img_w = img_info[0], img_info[1]
    scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
    bboxes /= scale
    return bboxes, classes, scores


class OCSortAdapter:
    """OC-SORT (Cao et al., CVPR'23) — observation-centric, motion-only (no ReID/CMC).
    Its update() signature already matches ours; we only convert the [x1,y1,x2,y2,id] output
    back to the harness contract and attach class post-hoc.
    Config = OC-SORT's standard MOT settings; det_thresh tied to our track_thresh so the
    high/low BYTE split matches DARE's."""

    def __init__(self, args):
        from ocsort.ocsort import OCSort
        self.tracker = OCSort(
            det_thresh=args.track_thresh,
            iou_threshold=0.3,          # OC-SORT paper default (actual IoU, not 1-IoU)
            use_byte=True,              # keep the low-score second association (fair vs our ByteTrack)
            asso_func="iou",
        )

    def update(self, output_results, img_info, img_size):
        det_tlbr, det_cls, det_scores = _parse_dets_original_coords(output_results, img_info, img_size)
        out = self.tracker.update(output_results, img_info, img_size)  # [[x1,y1,x2,y2,id],...] orig coords
        if out is None or len(out) == 0:
            return []
        out = np.asarray(out, dtype=np.float64)
        out_tlbr = out[:, :4]
        ids = out[:, 4]
        classes, scores = assign_classes_by_iou(out_tlbr, det_tlbr, det_cls, det_scores)
        tracks = []
        for k in range(len(out)):
            x1, y1, x2, y2 = out_tlbr[k]
            tracks.append(_Trk([x1, y1, x2 - x1, y2 - y1], ids[k], scores[k], classes[k]))
        return tracks
