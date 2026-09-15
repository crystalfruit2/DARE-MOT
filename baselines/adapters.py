"""Adapters that let third-party trackers run inside DARE-MOT's own eval harness for a
CONTROLLED Path-1 comparison (same detector, same preprocessing, same scoring; only the
association logic changes). See [[Projects/Dare_Mot/path1-baseline-integration-2026-07-23]].

Every adapter exposes the harness contract used by mot_evaluator.evaluate():
    .update(output_results, img_info, img_size) -> list of objects with
        .tlwh (x,y,w,h)  .track_id (int)  .score (float)  .cls (int, model head id 0..N-1)
    .print_diag_summary(video_name)  (called by the harness between sequences; no-op here)

Class is assigned POST-HOC and uniformly (IoU-match each output box back to the frame's input
detections, take that detection's class) so the upstream tracker code stays untouched -- a fair
"faithful public implementation" and class never influences association (tracking stays
class-agnostic; only scoring splits by class, same convention as DARE's own STrack.cls).

Shared front end (_prep), identical for every baseline:
  * DARE_MAX_CLASS: detections with model class id above it are dropped BEFORE the tracker sees
    them (the 10-class detector keeps the 5 evaluated classes at 0..4), exactly as BYTETracker does.
  * The harness passes img_info[0] = the raw BGR frame (mot_evaluator raw-frame path, 2026-07-15),
    not the height. Frame height/width are taken from the frame itself.
  * Boxes are rescaled to original image coordinates once, here, with the harness's own formula.

Shared appearance (SharedOSNet): BoT-SORT and Deep OC-SORT embed with DARE-MOT's OSNet-AIN
(DARE_REID=osnet, DARE_REID_MODEL, DARE_REID_WEIGHTS), crop logic copied from
BYTETracker._extract_features_osnet, so embedding quality is held fixed across trackers.

Vendored sources live under baselines/<tracker>/ with attribution (SOURCE.md); the large reference
clones are in _baselines/ (gitignored).

2026-09-15: OC-SORT adapter fixed (read img_info[0] as a height -> crashed under the raw-frame
harness; ignored DARE_MAX_CLASS; had no print_diag_summary) and BoT-SORT / Deep OC-SORT added.
"""
import os
import sys
from types import SimpleNamespace

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


def _prep(output_results, img_info, img_size):
    """Shared front end. Returns (dets, frame, h, w, scores, classes):
    dets = copy of the detector rows [x1,y1,x2,y2,obj,cls_conf,cls_id] in ORIGINAL image coords,
    after the DARE_MAX_CLASS filter; dtype kept as the detector produced it (as BYTETracker does)."""
    o = output_results
    if hasattr(o, "cpu"):
        o = o.detach().cpu().numpy()
    o = np.array(o, copy=True)
    if o.ndim == 1:
        o = o.reshape(0, 7) if o.size == 0 else o.reshape(1, -1)
    max_class = int(os.environ.get("DARE_MAX_CLASS", "-1"))
    if max_class >= 0 and o.shape[1] > 6:
        o = o[o[:, 6] <= max_class]
    frame = img_info[0] if isinstance(img_info[0], np.ndarray) else None
    if frame is not None:
        h, w = frame.shape[:2]
    else:
        h, w = int(img_info[0]), int(img_info[1])
    scale = min(img_size[0] / float(h), img_size[1] / float(w))
    o[:, :4] /= scale
    if o.shape[1] == 5:
        scores = o[:, 4]
        classes = np.full(len(o), -1)
    else:
        scores = o[:, 4] * o[:, 5]
        classes = o[:, 6] if o.shape[1] > 6 else np.full(len(o), -1)
    return o, frame, h, w, scores, classes


class _TensorLike:
    """Minimal .cpu().numpy() / .shape shim for upstream code that expects a torch tensor."""
    def __init__(self, arr):
        self._a = arr
        self.shape = arr.shape

    def cpu(self):
        return self

    def numpy(self):
        return self._a


def _require_lap():
    """Every baseline must solve with lap.lapjv (lap 0.5.12 on PYTHONPATH on the Windows box)."""
    try:
        import lap
        lap.lapjv
    except Exception as e:  # SAC block surfaces as ImportError
        raise ImportError("Path-1 baselines need lap.lapjv (put DARE-MOT-pylibs\\lap0512 first on "
                          "PYTHONPATH); no scipy fallback is allowed") from e


class _Base:
    def print_diag_summary(self, *args, **kwargs):
        pass

    @staticmethod
    def _wrap(out_tlbr, ids, track_scores, dets, classes, scores):
        if len(out_tlbr) == 0:
            return []
        out_tlbr = np.asarray(out_tlbr, dtype=np.float64).reshape(-1, 4)
        cls, det_sc = assign_classes_by_iou(out_tlbr, dets[:, :4], classes, scores)
        sc = det_sc if track_scores is None else track_scores
        return [_Trk([b[0], b[1], b[2] - b[0], b[3] - b[1]], i, s, c)
                for b, i, s, c in zip(out_tlbr, ids, sc, cls)]


# ------------------------------------------------------------------------------ shared appearance
_REID_CACHE = {}


class SharedOSNet:
    """DARE-MOT's OSNet-AIN embedding for the appearance baselines. Crop logic is a copy of
    BYTETracker._extract_features_osnet (int() truncation, clip to frame, BGR->RGB, optional
    DARE_CROP_SHRINK); the torchreid FeatureExtractor does resize/normalise. Returns raw
    (unnormalised) 512-d features, zeros for empty crops -- exactly what DARE's tracker stores."""
    feat_dim = 512

    def __init__(self):
        if os.environ.get("DARE_REID", "") != "osnet":
            raise RuntimeError("appearance baselines share DARE's embedding: set DARE_REID=osnet, "
                               "DARE_REID_MODEL and DARE_REID_WEIGHTS")
        model = os.environ.get("DARE_REID_MODEL", "osnet_x1_0")
        weights = os.environ["DARE_REID_WEIGHTS"]
        self.crop_shrink = float(os.environ.get("DARE_CROP_SHRINK", "0.0"))
        key = (model, weights)
        if key not in _REID_CACHE:
            import torch
            from torchreid.reid.utils import FeatureExtractor
            dev = os.environ.get("DARE_P1_REID_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
            _REID_CACHE[key] = FeatureExtractor(model_name=model, model_path=weights, device=dev, verbose=False)
        self.extractor = _REID_CACHE[key]

    def __call__(self, frame, tlbr):
        import cv2
        import torch
        tlbr = np.asarray(tlbr).reshape(-1, 4)
        feats = np.zeros((len(tlbr), self.feat_dim), dtype=np.float32)
        if len(tlbr) == 0:
            return feats
        H_img, W_img = frame.shape[:2]
        crops, idxs = [], []
        for i, b in enumerate(tlbr):
            tlwh = b.copy()
            tlwh[2:] -= tlwh[:2]                        # STrack.tlbr_to_tlwh, in the detector's dtype
            x, y, w, h = tlwh.astype(np.float64)        # STrack stores float64
            if self.crop_shrink > 0.0:
                dx, dy = w * self.crop_shrink, h * self.crop_shrink
                x, y, w, h = x + dx, y + dy, w - 2 * dx, h - 2 * dy
            x, y, w, h = int(x), int(y), int(w), int(h)
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(W_img, x + w), min(H_img, y + h)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                if crop.ndim == 3 and crop.shape[2] == 3:
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crops.append(crop)
                idxs.append(i)
        if crops:
            with torch.no_grad():
                f = self.extractor(crops).cpu().numpy()
            for j, i in enumerate(idxs):
                feats[i] = f[j].astype(np.float32)
        return feats


def _unit_rows(f):
    """L2-normalise rows; an all-zero row (empty crop) becomes a fixed uniform unit vector so a
    degenerate crop cannot produce NaN inside upstream code that normalises in place."""
    f = np.asarray(f, dtype=np.float64)
    n = np.linalg.norm(f, axis=1, keepdims=True)
    out = np.where(n > 0, f / np.maximum(n, 1e-12), 1.0 / np.sqrt(f.shape[1]))
    return out


class _BotEncoder:
    """BoT-SORT's encoder interface: inference(img, dets_tlbr) -> [N, D]."""
    def __init__(self, emb):
        self.emb = emb

    def inference(self, img, dets):
        if len(dets) == 0:
            return np.zeros((0, self.emb.feat_dim), dtype=np.float64)
        return _unit_rows(self.emb(img, dets))


class _DeepOCEmbedder:
    """Deep OC-SORT's EmbeddingComputer interface (grid_off): compute_embedding(img, bbox, tag)."""
    def __init__(self, emb):
        self.emb = emb

    def compute_embedding(self, img, bbox, tag):
        return _unit_rows(self.emb(img, bbox))

    def dump_cache(self):
        pass


class _GMCAffine:
    """Deep OC-SORT's CMCComputer interface backed by BoT-SORT's GMC run live. Upstream's published
    runs read precomputed BoT-SORT GMC affines from files (method='file'); none exist for VisDrone."""
    def __init__(self, method="sparseOptFlow"):
        from baselines.gmc import GMC
        self.gmc = GMC(method=method, downscale=2)

    def compute_affine(self, img, bbox, tag):
        return self.gmc.apply(img, None)

    def dump_cache(self):
        pass


# ------------------------------------------------------------------------------------- adapters
class OCSortAdapter(_Base):
    """OC-SORT (Cao et al., CVPR'23) -- observation-centric, motion-only (no ReID/CMC).
    Config = OC-SORT's standard MOT settings; det_thresh tied to our track_thresh so the
    high/low BYTE split matches DARE's."""

    def __init__(self, args):
        _require_lap()
        from ocsort.ocsort import OCSort
        self.tracker = OCSort(
            det_thresh=args.track_thresh,
            iou_threshold=0.3,          # OC-SORT paper default (actual IoU, not 1-IoU)
            use_byte=True,              # keep the low-score second association (fair vs our ByteTrack)
            asso_func="iou",
        )

    def update(self, output_results, img_info, img_size):
        dets, frame, h, w, scores, classes = _prep(output_results, img_info, img_size)
        # upstream rescales by min(img_size / (h, w)); boxes are already original coords -> scale 1
        out = self.tracker.update(_TensorLike(dets), (h, w), (h, w))  # [[x1,y1,x2,y2,id],...]
        if out is None or len(out) == 0:
            return []
        out = np.asarray(out, dtype=np.float64)
        return self._wrap(out[:, :4], out[:, 4], None, dets, classes, scores)


class BoTSORTAdapter(_Base):
    """BoT-SORT (Aharon et al., 2022), ReID + GMC ("BoT-SORT-ReID"), vendored in baselines/botsort.
    Shared with the harness: track_thresh (high), 0.1 (low), track_buffer, match_thresh, and the
    new-track threshold track_thresh + 0.1 (= upstream's 0.7 at 0.6, and ByteTrack's det_thresh).
    BoT-SORT-specific, upstream defaults: proximity_thresh 0.5, appearance_thresh 0.25,
    fuse_score on (mot20 off), GMC sparseOptFlow at downscale 2. Appearance = SharedOSNet."""

    def __init__(self, args, embedder=None):
        _require_lap()
        from botsort.bot_sort import BoTSORT
        emb = embedder if embedder is not None else SharedOSNet()
        a = SimpleNamespace(
            track_high_thresh=args.track_thresh,
            track_low_thresh=0.1,
            new_track_thresh=args.track_thresh + 0.1,
            track_buffer=args.track_buffer,
            match_thresh=args.match_thresh,
            proximity_thresh=0.5,
            appearance_thresh=0.25,
            with_reid=True,
            cmc_method=os.environ.get("DARE_P1_CMC", "sparseOptFlow"),
            mot20=bool(getattr(args, "mot20", False)),
            encoder=_BotEncoder(emb),
        )
        self.tracker = BoTSORT(a, frame_rate=30)

    def update(self, output_results, img_info, img_size):
        dets, frame, h, w, scores, classes = _prep(output_results, img_info, img_size)
        if frame is None:
            raise RuntimeError("BoT-SORT adapter needs the raw frame in img_info[0]")
        out = self.tracker.update(dets, frame)
        if not out:
            return []
        return self._wrap([t.tlbr for t in out], [t.track_id for t in out],
                          [float(t.score) for t in out], dets, classes, scores)


class DeepOCSORTAdapter(_Base):
    """Deep OC-SORT (Maggiolino et al., ICIP 2023), vendored in baselines/deepocsort, upstream
    defaults: det_thresh = track_thresh, iou 0.3, min_hits 3, delta_t 3, inertia 0.2,
    w_association_emb 0.75, alpha_fixed_emb 0.95, adaptive weighting (aw_param 0.5), new KF on,
    CMC on, max_age = track_buffer. No BYTE low-score association (upstream has none).
    Appearance = SharedOSNet, one vector per box (grid_off); CMC = BoT-SORT GMC live."""

    def __init__(self, args, embedder=None):
        _require_lap()
        from deepocsort.ocsort import OCSort as DeepOCSort
        emb = embedder if embedder is not None else SharedOSNet()
        self.tracker = DeepOCSort(
            det_thresh=args.track_thresh, max_age=args.track_buffer, min_hits=3, iou_threshold=0.3,
            delta_t=3, asso_func="iou", inertia=0.2, w_association_emb=0.75, alpha_fixed_emb=0.95,
            aw_param=0.5, embedding_off=False, cmc_off=False, aw_off=False, new_kf_off=False,
            grid_off=True, embedder=_DeepOCEmbedder(emb),
            cmc=_GMCAffine(os.environ.get("DARE_P1_CMC", "sparseOptFlow")))

    def update(self, output_results, img_info, img_size):
        dets, frame, h, w, scores, classes = _prep(output_results, img_info, img_size)
        if frame is None:
            raise RuntimeError("Deep OC-SORT adapter needs the raw frame in img_info[0]")
        # upstream scale = min(tensor_hw / frame_hw); boxes are already original coords -> scale 1
        shape_1 = SimpleNamespace(shape=(1, 3, h, w))
        out = self.tracker.update(dets, shape_1, frame, "")
        if out is None or len(out) == 0:
            return []
        out = np.asarray(out, dtype=np.float64)
        return self._wrap(out[:, :4], out[:, 4], None, dets, classes, scores)
