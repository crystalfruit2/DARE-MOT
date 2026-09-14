import os

import cv2
import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

from yolox.tracker import kalman_filter
import time

# 2026-09-14: Smart App Control started blocking site-packages/cython_bbox*.pyd on this machine
# ("Uygulama Denetimi ilkesi bu dosyayi engelledi"), the same class of block that hit lapx 0.9.4.
# The .pyd is unchanged since 2026-07-10, so it is the SAC policy that moved -- and because this import
# sits in yolox.evaluators' import chain, it took DETECTOR TRAINING down with it, not just tracking.
# The fallback below is a float64 transcription of cython_bbox 0.1.5's bbox_overlaps, including its
# +1 pixel convention and its "0.0 when the intersection is empty" branches. Same IEEE ops on the same
# float64 inputs, so it should reproduce the cython result exactly -- but "should" is not "measured":
# it is UNVALIDATED against a cached tracker run. Do not report a tracker number produced with the
# numpy path until the byte-identical repro (_run_lap0512_repro.ps1 style) has been run.
try:
    from cython_bbox import bbox_overlaps as _cython_bbox_ious
except ImportError as _e:          # SAC block, or no compiled extension
    _cython_bbox_ious, _cython_bbox_err = None, _e


def _numpy_bbox_ious(boxes, query_boxes):
    """float64 transcription of cython_bbox.bbox_overlaps: (N,4) x (K,4) -> (N,K)."""
    boxes = np.ascontiguousarray(boxes, dtype=np.float64)
    query_boxes = np.ascontiguousarray(query_boxes, dtype=np.float64)
    iw = (np.minimum(boxes[:, None, 2], query_boxes[None, :, 2])
          - np.maximum(boxes[:, None, 0], query_boxes[None, :, 0]) + 1.0)
    ih = (np.minimum(boxes[:, None, 3], query_boxes[None, :, 3])
          - np.maximum(boxes[:, None, 1], query_boxes[None, :, 1]) + 1.0)
    inter = np.where((iw > 0) & (ih > 0), iw * ih, 0.0)
    area_b = (boxes[:, 2] - boxes[:, 0] + 1.0) * (boxes[:, 3] - boxes[:, 1] + 1.0)
    area_q = (query_boxes[:, 2] - query_boxes[:, 0] + 1.0) * (query_boxes[:, 3] - query_boxes[:, 1] + 1.0)
    ua = area_b[:, None] + area_q[None, :] - inter
    return np.where(inter > 0, inter / ua, 0.0)


def bbox_ious(boxes, query_boxes):
    """cython_bbox when SAC allows it (the numbers of record), numpy transcription otherwise."""
    if _cython_bbox_ious is not None and os.environ.get("DARE_BBOX_IOU", "cython") != "numpy":
        return _cython_bbox_ious(boxes, query_boxes)
    return _numpy_bbox_ious(boxes, query_boxes)

def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float64)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float64),
        np.ascontiguousarray(btlbrs, dtype=np.float64)
    )

    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def v_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def embedding_distance_safe(tracks, detections, metric='cosine'):
    """
    Like embedding_distance but handles None smooth_feat / curr_feat gracefully.
    Tracks or detections without features get cost 1.0 (max) so IoU still decides them.
    """
    cost_matrix = np.ones((len(tracks), len(detections)), dtype=np.float64)
    if cost_matrix.size == 0:
        return cost_matrix

    track_indices = [i for i, t in enumerate(tracks) if t.smooth_feat is not None]
    det_indices   = [j for j, d in enumerate(detections) if d.curr_feat is not None]

    if not track_indices or not det_indices:
        return cost_matrix

    track_feats = np.asarray([tracks[i].smooth_feat for i in track_indices], dtype=np.float64)
    det_feats   = np.asarray([detections[j].curr_feat for j in det_indices], dtype=np.float64)

    sub = np.maximum(0.0, cdist(track_feats, det_feats, metric))
    cost_matrix[np.ix_(track_indices, det_indices)] = sub

    return cost_matrix


def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float64)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float64)
    #for i, track in enumerate(tracks):
        #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float64)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    #fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost