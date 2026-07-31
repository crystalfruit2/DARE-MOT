"""Zero-GPU smoke test for the Adaptive-Weighting (AW) comparison gate (2026-07-31).
Validates the per-detection margin math and boost direction, and composition with the
density gate, on synthetic reid_dists before spending any GPU time. No detector, no
tracker run."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

os.environ['DARE_LAMBDA'] = '0.5'

from yolox.tracker.byte_tracker import BYTETracker, STrack


class Args:
    track_thresh = 0.6
    track_buffer = 30
    match_thresh = 0.9
    mot20 = False
    min_box_area = 100


tracker = BYTETracker(Args())
tracker.aw_gate = 'boost'
tracker.aw_strength = 0.15

# 3 tracks x 4 detections synthetic appearance-COST matrix (lower = more similar).
# det 0: unambiguous best match (large margin, track0 cost 0.05 vs next 0.9)
# det 1: ambiguous (two tracks nearly tied, margin ~0.02)
# det 2: unambiguous best match (margin ~0.6)
# det 3: only 1 track has a real cost, others are the embedding_distance_safe fallback 1.0
reid_dists = np.array([
    [0.05, 0.40, 0.10, 0.30],
    [0.90, 0.42, 0.70, 1.00],
    [0.95, 0.99, 0.95, 1.00],
], dtype=np.float32)

margin = tracker._aw_factor(reid_dists)
print("aw margin factors:", np.round(margin, 3))
assert margin[0] > margin[1], "det 0 (unambiguous) must have a larger AW factor than det 1 (ambiguous)"
assert margin[2] > margin[1], "det 2 (unambiguous) must have a larger AW factor than det 1 (ambiguous)"
print("(a) margin ordering OK (unambiguous > ambiguous)")

# edge case: single track (no second-best) -> factor must be 0, not spuriously "trustworthy"
single_track_dists = np.array([[0.05, 0.4, 0.1, 0.3]], dtype=np.float32)
margin_single = tracker._aw_factor(single_track_dists)
assert np.allclose(margin_single, 0.0), "a single candidate track must not be treated as unambiguous"
print("(b) single-track edge case OK:", margin_single)

# boost mode: lambda should rise toward reid_lambda for unambiguous detections. Needs a size
# gate active first (same as real headline configs) so lam starts BELOW reid_lambda -- with
# no size gate, lam is already at reid_lambda everywhere and there is nothing to boost toward
# (same property the density gate has).
dets = [STrack([1000.0 * i, 1000.0, 100.0, 50.0], 0.9) for i in range(4)]
tracker.lambda_gate = 'size'
tracker.gate_lo = 0.0
tracker.gate_hi = 100000.0  # area 5000 -> ramp = 0.05, so lam starts near 0, well below reid_lambda
tracker.lambda_class_exclude = set()
tracker.density_gate = 'none'
lam_pre_aw = tracker.reid_lambda * (5000.0 / 100000.0)
lam = tracker._gated_lambda(dets, reid_dists)
print("(c) fused lambda with AW boost (pre-AW size-gated value ~", round(lam_pre_aw, 3), "):", np.round(lam, 3))
assert lam[0] > lam[1], "unambiguous det 0 must get more appearance weight than ambiguous det 1"
assert lam[2] > lam[1], "unambiguous det 2 must get more appearance weight than ambiguous det 1"
assert np.all(lam >= lam_pre_aw - 1e-6), "AW boost must never lower lambda below the pre-AW value"
assert np.all(lam <= tracker.reid_lambda + 1e-6), "AW boost must never exceed reid_lambda"

# composes with the density gate: both active should stack via successive pulls, not override
tracker.density_gate = 'boost'
tracker.density_radius = 3.0
tracker.density_strength = 2.0
lam_composed = tracker._gated_lambda(dets, reid_dists)
print("(d) composes with density gate OK:", np.round(lam_composed, 3))

# no reid_dists passed (e.g. a call site that forgot) must not crash -- degrades to no AW effect
tracker.density_gate = 'none'
tracker.lambda_gate = 'none'
lam_no_reid = tracker._gated_lambda(dets, None)
assert np.allclose(lam_no_reid, tracker.reid_lambda), "AW must no-op safely when reid_dists is None"
print("(e) graceful no-op without reid_dists OK")

print("\nALL AW-GATE SMOKE CHECKS PASSED")
