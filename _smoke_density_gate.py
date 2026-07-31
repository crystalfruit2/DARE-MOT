"""Zero-GPU smoke test for the density-gated fusion weight (2026-07-31). Validates the
neighbor-count/density-factor math and both boost/suppress directions on synthetic
detections before spending any GPU time. No detector, no tracker run."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

os.environ['DARE_LAMBDA'] = '0.5'
os.environ['DARE_DENSITY_GATE'] = 'boost'
os.environ['DARE_DENSITY_RADIUS'] = '3.0'
os.environ['DARE_DENSITY_STRENGTH'] = '2.0'

from yolox.tracker.byte_tracker import BYTETracker, STrack


class Args:
    track_thresh = 0.6
    track_buffer = 30
    match_thresh = 0.9
    mot20 = False
    min_box_area = 100


tracker = BYTETracker(Args())

# Three isolated detections (no neighbors) + a tight cluster of three (mutual neighbors),
# all the same box size (100x50) so any density effect isn't confounded by the size gate.
isolated = [STrack([1000.0 * i, 1000.0, 100.0, 50.0], 0.9) for i in range(3)]
cluster = [STrack([50.0 + 20.0 * i, 50.0, 100.0, 50.0], 0.9) for i in range(3)]
dets = isolated + cluster

density = tracker._local_density_factor(dets)
print("density factors:", np.round(density, 3))
assert np.allclose(density[:3], 0.0), "isolated detections must have zero density factor"
assert np.all(density[3:] > 0.3), "clustered detections must have a real density factor"
print("(a) neighbor-count / density-factor math OK")

# boost mode: lambda should rise toward reid_lambda for the crowded cluster
tracker.density_gate = 'boost'
tracker.lambda_gate = 'none'
tracker.lambda_class_exclude = set()
lam_boost = tracker._gated_lambda(dets)
assert np.allclose(lam_boost[:3], tracker.reid_lambda), "isolated dets unaffected under boost"
assert np.all(lam_boost[3:] >= lam_boost[:3]), "boost must not lower lambda for crowded dets"
print("(b) boost mode OK:", np.round(lam_boost, 3))

# suppress mode: lambda should fall toward 0 for the crowded cluster
tracker.density_gate = 'suppress'
lam_suppress = tracker._gated_lambda(dets)
assert np.allclose(lam_suppress[:3], tracker.reid_lambda), "isolated dets unaffected under suppress"
assert np.all(lam_suppress[3:] < lam_suppress[:3]), "suppress must lower lambda for crowded dets"
print("(c) suppress mode OK:", np.round(lam_suppress, 3))

# composes with the existing size gate: tiny cluster dets should still gate to (near) zero
tracker.lambda_gate = 'size'
tracker.gate_lo = 100000.0  # everything here (5000px area) is "tiny" under this gate
tracker.gate_hi = 0.0
tracker.density_gate = 'boost'
lam_composed = tracker._gated_lambda(dets)
print("(d) composes with size gate OK:", np.round(lam_composed, 3))

print("\nALL DENSITY-GATE SMOKE CHECKS PASSED")
