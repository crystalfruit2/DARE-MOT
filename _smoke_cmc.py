"""Zero-GPU smoke test for the Move 1 CMC wiring. Validates (a) the GMC module imports
and .apply() returns a 2x3 affine on a dummy frame, (b) byte_tracker imports with the
three edits (syntax), (c) STrack.multi_gmc is a no-op under identity H and a correct
translation under a translation H. No detector / no GPU."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

# (a) GMC import + apply
from baselines.gmc import GMC
for m in ("ecc", "sparseOptFlow", "none"):
    g = GMC(method=m, downscale=2)
    H = g.apply(np.zeros((128, 128, 3), dtype=np.uint8))   # first frame -> identity
    assert H.shape == (2, 3), f"{m}: bad shape {H.shape}"
print("(a) GMC import + apply OK")

# (b) byte_tracker imports with the edits
from yolox.tracker.byte_tracker import STrack
print("(b) byte_tracker import OK")

# (c) multi_gmc correctness via a lightweight fake track
class FakeTrack:
    def __init__(self):
        self.mean = np.arange(8, dtype=float)          # [x,y,a,h,vx,vy,va,vh]
        self.covariance = np.eye(8, dtype=float)

# identity -> no-op
t1 = FakeTrack(); before = t1.mean.copy()
STrack.multi_gmc([t1], np.eye(2, 3))
assert np.allclose(t1.mean, before), "identity H must be a no-op"

# pure translation (+5, -3) -> only x,y shift
t2 = FakeTrack(); before = t2.mean.copy()
Ht = np.eye(2, 3); Ht[0, 2] = 5.0; Ht[1, 2] = -3.0
STrack.multi_gmc([t2], Ht)
assert t2.mean[0] == before[0] + 5 and t2.mean[1] == before[1] - 3, "translation on x,y"
assert np.allclose(t2.mean[2:], before[2:]), "translation must not touch a,h,velocities"
print("(c) multi_gmc identity + translation OK")
print("\nALL SMOKE CHECKS PASSED")
