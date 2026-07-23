# Vendored: GMC (Global / Camera Motion Compensation)

`baselines/gmc.py` is adapted from BoT-SORT (Aharon, Orfaig, Bobrovsky,
"BoT-SORT: Robust Associations Multi-Pedestrian Tracking", arXiv:2206.14651),
https://github.com/NirAharon/BoT-SORT, `tracker/gmc.py` (MIT License).

Vendored 2026-07-23 for DARE-MOT. Two uses:
1. **BoT-SORT baseline adapter** (Path 1) — BoT-SORT's real differentiator over DARE is
   exactly this camera-motion compensation (we are already ByteTrack + appearance +
   proximity-mask ≈ BoT-SORT **minus** GMC).
2. **Path 2 (CMC on DARE's own KF)** — the affine warp is applied to DARE's Kalman-filter
   states before IoU matching to compensate for VisDrone's severe drone-camera motion.

## Modifications from upstream (NOT a verbatim copy)
- Removed the hard `import matplotlib.pyplot as plt` (was only used by a dead debug block).
- Removed the `if 0:` keypoint-match debug/visualisation block in `applyFeaures`.
- Removed the `'file'` / `'files'` GMC method + `applyFile` — it read precomputed
  GMC-<seq>.txt files keyed to hardcoded MOTChallenge paths, irrelevant to VisDrone.
- Hardened a couple of edge cases (`np.asarray(det[:4])` before the mask crop;
  `except Exception`; store per-call runtime in `self.last_runtime_ms` instead of a file).

The numerical estimators — `sparseOptFlow` (BoT-SORT's default), `ecc`, `orb`, `sift`,
`none` — are otherwise unchanged. Full reference clone (with LICENSE) is under
`_baselines/BoT-SORT/`.
