# baselines/botsort: source and modifications

Upstream: https://github.com/NirAharon/BoT-SORT (MIT, Copyright (c) 2022 Nir Aharon), reference clone `_baselines/BoT-SORT` @ 2519854.
Files: `tracker/bot_sort.py`, `tracker/matching.py`, `tracker/kalman_filter.py`, `tracker/basetrack.py` (single-class BoT-SORT; class is attached post-hoc by `baselines/adapters.py`, uniform across baselines).

Modifications (2026-09-15, all mechanical; association logic, thresholds semantics, KF and GMC application untouched):
1. `bot_sort.py`: dropped the unused `matplotlib` import; package-relative imports; GMC from `baselines/gmc.py` (the same vendored BoT-SORT estimator, see `baselines/gmc/SOURCE.md`), constructed with upstream's default `downscale=2`.
2. `bot_sort.py`: `FastReIDInterface` replaced by an injected encoder (`args.encoder.inference(img, dets)`) so every appearance baseline shares DARE-MOT's OSNet-AIN embedding (controlled comparison).
3. `matching.py`: `cython_bbox.bbox_overlaps` -> `yolox.tracker.matching.bbox_ious` (cython_bbox when loadable, else its bit-identical float64 transcription; same +1 px convention as upstream).
4. `np.float` -> `np.float64` (removed alias in NumPy >= 1.24).
5. Sensitivity knobs added the same evening (2026-09-15); every default reproduces upstream behaviour:
   - `DARE_P1_BOT_WARP=scale` in `bot_sort.py::multi_gmc` applies the scale-exact warp on the xywh state (R on x,y and vx,vy; isotropic s on w,h,vw,vh) instead of upstream's `kron(I4, sR)`.
   - `DARE_P1_BOT_REID=0` (adapter) runs without ReID (GMC + BYTE only).
   - `DARE_P1_BOT_MATCH` (adapter) overrides the first-association match threshold; the harness uses 0.9, upstream 0.8.
