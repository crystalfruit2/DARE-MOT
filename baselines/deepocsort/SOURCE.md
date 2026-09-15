# baselines/deepocsort: source and modifications

Upstream: https://github.com/GerardMaggiolino/Deep-OC-SORT (MIT, Copyright (c) 2023 Gerard Maggiolino), reference clone `_baselines/Deep-OC-SORT` @ 6bb51d0.
Files: `trackers/integrated_ocsort_embedding/{ocsort.py, association.py, kalmanfilter.py}`.

Modifications (2026-09-15; association, AW, OCR, KF and affine correction untouched):
1. `ocsort.py`: unused imports dropped; `EmbeddingComputer` and `CMCComputer` are injected (`embedder=`, `cmc=` kwargs).
   - Embedding: DARE-MOT's OSNet-AIN (shared across all appearance baselines), one L2-normalised vector per box (`grid_off=True`).
   - CMC: upstream's published runs use `method="file"`, i.e. precomputed BoT-SORT GMC affines for MOTChallenge/DanceTrack, which do not exist for VisDrone. The faithful equivalent is BoT-SORT's GMC run live (`baselines/gmc.py`, sparseOptFlow, downscale 2).
2. `association.py`: `linear_assignment`'s silent `scipy.optimize.linear_sum_assignment` fallback now raises instead (lapjv semantics must not change under a DLL block).
