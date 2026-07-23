# Vendored: OC-SORT tracker

`ocsort.py`, `association.py`, `kalmanfilter.py` are copied **unmodified** from
OC-SORT (Cao et al., "Observation-Centric SORT", CVPR 2023),
https://github.com/noahcao/OC_SORT, `trackers/ocsort_tracker/` (MIT License).

Vendored 2026-07-23 for a controlled Path-1 baseline comparison in DARE-MOT — run through
DARE-MOT's own eval harness (same detector/preprocessing/scoring) via `baselines/adapters.py`
(`OCSortAdapter`). Upstream code is intentionally left untouched; class labels are attached
post-hoc in the adapter. Full reference clone (with LICENSE) is under `_baselines/OC_SORT/`.
