# DARE-MOT: Edge-Optimized Multi-Drone Target Tracking

**Dynamic Aggregation for Real-time Edge MOT**

> Research Report submitted to the 4th EELISA Scientific Competition — April 2026  
> **Author:** Alp Eldam — Istanbul Technical University  
> **Supervisor:** Prof. Farzad Hashemzadeh  
> **Funding:** TÜBİTAK 1001, Project No. 125E327

---

## What is DARE-MOT?

DARE-MOT is a multi-object tracking system designed specifically for **UAV (drone) swarms running on edge hardware** like the NVIDIA Jetson. It extends [ByteTrack](https://github.com/ifzhang/ByteTrack) with a novel memory update mechanism that prevents identity loss during occlusions and motion blur — the two most common failure modes in aerial tracking.

The key insight: standard trackers blindly update their memory of a target even when the camera image is corrupted (tree occlusion, motion blur, sudden maneuver). DARE-MOT detects when an observation is unreliable and freezes the memory instead, preserving a clean historical identity template.

---

## The Problem: Template Pollution

In a standard JDE (Joint Detection and Embedding) tracker, each target maintains a "memory template" that is updated every frame via **Exponential Moving Average (EMA)**:

$$F^t = (1 - \gamma) F^{t-1} + \gamma f^t$$

where $f^t$ is the raw appearance feature extracted from the current frame and $\gamma$ is a **fixed** update rate (e.g. 0.9).

**The flaw:** When a drone passes under a tree or makes a sharp turn, the raw feature $f^t$ becomes noisy and degraded. Because $\gamma$ is static, this corrupted observation gets permanently baked into the template $F^t$. The tracker's "memory" of the target is now polluted — leading to **Identity Switches (IDSw)** when it tries to re-identify the target after the occlusion ends.

This is called **Template Pollution**, and it is the primary cause of identity failure in nadir-view (top-down) aerial tracking.

---

## The Solution: Kinematic-Aware Dynamic Aggregation

Instead of a fixed $\gamma$, DARE-MOT computes **dynamic weights** from the detection confidence scores, and uses a **second-order memory** (looking back two frames instead of one):

$$F^t = \gamma_0 f^t + \gamma_1 F^{t-1} + \gamma_2 F^{t-2} \quad \text{subject to} \quad \sum_{i=0}^{2} \gamma_i = 1$$

The weights $\gamma_0, \gamma_1, \gamma_2$ are computed automatically from the YOLOX bounding box confidence scores $c^t, c^{t-1}, c^{t-2}$:

- **High confidence frame** → $\gamma_0$ is large → memory updates normally
- **Low confidence frame** (occlusion/blur) → $\gamma_0 \to 0$ → memory freezes, relying on clean historical templates

### Two normalization options (set via `agg_option` in `byte_tracker.py`)

**Option A — L₁ Normalization** (`agg_option = 'A'`, faster):
$$\gamma_0 = \frac{c^t}{c^t + \beta(c^{t-1} + c^{t-2})}$$

The parameter $\beta$ (default: 4) is the *Inertia of Memory* — it biases the weights toward historical templates. At $\beta=4$ with equal confidences, the current frame contributes only ~11% to the update, making the template highly resistant to a single bad frame.

**Option B — Temperature-Scaled Softmax** (`agg_option = 'B'`, default, more aggressive):
$$\gamma_i = \frac{\exp(c^{t-i} / \tau)}{\sum_{j=0}^{2} \exp(c^{t-j} / \tau)}$$

With $\tau=0.5$, even a small confidence drop is non-linearly amplified, pushing $\gamma_0$ aggressively toward zero.

### Hard Lock: Kinematic Divergence Trigger

Confidence alone doesn't catch all occlusion events (e.g. a target that partially disappears but still gets a high-confidence detection). DARE-MOT adds a second safety check:

The **Kalman Filter** predicts where the target *should* be geometrically. If the detected bounding box diverges significantly from this prediction (IoU < 0.3), the system declares a kinematic divergence event and applies a hard penalty:

$$c^t_{\text{penalized}} = c^t \times \rho \quad (\rho \to 0)$$

This forces $\gamma_0 \to 0$, completely locking the memory bank.

---

## What Changed from ByteTrack

ByteTrack is a purely IoU-based tracker with no appearance features. DARE-MOT adds three things:

### 1. Dynamic Memory Update (`yolox/tracker/byte_tracker.py`)

**`STrack` class changes:**
- Added `smooth_history` — a deque of the last 2 **aggregated** templates $F^{t-1}$ and $F^{t-2}$ (not raw features)
- Added `_calculate_gammas()` — computes dynamic weights from confidence history
- Added `update_features()` — replaces static EMA with DARE-MOT second-order aggregation

```python
# Old (ByteTrack standard EMA — not in this repo):
# smooth_feat = (1 - 0.9) * smooth_feat + 0.9 * new_feature

# New (DARE-MOT):
F_t = gamma_0 * f_t + gamma_1 * F_t1 + gamma_2 * F_t2
```

### 2. Kinematic Hard Lock (`BYTETracker.update()`)

Before calling `track.update()` (which would overwrite the Kalman prediction), the predicted bbox is saved. After the update, IoU between prediction and detection is computed. If IoU < 0.3 **or** detection score < 0.4, `update_features()` is not called — the memory stays frozen.

```python
pred_tlwh = track.tlwh.copy()          # save KF prediction
track.update(det, self.frame_id)        # update position

# compute IoU between prediction and detection
kf_iou = compute_iou(pred_tlwh, det.tlwh)
if det.score >= 0.4 and kf_iou >= 0.3:
    track.update_features(det.curr_feat, det.score)
# else: memory stays frozen
```

### 3. Appearance-Fused Association (`yolox/tracker/matching.py`)

ByteTrack matches detections to tracks using IoU only. DARE-MOT adds appearance (ReID) to the first association pass:

```
cost = 0.5 × IoU_distance + 0.5 × ReID_distance
```

ReID features are extracted from each detection crop using **MobileNetV2** (pretrained, 1280-dim, GAP-pooled). The `embedding_distance_safe` function handles tracks without features gracefully (cost = 1.0 fallback to IoU).

The second association pass (low-confidence detections) still uses IoU only — appearance is unreliable at low confidence.

---

## Repository Structure

```
DARE-MOT/
├── yolox/
│   └── tracker/
│       ├── byte_tracker.py     ← DARE-MOT core (STrack + BYTETracker)
│       ├── matching.py         ← matching utils + embedding_distance_safe
│       └── kalman_filter.py    ← unchanged from ByteTrack
├── tools/
│   ├── track.py                ← evaluation script
│   └── mota.py                 ← metric computation
├── convert_visdrone.py         ← VisDrone → MOTChallenge format
├── exps/                       ← YOLOX experiment configs
└── requirements.txt
```

---

## Datasets

DARE-MOT is evaluated on aerial UAV benchmarks:

- **VisDrone2019-MOT** — primary benchmark. Nadir-view footage with severe occlusions, small targets (<30×30px), high density (100+ objects/frame).
- **UAVDT** — secondary benchmark. Similar aerial characteristics.

Download and convert VisDrone to MOTChallenge format:
```bash
python convert_visdrone.py \
    --src /path/to/VisDrone/VisDrone2019-MOT-val \
    --dst /path/to/datasets/visdrone/val
```

---

## Installation

```bash
git clone https://github.com/crystalfruit2/DARE-MOT.git
cd DARE-MOT
pip install -r requirements.txt
python setup.py develop
```

**Requirements:** Python 3.8+, PyTorch, torchvision, OpenCV  
**GPU:** CUDA recommended for benchmarks. CPU / MPS (Apple Silicon) works for development.

---

## Running

```bash
python tools/track.py \
    --exp_file exps/example/mot/yolox_x_mix_det.py \
    --ckpt /path/to/model.pth \
    --path /path/to/dataset \
    --fp16 --fuse
```

---

## Hyperparameters

All DARE-MOT hyperparameters are set inside `STrack.__init__()` in `yolox/tracker/byte_tracker.py`:

| Parameter | Default | Meaning |
|---|---|---|
| `agg_option` | `'B'` | `'A'` = L₁ normalization, `'B'` = Softmax |
| `beta` | `4.0` | Inertia of Memory (Option A only) |
| `tau` | `0.5` | Softmax temperature (Option B only) |
| Score threshold | `0.4` | Min confidence to allow memory update |
| IoU threshold | `0.3` | Min KF-detection IoU before hard lock |

---

## SOTA Comparison (VisDrone-MOT)

| Model | MOTA ↑ | IDF1 ↑ | IDSw ↓ | Domain |
|---|---|---|---|---|
| DeepSORT | 31.4% | 34.2% | High | Ground |
| FairMOT | 38.6% | 42.1% | Medium | Ground |
| ByteTrack | 43.5% | 46.8% | Medium | Edge/Aerial |
| **DARE-MOT** | **TBD** | **TBD** | **TBD** | **Edge/Aerial** |

---

## Base Framework

DARE-MOT is built on top of **ByteTrack** (ECCV 2022).  
→ [Original ByteTrack README and paper](docs/BYTETRACK_README.md)

> Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box," ECCV 2022. [arXiv:2110.06864](https://arxiv.org/abs/2110.06864)

---

## License

Apache 2.0 — see [LICENSE](LICENSE)
