from collections import deque
import numpy as np
import os
import os.path as osp
import copy
import cv2
import torch
import torchvision.transforms as T
from torchvision.models import mobilenet_v2
import torch.nn.functional as F

from .kalman_filter import KalmanFilter
from yolox.tracker import matching
from .basetrack import BaseTrack, TrackState

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, feature=None, cls=-1):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        # Detector class of the most recent matched detection (model head id, 0..N-1;
        # -1 = unknown). Threaded through so per-class MOT scoring is possible; class is
        # NOT used in association — tracking stays class-agnostic, only scoring splits by it.
        self.cls = int(cls)
        self.tracklet_len = 0

        # Hyperparameters — env-overridable for ablation; defaults reproduce the validated config.
        self.agg_option = os.environ.get('DARE_AGG', 'B')          # 'A' L1 Bias, 'B' Softmax
        self.beta = float(os.environ.get('DARE_BETA', '4.0'))      # Inertia of Memory (Option A)
        self.tau = float(os.environ.get('DARE_TAU', '0.5'))        # Softmax Temperature (Option B)
        self.static_ema = float(os.environ.get('DARE_STATIC_EMA', '-1'))  # >=0: static-EMA control gamma
        # Ablation knobs (meeting-notes-2026-07-16). agg_order = N, the number of historical
        # aggregated templates in the memory window: N=2 default -> [t-2,t-1,t]; N=1 -> [t-1,t];
        # general N -> [t-N,...,t-1,t]. Generalized 2026-07-28 to test N=5/N=10 (was hardcoded 1/2).
        self.agg_order = int(os.environ.get('DARE_AGG_ORDER', '2'))
        _sg = os.environ.get('DARE_STATIC_GAMMAS', '').strip()       # fixed weights -> static-EMA at N=order
        self.static_gammas = np.array([float(x) for x in _sg.split(',')]) if _sg else None

        # smooth_history stores the last N aggregated templates F^{t-N}..F^{t-1}
        # (NOT raw features — we store the output of update_features each step)
        self.smooth_history = deque(maxlen=self.agg_order)
        self.scores = deque(maxlen=self.agg_order + 1)
        self.curr_feat = None
        self.smooth_feat = None

        # Two-frame dynamic IoU gate (2026-07-28, Prof Farzad's idea — meeting-notes-2026-07-28
        # item 8). Confirmed NOT a novelty claim (STC-SORT, Applied Sciences 2026, already ships
        # a more general learned version of this over a much longer window — see
        # novelty-triage-2026-07-28 §6). Implemented anyway as an honest ablation/negative-result.
        # kf_history stores (mean, covariance) snapshots taken right after each KF correction
        # (activate/re_activate/update), oldest -> newest, so a second "t-2 coast" prediction can
        # be derived by propagating the OLDER snapshot two steps forward, ignoring the
        # intermediate t-1 detection entirely. Disabled by default -- see BYTETracker.twoframe_gate.
        self.kf_history = deque(maxlen=2)
        # Reliability statistic for 'resid' mode: IoU between the KF prediction and the detection
        # that was fused in at the track's most recent correction (same tau_shape statistic
        # already computed for the lock-on feature, just persisted across frames). Defaults to
        # 1.0 (assume consistent) until a correction has actually happened.
        self.last_kf_iou = 1.0

        if feature is not None:
            feat_norm = feature / (np.linalg.norm(feature) + 1e-6)
            self.smooth_feat = feat_norm.copy()
            self.smooth_history.append(feat_norm.copy())
            self.scores.append(score)
            self.curr_feat = feat_norm

    def _calculate_gammas(self):
        """
        Calculates dynamic weights based on the historical confidences.
        General N-tap: c_hist = [c_{t-N}, ..., c_{t-1}, c_t], oldest -> newest,
        length self.agg_order + 1 (self.scores is already exactly this window).
        """
        c_hist = list(self.scores)

        if self.agg_option == 'A':
            # Option A: L1 Normalization with Historical Bias
            c_t = c_hist[-1]
            older = c_hist[:-1]
            denom = c_t + self.beta * sum(older) + 1e-6  # epsilon prevents div by zero
            gammas = [self.beta * c / denom for c in older] + [c_t / denom]
            return np.array(gammas)

        elif self.agg_option == 'B':
            # Option B: Temperature-Scaled Softmax
            scores_array = np.array(c_hist)
            scaled_scores = scores_array / self.tau
            # Numerical stability: subtract max before exponentiating
            exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
            return exp_scores / np.sum(exp_scores)

        else:
            raise ValueError("agg_option must be 'A' or 'B'")

    def update_features(self, new_feature, new_score):
        """
        Replaces standard EMA with N-th order dynamic aggregation (DARE-MOT).
        F^t = gamma_0*f^t + sum_{i=1..N} gamma_i*F^{t-i}
        smooth_history stores the aggregated templates, not raw features.
        """
        self.scores.append(new_score)
        f_t = new_feature / (np.linalg.norm(new_feature) + 1e-6)  # normalize raw feature
        N = self.agg_order

        if self.static_ema >= 0.0:
            # Static-EMA control (ablation): F^t = g*F^{t-1} + (1-g)*f^t
            if self.smooth_feat is None:
                new_smooth = f_t.copy()
            else:
                g = self.static_ema
                new_smooth = g * self.smooth_feat + (1.0 - g) * f_t
        elif self.static_gammas is not None:
            # Static-EMA at N=order (ablation): fixed-weight (N+1)-tap over [F^{t-N},...,F^{t-1}, f^t].
            # Same window as the dynamic path, but weights are constant (not confidence-derived) —
            # isolates the value of dynamic weighting at equal order.
            if len(self.smooth_history) < N:
                new_smooth = f_t.copy()
            else:
                feats_array = np.array(list(self.smooth_history) + [f_t])
                new_smooth = np.average(feats_array, axis=0, weights=self.static_gammas)
        elif len(self.smooth_history) < N or len(self.scores) < N + 1:
            # Warmup: not enough history yet, use raw feature directly
            new_smooth = f_t.copy()
        else:
            # Full N-th order aggregation using aggregated historical templates
            gammas = self._calculate_gammas()  # [gamma_N,...,gamma_1,gamma_0], oldest -> newest
            feats_array = np.array(list(self.smooth_history) + [f_t])  # [F^{t-N},...,F^{t-1}, f_t]
            new_smooth = np.average(feats_array, axis=0, weights=gammas)
            if getattr(STrack, 'dare_diag', False):
                # how far the dynamic gate's newest-frame weight deviates from static EMA's implicit 0.9
                STrack.gamma_devs.append(abs(float(gammas[-1]) - 0.9))

        # Re-normalize to keep on unit hypersphere
        norm = np.linalg.norm(new_smooth)
        if norm > 0:
            new_smooth /= norm

        self.smooth_feat = new_smooth
        self.smooth_history.append(new_smooth.copy())  # store smooth template, not raw


    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.kf_history.append((self.mean.copy(), self.covariance.copy(), self.frame_id))

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls
        self.kf_history.append((self.mean.copy(), self.covariance.copy(), self.frame_id))

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.cls = new_track.cls
        self.kf_history.append((self.mean.copy(), self.covariance.copy(), self.frame_id))

    def coast_box_two_steps(self, kalman_filter, current_frame_id, max_horizon=3):
        """Two-frame dynamic IoU gate (2026-07-28): predicts the box at the CURRENT frame from
        the KF state as it stood right after the t-2 correction -- propagated forward the actual
        number of elapsed frames, ignoring every intermediate detection entirely -- for comparison
        against the standard 1-step-ahead prediction used by the existing single-frame gate.

        BUGFIX (found by an Opus brainstorm pass, 2026-07-28): the original version hard-coded
        "propagate 2 steps" regardless of how stale kf_history[0] actually was. That's correct
        for a continuously-matched track (corrected every frame, so kf_history[0] really is
        exactly 2 frames back) but silently WRONG for any track that missed frames before its
        last two corrections (lost/re-associated tracks, up to DARE_REASSOC_MAX/max_time_lost
        frames stale) -- exactly the population where a second geometric opinion could plausibly
        matter. Now propagates the true elapsed step count and bails (returns None, falling back
        to the single-frame gate) if that exceeds max_horizon, since projecting an already-stale
        state many steps forward stops being a meaningful "t-2 opinion" at all.

        Returns (tlbr, covariance) for the coasted prediction, or (None, None) if this track
        doesn't yet have a usable 2-correction-old snapshot."""
        if len(self.kf_history) < 2:
            return None, None
        mean, cov, snap_frame = self.kf_history[0]  # state as of t-2, post-correction (oldest stored)
        steps = current_frame_id - snap_frame
        if steps < 1 or steps > max_horizon:
            return None, None
        mean, cov = mean.copy(), cov.copy()
        for _ in range(steps):
            mean, cov = kalman_filter.predict(mean, cov)
        tlwh = mean[:4].copy()
        tlwh[2] *= tlwh[3]
        tlwh[:2] -= tlwh[2:] / 2
        tlbr = tlwh.copy()
        tlbr[2:] += tlbr[:2]
        return tlbr, cov

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        #self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        # Ablation knobs (env-overridable; defaults reproduce validated behavior)
        self.lock_on = os.environ.get('DARE_LOCK', '1') == '1'     # kinematic/confidence hard lock
        self.reid_lambda = float(os.environ.get('DARE_LAMBDA', '0.5'))  # appearance weight in fused cost

        # Scale-gated appearance fusion. The ablation shows appearance helps LARGE targets
        # and hurts tiny ones (tens-of-pixel UAV crops carry no reliable identity), so gate
        # the per-detection appearance weight by box AREA (original-image px):
        #   DARE_LAMBDA_GATE='size' -> lambda ramps 0 (area<=GATE_LO) .. reid_lambda (area>=GATE_HI).
        #   GATE_HI<=GATE_LO gives a hard gate at GATE_LO. Default 'none' = constant lambda (unchanged).
        self.lambda_gate = os.environ.get('DARE_LAMBDA_GATE', 'none')  # 'none' | 'size'
        self.gate_lo = float(os.environ.get('DARE_GATE_LO', '0'))      # area px; below -> lambda 0
        self.gate_hi = float(os.environ.get('DARE_GATE_HI', '0'))      # area px; above -> full lambda

        # Class-exclude gate (2026-07-24 diagnostic): force lambda=0 (IoU-only, i.e. ByteTrack
        # matching) for specific detection classes, composable with the size gate above. Model
        # head ids are 0-indexed (0=pedestrian,1=car,2=van,3=truck,4=bus per convert_visdrone_mc.py).
        # e.g. DARE_LAMBDA_CLASS_EXCLUDE=3,4 turns appearance off for truck+bus only. Empty (default)
        # = no exclusion, unchanged behavior.
        self.lambda_class_exclude = set(
            int(c) for c in os.environ.get('DARE_LAMBDA_CLASS_EXCLUDE', '').split(',') if c.strip() != ''
        )

        # IoU-feasibility gate on the first-pass fused cost. Decomposition (2026-07-20) showed
        # the appearance term manufactures FP by letting a low ReID distance pull a
        # geometrically-implausible (low-IoU) pair under match_thresh. This masks any pair whose
        # RAW IoU distance (1-IoU, pre fuse_score) exceeds the gate to inf, so appearance can only
        # re-rank IoU-feasible candidates, never rescue a non-overlapping one.
        #   DARE_IOU_GATE = max allowed 1-IoU for an appearance-eligible match (e.g. 0.7 => IoU>=0.3).
        #   Default 0.95 (IoU>=0.05) = HEADLINE: Pareto-dominates ByteTrack (IDF1 66.0/65.0,
        #   MOTA 54.6/54.5, IDSw 421/550 = -23%), zero per-seq regression. Mid-plateau (robust
        #   across IoU floor [0.03,0.10]), not edge-fished toward the off-cliff. Set 1.0 to disable.
        self.iou_gate = float(os.environ.get('DARE_IOU_GATE', '0.95'))

        # Two-frame dynamic IoU gate (2026-07-28, Prof Farzad's idea — meeting-notes-2026-07-28
        # item 8). Confirmed NOT a novelty claim: STC-SORT (Applied Sciences 2026) already ships
        # a strictly more general learned version of this (4D cost volume over a 25-30 frame
        # window, graph-attention softmax weights) on VisDrone — see novelty-triage-2026-07-28 §6.
        # Implemented anyway per Alp's instruction, as an honest ablation entry regardless of sign.
        #   DARE_TWOFRAME_GATE=1 enables; default 0 = exactly current (validated) behavior.
        #   DARE_TWOFRAME_MODE: 'static' (fixed blend, DARE_TWOFRAME_ALPHA weights the t-1 term)
        #     | 'dynamic' (default; softmax-weights the t-2 vs t-1 IoU term from the track's own
        #     last two detection confidences [c_{t-2}, c_{t-1}], same pattern as the appearance
        #     dynamic-aggregation's _calculate_gammas, temperature DARE_TWOFRAME_TAU).
        self.twoframe_gate = os.environ.get('DARE_TWOFRAME_GATE', '0') == '1'
        self.twoframe_mode = os.environ.get('DARE_TWOFRAME_MODE', 'dynamic')
        self.twoframe_alpha = float(os.environ.get('DARE_TWOFRAME_ALPHA', '0.7'))
        self.twoframe_tau = float(os.environ.get('DARE_TWOFRAME_TAU', '0.5'))
        # Extended variant sweep (2026-07-28 evening, after an Opus brainstorm pass). See
        # _two_frame_gate's docstring for what each knob controls; all default to values that
        # only matter when the corresponding DARE_TWOFRAME_MODE is selected.
        self.twoframe_tau2_gate = float(os.environ.get('DARE_TWOFRAME_TAU2_GATE', '-1'))  # 'max' mode; -1 = fall back to DARE_IOU_GATE
        self.twoframe_resid_thresh = float(os.environ.get('DARE_TWOFRAME_RESID_THRESH', '0.3'))  # 'resid' mode
        self.twoframe_age_min = int(os.environ.get('DARE_TWOFRAME_AGE_MIN', '10'))  # 'age' mode
        self.twoframe_scale_min = float(os.environ.get('DARE_TWOFRAME_SCALE_MIN', '2500'))  # 'scale' mode, px^2
        self.twoframe_disagree_delta = float(os.environ.get('DARE_TWOFRAME_DISAGREE_DELTA', '0.3'))  # 'disagree' mode
        self.twoframe_diag_total = 0    # V9-lite diagnostic counters (printed iff DARE_DIAG=1)
        self.twoframe_diag_coast_wins = 0
        self.gate_reject_count = 0      # V2 diagnostic: how many pairs the IoU gate actually vetoes
        self.gate_total_count = 0

        # Class-blocked association (2026-07-29, validity check raised in novelty-triage-2026-07-28
        # -- byte_tracker's association has always been class-agnostic by design (see STrack.cls
        # comment above), but mmtracking/BoxMOT default to class-BLOCKED cost (mmtracking's
        # cate_cost, PR #548: (1-match)*1e6 added into the joint Hungarian cost) -- the field's
        # actual default on multi-class benchmarks. Both arms (DARE and ByteTrack) share whichever
        # mode is active so the DELTA stays fair either way; this knob lets us re-measure the
        # headline under the field-standard class-blocked setting instead of just asserting the
        # agnostic delta is fair.
        #   DARE_CLASS_BLOCK=1 forbids cross-class matches in every association stage (inf cost,
        #   same mechanism as the IoU-feasibility gate). Default 0 = unchanged agnostic behavior.
        self.class_block = os.environ.get('DARE_CLASS_BLOCK', '0') == '1'

        # Fix #1 (foreground-focused appearance) & Fix #3 (re-association age cap) — meeting-brief-2026-07-16.
        # All default to reproduce the current real-appearance baseline exactly (no change when unset).
        self.crop_shrink = float(os.environ.get('DARE_CROP_SHRINK', '0.0'))  # shrink box each side toward center before ReID crop (fraction 0..0.4)
        self.pool_mode = os.environ.get('DARE_POOL', 'mean')                 # 'mean' (uniform GAP) | 'center' (Gaussian center-weighted pooling)
        self.pool_sigma = float(os.environ.get('DARE_POOL_SIGMA', '0.5'))    # center-pool Gaussian sigma as fraction of half-size
        self.reassoc_max = int(os.environ.get('DARE_REASSOC_MAX', '-1'))     # cap lost-track re-association age in frames; -1 = use max_time_lost

        # Phase 0.2 gate-activity diagnostics (results-improvement-plan-2026-07-13) — DARE_DIAG=1 to enable
        self.dare_diag = os.environ.get('DARE_DIAG', '0') == '1'
        self.lock_fires = 0
        STrack.dare_diag = self.dare_diag
        STrack.gamma_devs = []

        # --- FEATURE EXTRACTOR (ReID appearance embedding) ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # DARE_REID selects the embedding:
        #   'mobilenet' (default, legacy) — frozen ImageNet MobileNetV2 features. An
        #       object-*classification* backbone pressed into *instance* ReID: wrong
        #       objective, wrong domain. Kept as the baseline for the paper's
        #       before/after (naive CNN features vs a real ReID embedding).
        #   'osnet' — a purpose-built person-ReID embedding (torchreid OSNet,
        #       instance-discriminative). The intended fix. Weights default to the
        #       verified osnet_x1_0 MSMT17 checkpoint in reid_weights/.
        self.reid_backend = os.environ.get('DARE_REID', 'mobilenet')
        if self.reid_backend == 'osnet':
            from torchreid.reid.utils import FeatureExtractor
            model_name = os.environ.get('DARE_REID_MODEL', 'osnet_x1_0')
            default_w = osp.join(osp.dirname(__file__), '..', '..',
                                 'reid_weights', 'osnet_x1_0_msmt17.pth')
            weights = os.environ.get('DARE_REID_WEIGHTS', default_w)
            self.reid_extractor = FeatureExtractor(
                model_name=model_name, model_path=weights,
                device=str(self.device), verbose=False)
            self.feat_dim = 512
        else:
            self.extractor = mobilenet_v2(pretrained=True).features.to(self.device).eval()
            self.transform = T.Compose([
                T.ToTensor(),
                T.Resize((128, 64)),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.feat_dim = 1280

    def _pool(self, fmap):
        """Pool a [1, C, H, W] feature map to [C].
        Default 'mean' is uniform global-average-pool (bit-identical to the old
        `.mean([2,3]).squeeze()`). 'center' applies a separable Gaussian weight
        centered on the map so the target (usually box-centered) dominates and
        peripheral background (barriers, adjacent bodies) is down-weighted."""
        if self.pool_mode == 'center':
            _, _, H, W = fmap.shape
            ys = torch.arange(H, device=fmap.device, dtype=torch.float32)
            xs = torch.arange(W, device=fmap.device, dtype=torch.float32)
            cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
            sy = max(self.pool_sigma * (H / 2.0), 1e-3)
            sx = max(self.pool_sigma * (W / 2.0), 1e-3)
            wy = torch.exp(-0.5 * ((ys - cy) / sy) ** 2)
            wx = torch.exp(-0.5 * ((xs - cx) / sx) ** 2)
            w = wy[:, None] * wx[None, :]           # [H, W]
            w = w / (w.sum() + 1e-6)
            feat = (fmap[0] * w).sum(dim=(1, 2))    # [C]
            return feat.cpu().numpy()
        return fmap.mean([2, 3]).squeeze().cpu().numpy()

    def _gated_lambda(self, detections):
        """Appearance weight for the fused cost. Constant self.reid_lambda unless
        DARE_LAMBDA_GATE='size' and/or DARE_LAMBDA_CLASS_EXCLUDE is set, in which case a
        per-detection [n_det] vector is returned (broadcasts over the columns/detections
        of the cost matrix):
          size gate: tiny targets (area<=gate_lo) get lambda 0 (IoU only — no reliable
            identity in a few-pixel crop); large targets (area>=gate_hi) get the full
            lambda; linear ramp between. gate_hi<=gate_lo => hard gate at gate_lo.
          class-exclude: forces lambda 0 for detections whose class is in the exclude
            set, on top of (composes with) whatever the size gate already produced."""
        if len(detections) == 0:
            return self.reid_lambda
        if self.lambda_gate == 'size':
            areas = np.array([d.tlwh[2] * d.tlwh[3] for d in detections], dtype=np.float32)
            if self.gate_hi <= self.gate_lo:
                ramp = (areas >= self.gate_lo).astype(np.float32)
            else:
                ramp = np.clip((areas - self.gate_lo) / (self.gate_hi - self.gate_lo), 0.0, 1.0)
            lam = self.reid_lambda * ramp
        elif self.lambda_class_exclude:
            lam = np.full(len(detections), self.reid_lambda, dtype=np.float32)
        else:
            return self.reid_lambda
        if self.lambda_class_exclude:
            cls_arr = np.array([d.cls for d in detections], dtype=np.int64)
            lam = np.where(np.isin(cls_arr, list(self.lambda_class_exclude)), 0.0, lam)
        return lam

    def _class_block(self, dists, tracks, dets):
        """In-place cross-class veto (DARE_CLASS_BLOCK=1 only): dists[i,j] = inf wherever
        track i's class differs from detection j's class. Unknown class (-1, the 5-col
        no-class path) never blocks -- doesn't occur on the MC benchmark this targets.
        No-op (returns dists unchanged) unless self.class_block is set."""
        if not self.class_block or len(tracks) == 0 or len(dets) == 0:
            return dists
        t_cls = np.array([t.cls for t in tracks], dtype=np.int64)
        d_cls = np.array([d.cls for d in dets], dtype=np.int64)
        mismatch = t_cls[:, None] != d_cls[None, :]
        known = (t_cls[:, None] != -1) & (d_cls[None, :] != -1)
        dists[mismatch & known] = np.inf
        return dists

    def _two_frame_gate(self, strack_pool, detections, iou_t1):
        """Two-frame dynamic IoU gate (2026-07-28, ablation only — see __init__ docstring above
        for the novelty verdict; extended 2026-07-28 evening after an Opus brainstorm pass added
        a bugfix (coast_box_two_steps now uses the true elapsed-frame horizon, not a hard-coded
        2) and a wider variant sweep. Theoretical note carried from that pass: under a linear-
        Gaussian constant-velocity model, the t-1 posterior is conditioned on a strict information
        superset of the t-2 posterior (same prior + one more measurement), so t-1's 1-step
        prediction is MMSE-optimal and no *blend* of (iou_t1, iou_t2) can beat iou_t1 alone unless
        the t-1 measurement itself was an outlier / a CV-model violation -- which is exactly what
        'resid' mode targets and the other modes (by construction) cannot detect.

        DARE_TWOFRAME_MODE:
          'dynamic' (default) -- per-track softmax over [c_{t-2},c_{t-1}] weights iou_t1 vs iou_t2
              (mirrors the appearance dynamic-aggregation's _calculate_gammas).
          'static'  -- fixed blend, DARE_TWOFRAME_ALPHA weights iou_t1 (alpha=0 -> pure iou_t2,
              alpha=1 -> pure iou_t1; sweeping this closes the whole convex-combination family).
          'min'     -- OR-gate / "take the most optimistic": elementwise min(iou_t1, iou_t2) --
              relaxes the validated gate (can only re-admit pairs it used to reject).
          'max'     -- AND-gate / "take the most conservative": rejects if iou_t1 exceeds the
              main gate OR iou_t2 exceeds DARE_TWOFRAME_TAU2_GATE (defaults to the same value as
              DARE_IOU_GATE -- an uncalibrated but honest starting point; strictly tightens the gate).
          'cov'     -- inverse-covariance-trace weighted blend (principled-looking, but the coast
              path always has strictly more accumulated process noise than the 1-step path, so
              this is predicted to degenerate to a near-constant effective alpha -- included to
              pre-empt "did you weight by the KF's own uncertainty" rather than to win).
          'resid'   -- per-track HARD selection: use iou_t2 for the whole row if the track's most
              recent correction was itself a kinematic outlier (last_kf_iou < DARE_TWOFRAME_RESID_
              THRESH, same statistic already used for the lock-on feature), else iou_t1. The only
              mode that fires in the theoretically-live regime above.
          'age'     -- only tracks with tracklet_len >= DARE_TWOFRAME_AGE_MIN get blended with
              iou_t2 (younger tracks' CV velocity estimate hasn't converged, so their coast is
              worse than usual); younger tracks stay pure iou_t1.
          'scale'   -- only tracks whose own box area exceeds DARE_TWOFRAME_SCALE_MIN get blended
              with iou_t2 (a fixed pixel coast-displacement is a bigger fraction of a small box's
              own size, collapsing its IoU faster -- same mechanism as the project's scale-
              conditional CMC finding); smaller tracks stay pure iou_t1.
          'disagree'-- veto (force-reject) any pair where |iou_t1 - iou_t2| > DARE_TWOFRAME_
              DISAGREE_DELTA, treating strong disagreement between the two signals as an
              uncertainty flag rather than fusing them into a point estimate.

        Tracks without a usable t-2 snapshot (too young, or too stale -- see coast_box_two_steps'
        max_horizon bail-out) fall back to iou_t1 alone in every mode."""
        if len(strack_pool) == 0 or len(detections) == 0:
            return iou_t1

        tlbrs_t2 = []
        covs_t2 = []
        have_t2 = np.zeros(len(strack_pool), dtype=bool)
        for i, st in enumerate(strack_pool):
            box, cov2 = st.coast_box_two_steps(self.kalman_filter, self.frame_id)
            if box is not None:
                tlbrs_t2.append(box)
                covs_t2.append(cov2)
                have_t2[i] = True
            else:
                tlbrs_t2.append(st.tlbr)  # placeholder; masked out by have_t2 below
                covs_t2.append(None)

        iou_t2 = matching.iou_distance(tlbrs_t2, [d.tlbr for d in detections])
        combined = iou_t1.copy()
        mode = self.twoframe_mode

        if mode == 'static':
            a = self.twoframe_alpha
            combined[have_t2] = a * iou_t1[have_t2] + (1.0 - a) * iou_t2[have_t2]

        elif mode == 'min':
            combined[have_t2] = np.minimum(iou_t1[have_t2], iou_t2[have_t2])

        elif mode == 'max':
            tau2 = self.twoframe_tau2_gate if self.twoframe_tau2_gate >= 0 else self.iou_gate
            rows = np.where(have_t2)[0]
            for i in rows:
                fails = iou_t2[i] > tau2
                combined[i] = np.where(fails, np.inf, iou_t1[i])

        elif mode == 'disagree':
            rows = np.where(have_t2)[0]
            for i in rows:
                disagree = np.abs(iou_t1[i] - iou_t2[i]) > self.twoframe_disagree_delta
                combined[i] = np.where(disagree, np.inf, iou_t1[i])

        elif mode == 'cov':
            for i, st in enumerate(strack_pool):
                if not have_t2[i]:
                    continue
                p1 = float(np.trace(st.covariance[:2, :2]))
                p2 = float(np.trace(covs_t2[i][:2, :2]))
                w1 = 1.0 / max(p1, 1e-9)
                w2 = 1.0 / max(p2, 1e-9)
                a = w1 / (w1 + w2)  # weight on iou_t1
                combined[i] = a * iou_t1[i] + (1.0 - a) * iou_t2[i]

        elif mode == 'resid':
            thresh = self.twoframe_resid_thresh
            for i, st in enumerate(strack_pool):
                if not have_t2[i]:
                    continue
                if st.last_kf_iou < thresh:
                    combined[i] = iou_t2[i]
                # else: leave as iou_t1 (already the default in `combined`)

        elif mode == 'age':
            min_len = self.twoframe_age_min
            a = self.twoframe_alpha
            for i, st in enumerate(strack_pool):
                if not have_t2[i] or st.tracklet_len < min_len:
                    continue
                combined[i] = a * iou_t1[i] + (1.0 - a) * iou_t2[i]

        elif mode == 'scale':
            min_area = self.twoframe_scale_min
            a = self.twoframe_alpha
            for i, st in enumerate(strack_pool):
                if not have_t2[i]:
                    continue
                tlwh = st.tlwh
                area = tlwh[2] * tlwh[3]
                if area < min_area:
                    continue
                combined[i] = a * iou_t1[i] + (1.0 - a) * iou_t2[i]

        else:  # 'dynamic' (default)
            for i, st in enumerate(strack_pool):
                if not have_t2[i]:
                    continue
                c_hist = list(st.scores)[-2:]  # [c_{t-2}, c_{t-1}], oldest -> newest
                if len(c_hist) < 2:
                    continue
                w = np.exp(np.array(c_hist) / self.twoframe_tau)
                w = w / w.sum()
                # w[1] weights the newer (t-1-based, iou_t1) term; w[0] weights the older
                # (t-2-based coast, iou_t2) term.
                combined[i] = w[1] * iou_t1[i] + w[0] * iou_t2[i]

        # V9-lite diagnostic (DARE_DIAG=1 only): per-row, does the coast prediction's best
        # candidate column look more confident than the standard prediction's? A cheap proxy
        # for "how often would trusting iou_t2 change anything," not a matched-pair comparison.
        if self.dare_diag:
            rows = np.where(have_t2)[0]
            for i in rows:
                self.twoframe_diag_total += 1
                if iou_t2[i].min() < iou_t1[i].min():
                    self.twoframe_diag_coast_wins += 1

        return combined

    def _extract_features_osnet(self, detections, raw_frame):
        """ReID features via a real person-ReID embedding (torchreid OSNet).
        The model does its own resize/normalize/pooling, so DARE_POOL and the
        feature-map pooling knobs do not apply here; DARE_CROP_SHRINK still does
        (shrinks the box before cropping to cut peripheral background). All crops
        are embedded in one batched forward pass."""
        H_img, W_img = raw_frame.shape[:2]
        crops, idxs = [], []
        for i, det in enumerate(detections):
            x, y, w, h = det.tlwh
            if self.crop_shrink > 0.0:
                dx, dy = w * self.crop_shrink, h * self.crop_shrink
                x, y, w, h = x + dx, y + dy, w - 2 * dx, h - 2 * dy
            x, y, w, h = int(x), int(y), int(w), int(h)
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(W_img, x + w), min(H_img, y + h)
            crop = raw_frame[y1:y2, x1:x2]
            if crop.size > 0:
                if crop.ndim == 3 and crop.shape[2] == 3:
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crops.append(crop)
                idxs.append(i)
            else:
                det.curr_feat = np.zeros(self.feat_dim, dtype=np.float32)
        if crops:
            with torch.no_grad():
                feats = self.reid_extractor(crops).cpu().numpy()
            for j, i in enumerate(idxs):
                detections[i].curr_feat = feats[j].astype(np.float32)

    def _extract_features(self, detections, raw_frame):
        """Extract a ReID feature per detection from the raw frame.
        With DARE_CROP_SHRINK>0 the box is shrunk toward its center before
        cropping (cuts peripheral background before it enters the CNN);
        pooling is governed by DARE_POOL. Behavior is bit-identical to the
        old inline block when both knobs are at their defaults."""
        if self.reid_backend == 'osnet':
            self._extract_features_osnet(detections, raw_frame)
            return
        H_img, W_img = raw_frame.shape[:2]
        for det in detections:
            x, y, w, h = det.tlwh
            if self.crop_shrink > 0.0:
                dx, dy = w * self.crop_shrink, h * self.crop_shrink
                x, y, w, h = x + dx, y + dy, w - 2 * dx, h - 2 * dy
            x, y, w, h = int(x), int(y), int(w), int(h)
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(W_img, x + w), min(H_img, y + h)
            crop = raw_frame[y1:y2, x1:x2]
            if crop.size > 0:
                if crop.ndim == 3 and crop.shape[2] == 3:
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop_t = self.transform(crop).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    fmap = self.extractor(crop_t)
                det.curr_feat = self._pool(fmap)
            else:
                det.curr_feat = np.zeros(self.feat_dim, dtype=np.float32)

    def update(self, output_results, img_info, img_size):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        raw_frame = img_info[0] if isinstance(img_info[0], np.ndarray) else None

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
            classes = np.full(len(output_results), -1)  # no class info in the 5-col path
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
            # YOLOX postprocess col 6 = predicted class (model head id, 0..num_classes-1)
            classes = output_results[:, 6] if output_results.shape[1] > 6 else np.full(len(output_results), -1)
        if raw_frame is not None:
            img_h, img_w = raw_frame.shape[:2]
        else:
            img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        classes_keep = classes[remain_inds]
        classes_second = classes[inds_second]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, cls=c) for
                          (tlbr, s, c) in zip(dets, scores_keep, classes_keep)]
        else:
            detections = []

        if raw_frame is not None and len(detections) > 0:
            self._extract_features(detections, raw_frame)

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        # Fix #3: cap how stale a lost track may be to remain re-associable. Beyond
        # DARE_REASSOC_MAX frames it's excluded from matching (still removed on the
        # normal max_time_lost schedule). -1 disables the cap (== old behavior).
        if self.reassoc_max >= 0:
            eligible_lost = [t for t in self.lost_stracks
                             if self.frame_id - t.end_frame <= self.reassoc_max]
        else:
            eligible_lost = self.lost_stracks
        strack_pool = joint_stracks(tracked_stracks, eligible_lost)
        STrack.multi_predict(strack_pool)

        iou_dists = matching.iou_distance(strack_pool, detections)
        raw_iou_dists = iou_dists.copy()  # geometry only (1-IoU), before score fusion; used by the IoU gate
        if not self.args.mot20:
            iou_dists = matching.fuse_score(iou_dists, detections)

        # Fuse ReID distance with IoU — smooth_feat is the DARE-MOT aggregated template.
        # embedding_distance_safe falls back to cost=1.0 for any track/det without features,
        # so the fused matrix degrades gracefully to IoU-only for those pairs.
        reid_dists = matching.embedding_distance_safe(strack_pool, detections)
        lam = self._gated_lambda(detections)  # scalar, or per-detection [n_det] when size-gated
        dists = (1.0 - lam) * iou_dists + lam * reid_dists

        # Two-frame dynamic IoU gate (ablation, default off — see DARE_TWOFRAME_GATE above).
        gate_dists = raw_iou_dists
        if self.twoframe_gate:
            gate_dists = self._two_frame_gate(strack_pool, detections, raw_iou_dists)

        # IoU-feasibility gate: appearance may re-rank but not rescue non-overlapping pairs.
        if self.iou_gate < 1.0:
            reject_mask = gate_dists > self.iou_gate
            if self.dare_diag:
                self.gate_reject_count += int(reject_mask.sum())
                self.gate_total_count += reject_mask.size
            dists[reject_mask] = np.inf

        dists = self._class_block(dists, strack_pool, detections)

        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                # Save Kalman-predicted bbox BEFORE update overwrites track.mean
                pred_tlwh = track.tlwh.copy()

                track.update(det, self.frame_id)

                # Kinematic divergence check: IoU between KF-predicted and detected bbox
                p = pred_tlwh
                d = det.tlwh
                ix1 = max(p[0], d[0]);         iy1 = max(p[1], d[1])
                ix2 = min(p[0]+p[2], d[0]+d[2]); iy2 = min(p[1]+p[3], d[1]+d[3])
                inter = max(0, ix2-ix1) * max(0, iy2-iy1)
                union = p[2]*p[3] + d[2]*d[3] - inter
                kf_iou = inter / (union + 1e-6)
                is_kinematic_divergence = kf_iou < 0.3  # tau_shape threshold
                track.last_kf_iou = kf_iou  # persisted for the two-frame gate's 'resid' mode

                if det.curr_feat is not None:
                    should_update = not self.lock_on or (det.score >= 0.4 and not is_kinematic_divergence)
                    if should_update:
                        track.update_features(det.curr_feat, det.score)
                    elif self.dare_diag:
                        self.lock_fires += 1
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, cls=c) for
                          (tlbr, s, c) in zip(dets_second, scores_second, classes_second)]
        else:
            detections_second = []

        if raw_frame is not None and len(detections_second) > 0:
            self._extract_features(detections_second, raw_frame)
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        dists = self._class_block(dists, r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                # Save Kalman-predicted bbox BEFORE update overwrites track.mean
                pred_tlwh = track.tlwh.copy()

                track.update(det, self.frame_id)

                # Kinematic divergence check: IoU between KF-predicted and detected bbox
                p = pred_tlwh
                d = det.tlwh
                ix1 = max(p[0], d[0]);         iy1 = max(p[1], d[1])
                ix2 = min(p[0]+p[2], d[0]+d[2]); iy2 = min(p[1]+p[3], d[1]+d[3])
                inter = max(0, ix2-ix1) * max(0, iy2-iy1)
                union = p[2]*p[3] + d[2]*d[3] - inter
                kf_iou = inter / (union + 1e-6)
                is_kinematic_divergence = kf_iou < 0.3  # tau_shape threshold
                track.last_kf_iou = kf_iou  # persisted for the two-frame gate's 'resid' mode

                if det.curr_feat is not None:
                    should_update = not self.lock_on or (det.score >= 0.4 and not is_kinematic_divergence)
                    if should_update:
                        track.update_features(det.curr_feat, det.score)
                    elif self.dare_diag:
                        self.lock_fires += 1
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        dists = self._class_block(dists, unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks

    def print_diag_summary(self, seq_name=None):
        """Phase 0.2 (results-improvement-plan-2026-07-13): print gate-activity counters
        accumulated over this tracker's lifetime. No-op unless DARE_DIAG=1."""
        if not self.dare_diag:
            return
        devs = STrack.gamma_devs
        mean_dev = float(np.mean(devs)) if devs else 0.0
        coast_rate = (self.twoframe_diag_coast_wins / self.twoframe_diag_total
                      if self.twoframe_diag_total else 0.0)
        reject_rate = (self.gate_reject_count / self.gate_total_count
                       if self.gate_total_count else 0.0)
        print(f"[DARE_DIAG] seq={seq_name or '?'} lock_fires={self.lock_fires} "
              f"gamma_dev_mean={mean_dev:.4f} (n={len(devs)}) "
              f"twoframe_coast_win_rate={coast_rate:.4f} (n={self.twoframe_diag_total}) "
              f"gate_reject_rate={reject_rate:.4f} (rejects={self.gate_reject_count}/{self.gate_total_count})")


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
