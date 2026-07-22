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
    def __init__(self, tlwh, score, feature=None):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        # smooth_history stores the last 2 aggregated templates F^{t-2}, F^{t-1}
        # (NOT raw features — we store the output of update_features each step)
        self.smooth_history = deque(maxlen=2)
        self.scores = deque(maxlen=3)
        self.curr_feat = None
        self.smooth_feat = None

        if feature is not None:
            feat_norm = feature / (np.linalg.norm(feature) + 1e-6)
            self.smooth_feat = feat_norm.copy()
            self.smooth_history.append(feat_norm.copy())
            self.scores.append(score)
            self.curr_feat = feat_norm
            
        # Hyperparameters — env-overridable for ablation; defaults reproduce the validated config.
        self.agg_option = os.environ.get('DARE_AGG', 'B')          # 'A' L1 Bias, 'B' Softmax
        self.beta = float(os.environ.get('DARE_BETA', '4.0'))      # Inertia of Memory (Option A)
        self.tau = float(os.environ.get('DARE_TAU', '0.5'))        # Softmax Temperature (Option B)
        self.static_ema = float(os.environ.get('DARE_STATIC_EMA', '-1'))  # >=0: static-EMA control gamma
        # Ablation knobs (meeting-notes-2026-07-16). Defaults reproduce the validated N=2 dynamic config.
        self.agg_order = int(os.environ.get('DARE_AGG_ORDER', '2'))  # 2 = [t-2,t-1,t]; 1 = [t-1,t] dynamic
        _sg = os.environ.get('DARE_STATIC_GAMMAS', '').strip()       # fixed weights -> static-EMA at N=2
        self.static_gammas = np.array([float(x) for x in _sg.split(',')]) if _sg else None
    def _calculate_gammas(self):
        """
        Calculates dynamic weights based on the historical confidences.
        Order 2 (default): 3-tap over [c_{t-2}, c_{t-1}, c_t] -> [gamma_2, gamma_1, gamma_0].
        Order 1 (ablation): 2-tap over [c_{t-1}, c_t]        -> [gamma_1, gamma_0].
        """
        if self.agg_order == 1:
            c_hist = [self.scores[-2], self.scores[-1]]        # [c_{t-1}, c_t]
        else:
            c_hist = [self.scores[0], self.scores[1], self.scores[2]]  # [c_{t-2}, c_{t-1}, c_t]

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
        Replaces standard EMA with second-order dynamic aggregation (DARE-MOT).
        F^t = gamma_0*f^t + gamma_1*F^{t-1} + gamma_2*F^{t-2}
        smooth_history stores the aggregated templates, not raw features.
        """
        self.scores.append(new_score)
        f_t = new_feature / (np.linalg.norm(new_feature) + 1e-6)  # normalize raw feature

        if self.static_ema >= 0.0:
            # Static-EMA control (ablation): F^t = g*F^{t-1} + (1-g)*f^t
            if self.smooth_feat is None:
                new_smooth = f_t.copy()
            else:
                g = self.static_ema
                new_smooth = g * self.smooth_feat + (1.0 - g) * f_t
        elif self.static_gammas is not None:
            # Static-EMA at N=2 (ablation): fixed-weight 3-tap over [F^{t-2}, F^{t-1}, f^t].
            # Same window as the dynamic path, but weights are constant (not confidence-derived) —
            # isolates the value of dynamic weighting at equal order.
            if len(self.smooth_history) < 2:
                new_smooth = f_t.copy()
            else:
                feats_array = np.array([self.smooth_history[0], self.smooth_history[1], f_t])
                new_smooth = np.average(feats_array, axis=0, weights=self.static_gammas)
        elif self.agg_order == 1:
            # Order-1 dynamic aggregation (ablation): 2-tap over [F^{t-1}, f^t] with dynamic gammas.
            if len(self.smooth_history) < 1 or len(self.scores) < 2:
                new_smooth = f_t.copy()
            else:
                gammas = self._calculate_gammas()   # [gamma_1, gamma_0]
                feats_array = np.array([self.smooth_history[-1], f_t])
                new_smooth = np.average(feats_array, axis=0, weights=gammas)
        elif len(self.smooth_history) < 2:
            # Warmup: not enough history yet, use raw feature directly
            new_smooth = f_t.copy()
        else:
            # Full second-order aggregation using aggregated historical templates
            gammas = self._calculate_gammas()
            F_t2 = self.smooth_history[0]  # F^{t-2} — aggregated template
            F_t1 = self.smooth_history[1]  # F^{t-1} — aggregated template
            feats_array = np.array([F_t2, F_t1, f_t])
            # gammas = [gamma_2, gamma_1, gamma_0] — ordered oldest to newest
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
        DARE_LAMBDA_GATE='size', in which case it is gated per-detection by box area:
        tiny targets (area<=gate_lo) get lambda 0 (IoU only — no reliable identity in a
        few-pixel crop); large targets (area>=gate_hi) get the full lambda; linear ramp
        between. gate_hi<=gate_lo => hard gate at gate_lo. Returned as a [n_det] vector
        that broadcasts over the columns (detections) of the cost matrix."""
        if self.lambda_gate != 'size' or len(detections) == 0:
            return self.reid_lambda
        areas = np.array([d.tlwh[2] * d.tlwh[3] for d in detections], dtype=np.float32)
        if self.gate_hi <= self.gate_lo:
            ramp = (areas >= self.gate_lo).astype(np.float32)
        else:
            ramp = np.clip((areas - self.gate_lo) / (self.gate_hi - self.gate_lo), 0.0, 1.0)
        return self.reid_lambda * ramp

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
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
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

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
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

        # IoU-feasibility gate: appearance may re-rank but not rescue non-overlapping pairs.
        if self.iou_gate < 1.0:
            dists[raw_iou_dists > self.iou_gate] = np.inf

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
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []

        if raw_frame is not None and len(detections_second) > 0:
            self._extract_features(detections_second, raw_frame)
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
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
        print(f"[DARE_DIAG] seq={seq_name or '?'} lock_fires={self.lock_fires} "
              f"gamma_dev_mean={mean_dev:.4f} (n={len(devs)})")


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
