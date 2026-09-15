"""CPU test for the Path-1 adapters (baselines/adapters.py) -- no GPU, no detector (2026-09-15).

Drives every adapter exactly the way mot_evaluator does: img_info[0] = the raw BGR frame, 7-column
detector rows in NETWORK-INPUT scale as a torch tensor, img_size = the exp's test_size. Detections are
the real val7 ground-truth boxes of uav0000086 frames 1..8 (so identities are known), plus look-alike
rows with model class id 7 that DARE_MAX_CLASS=4 must remove.

Checks per adapter (ocsort, botsort, deepocsort):
  (A) contract: .tlwh (4,), int .track_id, float .score, .cls in {-1, 0..4}; print_diag_summary exists
  (B) DARE_MAX_CLASS: no output box sits on a class-7 input row; no output class > 4
  (C) coordinates: outputs are in ORIGINAL image coords (>= 90% IoU-match a GT box at IoU >= 0.5)
  (D) identity: on frames 4..8 the track_id -> GT target id mapping is stable (>= 95% of matched boxes)
  (E) appearance adapters: the injected embedder is actually called
Plus (F) SharedOSNet crop logic == BYTETracker._extract_features_osnet on the same extractor (bitwise),
run only when DARE_REID_WEIGHTS points to a file (CPU forward of a few crops).

Run (lap 0.5.12 first on the path, as in the run scripts):
  $env:PYTHONPATH="C:\\Users\\User\\Desktop\\projects\\DARE-MOT-pylibs\\lap0512;C:\\Users\\User\\Desktop\\Projects\\DARE-MOT"
  python baselines/test_adapters.py
"""
import os
import sys
from collections import defaultdict
from types import SimpleNamespace

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["DARE_MAX_CLASS"] = "4"

from baselines import adapters as A  # noqa: E402

VAL = r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone2019-MOT-val"
SEQ = "uav0000086_00000_v"
TEST_SIZE = (800, 1440)
RAW_TO_MODEL = {1: 0, 4: 1, 5: 2, 6: 3, 9: 4}        # pedestrian car van truck bus -> head ids 0..4
N_FRAMES = 8
FAILS = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def load_gt():
    gt = defaultdict(list)
    for ln in open(os.path.join(VAL, "annotations", SEQ + ".txt")):
        v = ln.strip().split(",")
        fr, tid, c = int(v[0]), int(v[1]), int(v[7])
        if fr > N_FRAMES:
            continue
        x, y, w, h = map(float, v[2:6])
        if c in RAW_TO_MODEL:
            gt[fr].append((x, y, x + w, y + h, tid, RAW_TO_MODEL[c]))
        elif c in (2, 3, 7, 8, 10):                   # look-alikes -> pretend 10-class head id 7
            gt[fr].append((x, y, x + w, y + h, -1, 7))
    return gt


def frame_inputs(gt, fr, frame):
    h, w = frame.shape[:2]
    scale = min(TEST_SIZE[0] / float(h), TEST_SIZE[1] / float(w))
    rows = [[x0 * scale, y0 * scale, x1 * scale, y1 * scale, 0.95, 0.9, c] for x0, y0, x1, y1, _, c in gt[fr]]
    out = torch.tensor(rows, dtype=torch.float32) if rows else torch.zeros((0, 7))
    img_info = [frame, w, fr, 1, f"{SEQ}/{fr:07d}.jpg"]  # mot_evaluator: [raw_img, width, frame_id, video_id, file]
    return out, img_info


class StubEmb:
    """Deterministic cheap appearance: 8x4 colour thumbnail, padded to 512."""
    feat_dim = 512

    def __init__(self):
        self.calls = 0

    def __call__(self, frame, tlbr):
        self.calls += 1
        tlbr = np.asarray(tlbr).reshape(-1, 4)
        f = np.zeros((len(tlbr), 512), np.float32)
        H, W = frame.shape[:2]
        for i, (x0, y0, x1, y1) in enumerate(tlbr):
            c = frame[max(0, int(y0)):min(H, int(y1)), max(0, int(x0)):min(W, int(x1))]
            if c.size:
                f[i, :96] = cv2.resize(c, (4, 8)).reshape(-1)
        return f


def run_adapter(name, make, gt, frames):
    print(f"\n[{name}]")
    trk = make()
    check(hasattr(trk, "print_diag_summary"), "has print_diag_summary")
    id_map = defaultdict(set)
    n_out = n_match = 0
    on_lookalike = bad_cls = 0
    for fr in range(1, N_FRAMES + 1):
        out, info = frame_inputs(gt, fr, frames[fr])
        res = trk.update(out.cuda() if torch.cuda.is_available() and False else out, info, TEST_SIZE)
        g = np.array([r[:4] for r in gt[fr]]) if gt[fr] else np.zeros((0, 4))
        gid = [r[4] for r in gt[fr]]
        gcls = [r[5] for r in gt[fr]]
        for t in res:
            ok_contract = (np.asarray(t.tlwh).shape == (4,) and isinstance(t.track_id, int)
                           and isinstance(t.score, float) and t.cls in (-1, 0, 1, 2, 3, 4))
            if not ok_contract:
                check(False, f"contract violated at frame {fr}: {t.tlwh} {t.track_id!r} {t.score!r} {t.cls!r}")
                return
            if t.cls > 4:
                bad_cls += 1
            tlbr = np.array([t.tlwh[0], t.tlwh[1], t.tlwh[0] + t.tlwh[2], t.tlwh[1] + t.tlwh[3]])
            n_out += 1
            if len(g):
                iou = A._iou_matrix(tlbr[None], g)[0]
                j = int(iou.argmax())
                if iou[j] >= 0.5:
                    if gcls[j] == 7:
                        on_lookalike += 1
                    else:
                        n_match += 1
                        if fr >= 4:
                            id_map[t.track_id].add(gid[j])
    check(n_out > 0, f"produced outputs ({n_out} boxes over {N_FRAMES} frames)")
    check(bad_cls == 0, "no output class > 4")
    check(on_lookalike == 0, f"no output on a class-7 (filtered) row ({on_lookalike})")
    check(n_match >= 0.9 * max(1, n_out - 0), f"outputs in original coords: {n_match}/{n_out} match GT at IoU>=0.5")
    stable = sum(1 for v in id_map.values() if len(v) == 1)
    check(len(id_map) > 0 and stable / len(id_map) >= 0.95,
          f"identity stable frames 4..{N_FRAMES}: {stable}/{len(id_map)} track ids map to one GT id")
    return trk


def main():
    gt = load_gt()
    frames = {fr: cv2.imread(os.path.join(VAL, "sequences", SEQ, f"{fr:07d}.jpg")) for fr in range(1, N_FRAMES + 1)}
    assert all(f is not None for f in frames.values())
    args = SimpleNamespace(track_thresh=0.6, track_buffer=30, match_thresh=0.9, mot20=False)
    print(f"GT rows frame 1: {len(gt[1])} ({sum(1 for r in gt[1] if r[5] == 7)} look-alike rows with class id 7)")

    run_adapter("ocsort", lambda: A.OCSortAdapter(args), gt, frames)
    e1 = StubEmb()
    run_adapter("botsort", lambda: A.BoTSORTAdapter(args, embedder=e1), gt, frames)
    check(e1.calls >= N_FRAMES, f"botsort called the injected embedder ({e1.calls} calls)")
    e2 = StubEmb()
    run_adapter("deepocsort", lambda: A.DeepOCSORTAdapter(args, embedder=e2), gt, frames)
    check(e2.calls >= N_FRAMES - 1, f"deepocsort called the injected embedder ({e2.calls} calls)")

    # (F) SharedOSNet crop parity with BYTETracker._extract_features_osnet
    w = os.environ.get("DARE_REID_WEIGHTS", "")
    if os.path.isfile(w):
        print("\n[SharedOSNet parity vs BYTETracker._extract_features_osnet]")
        os.environ.setdefault("DARE_REID", "osnet")
        os.environ.setdefault("DARE_P1_REID_DEVICE", "cpu")
        from yolox.tracker.byte_tracker import BYTETracker, STrack
        emb = A.SharedOSNet()
        out, info = frame_inputs(gt, 1, frames[1])
        dets, frame, h, wd, scores, classes = A._prep(out, info, TEST_SIZE)
        tlbr = dets[:12, :4]
        mine = emb(frame, tlbr)
        fake = SimpleNamespace(crop_shrink=emb.crop_shrink, feat_dim=512, reid_extractor=emb.extractor)
        stracks = [STrack(STrack.tlbr_to_tlwh(b), 0.9) for b in tlbr]
        BYTETracker._extract_features_osnet(fake, stracks, frame)
        ref = np.stack([s.curr_feat for s in stracks])
        check(np.array_equal(mine, ref), f"bitwise equal on {len(tlbr)} real crops (max |diff| {np.abs(mine - ref).max():.3g})")
    else:
        print("\n[SharedOSNet parity] skipped (DARE_REID_WEIGHTS not set to a file)")

    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAIL(S)'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
