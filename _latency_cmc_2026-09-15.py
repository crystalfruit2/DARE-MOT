"""Per-frame tracker latency WITH camera motion compensation, split into its parts (2026-09-15).

Why this script exists
----------------------
Paper limitation 4: "The per-frame cost of camera motion compensation has not been measured, and the
8.40 ms/frame figure was taken with compensation off." The 09-14 run (_latency_assoc_2026-09-14.py,
main worktree) timed association with CMC off. CMC is part of the baseline in both arms now, and it only
exists in the exp/cmc-fixed worktree, so this is the same instrument, run from here, on the paper's
provenance-clean detector (D3, 10-class, DARE_MAX_CLASS=4).

What is timed
-------------
BYTETracker.update() contains, per frame:
  t_embed   OSNet crop + forward (_extract_features_osnet)        -- GPU, synced
  t_gmc     GMC.apply(raw_frame): global transform from image content (sparse optical flow) -- CPU
  t_warp    STrack.multi_gmc(): applying that transform to track states/covariances       -- CPU
  t_assoc   everything else: KF predict/update, cost build, IoU gate, memory gate, LAP
            = t_update - t_embed - t_gmc - t_warp
Each part is measured around the exact call the tracker makes, so the subtraction is exact. Monkeypatching
changes timing only, never decisions: the run script SHA-256 compares track_results against the cached
09-11 eval runs to prove the timed configuration is the reported one.

Usage (wraps tools/track.py; every flag after -- is passed through unchanged):
  python _latency_cmc_2026-09-15.py --tag <arm> -- -f <exp> -c <ckpt> ... -expn <run>
Outputs: <main>/_scratch/latency_cmc/latency_<tag>.json + .csv
"""
import argparse
import json
import os
import os.path as osp
import runpy
import statistics
import sys
from time import perf_counter

HERE = osp.dirname(osp.abspath(__file__))
OUT = r"C:\Users\User\Desktop\projects\DARE-MOT\_scratch\latency_cmc"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    passthrough = args.rest[1:] if args.rest and args.rest[0] == "--" else args.rest
    os.makedirs(OUT, exist_ok=True)

    import torch
    from yolox.tracker import byte_tracker as bt
    import baselines.gmc as gmc_mod

    cuda = torch.cuda.is_available()
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    acc = {"embed": 0.0, "gmc": 0.0, "warp": 0.0}
    rows = []                                   # (update, embed, gmc, warp, n_dets_high)

    _orig_update = bt.BYTETracker.update
    # v2 (09-15 14:00): time the DISPATCH (_extract_features), not only the OSNet branch. With DARE_REID unset
    # (ByteTrack arms, lambda = 0) update() still runs the legacy MobileNetV2 per-crop extractor, and v1
    # silently charged that ~360 ms/frame to association. Patching only the dispatch avoids double counting.
    _orig_embed = bt.BYTETracker._extract_features
    _orig_apply = gmc_mod.GMC.apply
    _orig_warp = bt.STrack.multi_gmc            # staticmethod -> plain function on attribute access

    def timed_embed(self, detections, raw_frame):
        sync(); t0 = perf_counter()
        r = _orig_embed(self, detections, raw_frame)
        sync(); acc["embed"] += perf_counter() - t0
        return r

    def timed_apply(self, raw_frame, detections=None):
        t0 = perf_counter()
        r = _orig_apply(self, raw_frame, detections)
        acc["gmc"] += perf_counter() - t0
        return r

    def timed_warp(stracks, H=None, mode="parity"):
        t0 = perf_counter()
        r = _orig_warp(stracks, H, mode) if H is not None else _orig_warp(stracks, mode=mode)
        acc["warp"] += perf_counter() - t0
        return r

    def timed_update(self, output_results, img_info, img_size):
        for k in acc:
            acc[k] = 0.0
        sync(); t0 = perf_counter()
        r = _orig_update(self, output_results, img_info, img_size)
        sync(); dt = perf_counter() - t0
        rows.append((dt, acc["embed"], acc["gmc"], acc["warp"]))
        return r

    bt.BYTETracker.update = timed_update
    bt.BYTETracker._extract_features = timed_embed
    gmc_mod.GMC.apply = timed_apply
    bt.STrack.multi_gmc = staticmethod(timed_warp)

    sys.argv = [osp.join(HERE, "tools", "track.py")] + passthrough
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    except SystemExit:
        pass

    n = len(rows)
    if n <= args.warmup:
        print("LATENCY: too few frames (%d) for warmup %d" % (n, args.warmup))
        sys.exit(2)
    rows = rows[args.warmup:]
    cols = {
        "update_ms": [r[0] for r in rows],
        "embed_ms": [r[1] for r in rows],
        "gmc_ms": [r[2] for r in rows],
        "warp_ms": [r[3] for r in rows],
        "assoc_ms": [r[0] - r[1] - r[2] - r[3] for r in rows],
    }

    def stats(xs):
        xs = sorted(x * 1000.0 for x in xs)
        return {"mean": statistics.fmean(xs), "median": statistics.median(xs),
                "p95": xs[int(0.95 * (len(xs) - 1))], "n": len(xs)}

    from yolox.tracker import matching
    iou_path = "cython_bbox" if (getattr(matching, "_cython_bbox_ious", None) is not None
                                 and os.environ.get("DARE_BBOX_IOU", "cython") != "numpy") else \
        ("cython_bbox (no fallback in this tree)" if not hasattr(matching, "_cython_bbox_ious") else "numpy-fallback")
    out = {"tag": args.tag, "frames_total": n, "warmup_discarded": args.warmup,
           **{k: stats(v) for k, v in cols.items()},
           "iou_path": iou_path,
           "gpu": torch.cuda.get_device_name(0) if cuda else "cpu",
           "env": {k: v for k, v in os.environ.items() if k.startswith("DARE_")}}
    with open(osp.join(OUT, "latency_%s.json" % args.tag), "w") as f:
        json.dump(out, f, indent=2)
    with open(osp.join(OUT, "latency_%s.csv" % args.tag), "w") as f:
        f.write("frame," + ",".join(cols) + "\n")
        for i in range(len(rows)):
            f.write("%d," % i + ",".join("%.6f" % (cols[k][i] * 1000) for k in cols) + "\n")

    print("\n=== LATENCY+CMC [%s] === frames %d (first %d discarded), IoU: %s" % (args.tag, n, args.warmup, iou_path))
    for k in cols:
        s = out[k]
        print("  %-10s mean %8.3f  median %8.3f  p95 %8.3f ms" % (k, s["mean"], s["median"], s["p95"]))


if __name__ == "__main__":
    main()
