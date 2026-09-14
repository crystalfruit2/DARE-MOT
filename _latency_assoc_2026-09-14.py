"""Per-frame ASSOCIATION latency for the DARE arm and the static-EMA control.

Why this script exists
----------------------
The report claims "8.40 ms/frame against a static-EMA control at 8.17 ms/frame, a marginal
cost of 0.23 ms/frame (~2.8%)". The 8.40 figure has a record; the static-EMA control and
therefore the 0.23 ms marginal cost do NOT appear in any logged run (audit, 2026-09-14).
This script produces both arms under one protocol so the claim has a basis.

What is timed, and why it is not simply update()
------------------------------------------------
The report's quantity is "candidate matching, the Kalman predict/update step and the
memory-gate computation together" -- it does NOT include appearance extraction.
BYTETracker.update() DOES include it: _extract_features_osnet() crops every detection out of
the raw frame and runs OSNet on the GPU from inside update(). Timing update() wholesale would
therefore over-count by the entire embedding cost. So both are timed and subtracted:

    t_assoc = t_update - t_embed

t_embed is measured around the same call the tracker makes, so the subtraction is exact
rather than modelled. Both arms are instrumented identically, so whatever residual overhead
the instrumentation itself adds is common-mode and cancels in the marginal cost.

Known limitation, stated up front
---------------------------------
Smart App Control blocks cython_bbox on this machine, so association IoU runs through the
NumPy fallback, which is slower than the compiled path. The ABSOLUTE ms/frame reported here
is therefore an upper bound for the shipped pipeline. The MARGINAL cost is unaffected: both
arms traverse the identical IoU path, so it cancels in the difference. Re-measure the
absolute where cython_bbox is available before quoting it as the pipeline's latency.

Usage (wraps tools/track.py; every flag after -- is passed through unchanged):
  python _latency_assoc_2026-09-14.py --tag dare -- -f <exp> -c <ckpt> ... -expn lat_dare
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
SCRATCH = osp.join(HERE, "_scratch")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="arm name, used for the output filenames")
    ap.add_argument("--warmup", type=int, default=50,
                    help="leading frames discarded (CUDA context, cuDNN autotune, page cache)")
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    passthrough = args.rest[1:] if args.rest and args.rest[0] == "--" else args.rest

    os.makedirs(SCRATCH, exist_ok=True)

    import torch
    from yolox.tracker import byte_tracker as bt

    t_update, t_embed = [], []
    _orig_update = bt.BYTETracker.update
    _orig_embed = bt.BYTETracker._extract_features_osnet
    embed_acc = {"v": 0.0}

    def timed_embed(self, detections, raw_frame):
        # CUDA is asynchronous: without a sync the embedding cost silently lands in the
        # association slice instead of its own.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = perf_counter()
        r = _orig_embed(self, detections, raw_frame)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        embed_acc["v"] += perf_counter() - t0
        return r

    def timed_update(self, output_results, img_info, img_size):
        embed_acc["v"] = 0.0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = perf_counter()
        r = _orig_update(self, output_results, img_info, img_size)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_update.append(perf_counter() - t0)
        t_embed.append(embed_acc["v"])
        return r

    bt.BYTETracker.update = timed_update
    bt.BYTETracker._extract_features_osnet = timed_embed

    sys.argv = [osp.join(HERE, "tools", "track.py")] + passthrough
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    except SystemExit:
        pass

    n = len(t_update)
    if n <= args.warmup:
        print("LATENCY: too few frames (%d) for warmup %d" % (n, args.warmup))
        return
    upd = t_update[args.warmup:]
    emb = t_embed[args.warmup:]
    assoc = [u - e for u, e in zip(upd, emb)]
    ms = lambda xs: [x * 1000.0 for x in xs]

    def stats(xs):
        xs = sorted(ms(xs))
        return {
            "mean": statistics.fmean(xs),
            "median": statistics.median(xs),
            "p95": xs[int(0.95 * (len(xs) - 1))],
            "n": len(xs),
        }

    out = {
        "tag": args.tag,
        "frames_total": n,
        "warmup_discarded": args.warmup,
        "update_ms": stats(upd),
        "embed_ms": stats(emb),
        "assoc_ms": stats(assoc),
        "iou_path": "numpy-fallback" if bt.__dict__.get("_dummy") is None else "unknown",
        "env": {k: v for k, v in os.environ.items() if k.startswith("DARE_")},
    }
    # record which IoU implementation actually ran -- the absolute number depends on it
    try:
        from yolox.tracker.matching import _cython_bbox_ious
        out["iou_path"] = "cython_bbox" if (
            _cython_bbox_ious is not None
            and os.environ.get("DARE_BBOX_IOU", "cython") != "numpy"
        ) else "numpy-fallback"
    except Exception as e:  # pragma: no cover
        out["iou_path"] = "unknown (%s)" % e

    jpath = osp.join(SCRATCH, "latency_%s.json" % args.tag)
    with open(jpath, "w") as f:
        json.dump(out, f, indent=2)
    with open(osp.join(SCRATCH, "latency_%s.csv" % args.tag), "w") as f:
        f.write("frame,update_ms,embed_ms,assoc_ms\n")
        for i, (u, e, a) in enumerate(zip(upd, emb, assoc)):
            f.write("%d,%.6f,%.6f,%.6f\n" % (i, u * 1000, e * 1000, a * 1000))

    print("\n=== LATENCY [%s] ===" % args.tag)
    print("frames %d (first %d discarded), IoU path: %s" % (n, args.warmup, out["iou_path"]))
    for k in ("update_ms", "embed_ms", "assoc_ms"):
        s = out[k]
        print("  %-10s mean %7.3f  median %7.3f  p95 %7.3f ms" % (k, s["mean"], s["median"], s["p95"]))
    print("wrote %s" % jpath)


if __name__ == "__main__":
    main()
