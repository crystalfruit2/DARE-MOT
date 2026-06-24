# Shared MOT metric script — use the SAME one for DARE-MOT and the ByteTrack baseline.
# Usage: python eval_mot.py <results.txt> <gt.txt> [name]
import sys
import motmetrics as mm

res_path = sys.argv[1]
gt_path  = sys.argv[2]
name     = sys.argv[3] if len(sys.argv) > 3 else "seq"

gt = mm.io.loadtxt(gt_path,  fmt="mot15-2D", min_confidence=1)
ts = mm.io.loadtxt(res_path, fmt="mot15-2D", min_confidence=-1)
acc = mm.utils.compare_to_groundtruth(gt, ts, "iou", distth=0.5)

mh = mm.metrics.create()
summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name=name)
print(mm.io.render_summary(summary, formatters=mh.formatters,
                           namemap=mm.io.motchallenge_metric_names))
