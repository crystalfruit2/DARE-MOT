"""Build the initial checkpoint for the 10-class detector (2026-09-10) and verify it on CPU.

- 5-class source (yolox_x_visdrone_mc): head.cls_preds.* grow from 5 to 10 rows. Rows 0..4 are COPIED
  (the class order is preserved, see tools/convert_visdrone_10c.py), rows 5..9 get YOLOX's own init
  (weight N(0, 0.01), bias = -log((1-p)/p), p = 0.01). Everything else is copied as is.
- any other source (e.g. bytetrack_x_mot17, 1 class): cls_preds are DROPPED so they reinitialise;
  backbone, neck, reg/obj heads are kept.
Check: builds the 10-class model the way tools/train.py does (load_ckpt), asserts nothing was skipped,
and -- for a 5-class source -- runs one val image through both models and asserts boxes, objectness and
class channels 0..4 agree.
Usage: python _make_10c_init.py <src ckpt> <out ckpt>
"""
import math
import sys
import torch
from yolox.exp import get_exp
from yolox.utils import load_ckpt

NEW = 10
src_path, out_path = sys.argv[1], sys.argv[2]
sd = torch.load(src_path, map_location="cpu")["model"]
cls_keys = [k for k in sd if k.startswith("head.cls_preds.")]
n_src = sd[cls_keys[0]].shape[0]
print(f"source {src_path}: {n_src} classes, {len(sd)} tensors")

out = dict(sd)
if n_src == 5:
    g = torch.Generator().manual_seed(0)
    prior = -math.log((1 - 0.01) / 0.01)
    for k in cls_keys:
        t = sd[k]
        if k.endswith("weight"):
            new = torch.empty((NEW,) + tuple(t.shape[1:]), dtype=t.dtype).normal_(0, 0.01, generator=g)
        else:
            new = torch.full((NEW,), prior, dtype=t.dtype)
        new[:5] = t
        out[k] = new
    print("cls_preds grown 5 -> 10 (rows 0..4 copied)")
else:
    for k in cls_keys:
        del out[k]
    print(f"cls_preds dropped ({n_src}-class source) -> reinitialised by the model")
torch.save({"model": out, "source": src_path}, out_path)
print(f"saved {out_path}")

# ---- verification, exactly the trainer's load path ----
exp10 = get_exp("exps/example/mot/yolox_x_visdrone_10c.py", None)
m10 = exp10.get_model().eval()
own = m10.state_dict()
missing = [k for k in own if k not in out]
shape_bad = [k for k in own if k in out and own[k].shape != out[k].shape]
m10 = load_ckpt(m10, out)
print(f"10-class model: {len(own)} tensors, missing from init {len(missing)}, shape mismatch {len(shape_bad)}")
if n_src == 5:
    assert not missing and not shape_bad, (missing[:5], shape_bad[:5])
else:
    assert all(k.startswith("head.cls_preds.") for k in missing) and not shape_bad, (missing[:5], shape_bad[:5])

if n_src == 5:
    exp5 = get_exp("exps/example/mot/yolox_x_visdrone_mc_val7.py", None)
    m5 = exp5.get_model().eval()
    m5.load_state_dict(sd)
    exp10.data_num_workers = 0
    img = next(iter(exp10.get_eval_loader(batch_size=1, is_distributed=False)))[0]
    with torch.no_grad():
        o5, o10 = m5(img), m10(img)          # [1, anchors, 5 + num_classes]
    d_box = (o5[..., :5] - o10[..., :5]).abs().max().item()
    d_cls = (o5[..., 5:10] - o10[..., 5:10]).abs().max().item()
    new_mean = o10[..., 10:].mean().item()
    print(f"one val image: max |diff| box+obj {d_box:.2e}, classes 0..4 {d_cls:.2e}; "
          f"new classes 5..9 mean prob {new_mean:.4f} (prior 0.01)")
    assert d_box < 1e-4 and d_cls < 1e-4
print("OK")
