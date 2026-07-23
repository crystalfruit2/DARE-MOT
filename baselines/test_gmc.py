"""CPU unit test for baselines/gmc.py — no GPU, no detector needed.

Two checks per estimator:
  (A) KNOWN-WARP RECOVERY (the real correctness test): warp a real val7 frame by a known
      affine (translation, then translation+rotation) and confirm GMC recovers it within
      tolerance. This proves GMC estimates the *right* transform, not just that it runs.
  (B) REAL CONSECUTIVE FRAMES (sanity): feed two adjacent val7 frames; frame 1 must return
      exact identity, frame 2 must return a well-formed 2x3 affine with ~unit scale and a
      finite, plausible translation (camera drift is small frame-to-frame at 30 fps).

Run:  conda run -n dare_mot python baselines/test_gmc.py
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gmc import GMC  # noqa: E402

VAL = r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone2019-MOT-val\sequences"
SEQ = "uav0000086_00000_v"  # 1344x756, real drone camera motion


def _load(seq, idx):
    p = os.path.join(VAL, seq, f"{idx:07d}.jpg")
    img = cv2.imread(p)
    if img is None:
        raise FileNotFoundError(p)
    return img


def _translation(H):
    return float(H[0, 2]), float(H[1, 2])


def _scale_rot(H):
    """Recover (scale, rotation_deg) from the 2x2 part of a similarity affine."""
    a, b = H[0, 0], H[0, 1]
    s = float(np.hypot(a, b))
    ang = float(np.degrees(np.arctan2(-b, a)))
    return s, ang


PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, ok, detail=""):
    results.append((name, ok))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f"  ({detail})" if detail else ""))


def test_known_warp(method):
    """(A) Recover a synthetic affine applied to a real frame. downscale=1 for exactness."""
    print(f"\n== (A) known-warp recovery :: method={method} ==")
    base = _load(SEQ, 1)
    h, w = base.shape[:2]

    # A.1 pure translation (+24, -16)px
    tx, ty = 24.0, -16.0
    M = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
    warped = cv2.warpAffine(base, M, (w, h))
    g = GMC(method=method, downscale=1)
    g.apply(base)                 # frame 1: initializes, returns identity
    H = g.apply(warped)           # frame 2: should recover (tx, ty)
    rtx, rty = _translation(H)
    # GMC estimates prev->curr background motion; content shifted by (tx,ty) => recover ~(tx,ty)
    tol = 3.0 if method != "ecc" else 6.0
    ok = abs(rtx - tx) < tol and abs(rty - ty) < tol
    check(f"{method}: translation recovered", ok,
          f"want~({tx:+.0f},{ty:+.0f}) got=({rtx:+.1f},{rty:+.1f}) tol={tol}")

    # A.2 translation + small rotation (2 deg about center)
    ang = 2.0
    Mrot = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    Mrot[0, 2] += 10.0
    Mrot[1, 2] += 6.0
    warped2 = cv2.warpAffine(base, Mrot, (w, h))
    g2 = GMC(method=method, downscale=1)
    g2.apply(base)
    H2 = g2.apply(warped2)
    s, rot = _scale_rot(H2)
    # cv2.getRotationMatrix2D(angle=+2) rotates counter-clockwise; recovered sign may flip by convention
    ok_s = 0.95 < s < 1.05
    ok_rot = abs(abs(rot) - ang) < 1.5
    check(f"{method}: scale ~1 under rotation", ok_s, f"scale={s:.3f}")
    check(f"{method}: rotation magnitude recovered", ok_rot, f"want~{ang} got={rot:+.2f}deg")


def test_real_frames(method):
    """(B) Sanity on two consecutive real val7 frames."""
    print(f"\n== (B) real consecutive frames :: method={method} ==")
    f1, f2 = _load(SEQ, 1), _load(SEQ, 2)
    g = GMC(method=method, downscale=2)
    H1 = g.apply(f1)
    ok_id = np.allclose(H1, np.eye(2, 3))
    check(f"{method}: frame 1 returns identity", ok_id)
    H2 = g.apply(f2)
    ok_shape = H2.shape == (2, 3)
    s, rot = _scale_rot(H2)
    tx, ty = _translation(H2)
    ok_scale = 0.9 < s < 1.1
    ok_trans = abs(tx) < 100 and abs(ty) < 100          # 30fps drone: small inter-frame drift
    ok_finite = np.all(np.isfinite(H2))
    check(f"{method}: well-formed 2x3 affine", ok_shape and ok_finite)
    check(f"{method}: plausible inter-frame motion", ok_scale and ok_trans,
          f"scale={s:.3f} rot={rot:+.2f}deg trans=({tx:+.1f},{ty:+.1f})")


def test_none():
    print("\n== (C) method='none' is a no-op identity ==")
    g = GMC(method="none")
    check("none: returns identity", np.allclose(g.apply(_load(SEQ, 1)), np.eye(2, 3)))


if __name__ == "__main__":
    print(f"GMC unit test | cv2 {cv2.__version__} | numpy {np.__version__}")
    print(f"sequence: {SEQ}")
    for m in ("sparseOptFlow", "orb", "ecc"):
        try:
            test_known_warp(m)
            test_real_frames(m)
        except Exception as e:
            check(f"{m}: raised {type(e).__name__}", False, str(e)[:120])
    test_none()

    n_pass = sum(1 for _, ok in results if ok)
    n = len(results)
    print(f"\n=== {n_pass}/{n} checks passed ===")
    sys.exit(0 if n_pass == n else 1)
