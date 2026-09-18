"""Unit check for the corrected STrack.multi_gmc port (2026-09-10). CPU only, no solver."""
import importlib.util
import numpy as np
from yolox.tracker.byte_tracker import STrack

spec = importlib.util.spec_from_file_location(
    "ref", r"C:\Users\User\Desktop\projects\DARE-MOT\_multi_gmc_fix.py")
ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)


class T:
    def __init__(self, mean, cov=None):
        self.mean = np.asarray(mean, float)
        self.covariance = np.eye(len(mean)) * 1e-2 if cov is None else cov


th, s = 1.65e-3, 0.9995                           # uav0000086-like median frame
R = s * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
H = np.hstack([R, [[1.7], [-0.6]]])
cv = [960., 540., 38 / 82, 82., 1.2, -0.4, 0.001, 0.3]
ca = cv + [0.05, -0.02, 1e-4, 0.01]
ok = True

# 1. identity warp is a no-op in every mode, both state sizes
for mode in ("parity", "scale", "bug"):
    for m0 in (cv, ca):
        t = T(m0); c0 = t.covariance.copy()
        STrack.multi_gmc([t], np.eye(2, 3), mode)
        good = np.allclose(t.mean, m0, atol=1e-9) and np.allclose(t.covariance, c0, atol=1e-8)
        ok &= good
        print(f"identity  {mode:6s} d={len(m0):2d}  {'OK' if good else 'FAIL'}")

# 2. CV parity == the verified reference implementation (analytic Jacobians)
a = T(cv); b = T(cv)
STrack.multi_gmc([a], H, "parity"); ref.multi_gmc_fixed([b], H)
good = np.allclose(a.mean, b.mean, rtol=1e-10, atol=1e-10) and np.allclose(a.covariance, b.covariance, rtol=1e-5, atol=1e-9)
ok &= good
print(f"CV parity vs reference: mean maxdiff {np.abs(a.mean-b.mean).max():.2e}, "
      f"cov maxdiff {np.abs(a.covariance-b.covariance).max():.2e}  {'OK' if good else 'FAIL'}")

# 3. width error per mode (the bug should reproduce ~ -theta*h^2/w)
for mode in ("bug", "parity", "scale"):
    t = T(cv); STrack.multi_gmc([t], H, mode)
    w = t.mean[2] * t.mean[3]
    print(f"width change {mode:6s}: {100*(w-38)/38:+.3f}%   (pure scale would be {100*(s-1):+.3f}%)")

# 4. CA: xyah<->xywh round trip is exact, and parity keeps position/height consistent with CV
z = STrack._xywh_to_xyah(STrack._xyah_to_xywh(np.array(ca)))
good = np.allclose(z, ca, rtol=1e-12, atol=1e-12); ok &= good
print(f"CA round trip  {'OK' if good else 'FAIL'}")
tc, tv = T(ca), T(cv)
STrack.multi_gmc([tc], H, "parity"); STrack.multi_gmc([tv], H, "parity")
good = np.allclose(tc.mean[:8], tv.mean[:8], rtol=1e-9, atol=1e-9)
ok &= good
print(f"CA parity first 8 dims == CV parity (accel terms don't leak into lower orders)  {'OK' if good else 'FAIL'}")
print(f"CA cov symmetric PSD: sym={np.allclose(tc.covariance, tc.covariance.T)}, "
      f"min eig={np.linalg.eigvalsh(tc.covariance).min():.2e}")
print("ALL OK" if ok else "SOME FAILED")
