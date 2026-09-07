"""FIX for the CMC state-parameterisation bug + a numerical check of its size.

WHERE THIS GOES: `yolox/tracker/byte_tracker.py`, `STrack.multi_gmc`, on branch
`exp/cmc-move1` (main has no GMC). Not applied in place because the GMC code lives on
another branch and this working tree is dirty with the R2-1/R3-3 diagnostics; drop the
`multi_gmc_fixed` body over the existing one when that branch is checked out.

THE BUG. The current implementation is:

    R = H[:2, :2]                      # = s*Rot(theta), from estimateAffinePartial2D
    R8x8 = np.kron(np.eye(4), R)
    mean = R8x8.dot(mean); mean[:2] += t
    cov  = R8x8.dot(cov).dot(R8x8.T)

and its own docstring names the state "the shared **xyah** state [x,y,a,h,vx,vy,va,vh]",
calling this "standard BoT-SORT recipe". It is not. BoT-SORT replaced (a,h) with (w,h)
precisely so that kron(I4, sR) is dimensionally coherent -- both components are pixel
lengths. Applied to xyah, the same 2x2 block rotates a DIMENSIONLESS aspect ratio into a
PIXEL height:

    new_a = s*cos(th)*a - s*sin(th)*h   ->  da/a ~= (s-1) - th*h/a = (s-1) - th*h^2/w
    new_h = s*sin(th)*a + s*cos(th)*h   ->  dh/h ~= (s-1) + th*a/h   (negligible, a/h ~ 0.01)

Position and height stay correct; the whole error lands on WIDTH and scales with h^2/w --
worst for tall, thin, large-in-pixels targets, i.e. low-altitude aerial pedestrians.

THE FIX. Convert xyah -> xywh, apply the warp exactly as BoT-SORT does, convert back. This
reproduces the prior art bit-for-bit, which is what makes the resulting ablation
interpretable: any remaining CMC damage is then a property of the method, not of our port.
Velocity conversion is the chain rule (w = a*h  =>  vw = va*h + a*vh) and the covariance is
mapped through the corresponding Jacobians rather than copied, since the map is nonlinear.

A note on the alternative: for a similarity warp acting on an AXIS-ALIGNED box, the
physically exact shape update is "scale w and h by s, leave the aspect ratio alone" --
rotation of an axis-aligned extent is not representable without growing the box. That is
arguably more correct than BoT-SORT itself, but it is NOT what the prior art does, so it is
provided here as `multi_gmc_scale_only` for a second ablation arm rather than as the
default. Measure both; report the BoT-SORT-parity arm as the comparison.
"""
import numpy as np


# --------------------------------------------------------------------------------------
# Drop-in replacement for STrack.multi_gmc  (BoT-SORT parity)
# --------------------------------------------------------------------------------------
def multi_gmc_fixed(stracks, H=np.eye(2, 3)):
    """Camera Motion Compensation. Warp each track's predicted KF state by the inter-frame
    affine H (prev->curr).

    The KF state is xyah = [x, y, a, h, vx, vy, va, vh] with a = w/h dimensionless. The
    BoT-SORT warp recipe is defined on xywh, so the state is converted to xywh, warped, and
    converted back. Applying kron(I4, sR) directly to xyah mixes a dimensionless ratio with
    a pixel length and corrupts predicted width by theta*h^2/w.
    """
    if len(stracks) == 0:
        return
    R = H[:2, :2]
    R8x8 = np.kron(np.eye(4, dtype=float), R)
    t = H[:2, 2]

    for st in stracks:
        mean = st.mean.copy()
        cov = st.covariance.copy()

        x, y, a, h, vx, vy, va, vh = mean

        # ---- xyah -> xywh (mean) ----
        w = a * h
        vw = va * h + a * vh
        m_xywh = np.array([x, y, w, h, vx, vy, vw, vh], dtype=float)

        # ---- xyah -> xywh (covariance), via the Jacobian J = d(xywh)/d(xyah) ----
        J = np.eye(8)
        J[2, 2] = h        # dw/da
        J[2, 3] = a        # dw/dh
        J[6, 2] = vh       # dvw/da
        J[6, 3] = va       # dvw/dh
        J[6, 6] = h        # dvw/dva
        J[6, 7] = a        # dvw/dvh
        cov_xywh = J @ cov @ J.T

        # ---- warp, exactly as BoT-SORT ----
        m_xywh = R8x8.dot(m_xywh)
        m_xywh[:2] += t
        cov_xywh = R8x8 @ cov_xywh @ R8x8.T

        # ---- xywh -> xyah (mean) ----
        xw, yw, ww, hw, vxw, vyw, vww, vhw = m_xywh
        hw = hw if abs(hw) > 1e-6 else (1e-6 if hw >= 0 else -1e-6)
        a2 = ww / hw
        va2 = (vww * hw - ww * vhw) / (hw * hw)
        st.mean = np.array([xw, yw, a2, hw, vxw, vyw, va2, vhw], dtype=float)

        # ---- xywh -> xyah (covariance), via K = d(xyah)/d(xywh) ----
        K = np.eye(8)
        K[2, 2] = 1.0 / hw                     # da/dw
        K[2, 3] = -ww / (hw * hw)              # da/dh
        K[6, 2] = -vhw / (hw * hw)             # dva/dw
        K[6, 3] = (-vww * hw + 2.0 * ww * vhw) / (hw ** 3)   # dva/dh
        K[6, 6] = 1.0 / hw                     # dva/dvw
        K[6, 7] = -ww / (hw * hw)              # dva/dvh
        st.covariance = K @ cov_xywh @ K.T


# --------------------------------------------------------------------------------------
# Second ablation arm: exact similarity update for an axis-aligned extent
# --------------------------------------------------------------------------------------
def multi_gmc_scale_only(stracks, H=np.eye(2, 3)):
    """Warp the centre with the full affine; scale h by the similarity scale s; leave the
    aspect ratio untouched. Physically exact for an axis-aligned box under a similarity,
    but NOT what BoT-SORT does -- use as a second arm, not as the comparison baseline."""
    if len(stracks) == 0:
        return
    R = H[:2, :2]
    t = H[:2, 2]
    s = float(np.sqrt(abs(np.linalg.det(R))))
    for st in stracks:
        m = st.mean.copy()
        m[:2] = R.dot(m[:2]) + t
        m[4:6] = R.dot(m[4:6])
        m[3] *= s          # h
        m[7] *= s          # vh
        # a and va are scale-invariant, so they are left alone
        st.mean = m
        S = np.diag([1., 1., 1., s, 1., 1., 1., s])
        A = np.eye(8)
        A[:2, :2] = R
        A[4:6, 4:6] = R
        st.covariance = (S @ A) @ st.covariance @ (S @ A).T


# --------------------------------------------------------------------------------------
# Numerical check: how big is the bug, on real val7 affines and real box geometry?
# --------------------------------------------------------------------------------------
class _T:
    def __init__(self, mean):
        self.mean = np.asarray(mean, float)
        self.covariance = np.eye(8) * 1e-3


def _buggy(stracks, H):
    R = H[:2, :2]
    R8x8 = np.kron(np.eye(4, dtype=float), R)
    t = H[:2, 2]
    for st in stracks:
        m = R8x8.dot(st.mean)
        m[:2] += t
        st.mean = m


def main():
    import os.path as osp
    import pandas as pd

    rng = np.random.default_rng(0)
    print("box: median val7 pedestrian, w=38 h=82 (a=0.463); one frame of warp\n")
    print(f"{'sequence':<20}{'|theta|':>10}{'s':>9}   "
          f"{'buggy dW/W':>12}{'fixed dW/W':>12}{'scale dW/W':>12}")
    for seq in ["uav0000086_00000_v", "uav0000117_02622_v", "uav0000137_00458_v",
                "uav0000182_00000_v", "uav0000305_00000_v", "uav0000339_00001_v"]:
        p = osp.join("_scratch/_r33_affines", seq + ".csv")
        if not osp.exists(p):
            print(f"{seq:<20}  (affines not found)")
            continue
        d = pd.read_csv(p)
        A = d[["a", "b", "c", "d"]].to_numpy().reshape(-1, 2, 2)
        th = np.arctan2(d.c.to_numpy(), d.a.to_numpy())
        sc = np.sqrt(np.abs(np.linalg.det(A)))
        i = int(np.argsort(np.abs(th))[len(th) // 2])          # the median-|theta| frame
        H = np.array([[d.a[i], d.b[i], d.tx[i]], [d.c[i], d.d[i], d.ty[i]]], float)

        w0, h0 = 38.0, 82.0
        a0 = w0 / h0
        base = [960., 540., a0, h0, 1., 1., 0., 0.]
        outs = {}
        for name, fn in (("buggy", _buggy), ("fixed", multi_gmc_fixed),
                         ("scale", multi_gmc_scale_only)):
            tr = _T(base)
            fn([tr], H)
            outs[name] = tr.mean[2] * tr.mean[3]               # w' = a'*h'
        print(f"{seq:<20}{abs(th[i]):>10.2e}{sc[i]:>9.5f}   "
              + "".join(f"{100*(outs[k]-w0)/w0:>11.2f}%" for k in ("buggy", "fixed", "scale")))

    print("\nA one-frame width error compounds every frame a track coasts.")
    print("The `fixed` column is BoT-SORT parity; `scale` is the exact axis-aligned update.")


if __name__ == "__main__":
    main()
