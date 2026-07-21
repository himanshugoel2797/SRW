"""Separable (rank-N_s) evaluator for the SRW undulator radiation integral.

Phase(s,xO,zO) = A(s) + xO^2 B(s) - 2 xO Cx(s) + zO^2 B(s) - 2 zO Cz(s)
  => exp(i Phase) = e^{iA} * Gx(s,xO) * Gz(s,zO)   [EXACTLY separable]
Amplitude Ax = P(s) + xO Q(s) (linear in xO only), Az = Pz(s) + zO Qz(s).

Hence   Ex[ix,iz] = sum_s Gx[s,ix] (w_s P_s e^{iA_s}) Gz[s,iz]
                  + xO[ix] * sum_s Gx[s,ix] (w_s Q_s e^{iA_s}) Gz[s,iz]
i.e. two complex GEMMs (N_x x N_s)(N_s x N_z) per polarisation.
"""
import sys, numpy as np
from srwpy import srwlib as sl
from srwpy import srwlpy as srwl

print("module:", srwl.__file__)

N     = int(sys.argv[1]) if len(sys.argv) > 1 else 64
NPER  = int(sys.argv[2]) if len(sys.argv) > 2 else 20
NS    = int(sys.argv[3]) if len(sys.argv) > 3 else 40001   # odd -> Simpson
TIGHT = 1e-4
PER, BPEAK, EPH, YOBS, HALF = 0.021, 0.88, 1.2e3, 20.0, 0.4e-3


def undulator():
    h = sl.SRWLMagFldH(1, 'v', BPEAK, 0, 1, 1)
    und = sl.SRWLMagFldU([h], PER, NPER)
    return sl.SRWLMagFldC([und], sl.array('d', [0]), sl.array('d', [0]), sl.array('d', [0]))


def make_wfr(n):
    wfr = sl.SRWLWfr()
    wfr.allocate(1, n, n)
    wfr.mesh.set_from_other(sl.SRWLRadMesh(
        _eStart=EPH, _eFin=EPH, _ne=1,
        _xStart=-HALF, _xFin=HALF, _nx=n,
        _yStart=-HALF, _yFin=HALF, _ny=n,
        _zStart=YOBS))
    eb = sl.SRWLPartBeam()
    eb.Iavg = 0.5
    eb.partStatMom1.z = -0.5 * PER * (NPER + 4)
    eb.partStatMom1.gamma = 3.0 / 0.51099890221e-03
    wfr.partBeam = eb
    return wfr


def srw_ref(relPrec, dev=None):
    wfr = make_wfr(N)
    prec = [1, relPrec, 0, 0, 50000, 1, 0]
    args = (wfr, 0, undulator(), prec) + ((dev,) if dev is not None else ())
    srwl.CalcElecFieldSR(*args)
    ex = np.frombuffer(wfr.arEx, dtype=np.float32).astype(np.float64)
    ez = np.frombuffer(wfr.arEy, dtype=np.float32).astype(np.float64)
    Ex = (ex[0::2] + 1j * ex[1::2]).reshape(N, N)   # [iz, ix]
    Ez = (ez[0::2] + 1j * ez[1::2]).reshape(N, N)
    return Ex, Ez


def trajectory(ns):
    """Uniform-s trajectory from SRW itself (so x(s),z(s),Btx,Btz match exactly)."""
    trj = sl.SRWLPrtTrj(_np=ns)
    trj.partInitCond.gamma = 3.0 / 0.51099890221e-03
    trj.partInitCond.z = -0.5 * PER * (NPER + 4)
    trj.ctStart = 0.0
    trj.ctEnd = PER * (NPER + 4)
    srwl.CalcPartTraj(trj, undulator(), [1])
    s = np.asarray(trj.arZ, dtype=np.float64)          # longitudinal position [m]
    x = np.asarray(trj.arX, dtype=np.float64)
    z = np.asarray(trj.arY, dtype=np.float64)          # SRW lab Y = internal z
    btx = np.asarray(trj.arXp, dtype=np.float64)
    btz = np.asarray(trj.arYp, dtype=np.float64)
    return s, x, z, btx, btz


def cumint_simpson(f, s):
    """cumulative integral of f ds on a (near-)uniform grid, trapezoid+correction."""
    d = np.diff(s)
    inc = 0.5 * (f[1:] + f[:-1]) * d
    out = np.empty_like(f)
    out[0] = 0.0
    np.cumsum(inc, out=out[1:])
    return out


def separable(N, s, x, z, btx, btz, gam):
    K = np.pi * 1e9 / (1.239841984e-6 / EPH * 1e9)     # pi*1e9/lambda[nm] = pi/lambda[m]
    g2 = 1.0 / (gam * gam)
    Ix = cumint_simpson(btx * btx, s)
    Iz = cumint_simpson(btz * btz, s)

    u = 1.0 / (YOBS - s)
    A  = K * (s * g2 + Ix + Iz + (x * x + z * z) * u)
    B  = K * u
    Cx = K * x * u
    Cz = K * z * u
    P  = btx * u + x * u * u;  Q  = -u * u          # Ax = P + xO*Q
    Pz = btz * u + z * u * u;  Qz = -u * u          # Az = Pz + zO*Qz

    # quadrature weights (composite Simpson on uniform s)
    ns = len(s); h = (s[-1] - s[0]) / (ns - 1)
    w = np.ones(ns); w[1:-1:2] = 4.0; w[2:-1:2] = 2.0; w *= h / 3.0

    xo = np.linspace(-HALF, HALF, N)
    zo = np.linspace(-HALF, HALF, N)

    eA = np.exp(1j * A)
    # Gx[s, ix], Gz[s, iz]
    Gx = np.exp(1j * (np.outer(B, xo * xo) - 2.0 * np.outer(Cx, xo)))
    Gz = np.exp(1j * (np.outer(B, zo * zo) - 2.0 * np.outer(Cz, zo)))

    def gemm(amp):
        return Gx.T @ (amp[:, None] * Gz)            # -> [ix, iz]

    wA = w * eA
    Ex = gemm(wA * P) + xo[:, None] * gemm(wA * Q)
    Ez = gemm(wA * Pz) + zo[None, :] * gemm(wA * Qz)

    # --- endpoint residual (terminating) terms: O(1) per obs point, NOT separable,
    # but that is fine: they cost O(N_obs), not O(N_obs*N_s).
    # Outside the undulator Bz = dBzds = Bx = dBxds = 0, so the 2nd/3rd expansion
    # terms collapse to the field-free form.
    X, Z = xo[:, None], zo[None, :]

    def resid(i):
        si, xi, zi, btxi, btzi = s[i], x[i], z[i], btx[i], btz[i]
        ui = 1.0 / (YOBS - si)
        dx, dz = X - xi, Z - zi
        Ph = K * (si * g2 + Ix[i] + Iz[i] + (dx * dx + dz * dz) * ui)
        Nx_, Nz_ = dx * ui, dz * ui
        bx, bz = btxi - Nx_, btzi - Nz_
        dPhds = K * (g2 + bx * bx + bz * bz)
        Axl, Azl = bx * ui, bz * ui
        d2 = 2.0 * K * (bx * Axl + bz * Azl)          # Bz=Bx=0 at the ends
        dAx, dAz = 2.0 * Axl * ui, 2.0 * Azl * ui
        t2x = (dAx - Axl * d2 / dPhds) / dPhds
        t2z = (dAz - Azl * d2 / dPhds) / dPhds
        d3 = 2.0 * K * (Axl * Axl + Azl * Azl
                        + bx * ui * 2.0 * bx * ui + bz * ui * 2.0 * bz * ui)
        d2Ax, d2Az = 3.0 * dAx * ui, 3.0 * dAz * ui
        t3x = (-d2Ax + (3.0 * dAx * d2 + Axl * d3) / dPhds
               - 3.0 * Axl * d2 * d2 / dPhds**2) / dPhds**2
        t3z = (-d2Az + (3.0 * dAz * d2 + Azl * d3) / dPhds
               - 3.0 * Azl * d2 * d2 / dPhds**2) / dPhds**2
        preX = (Axl + t3x) + 1j * t2x
        preZ = (Azl + t3z) + 1j * t2z
        return (1j * preX / dPhds) * np.exp(1j * Ph), (1j * preZ / dPhds) * np.exp(1j * Ph)

    RxF, RzF = resid(len(s) - 1)
    RxS, RzS = resid(0)
    Ex = Ex + (RxF - RxS)
    Ez = Ez + (RzF - RzS)
    return Ex.T, Ez.T                                # -> [iz, ix] to match SRW


if __name__ == "__main__":
    print(f"mesh {N}x{N}, {NPER} periods, Ns={NS}")
    Ex_ref, Ez_ref = srw_ref(TIGHT)
    gam = 3.0 / 0.51099890221e-03
    s, x, z, btx, btz = trajectory(NS)
    print(f"traj: s in [{s[0]:.4f},{s[-1]:.4f}], |x|max={np.abs(x).max():.3e}, "
          f"|btx|max={np.abs(btx).max():.3e}")
    Ex_s, Ez_s = separable(N, s, x, z, btx, btz, gam)

    # SRW applies an overall real normalisation constant; fit the single global
    # complex scale from the data and report the residual everywhere else.
    for nm, a, b in (("Ex", Ex_s, Ex_ref), ("Ez", Ez_s, Ez_ref)):
        if np.abs(b).max() < 1e-30:
            print(f"{nm}: reference is zero (max {np.abs(b).max():.2e})"); continue
        c = np.vdot(a.ravel(), b.ravel()) / np.vdot(a.ravel(), a.ravel())
        r = np.abs(c * a - b).max() / np.abs(b).max()
        print(f"{nm}: global scale = {c:.6e}  |c*sep - srw|_inf / peak = {r:.3e}")

    # --- error distribution diagnostics ---
    print("\n--- distribution of |c*sep - srw|/peak ---")
    for nm, a, b in (("Ex", Ex_s, Ex_ref), ("Ez", Ez_s, Ez_ref)):
        c = np.vdot(a.ravel(), b.ravel()) / np.vdot(a.ravel(), a.ravel())
        e = np.abs(c * a - b) / np.abs(b).max()
        q = np.percentile(e, [50, 90, 99, 99.9])
        j = np.unravel_index(np.argmax(e), e.shape)
        rel_int = np.abs(b)[j] / np.abs(b).max()
        print(f"{nm}: med={q[0]:.2e} p90={q[1]:.2e} p99={q[2]:.2e} p99.9={q[3]:.2e} "
              f"max={e.max():.2e} at {j} where |E|/peak={rel_int:.3f}")
        # error restricted to the bright region
        m = np.abs(b) > 0.1 * np.abs(b).max()
        print(f"     within |E|>0.1*peak ({m.sum()} pts): max={e[m].max():.2e}")
