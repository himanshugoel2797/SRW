"""GPU-native low-rank evaluator for the SRW undulator radiation integral.

  E[ix,iz] = sum_s Gx[s,ix] d_s Gz[s,iz]            (exact separation, see 02/03)
  Gx ~ Ux Vx (rank rx), Gz ~ Uz Vz (rank rz)
  =>  E = Vx^T (Ux^T diag(d) Uz) Vz

Cost  O(N_obs*r + N_s*(Nx+Nz)*r)   instead of   O(N_obs*N_s).
Every rank operation touches only N_s x Nx matrices, never N_obs x N_s.
"""
import sys, numpy as np
import scipy  # noqa: F401  (must precede cupy)
from scipy.integrate import cumulative_simpson
import cupy as cp
from srwpy import srwlib as sl
from srwpy import srwlpy as srwl

PER, BPEAK, EPH, YOBS = 0.021, 0.88, 1.2e3, 20.0
GAM = 3.0 / 0.51099890221e-03
LAM = 1.239841984e-6 / EPH
KK = np.pi / LAM


def undulator(nper):
    h = sl.SRWLMagFldH(1, 'v', BPEAK, 0, 1, 1)
    return sl.SRWLMagFldC([sl.SRWLMagFldU([h], PER, nper)],
                          sl.array('d', [0]), sl.array('d', [0]), sl.array('d', [0]))


def traj(nper, ns):
    t = sl.SRWLPrtTrj(_np=ns)
    t.partInitCond.gamma = GAM
    t.partInitCond.z = -0.5 * PER * (nper + 4)
    t.ctStart, t.ctEnd = 0.0, PER * (nper + 4)
    srwl.CalcPartTraj(t, undulator(nper), [1])
    return (np.asarray(t.arZ), np.asarray(t.arX), np.asarray(t.arY),
            np.asarray(t.arXp), np.asarray(t.arYp))


def srw_field(n, nper, half, relPrec, dev=None):
    wfr = sl.SRWLWfr(); wfr.allocate(1, n, n)
    wfr.mesh.set_from_other(sl.SRWLRadMesh(
        _eStart=EPH, _eFin=EPH, _ne=1, _xStart=-half, _xFin=half, _nx=n,
        _yStart=-half, _yFin=half, _ny=n, _zStart=YOBS))
    eb = sl.SRWLPartBeam(); eb.Iavg = 0.5
    eb.partStatMom1.z = -0.5 * PER * (nper + 4)
    eb.partStatMom1.gamma = GAM
    wfr.partBeam = eb
    prec = [1, relPrec, 0, 0, 50000, 1, 0]
    a = (wfr, 0, undulator(nper), prec) + ((dev,) if dev is not None else ())
    srwl.CalcElecFieldSR(*a)
    ex = np.frombuffer(wfr.arEx, dtype=np.float32).astype(np.float64)
    ez = np.frombuffer(wfr.arEy, dtype=np.float32).astype(np.float64)
    return ((ex[0::2] + 1j * ex[1::2]).reshape(n, n),
            (ez[0::2] + 1j * ez[1::2]).reshape(n, n))


def rand_lowrank(G, r, p=10, seed=0):
    """Randomized range finder: G (Ns x Nx) ~ U V, U: Ns x k, V: k x Nx."""
    k = min(r + p, G.shape[1])
    rs = cp.random.RandomState(seed)
    Om = (rs.standard_normal((G.shape[1], k)) +
          1j * rs.standard_normal((G.shape[1], k))).astype(cp.complex128)
    Y = G @ Om
    Q, _ = cp.linalg.qr(Y)
    return Q, Q.conj().T @ G


def lowrank_field(n, nper, half, ns, rx, timing=False):
    s, x, z, btx, btz = traj(nper, ns)
    # 4th-order cumulative integral: with ~560 rad of accumulated K*Ix at 200
    # periods, a trapezoidal rule here (not the algorithm) sets the error floor.
    hh = (s[-1]-s[0])/(ns-1)
    Ix = np.concatenate([[0.], cumulative_simpson(btx*btx, dx=hh)])
    Iz = np.concatenate([[0.], cumulative_simpson(btz*btz, dx=hh)])
    u = 1.0/(YOBS - s)
    A = KK*(s/GAM**2 + Ix + Iz + (x*x + z*z)*u)
    B, Cx, Cz = KK*u, KK*x*u, KK*z*u
    P, Q_ = btx*u + x*u*u, -u*u
    Pz, Qz = btz*u + z*u*u, -u*u
    h = (s[-1]-s[0])/(ns-1)
    w = np.ones(ns); w[1:-1:2] = 4.; w[2:-1:2] = 2.; w *= h/3.

    g = lambda a: cp.asarray(a)
    B, Cx, Cz, A = g(B), g(Cx), g(Cz), g(A)
    w, P, Q_, Pz, Qz = g(w), g(P), g(Q_), g(Pz), g(Qz)
    xo = cp.linspace(-half, half, n, dtype=cp.float64)

    ev = lambda: (cp.cuda.Event(), )
    st, en = cp.cuda.Event(), cp.cuda.Event()
    cp.cuda.Stream.null.synchronize(); st.record()

    Gx = cp.exp(1j*(cp.outer(B, xo*xo) - 2.0*cp.outer(Cx, xo)))
    Gz = cp.exp(1j*(cp.outer(B, xo*xo) - 2.0*cp.outer(Cz, xo)))
    Ux, Vx = rand_lowrank(Gx, rx)
    Uz, Vz = rand_lowrank(Gz, rx)
    del Gx, Gz

    eA = cp.exp(1j*A)*w
    def core(amp):                      # M = Ux^T diag(amp) Uz  (rx x rz)
        return Ux.T @ (amp[:, None]*Uz)
    def expand(M):                      # Vx^T M Vz  -> (Nx x Nz)   [the only O(N_obs*r)]
        return Vx.T @ (M @ Vz)

    Ex = expand(core(eA*P)) + xo[:, None]*expand(core(eA*Q_))
    Ez = expand(core(eA*Pz)) + xo[None, :]*expand(core(eA*Qz))

    # endpoint residual terms, O(N_obs)
    X, Z = xo[:, None], xo[None, :]
    def resid(i):
        si = float(s[i]); ui = 1.0/(YOBS-si)
        dx, dz = X - float(x[i]), Z - float(z[i])
        Ph = KK*(si/GAM**2 + float(Ix[i]) + float(Iz[i]) + (dx*dx+dz*dz)*ui)
        bx, bz = float(btx[i]) - dx*ui, float(btz[i]) - dz*ui
        dP = KK*(1.0/GAM**2 + bx*bx + bz*bz)
        Ax, Az = bx*ui, bz*ui
        d2 = 2.0*KK*(bx*Ax + bz*Az)
        dAx, dAz = 2.0*Ax*ui, 2.0*Az*ui
        t2x, t2z = (dAx-Ax*d2/dP)/dP, (dAz-Az*d2/dP)/dP
        d3 = 2.0*KK*(Ax*Ax+Az*Az+2*(bx*ui)**2+2*(bz*ui)**2)
        t3x = (-3.*dAx*ui + (3.*dAx*d2+Ax*d3)/dP - 3.*Ax*d2*d2/dP**2)/dP**2
        t3z = (-3.*dAz*ui + (3.*dAz*d2+Az*d3)/dP - 3.*Az*d2*d2/dP**2)/dP**2
        e = cp.exp(1j*Ph)
        return (1j*((Ax+t3x)+1j*t2x)/dP)*e, (1j*((Az+t3z)+1j*t2z)/dP)*e
    RxF, RzF = resid(ns-1); RxS, RzS = resid(0)
    Ex, Ez = Ex + (RxF-RxS), Ez + (RzF-RzS)

    en.record(); en.synchronize()
    return cp.asnumpy(Ex).T, cp.asnumpy(Ez).T, cp.cuda.get_elapsed_time(st, en)


def agree(a, b):
    c = np.vdot(a.ravel(), b.ravel())/np.vdot(a.ravel(), a.ravel())
    e = np.abs(c*a - b)/np.percentile(np.abs(b), 99.9)
    return np.percentile(e, 50), np.percentile(e, 99.9), e.max()


if __name__ == "__main__":
    n    = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    nper = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    half = float(sys.argv[3]) if len(sys.argv) > 3 else 0.4e-3
    print(f"module {srwl.__file__}\nmesh {n}^2  nper={nper}  half={half*1e3:.2f} mm")
    ref = srw_field(n, nper, half, 1e-5)[0]

    print(f"\n{'Ns':>7} {'r':>4} {'med':>10} {'p99.9':>10} {'max':>10} {'gpu_ms':>9}")
    for ns in (2001, 4001, 8001, 16001):
        for r in (16, 32, 64):
            Ex, Ez, ms = lowrank_field(n, nper, half, ns, r)
            m, p, mx = agree(Ex, ref)
            print(f"{ns:>7} {r:>4} {m:>10.2e} {p:>10.2e} {mx:>10.2e} {ms:>9.3f}")
