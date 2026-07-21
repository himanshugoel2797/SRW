"""Per-phase GPU timing of the low-rank evaluator, steady state (median of reps).

Separates the algorithm's ARITHMETIC (which a CUDA implementation would pay)
from CuPy/cuSOLVER launch overhead (which it would not).
"""
import sys, numpy as np, scipy  # noqa
import cupy as cp
import importlib
sys.path.insert(0, ".")
L = importlib.import_module("05_lowrank_gpu")

n    = int(sys.argv[1]); nper = int(sys.argv[2])
half = float(sys.argv[3]); ns = int(sys.argv[4]); r = int(sys.argv[5])
REPS = 11

s, x, z, btx, btz = L.traj(nper, ns)
d = np.diff(s)
Ix = np.concatenate([[0.], np.cumsum(0.5*(btx[1:]**2+btx[:-1]**2)*d)])
Iz = np.concatenate([[0.], np.cumsum(0.5*(btz[1:]**2+btz[:-1]**2)*d)])
u = 1.0/(L.YOBS - s)
A = L.KK*(s/L.GAM**2 + Ix + Iz + (x*x+z*z)*u)
B, Cx, Cz = L.KK*u, L.KK*x*u, L.KK*z*u
P, Q_ = btx*u + x*u*u, -u*u
h = (s[-1]-s[0])/(ns-1)
w = np.ones(ns); w[1:-1:2] = 4.; w[2:-1:2] = 2.; w *= h/3.
gB, gCx, gCz, gA, gw, gP, gQ = map(cp.asarray, (B, Cx, Cz, A, w, P, Q_))
xo = cp.linspace(-half, half, n, dtype=cp.float64)
xx = xo*xo

def timed(fn, reps=REPS):
    ts = []
    for _ in range(reps):
        st, en = cp.cuda.Event(), cp.cuda.Event()
        cp.cuda.Stream.null.synchronize(); st.record()
        out = fn()
        en.record(); en.synchronize()
        ts.append(cp.cuda.get_elapsed_time(st, en))
    return float(np.median(ts)), out

t_build, (Gx, Gz) = timed(lambda: (
    cp.exp(1j*(cp.outer(gB, xx) - 2.0*cp.outer(gCx, xo))),
    cp.exp(1j*(cp.outer(gB, xx) - 2.0*cp.outer(gCz, xo)))))

t_fact, (Ux, Vx, Uz, Vz) = timed(lambda: L.rand_lowrank(Gx, r) + L.rand_lowrank(Gz, r))

eA = cp.exp(1j*gA)*gw
t_core, M = timed(lambda: Ux.T @ ((eA*gP)[:, None]*Uz))
t_exp,  _ = timed(lambda: Vx.T @ (M @ Vz))

rk = Ux.shape[1]
print(f"mesh {n}^2 nper={nper} half={half*1e3:.2f}mm Ns={ns} r_req={r} r_used={rk}")
print(f"  build Gx,Gz      {t_build:8.3f} ms")
print(f"  rand factorize   {t_fact:8.3f} ms   (cuSOLVER QR - overhead-dominated)")
print(f"  core (per term)  {t_core:8.3f} ms   x4 terms")
print(f"  expand(per term) {t_exp:8.3f} ms   x4 terms  <- only O(N_obs*r) step")
print(f"  TOTAL(4 terms)   {t_build+t_fact+4*(t_core+t_exp):8.3f} ms")
print(f"  arithmetic-only  {t_build+4*(t_core+t_exp):8.3f} ms  (excl. cuSOLVER)")
gf = 8.0*n*n*rk*4/1e9
print(f"  expand flops     {gf:.3f} GFLOP -> {gf/(4*t_exp/1e3)/1e3:.2f} TFLOP/s")
