"""THE decisive measurement.

E[ix,iz] = sum_s Gx[s,ix] * d_s * Gz[s,iz]   (exact, established in 02/03)

Gx[s,ix] = exp(i*(xO^2*B(s) - 2*xO*Cx(s))),  B = K*u(s), Cx = K*x(s)*u(s)

If Gx = Ux Vx and Gz = Uz Vz to rank r, then
    E = Vx^T (Ux^T diag(d) Uz) Vz
costs O(N_s*r^2) + O(N_obs*r) instead of O(N_obs*N_s).

So the whole prize is: how big is r, and how does it scale with
  (a) number of undulator periods,  (b) mesh half-width?

Theory (mine): the s-variation of the observation-dependent phase is
    dPhi = K * Delta_s * theta_max^2 = 2*pi * (Delta_s*theta_max^2)/(2*lambda)
and with theta_cone^2 = 1/(gamma^2 N) and Delta_s = N*lambda_u and the resonance
condition lambda = lambda_u(1+K^2/2)/(2 gamma^2), this reduces to
    R := dPhi/2pi ~ (theta_max/theta_cone)^2      -- INDEPENDENT of N periods.
=> r ~ const + 2R, while N_s must grow linearly with N periods.
=> the advantage ratio N_s/r grows linearly with undulator length.
"""
import sys, numpy as np
from srwpy import srwlib as sl
from srwpy import srwlpy as srwl

PER, BPEAK, EPH, YOBS = 0.021, 0.88, 1.2e3, 20.0
GAM = 3.0 / 0.51099890221e-03
K = np.pi / (1.239841984e-6 / EPH)          # pi/lambda[m]
LAM = 1.239841984e-6 / EPH


def traj(nper, ns):
    h = sl.SRWLMagFldH(1, 'v', BPEAK, 0, 1, 1)
    und = sl.SRWLMagFldU([h], PER, nper)
    mag = sl.SRWLMagFldC([und], sl.array('d', [0]), sl.array('d', [0]), sl.array('d', [0]))
    t = sl.SRWLPrtTrj(_np=ns)
    t.partInitCond.gamma = GAM
    t.partInitCond.z = -0.5 * PER * (nper + 4)
    t.ctStart, t.ctEnd = 0.0, PER * (nper + 4)
    srwl.CalcPartTraj(t, mag, [1])
    return (np.asarray(t.arZ), np.asarray(t.arX), np.asarray(t.arY),
            np.asarray(t.arXp), np.asarray(t.arYp))


def rank_of_Gx(nper, half, ns=6001, nx=1024, tols=(1e-4, 1e-6)):
    s, x, z, btx, btz = traj(nper, ns)
    u = 1.0 / (YOBS - s)
    B, Cx = K * u, K * x * u
    xo = np.linspace(-half, half, nx)
    G = np.exp(1j * (np.outer(B, xo * xo) - 2.0 * np.outer(Cx, xo)))
    sv = np.linalg.svd(G, compute_uv=False)
    sv = sv / sv[0]
    return [int(np.searchsorted(-sv, -t)) for t in tols], s[-1] - s[0]


if __name__ == "__main__":
    theta_cone = lambda n: 1.0 / (GAM * np.sqrt(n))
    print(f"lambda = {LAM*1e9:.4f} nm, gamma = {GAM:.1f}")
    print(f"{'nper':>5} {'L[m]':>7} {'half[mm]':>9} {'th_max/th_cone':>15} "
          f"{'R_pred':>8} {'r@1e-4':>8} {'r@1e-6':>8}")
    for nper in (20, 60, 200):
        tc = theta_cone(nper)
        for half in (0.4e-3, 1.2e-3, 4.0e-3):
            th = half / YOBS
            (r4, r6), L = rank_of_Gx(nper, half)
            R = L * th * th / (2 * LAM)
            print(f"{nper:>5} {L:>7.3f} {half*1e3:>9.2f} {th/tc:>15.2f} "
                  f"{R:>8.2f} {r4:>8} {r6:>8}")
