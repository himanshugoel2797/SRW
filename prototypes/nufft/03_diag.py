"""Is the outlier at (7,46) my error or a CPU adaptive-integrator spike?

Test: tighten the CPU reference. If the separable result is the accurate one,
the CPU should move TOWARDS it as relPrec tightens, and the spike should move
or vanish (adaptive blowups are not stable under tolerance changes).
"""
import numpy as np
import importlib, sys
sys.argv = ["x", "64", "20", "40001"]
m = importlib.import_module("02_separable")

gam = 3.0 / 0.51099890221e-03
s, x, z, btx, btz = m.trajectory(40001)
Ex_s, Ez_s = m.separable(64, s, x, z, btx, btz, gam)

refs = {}
for p in (1e-3, 1e-4, 1e-5):
    refs[p] = m.srw_ref(p)
    print(f"CPU relPrec={p:g} done")

Ex_t, Ez_t = refs[1e-5]

# 1. is the reference peak location anomalous?
for p, (Ex_r, _) in refs.items():
    j = np.unravel_index(np.argmax(np.abs(Ex_r)), Ex_r.shape)
    a = np.abs(Ex_r)
    i0, i1 = j
    nb = np.median([a[max(i0-1,0), i1], a[min(i0+1,63), i1],
                    a[i0, max(i1-1,0)], a[i0, min(i1+1,63)]])
    print(f"relPrec={p:g}: argmax|Ex| at {j}, |Ex|={a[j]:.4e}, "
          f"median of 4 neighbours={nb:.4e}, ratio={a[j]/nb:.2f}")

# 2. robust peak (99th pct) as the normaliser instead of the max
def report(tag, a, b, norm):
    c = np.vdot(a.ravel(), b.ravel()) / np.vdot(a.ravel(), a.ravel())
    e = np.abs(c * a - b) / norm
    q = np.percentile(e, [50, 99, 99.9])
    print(f"  {tag}: med={q[0]:.2e} p99={q[1]:.2e} p99.9={q[2]:.2e} max={e.max():.2e}")

norm = np.percentile(np.abs(Ex_t), 99.9)
print(f"\nnormaliser = 99.9th pct of |Ex| @1e-5 = {norm:.4e} "
      f"(true max = {np.abs(Ex_t).max():.4e})")

print("\nseparable vs CPU at each tolerance (Ex):")
for p, (Ex_r, _) in refs.items():
    report(f"sep vs cpu@{p:g}", Ex_s, Ex_r, norm)

print("\nCPU self-convergence (does CPU move toward separable?):")
for p in (1e-3, 1e-4):
    e = np.abs(refs[p][0] - Ex_t) / norm
    q = np.percentile(e, [50, 99, 99.9])
    print(f"  cpu@{p:g} vs cpu@1e-5: med={q[0]:.2e} p99={q[1]:.2e} "
          f"p99.9={q[2]:.2e} max={e.max():.2e}")
