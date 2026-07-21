# Separable low-rank undulator radiation integral — measured assessment

Branch `agent/nufft`. Prototype only; **no C++/CUDA implementation was written.**
Everything below is measured, on one A100, with the commands in this directory.

## 1. The structure (verified, not assumed)

The orchestrator's decomposition is correct, and it is stronger than stated:
xO and zO separate **completely**, not just into monomials.

```
Phase(s,xO,zO) = A(s) + xO^2*B(s) - 2*xO*Cx(s)  +  zO^2*B(s) - 2*zO*Cz(s)
```

| | A(s) | B(s) | Cx(s) |
|---|---|---|---|
| near (CoordPres) | K(s/g^2 + Ix + Iz + (x^2+z^2)u) | K*u(s) | K*x(s)*u(s) |
| far (AngPres) | K(s/g^2 + Ix + Iz) | K*s | K*x(s) |

`01_algebra.py`: max relative error of the decomposition over 2000 random
(s,xO,zO) = **4.3e-13 (near), 3.1e-15 (far)**. Exact to roundoff.

Therefore `exp(i*Phase) = e^{iA} * Gx(s,xO) * Gz(s,zO)`, and since the amplitude
is linear in xO only (`Ax = P(s) + xO*Q(s)`, near: `P = Btx*u + x*u^2`, `Q = -u^2`):

```
E[ix,iz] = sum_s Gx[s,ix] * (w_s P_s e^{iA_s}) * Gz[s,iz]   (+ xO * same with Q)
         = (Gx^T D Gz)[ix,iz]
```

**This makes the whole mesh evaluation a matrix product, not a NUFFT.**
A NUFFT is unnecessary — and a plain GEMM is *not* the answer either, since it
still costs O(N_obs*N_s), the same complexity as today's kernel.

## 2. Where the win actually is: the rank of Gx

`Gx` is `N_s x N_x`. Factor `Gx ~ Ux Vx`, `Gz ~ Uz Vz` at rank r:

```
E = Vx^T (Ux^T D Uz) Vz        O(N_s*r^2) + O(N_obs*r)   instead of O(N_obs*N_s)
```

Critically, **every rank operation touches only N_s x N_x matrices (~1e7), never
N_obs x N_s (~1e10)**, so the factorization is affordable at runtime.

Measured singular-value ranks (`04_rank.py`), with the prediction
`R = (theta_max/theta_cone)^2` derived from the resonance condition:

| nper | L [m] | half [mm] | th_max/th_cone | R_pred | r@1e-4 | r@1e-6 |
|---|---|---|---|---|---|---|
| 20 | 0.504 | 0.40 | 0.53 | 0.10 | 5 | 7 |
| 20 | 0.504 | 4.00 | 5.25 | 9.76 | 32 | 38 |
| 60 | 1.344 | 1.20 | 2.73 | 2.34 | 15 | 19 |
| 200 | 4.284 | 0.40 | 1.66 | 0.83 | 9 | 13 |
| 200 | 4.284 | 1.20 | 4.98 | 7.46 | 28 | 33 |
| 200 | 4.284 | 4.00 | 16.61 | 82.93 | 186 | 196 |

`r ~ 2R + 5`, and **R does not depend on the number of periods** — it depends only
on how many central cones the mesh spans. Meanwhile N_s grows linearly with
undulator length. So the advantage ratio grows with undulator length; the heavy
regime is exactly where this pays.

## 3. Accuracy

**The algorithm itself is exact.** `08_isolate.py` compares the low-rank result
against a direct O(N_obs*N_s) sum of the *same* integral, same trajectory, same
quadrature:

| nper | rank used | \|lowrank - direct\|_inf / peak |
|---|---|---|
| 20 | 18 | **1.1e-14** |
| 200 | 18 | 7.1e-09 |
| 200 | 26 | **3.0e-13** |

Against SRW ground truth (**CPU @ relPrec=1e-5**, per KNOWN_ISSUES; never the
default), `03_diag.py` / `05_lowrank_gpu.py`, 256^2 mesh:

| case | median | p99.9 | max |
|---|---|---|---|
| 20 per, Ns=4001 | 6.7e-5 | 2.9e-4 | **3.0e-4** |
| 200 per, Ns=8001 | 8.6e-4 | 1.4e-3 | 1.7e-2 (*) |

(*) the max is a **CPU reference outlier, not ours**: `cpu@1e-4` vs `cpu@1e-5`
differ by **1.41e-1 of peak at a single bright point** while being bit-identical
(median error exactly 0) everywhere else. The prototype has no outlier there.
This is a *new* instance of the trap KNOWN_ISSUES warns about, now at
relPrec=1e-4 — worth recording independently of this work.

Error is **bit-identical for r=16/32/64**, confirming rank truncation contributes
nothing. The ~1e-3 floor at 200 periods is prototype trajectory fidelity: the
prototype rebuilds the trajectory via `CalcPartTraj` + cumulative Simpson, while
SRW's integrator uses `CompTotalTrjData`'s polynomial representation. A C++
implementation would reuse the arrays the existing kernel **already uploads**, so
this error source disappears by construction. (Switching trapezoid -> cumulative
Simpson for `Ix` already moved 200-period median 1.18e-3 -> 8.6e-4.)

## 4. Performance — measured, kernel-level only

Baseline measured by me with nsys, **not quoted**: 1024^2, 200 periods,
relPrec=1e-2, `--report cuda_gpu_kern_sum`:

```
RadIntAuto1Kernel   29,658,978 ns median   (3 instances, min 29.49 ms, max 30.09 ms)
```

Low-rank CuPy prototype, same mesh/periods, Ns=8001, r=26, steady-state medians
of 11 reps (`06_phases.py`):

| phase | ms | note |
|---|---|---|
| build Gx, Gz | 1.432 | N_s*(Nx+Nz) complex exp |
| randomized factorization | 3.050 | cuSOLVER QR, **overhead-dominated** |
| core `Ux^T D Uz` (x4) | 0.376 | O(N_s*r^2) |
| **expand `Vx^T M Vz` (x4)** | **0.288** | **the only O(N_obs*r) step** |
| total | **5.15** | |
| arithmetic only (excl. cuSOLVER) | 2.10 | |

| comparison | speedup vs 29.66 ms |
|---|---|
| full CuPy prototype (5.15 ms) | **5.8x** |
| excluding cuSOLVER overhead (2.10 ms) | 14x |
| the O(N_obs) step alone (0.288 ms) | 103x |

**These are kernel-level numbers only.** At 1024^2/200 periods the integral is
~31% of a 95 ms wall, so even an infinitely fast integral is ~1.45x end-to-end,
and 5.8x on the kernel is about **1.35x end-to-end**. In the common regime
(512^2, 20 periods, ~4% kernel share) it is worth ~1.03x. Do not quote the
kernel number as an end-to-end one.

## 5. Honest limitations / what is unverified

1. **No CUDA implementation.** Nothing was dispatched or profiled beyond the
   CuPy prototype; no scope gate, no `--no-cuda` build check, no pytest run.
2. **Setup cost is the binding constraint, and it is per-electron.** Build +
   factorize = 4.5 ms is paid once per trajectory. In
   `srwl_wfr_emit_prop_multi_e` every macro-electron has a different trajectory,
   so this repeats per electron. At 200 periods that is still ~6x vs 29.7 ms; at
   **20 periods (3.2 ms kernel) the setup alone exceeds the existing kernel and
   the method LOSES**. The win is confined to long undulators.
3. Most of that setup is avoidable and was not attempted: cuSOLVER QR on a
   tall-skinny matrix is ~3 ms of pure overhead for ~0.05 ms of arithmetic, and
   `Gx` need only be built at ~r Chebyshev nodes rather than all N_s. Getting
   setup under 1 ms is the single highest-value next step.
4. **Fixed common quadrature replaces per-point adaptivity.** N_s must be set
   from the worst-case phase gradient over the mesh. I verified convergence
   empirically (Ns=4001 at 20 per, 8001 at 200 per) but wrote **no automatic N_s
   selector**, and no automatic rank selector either.
5. Validated only for **CoordPres (near field), ne=1**, a single centred
   on-axis electron, symmetric mesh, one photon energy. `ne>1` needs one
   factorization per energy. AngPres is derived but untested.
6. The endpoint residual terms were re-derived assuming **B=0 outside the
   undulator** (true here); the general form must come from the existing
   `RadIntResidualOneSide_GPU`.

## 6. Recommendation

The structure is real and the measured rank is small, so the asymptotic claim
holds: **O(N_obs*r) with r~10-30 instead of O(N_obs*N_s)**, and it is exact to
1e-13. But the prize is bounded by the ~31% kernel share, and the per-electron
setup cost currently eats most of it outside the long-undulator regime.

Worth continuing **only** if the setup cost is attacked first (item 3). If it
drops below ~1 ms, the method is ~15x on the kernel at 200 periods (~1.4x
end-to-end) and roughly break-even at 20 periods. That is a real but narrow win,
and the host-side work the other agent is attacking is the larger prize.

## Files

- `01_algebra.py` — decomposition check (exact to roundoff)
- `02_separable.py` — CPU separable evaluator incl. endpoint residual terms
- `03_diag.py` — error distribution; exposes the CPU@1e-4 outlier
- `04_rank.py` — rank vs periods and mesh width (the key scaling table)
- `05_lowrank_gpu.py` — full CuPy low-rank evaluator + accuracy sweep
- `06_phases.py` — per-phase steady-state GPU timing
- `07_base.py` + `base1024_200.nsys-rep` — nsys baseline of the existing kernel
- `08_isolate.py` — low-rank vs direct sum (proves the algorithm is exact)
