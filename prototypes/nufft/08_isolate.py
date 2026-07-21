"""Isolate the error: low-rank vs DIRECT evaluation of the SAME integral,
same trajectory, same quadrature. If they agree to ~1e-12, the algorithm is
exact and everything seen vs SRW is prototype trajectory/setup fidelity."""
import sys, numpy as np, scipy, importlib
import cupy as cp
sys.path.insert(0,".")
L = importlib.import_module("05_lowrank_gpu")
from scipy.integrate import cumulative_simpson

n, nper, half, ns = 64, int(sys.argv[1]), 0.4e-3, 8001
s,x,z,btx,btz = L.traj(nper, ns)
hh=(s[-1]-s[0])/(ns-1)
Ix=np.concatenate([[0.],cumulative_simpson(btx*btx,dx=hh)])
Iz=np.concatenate([[0.],cumulative_simpson(btz*btz,dx=hh)])
u=1./(L.YOBS-s); KK=L.KK
A=KK*(s/L.GAM**2+Ix+Iz+(x*x+z*z)*u); B=KK*u; Cx=KK*x*u; Cz=KK*z*u
P=btx*u+x*u*u; Q=-u*u
w=np.ones(ns); w[1:-1:2]=4.; w[2:-1:2]=2.; w*=hh/3.
xo=np.linspace(-half,half,n)

# DIRECT: full O(N_obs*N_s) sum, no separation, no low rank
g=cp.asarray
Xo=g(xo)[:,None,None]; Zo=g(xo)[None,:,None]; S=g(s)[None,None,:]
Ph = g(A)[None,None,:] + (Xo*Xo+Zo*Zo)*g(B)[None,None,:] - 2*Xo*g(Cx)[None,None,:] - 2*Zo*g(Cz)[None,None,:]
amp = g(P)[None,None,:] + Xo*g(Q)[None,None,:]
Edir = cp.sum(g(w)[None,None,:]*amp*cp.exp(1j*Ph), axis=2)

# LOW-RANK path (same inputs)
gB,gCx,gA,gw,gP,gQ = map(cp.asarray,(B,Cx,A,w,P,Q))
gxo=cp.asarray(xo)
Gx=cp.exp(1j*(cp.outer(gB,gxo*gxo)-2*cp.outer(gCx,gxo)))
Gz=cp.exp(1j*(cp.outer(gB,gxo*gxo)-2*cp.outer(cp.asarray(Cz),gxo)))
for r in (8,16,32):
    Ux,Vx=L.rand_lowrank(Gx,r); Uz,Vz=L.rand_lowrank(Gz,r)
    ex=lambda a: Vx.T@((Ux.T@(a[:,None]*Uz))@Vz)
    Elr = ex(cp.exp(1j*gA)*gw*gP) + gxo[:,None]*ex(cp.exp(1j*gA)*gw*gQ)
    e=cp.abs(Elr-Edir).max()/cp.abs(Edir).max()
    print(f"nper={nper} r={r:>3} (k={Ux.shape[1]:>3}):  |lowrank - direct|_inf / peak = {float(e):.3e}")
