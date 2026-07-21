import numpy as np
rng = np.random.default_rng(0)
K, g2, yO = 3.0e10, 2.9e-8, 20.0
for case in ('near','far'):
    worst = 0.0
    for _ in range(2000):
        s,x,z,Ix,Iz = rng.normal(0,0.2), rng.normal(0,3e-7), rng.normal(0,3e-7), rng.normal(0,1e-9), rng.normal(0,1e-9)
        xO,zO = rng.normal(0,4e-4), rng.normal(0,4e-4)
        if case=='near':
            u = 1.0/(yO-s)
            direct = K*(s*g2 + Ix + Iz + ((xO-x)**2+(zO-z)**2)*u)
            A = K*(s*g2+Ix+Iz+(x*x+z*z)*u); B = K*u; Cx = K*x*u; Cz = K*z*u
        else:
            direct = K*(s*(g2+xO*xO+zO*zO) + Ix + Iz - 2.0*(xO*x+zO*z))
            A = K*(s*g2+Ix+Iz); B = K*s; Cx = K*x; Cz = K*z
        sep = A + (xO*xO)*B - 2*xO*Cx + (zO*zO)*B - 2*zO*Cz
        worst = max(worst, abs(direct-sep)/max(abs(direct),1e-30))
    print(f"{case:5s}  max rel err phase decomposition = {worst:.3e}")
