import sys, numpy as np, importlib
sys.path.insert(0,".")
L = importlib.import_module("05_lowrank_gpu")
n, nper, half = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
print("module", L.srwl.__file__)
for i in range(3):
    import time; t=time.time()
    L.srw_field(n, nper, half, 1e-2, dev=1)
    print(f"  call {i}: {time.time()-t:.3f} s wall")
