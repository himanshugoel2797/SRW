#!/usr/bin/env python

import os
try:
    __IPYTHON__
    import sys
    del sys.argv[1:]
except:
    pass

try: #OC15112022
    import sys
    sys.path.append('../')
    from srwlib import *
    from srwl_bl import *
    from srwl_uti_smp import *
    from uti_io_genesis import *
    from srwl_uti_detector import *
    #from uti_plot import *
except:
    from srwpy.srwlib import *
    from srwpy.srwl_uti_smp import *
    from srwpy.uti_plot import *
    from srwpy.srwl_uti_smp import *
    from srwpy.uti_io_genesis import *
    from srwpy.srwl_uti_detector import *

from time import *
import pickle
import numpy as np
import matplotlib.pyplot as plt
import shutil
import copy

# Compare FFT calculation results for a test signal between srwpy and numpy

# Generate a test signal
nx = 2100
nz = 756
signal = np.zeros(2*nx*nz)

#Read field.txt into signal
line = np.loadtxt('field_gpu.txt', delimiter=',', dtype=np.float32)
print(line.shape)
signal[:line.size] = line

dims = [-0.000991, 1e-6, nx, -0.000488, 1e-6, nz]

srw_fft_cpu = copy.deepcopy(signal)
srw_fft_gpu = copy.deepcopy(signal)

# Compute FFT using srwpy on cpu
srwl.UtiFFT(srw_fft_cpu, copy.deepcopy(dims), 1, 0)
srwl.UtiFFT(srw_fft_gpu, copy.deepcopy(dims), 1, 1)

srw_fft_cpu = srw_fft_cpu.reshape((nz, nx, 2))
srw_fft_cpu = np.abs((srw_fft_cpu[:,:,0] + 1j*srw_fft_cpu[:,:,1]) ** 2)

srw_fft_gpu = srw_fft_gpu.reshape((nz, nx, 2))
srw_fft_gpu = np.abs((srw_fft_gpu[:,:,0] + 1j*srw_fft_gpu[:,:,1]) ** 2)

max_v = max(np.max(np.abs(srw_fft_cpu)), np.max(np.abs(srw_fft_gpu)))
min_v = min(np.min(np.abs(srw_fft_cpu)), np.min(np.abs(srw_fft_gpu)))

# Compare results
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].set_title('SRWPy FFT (CPU)')
im = ax[0].imshow(srw_fft_cpu, vmin=min_v, vmax=max_v)
plt.colorbar(im, ax=ax[0])
ax[1].set_title('SRWPy FFT (GPU)')
im = ax[1].imshow(srw_fft_gpu, vmin=min_v, vmax=max_v)
plt.colorbar(im, ax=ax[1])
ax[2].set_title('Difference (CPU - GPU)')
im = ax[2].imshow(srw_fft_cpu - srw_fft_gpu)
plt.colorbar(im, ax=ax[2])
#ax[2].plot(np_fft, label='NumPy FFT', linestyle='dotted')
#ax[0].set_ylim()
#ax[1].set_ylim()
#ax[2].set_ylim()
ax[0].set_xlim(nx//2 - nz//2, nx//2 + nz//2)
ax[1].set_xlim(nx//2 - nz//2, nx//2 + nz//2)
ax[2].set_xlim(nx//2 - nz//2, nx//2 + nz//2)
plt.show()