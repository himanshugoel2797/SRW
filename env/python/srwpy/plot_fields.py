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

#Read field.txt into signal
def read_field(idx, dev, p='post'):
    data = np.loadtxt('base_rad_x_%s_%s_%d.txt'%(p, dev, idx), delimiter=',', dtype=np.float32)

    # Read the first line of the file
    with open('base_rad_x_%s_%s_%d.txt'%(p, dev, idx), 'r') as f:
        line0 = f.readline().strip().split(',')
        print(line0)

    return data, int(line0[0][1:]), int(line0[1])

idx = 1


srw_fft_cpu, nx, nz = read_field(idx, 'cpu')
srw_fft_gpu, _, _ = read_field(idx, 'gpu')
dims = [-0.000991, 1e-6, nx, -0.000488, 1e-6, nz]

srw_fft_cpu = srw_fft_cpu.reshape((nz, nx, 2))
srw_fft_cpu = np.abs((srw_fft_cpu[:,:,0] + 1j*srw_fft_cpu[:,:,1]) ** 2)

srw_fft_gpu = srw_fft_gpu.reshape((nz, nx, 2))
srw_fft_gpu = np.abs((srw_fft_gpu[:,:,0] + 1j*srw_fft_gpu[:,:,1]) ** 2)

max_v = max(np.max(np.abs(srw_fft_cpu)), np.max(np.abs(srw_fft_gpu)))
min_v = min(np.min(np.abs(srw_fft_cpu)), np.min(np.abs(srw_fft_gpu)))

# Compare results
fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
ax[0].set_title('SRW (CPU)')
im = ax[0].imshow(srw_fft_cpu)#, vmin=min_v, vmax=max_v)
plt.colorbar(im, ax=ax[0])
ax[1].set_title('SRW (GPU)')
im = ax[1].imshow(srw_fft_gpu)#, vmin=min_v, vmax=max_v)
plt.colorbar(im, ax=ax[1])

amax = np.argmax(srw_fft_cpu - srw_fft_gpu)
amin = np.argmin(srw_fft_cpu - srw_fft_gpu)

amax = [amax // nx, amax % nx]
amin = [amin // nx, amin % nx]

ax[2].set_title('Difference (CPU - GPU) Max=%f@(%d,%d) Min=%f@(%d,%d)' % (np.max(srw_fft_cpu - srw_fft_gpu), amax[0], amax[1], np.min(srw_fft_cpu - srw_fft_gpu), amin[0], amin[1]))
im = ax[2].imshow(np.abs(srw_fft_cpu - srw_fft_gpu))
plt.colorbar(im, ax=ax[2])
#ax[2].plot(np_fft, label='NumPy FFT', linestyle='dotted')
#ax[0].set_ylim()
#ax[1].set_ylim()
#ax[2].set_ylim()
#ax[0].set_xlim(nx//2 - nz//2, nx//2 + nz//2)
#ax[1].set_xlim(nx//2 - nz//2, nx//2 + nz//2)
#ax[2].set_xlim(nx//2 - nz//2, nx//2 + nz//2)
plt.show()