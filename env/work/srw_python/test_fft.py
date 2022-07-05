#############################################################################
# Test FFT
# v 0.01
#############################################################################

from __future__ import print_function  # Python 2.7 compatibility
from srwlib import *
from uti_plot import *
import os

import ctypes
import cupy as cp
import numpy as np
import cupy.cuda.memory
cp.cuda.set_allocator(cupy.cuda.memory.malloc_managed)


#**********************Input Parameters:

sig = 1
gpuEn = True
xNp = 100
xRange = 10
xStart = -0.5*xRange
xStep = xRange/(xNp)

if gpuEn:
    #ar_b = srwl.managedbuffer(xNp*2)#array('f', [0]*(xNp*2))
    #ar = np.asarray(ar_b, dtype=np.float32)
    ar_cp = cp.array(array('f', [0]*(xNp*2)))
    a_ptr = ctypes.cast(ar_cp.data.ptr, ctypes.POINTER(ctypes.c_float))
    ar = np.ctypeslib.as_array(a_ptr, (2 * xNp,))
else:
    ar_cp = array('f', [0]*(xNp*2))
    ar = ar_cp

x = xStart
for i in range(xNp):
    #ar[2*i] = exp(-x*x/(2*sig*sig))
    if abs(x) < 1:
        ar[2*i] = 1
    x += xStep

mesh = [xStart, xStep, xNp]

ar_Re = array('f', [0]*(xNp))
ar_Im = array('f', [0]*(xNp))
for i in range(xNp):
    ar_Re[i] = ar[2*i]
    ar_Im[i] = ar[2*i + 1]
uti_plot1d(ar_Re, [mesh[0], mesh[0] + mesh[1]*xNp, xNp],
           ['Qx', 'Re FT', 'Input'])
#input('Waiting for enter.')

if gpuEn:
    srwl.UtiFFT(ar, mesh, 1, 1)
else:
    srwl.UtiFFT(ar, mesh, 1, 0)

arFT_Re = array('f', [0]*(xNp))
arFT_Im = array('f', [0]*(xNp))
for i in range(xNp):
    arFT_Re[i] = ar[2*i]
    arFT_Im[i] = ar[2*i + 1]

    print(arFT_Re[i], arFT_Im[i])

uti_plot1d(arFT_Re, [mesh[0], mesh[0] + mesh[1]*xNp, xNp],
           ['Qx', 'Re FT', 'Test FFT {}'.format( 'GPU' if gpuEn else 'CPU')])
uti_plot1d(arFT_Im, [mesh[0], mesh[0] + mesh[1]*xNp, xNp],
           ['Qx', 'Im FT', 'Test FFT {}'.format( 'GPU' if gpuEn else 'CPU')])

uti_plot_show()