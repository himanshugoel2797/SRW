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
#import cupy.cuda.memory
#cp.cuda.set_allocator(cupy.cuda.memory.malloc_managed)


#**********************Input Parameters:

sig = 1
gpuEn = False
xNp = 100
xRange = 10
xStart = -0.5*xRange
xStep = xRange/(xNp)

ar_cp = array('f', [0]*(xNp*xNp*2))
ar = ar_cp

y = xStart
for i in range(xNp):
    x = xStart
    for j in range(xNp):
        #ar[2*i] = exp(-x*x/(2*sig*sig))
        if abs(x) < 1 and abs(y) < 1:
            ar[2*(i*xNp+j)] = 1
        x += xStep
    y += xStep

mesh = [xStart, xStep, xNp, xStart, xStep, xNp]

ar_Re = array('f', [0]*(xNp*xNp))
ar_Im = array('f', [0]*(xNp*xNp))
for i in range(xNp*xNp):
    ar_Re[i] = ar[2*i]
    ar_Im[i] = ar[2*i + 1]
uti_plot2d(ar_Re, [mesh[0], mesh[0] + mesh[1]*xNp, xNp], [mesh[3], mesh[3] + mesh[4]*xNp, xNp], ['Qx', 'Qy', 'Input'])
#input('Waiting for enter.')

if gpuEn:
    srwl.UtiFFT(ar, mesh, 1, 1)
else:
    srwl.UtiFFT(ar, mesh, 1, 0)

arFT_Re = array('f', [0]*(xNp*xNp))
arFT_Im = array('f', [0]*(xNp*xNp))
for i in range(xNp*xNp):
    arFT_Re[i] = ar[2*i]
    arFT_Im[i] = ar[2*i + 1]

    print(arFT_Re[i], arFT_Im[i])

#uti_plot1d(arFT_Re, [mesh[0], mesh[0] + mesh[1]*xNp, xNp],
#           ['Qx', 'Re FT', 'Test FFT {}'.format( 'GPU' if gpuEn else 'CPU')])
#uti_plot1d(arFT_Im, [mesh[0], mesh[0] + mesh[1]*xNp, xNp],
#           ['Qx', 'Im FT', 'Test FFT {}'.format( 'GPU' if gpuEn else 'CPU')])

uti_plot2d(arFT_Re, [mesh[0], mesh[0] + mesh[1]*xNp, xNp], [mesh[3], mesh[3] + mesh[4]*xNp, xNp], ['Qx', 'Qy', 'Re FT'])
uti_plot2d(arFT_Im, [mesh[0], mesh[0] + mesh[1]*xNp, xNp], [mesh[3], mesh[3] + mesh[4]*xNp, xNp], ['Qx', 'Qy', 'Im FT'])

uti_plot_show()