/************************************************************************//**
 * File: sroptelm_gpu.cu
 * Description: Optical element (general CUDA functions)
 * Project: Synchrotron Radiation Workshop
 * First release: 2023
 *
 * Copyright (C) Brookhaven National Laboratory
 * All Rights Reserved
 *
 * @author H.Goel
 * @version 1.0
 ***************************************************************************/

#ifdef _OFFLOAD_GPU
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_constants.h"
#include "auxgpu.h"

const int PerThread = 16;
template<typename T> __global__ void Memset_Kernel(T* p, T val, long long n)
{
    long long offset = blockIdx.x * blockDim.x + threadIdx.x;
    offset *= PerThread;
    long long dst = min(offset + PerThread, n);
    for (; offset < dst; offset++)
        p[offset] = val;
}

void CAuxGPU::Memset_GPU(float* p, float val, long long n, long long streamIdx)
{
    int minGridSize = 0;
    int bs = 256;
    long long n_orig = n;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &bs, Memset_Kernel<float>, 0, (n + PerThread - 1) / PerThread);
    n = ((n + PerThread - 1) / PerThread + bs - 1) / bs;
    Memset_Kernel<float> <<<n, bs, 0, (cudaStream_t)streamIdx >>> (p, val, n_orig);
}

void CAuxGPU::Memset_GPU(double* p, double val, long long n, long long streamIdx)
{
    int minGridSize = 0;
    int bs = 256;
    long long n_orig = n;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &bs, Memset_Kernel<double>, 0, (n + PerThread - 1) / PerThread);
    n = ((n + PerThread - 1) / PerThread + bs - 1) / bs;
    Memset_Kernel<double> <<<n, bs, 0, (cudaStream_t)streamIdx >>> (p, val, n_orig);
}

#endif